import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import os
from torchvision import transforms
from models import TaG_Net as TaG_Net
from data import VesselLabel
import utils.pytorch_utils as pt_utils
import data.data_utils as d_utils
import  graph_utils.utils as gutils
import argparse
import random
import yaml
import pptk
import warnings
warnings.filterwarnings('ignore')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
import  shutil
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

parser = argparse.ArgumentParser(description='TaG-Net for Centerline Labeling Training')
parser.add_argument('--config', default='cfgs/config_train.yaml', type=str)

def main():
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print("\n**************************")
    for k, v in config['common'].items():
        setattr(args, k, v)
        print('\n[%s]:' % (k), v)
    print("\n**************************\n")

    try:
        os.makedirs(args.save_path)

    except OSError:
        pass
    train_transforms = transforms.Compose([d_utils.PointcloudToTensor()])
    test_transforms = transforms.Compose([d_utils.PointcloudToTensor()])

    train_dataset = VesselLabel(root=args.data_root, 
                               num_points=args.num_points, 
                               split='train',
                               graph_dir = args.graph_dir, 
                               normalize=True,
                               transforms=train_transforms)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        pin_memory=True
    )

    global test_dataset
    test_dataset = VesselLabel(root=args.data_root, 
                                num_points=args.num_points, 
                                split='val', 
                                graph_dir = args.graph_dir,
                                normalize=True, 
                                transforms=test_transforms)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        pin_memory=True
    )


    ### model
    model = TaG_Net(num_classes=args.num_classes, 
                    input_channels=args.input_channels,
                    relation_prior=args.relation_prior, 
                    use_xyz=True)

    model.cuda()
    optimizer = optim.Adam(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)

    lr_lbmd = lambda e: max(args.lr_decay ** (e // args.decay_step), args.lr_clip / args.base_lr)
    bnm_lmbd = lambda e: max(args.bn_momentum * args.bn_decay ** (e // args.decay_step), args.bnm_clip)
    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd)
    bnm_scheduler = pt_utils.BNMomentumScheduler(model, bnm_lmbd)

    if args.checkpoint is not '':
        model.load_state_dict(torch.load(args.checkpoint))
        print('Load model successfully: %s' % (args.checkpoint))

    criterion = nn.CrossEntropyLoss() 
    num_batch = len(train_dataset) / args.batch_size

    # train
    train(train_dataloader, test_dataloader, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch)


def train(train_dataloader, test_dataloader, model, criterion, optimizer, lr_scheduler, bnm_scheduler, args, num_batch):
    PointcloudScaleAndTranslate = d_utils.PointcloudScaleAndTranslate()  
    global Class_mIoU, Inst_mIoU
    Class_mIoU, Inst_mIoU = 0, 0
    batch_count = 0
    model.train()
    for epoch in range(args.epochs):
        for i, data in enumerate(train_dataloader, 0):
            if lr_scheduler is not None:
                lr_scheduler.step(epoch)
            if bnm_scheduler is not None:
                bnm_scheduler.step(epoch - 1)
            
            points, target, cls, edges, points_ori = data


            print('train_true: Labels: {}'.format(np.unique(target)))
            points, target = points.cuda(), target.cuda()
            points, target = Variable(points), Variable(target)
            points.data = PointcloudScaleAndTranslate(points.data)

            optimizer.zero_grad()

            batch_one_hot_cls = np.zeros((len(cls), 1)) 
            for b in range(len(cls)):
                batch_one_hot_cls[b, int(cls[b])] = 1
            batch_one_hot_cls = torch.from_numpy(batch_one_hot_cls)
            batch_one_hot_cls = Variable(batch_one_hot_cls.float().cuda())

            pred = model(points, batch_one_hot_cls, edges)
            _, pred_clss_tensor = torch.max(pred, -1)
            print('train_pred: Labels: {}'.format(np.unique(pred_clss_tensor)))
            pred = pred.view(-1, args.num_classes)
            target = target.view(-1, 1)[:, 0]
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            print('[epoch %3d: %3d/%3d] \t train loss: %0.6f \t lr: %0.5f' % (
            epoch + 1, i, num_batch, loss.data.clone(), lr_scheduler.get_lr()[0]))
            batch_count += 1

            if (epoch < 3 or epoch > 10) and args.evaluate and batch_count % int(args.val_freq_epoch * num_batch) == 0:
                validate(test_dataloader, model, criterion, args, batch_count)


def validate(test_dataloader, model, criterion, args, iter):
    global Class_mIoU, Inst_mIoU, test_dataset
    model.eval()

    seg_classes = test_dataset.seg_classes
    shape_ious = {cat: [] for cat in seg_classes.keys()}
    seg_label_to_cat = {}  
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    losses = []
    for _, data in enumerate(test_dataloader, 0):
        points, target, cls, edges, point_ori = data
        print('val_true: Labels: {}'.format(np.unique(target)))
        with torch.no_grad():
            points, target = Variable(points), Variable(target)  
        points, target = points.cuda(), target.cuda()

        batch_one_hot_cls = np.zeros((len(cls), 1)) 
        for b in range(len(cls)):
            batch_one_hot_cls[b, int(cls[b])] = 1
        batch_one_hot_cls = torch.from_numpy(batch_one_hot_cls)
        batch_one_hot_cls = Variable(batch_one_hot_cls.float().cuda())

        pred = model(points, batch_one_hot_cls, edges)
        _, pred_clss_tensor = torch.max(pred, -1)
        print('val_pred: Labels: {}'.format(np.unique(pred_clss_tensor)))
        loss = criterion(pred.view(-1, args.num_classes), target.view(-1, 1)[:, 0])
        losses.append(loss.data.clone())
        pred = pred.data.cpu()
        target = target.data.cpu()
        pred_val = torch.zeros(len(cls), args.num_points).type(torch.LongTensor)
        for b in range(len(cls)):
            cat = seg_label_to_cat[target[b, 0].item()]  
            logits = pred[b, :, :] 
            pred_val[b, :] = logits[:, seg_classes[cat]].max(1)[1] + seg_classes[cat][0]

        for b in range(len(cls)):  
            segp = pred_val[b, :]  
            segl = target[b, :] 
            cat = seg_label_to_cat[segl[0].item()]  

            part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
            for l in seg_classes[cat]:
                if torch.sum((segl == l) | (segp == l)) == 0:  
                    part_ious[l - seg_classes[cat][0]] = 1.0  #
                else:
                    part_ious[l - seg_classes[cat][0]] = float(torch.sum((segl == l) & (segp == l))) / float(
                        torch.sum((segl == l) | (segp == l)))
            shape_ious[cat].append(part_ious)

    instance_ious = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            instance_ious.append(np.mean(iou))

    # each cls iou
    cls_ious = {l: [] for l in seg_classes[cat]}
    for cat in shape_ious.keys():
        for ious in shape_ious[cat]:
            for i in range(len(ious)):
                cls_ious[i].append(ious[i])

    for cls_l in sorted(cls_ious.keys()):
        print('************ %s: %0.6f' % (cls_l, np.array(cls_ious[cls_l]).mean()))

    print('************ Test Loss: %0.6f' % (np.array(losses).mean()))
    print('************ Instance_mIoU: %0.6f' % (np.mean(instance_ious))) 

    if np.mean(instance_ious) > Inst_mIoU:
        if np.mean(instance_ious) > Inst_mIoU:
            Inst_mIoU = np.mean(instance_ious)
        torch.save(model.state_dict(),
                   '%s/tag_net_iter_%d_ins_%0.6f_4r.pth' % (args.save_path, iter, np.mean(instance_ious)))
    model.train()


if __name__ == "__main__":
    main()
