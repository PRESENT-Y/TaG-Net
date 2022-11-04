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
from data import VesselLabelTest
import utils.pytorch_utils as pt_utils
import data.data_utils as d_utils
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

parser = argparse.ArgumentParser(description='TaG-Net for Centerline Labeling Voting Evaluate')
parser.add_argument('--config', default='cfgs/config_test.yaml', type=str)
dir_output_test = './TaG-Net/TaG-Net-Test/results/centerline_label_graph/' 
dir_output_test_gt = './TaG-Net/TaG-Net-Test/results/centerline_label_graph/gt/'
if not os.path.exists(dir_output_test_gt):
    os.mkdir(os.path.join(dir_output_test))
    os.mkdir(os.path.join(dir_output_test_gt))

NUM_REPEAT = 1
NUM_VOTE = 2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in config['common'].items():
        setattr(args, k, v)

    test_transforms = transforms.Compose([ d_utils.PointcloudToTensor()])

    test_dataset = VesselLabelTest(root=args.data_root, 
                                   num_points=args.num_points, 
                                   split='test', 
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

    model =TaG_Net(num_classes=args.num_classes, 
                   input_channels=args.input_channels,
                   relation_prior=args.relation_prior, 
                   use_xyz=True)
    model.cuda()

    if args.checkpoint is not '':
        model.load_state_dict(torch.load(args.checkpoint))
        print('Load model successfully: %s' % (args.checkpoint))

    # evaluate
    PointcloudScale = d_utils.PointcloudScale(scale_low=0.87, scale_high=1.15) 
    model.eval()
    global_Class_mIoU, global_Inst_mIoU = 0, 0
    seg_classes = test_dataset.seg_classes
    seg_label_to_cat = {}  
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    for i in range(NUM_REPEAT):
        num = 0
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        for _, data in enumerate(test_dataloader, 0):
            name_file_path = test_dataset.datapath[num][1][0].split('/')[6] 
            num += 1
            print(num)

            points, target, cls, edges, points_ori = data
            with torch.no_grad():
                points, target = Variable(points), Variable(target)
            points, target = points.cuda(), target.cuda()

            batch_one_hot_cls = np.zeros((len(cls), 1)) 
            for b in range(len(cls)):
                batch_one_hot_cls[b, int(cls[b])] = 1
            batch_one_hot_cls = torch.from_numpy(batch_one_hot_cls)
            batch_one_hot_cls = Variable(batch_one_hot_cls.float().cuda())

            pred = 0

            new_points = Variable(torch.zeros(points.size()[0], points.size()[1], points.size()[2]).cuda())
            for v in range(NUM_VOTE):
                if v > 0:
                    new_points.data = PointcloudScale(points.data)
                    pred = model(points, batch_one_hot_cls, edges)
            pred /= NUM_VOTE

            _, pred_clss_tensor = torch.max(pred, -1) 

       
            pred_clss = pred_clss_tensor.cpu().squeeze(0).numpy()
            pred_clss = pred_clss.reshape(-1, 1)
            pred_out = np.concatenate([points.cpu()[0], pred_clss], axis=1)

            target_clss = target.cpu().squeeze(0).numpy()
            target_clss = target_clss.reshape(-1, 1)
            gt = np.concatenate([points.cpu()[0], target_clss], axis=1)

            path_out = os.path.join(dir_output_test, name_file_path, 'point_clouds.txt')
            path_out_gt = os.path.join(dir_output_test_gt, name_file_path, 'point_clouds.txt')
            if not os.path.exists(path_out):
                os.mkdir(os.path.join(dir_output_test, name_file_path))
                os.mkdir(os.path.join(dir_output_test_gt, name_file_path))
            np.savetxt(path_out, pred_out)
            np.savetxt(path_out_gt, gt)


if __name__ == '__main__':
    main()