import os
import os.path
from types import DynamicClassAttribute
from dgl.convert import graph
import torch
import json
import numpy as np
import sys
import torchvision.transforms as transforms
import pickle
import graph_utils.utils as gutils


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc, m

class VesselLabelTest():
    def __init__(self, root, num_points=2048, split='train', graph_dir=None, normalize=True, transforms=None):
        self.transforms = transforms
        self.num_points = num_points
        self.root = root  # point cloud path
        self.graph_dir = graph_dir 
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.normalize = normalize

        self.cat = {}
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}

        self.meta = {}
 
        # txt
        with open(os.path.join(self.root, 'train_test_split', 'train_list.txt'), 'r') as f:
            train_ids = set([str(d.strip().split('/')[0]) for d in f.readlines()])
        with open(os.path.join(self.root, 'train_test_split', 'val_list.txt'), 'r') as f:
            val_ids = set([str(d.strip().split('/')[0]) for d in f.readlines()])
        with open(os.path.join(self.root, 'train_test_split', 'test_list.txt'), 'r') as f:
            test_ids = set([str(d.strip().split('/')[0]) for d in f.readlines()])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))

            point_clouds_name = 'cl.txt'  
            graph_name = 'CenterlineGraph'


            if split == 'trainval':
                fns_file = [('/'.join([fn,point_clouds_name])) for fn in fns if fn in train_ids or fn in val_ids]
                gns = [('/'.join([fn,graph_name])) for fn in fns if fn in train_ids or fn in val_ids]
                split = 'train' 
            elif split == 'train':
                fns_file = [('/'.join([fn,point_clouds_name])) for fn in fns if fn in train_ids]
                gns = [('/'.join([fn,graph_name])) for fn in fns if fn in train_ids]
            elif split == 'val':
                fns_file = [('/'.join([fn,point_clouds_name])) for fn in fns if fn in val_ids]
                gns = [('/'.join([fn,graph_name])) for fn in fns if fn  in val_ids]
                split = 'train'  
            elif split == 'test':
                fns_file = [('/'.join([fn,point_clouds_name])) for fn in fns if fn in test_ids]
                gns = [('/'.join([fn,graph_name])) for fn in fns if fn in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            for fn_i in range(len(fns_file)):
                graph_path = os.path.join(graph_dir, '{}_process'.format(split), gns[fn_i])
                pc_path = os.path.join(dir_point, fns_file[fn_i])
                
                self.meta[item].append([pc_path, graph_path])

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = dict(zip(self.cat, range(len(self.cat))))
       
        self.seg_classes = {'NeckHeadVessel': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]}
      
        self.cache = {}
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, seg, cls, edge_list, data = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int64)


            data = np.loadtxt(fn[1][0]).astype(np.float32)

            edge_list = gutils.load_pairs(fn[1][1])  
           

            point_set = data[:, 0:3]
            # gutils.vis_graph_colorful(point_set, edge_list)
            # print(len(point_set))

            if self.normalize:
                point_set,_ = pc_normalize(point_set)
            seg = data[:, -1].astype(np.int64)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, seg, cls, edge_list, data)

     
        # idx_selected, sp_index = gutils.gen_certain_number(point_set, edge_list, self.num_points)
        
        self.num_points = len(seg) ## test  
        idx_selected, edge_list_certain = gutils.topology_preserving_sampling(edge_list, self.num_points, point_set)
        point_set_selected = point_set[idx_selected]
        seg_selected = seg[idx_selected]

        # edge_list_certain = gutils.gen_certain_graph_from_graph(edge_list, idx_selected)
        # g = gutils.graph_construction(len(seg_selected), edge_list_certain)
        # gutils.vis_graph_colorful(point_set_selected, edge_list_certain)


        if self.transforms is not None:
            point_set_selected = self.transforms(point_set_selected)


        return  point_set_selected, torch.from_numpy(seg_selected), torch.from_numpy(cls), edge_list_certain, data

    def __len__(self):
        return len(self.datapath)


