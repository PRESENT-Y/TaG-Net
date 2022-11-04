import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))
import torch
import torch.nn as nn
from torch.autograd import Variable
import pytorch_utils as pt_utils
from pointnet2_modules import PointnetSAModule, PointnetFPModule, PointnetSAModuleMSG
import numpy as np
from .graph_module import *

class TaG_Net(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, num_classes, input_channels=0, relation_prior=1, use_xyz=True):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        c_in = input_channels
        self.SA_modules.append(     # 0 4096 96*2
            PointnetSAModuleMSG(
                npoint=4096,
                radii=[0.1],
                nsamples=[48],
                mlps=[[c_in, 96]],
                gcns=[96, 192, 96],
                first_layer=True,
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        c_out_0 = 96 * 2

        c_in = c_out_0
        self.SA_modules.append(    # 1 2048 192*2
            PointnetSAModuleMSG(
                npoint=2048,
                radii=[0.2],
                nsamples=[64],
                mlps=[[c_in, 192]],
                gcns=[192, 384, 192],
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
       

        c_out_1 = 192*2

        c_in = c_out_1
        self.SA_modules.append(    # 2 1024 384*2
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.4],
                nsamples=[80],
                mlps=[[c_in, 384]],
                gcns=[384, 768, 384],
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        c_out_2 = 384*2

        c_in = c_out_2
        self.SA_modules.append(    # 3 512 768*2
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.8],
                nsamples=[96],
                mlps=[[c_in, 768]],
                gcns=[768, 1536, 768],
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        c_out_3 = 768*2
        
        self.SA_modules.append(   # 4  global pooling 128
            PointnetSAModule(
                nsample = 16,
                mlp=[c_out_3, 128], use_xyz=use_xyz
            )
        )
        global_out = 128    

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[384 + input_channels, 128, 128])) # 3
        self.FP_modules.append(PointnetFPModule(mlp=[768 + c_out_0, 384, 384])) # 2
        self.FP_modules.append(PointnetFPModule(mlp=[1536 + c_out_1, 768, 768])) # 1
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3+c_out_2, 1536, 1536])) # 0


        self.FC_layer = nn.Sequential(
            pt_utils.Conv1d(128+global_out+1, 128, bn=True), nn.Dropout(),
            pt_utils.Conv1d(128, num_classes, activation=None)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, cls, edge_list):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)graph_related
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)
        
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, edge_list = self.SA_modules[i](l_xyz[i], l_features[i],  edge_list)
            if li_xyz is not None:
                random_index = np.arange(li_xyz.size()[1])
                np.random.shuffle(random_index)
                #edge reindex
                idx_map={j:i for i, j in enumerate(random_index)}
                edge_unordered = np.array(edge_list)
                edges = np.array(list(map(idx_map.get, edge_unordered.flatten())), dtype=np.int32).reshape(edge_unordered.shape)
                edges = [(edge[0],edge[1]) for edge in edges]
                edge_list = edges
                li_xyz = li_xyz[:, random_index, :]
                li_features = li_features[:, :, random_index]

            l_xyz.append(li_xyz)
            l_features.append(li_features)

        
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1 - 1] = self.FP_modules[i](
                l_xyz[i - 1 - 1], l_xyz[i - 1], l_features[i - 1 - 1], l_features[i - 1]
            )
        
        cls = cls.view(-1, 1, 1).repeat(1, 1, l_features[0].size()[2])
        l_features[0] = torch.cat((l_features[0], l_features[-1].repeat(1, 1, l_features[0].size()[2]), cls), 1)

        temp = self.FC_layer(l_features[0]).transpose(1, 2).contiguous()
        return temp
