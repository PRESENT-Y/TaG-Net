# -*- coding: utf-8 -*-
# @time:2021.08
# @Author:PRESENT

import os
import SimpleITK as sitk 
import numpy as np
from skimage import morphology
import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utils_base as butils
import utils_graph as gutils 
import scipy.ndimage as nd

def gen_cl_img(seg_path):
    
    segNumpy, segOrigin, segSpacing = butils.load_itk_image(seg_path)
    # segNumpy[segNumpy==100] = 0
    # segNumpy[segNumpy==101] = 0
    # segNumpy[segNumpy!=0] = 1
    # segNumpy = nd.binary_fill_holes(segNumpy)
    clNumpy= morphology.skeletonize_3d(segNumpy)
    # clNumpy[clNumpy!=0]=1
    butils.save_itk(clNumpy, segOrigin, segSpacing, cl_save_name)

    return clNumpy


def gen_img_to_pc(clNumpy, pc_save_name):

    cls_idx = np.nonzero(clNumpy == 1)
    pc = np.transpose(cls_idx)
    np.savetxt(pc_save_name, pc)

    return pc


if __name__ =='__main__':
    
    data_path = './SampleData'
    r_thresh = 1.75
    patients = sorted(os.listdir(data_path))
    
    patients = ['001']

    for patient in patients:
        # input
        seg_path = os.path.join(data_path, patient, 'seg.nii.gz')

        # output
        cl_save_name = os.path.join(os.path.dirname(seg_path), 'cl.nii.gz')
        pc_save_name = os.path.join(data_path, patient,'cl.txt')
        graph_save_name = os.path.join(data_path, patient, 'CenterlineGraph')
        
        # generate centerline
        if not os.path.exists(cl_save_name):
           clNumpy = gen_cl_img(seg_path)
        else:
           clNumpy, clOrigin, clSpacing = butils.load_itk_image(cl_save_name)

        # image to point set
        if not os.path.exists(pc_save_name):
            pc = gen_img_to_pc(clNumpy, pc_save_name)
        else:
            pc = np.loadtxt(pc_save_name)
       
        # centerline vascular graph construction
        if not os.path.exists(graph_save_name):
            edges =  gutils.gen_pairs(pc, r_thresh)
            butils.dump_pairs(graph_save_name, edges)
        else:
            edges = butils.load_pairs(graph_save_name)

        # remove isolated nodes
        graph_path_length_thresh = 1
        new_pc, new_edges = gutils.gen_isolate_removal(pc, edges, graph_path_length_thresh)
        pc_save_name = os.path.join(data_path, patient,'cl.txt')
        graph_save_name = os.path.join(data_path, patient, 'CenterlineGraph')
        np.savetxt(pc_save_name, new_pc)
        butils.dump_pairs(graph_save_name, new_edges)