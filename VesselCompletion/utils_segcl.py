
from re import S
import SimpleITK as sitk 
import os 
import datetime
from networkx.classes.function import degree
from networkx.utils import heaps
import numpy as np
import pickle
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import networkx as nx
from itertools import combinations 
import scipy.spatial as spt
from sklearn.manifold import Isomap
import shutil
import warnings
warnings.filterwarnings('ignore')

from skimage.segmentation import active_contour
from skimage.filters import gaussian 

from numpy import polyfit, poly1d

import  skimage.measure as measure

import dijkstra3d

import scipy.ndimage as nd

from scipy import ndimage

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

import GraphConstruction.utils_base as butils
import GraphConstruction.utils_graph as gutils

from skimage.morphology import binary_dilation, binary_opening, binary_erosion, ball

from collections import Counter





def gen_coordinates_pairs(pair, pc):

    coordinates_pairs = []
    start_points = pc[int(pair[0]),0:3]  #(z,y,x)  [373, 110, 225, 16]   
    end_points = pc[int(pair[1]),0:3]  

    start_point = (int(start_points[0]), int(start_points[1]), int(start_points[2]))
    end_point = (int(end_points[0]), int(end_points[1]), int(end_points[2]))
    
    coordinates_pairs.append(pair)
    coordinates_pairs.append(start_point)
    coordinates_pairs.append(end_point)

    return coordinates_pairs


def gen_start_end_label(pair, pc):
    label_list = [7,12]
    label_pc = pc[:,-1]
    if label_pc[int(pair[0])] == label_pc[int(pair[1])]:
        start_end_label = label_pc[int(pair[0])]

    if label_pc[int(pair[0])] != label_pc[int(pair[1])]:
        if label_pc[int(pair[0])] in label_list:
            start_end_label = label_pc[int(pair[0])]
        elif label_pc[int(pair[1])] in label_list:
            start_end_label = label_pc[int(pair[1])]
        else:
            start_end_label = label_pc[int(pair[0])]

    return start_end_label



def get_26_neighboring_points(center_point, ori_img):
    hu_all = []
    N, M, D = ori_img.shape
    z, y, x = center_point
    neighboring_points = []
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx = dx + x
                ny = dy + y
                nz = dz + z
                if (nz<N) and (ny <M) and (nx<D) and (nz>0) and (ny>0) and (nx>0):  # boundary
                    hu = ori_img[nz, ny, nx]
                    neighboring_points.append([nz, ny, nx])
                    hu_all.append(hu)

    return neighboring_points, hu_all


def gen_neighbor_hu_uper_lower(start_point, end_point, ori_img):
    
    _,  hu_all_start = get_26_neighboring_points(start_point, ori_img)

    _,  hu_all_end = get_26_neighboring_points(end_point, ori_img)

    hu_uper = np.max([np.max(hu_all_start), np.max(hu_all_end)])
    hu_lower = np.min([np.min(hu_all_start), np.min(hu_all_end)])
    hu_average = np.mean([np.mean(hu_all_start), np.mean(hu_all_end)])

    return hu_uper, hu_lower, hu_average 


def gen_crop_region(distance, start_points, end_points, oriNumpy):

    oriNumpy_temp = oriNumpy.copy()
    N, M, D = oriNumpy.shape
    oriNumpy_temp[oriNumpy_temp != 0] = 0
    # center_crop 
    center_point = [int((start_points[0] + end_points[0])/2), \
                    int((start_points[1] + end_points[1])/2), \
                    int((start_points[2]+ end_points[2])/2)]

    sub = int(distance)
    if distance < 50:
        sub = int(distance * 2) 
    elif distance > 100:
        sub = int(distance/3) 
    
    for z in range(center_point[0]-sub,center_point[0]+sub):
        for y in range(center_point[1]-sub,center_point[1]+sub):
            for x in range(center_point[2]-sub,center_point[2]+sub):
                # prob_np_temp[z, y, x] = prob_np[z, y, x]
                if (z<N) and (y <M) and (x<D) and (z>0) and (y>0) and (x>0):  # boundary
                    oriNumpy_temp[z, y, x] = oriNumpy[z, y, x]

    return oriNumpy_temp


def gen_crop_distance_map(ori_img_path, coordinates_pairs, pc):
    oriNumpy, _, spacing = butils.load_itk_image(ori_img_path)
    # hu ranges
    start_point = coordinates_pairs[1]
    end_point = coordinates_pairs[2]
    hu_uper, hu_lower, hu_average = gen_neighbor_hu_uper_lower(start_point, end_point, oriNumpy)
    
    # crop regions
    square_distance_matrix = squareform(pdist(pc[:,0:3], metric='euclidean'))
    pair = coordinates_pairs[0]
    Sdistance = square_distance_matrix[int(pair[0]), int(pair[1])]

    oriNumpy_temp = gen_crop_region(Sdistance, start_point, end_point, oriNumpy)
    oriNumpy_temp[(oriNumpy_temp>hu_uper) | ( oriNumpy_temp <hu_lower)] = 0
    oriNumpy_temp[oriNumpy_temp != 0] = 1
    # distance_map 
    distance_map = ndimage.morphology.distance_transform_edt(oriNumpy_temp, spacing)
    zero_map = distance_map.copy()
    zero_map [ zero_map != 0 ] = 0
    max_value =np.max(distance_map)
    max_map = zero_map.copy()
    max_map[ max_map == 0 ] = max_value
    distance_map_temp = max_map - distance_map 

    return distance_map_temp


