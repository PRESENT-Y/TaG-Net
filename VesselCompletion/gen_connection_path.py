from re import S
import SimpleITK as sitk 
import os 
import datetime
from networkx.algorithms.distance_measures import center
from networkx.classes.function import degree
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

from skimage.morphology import binary_dilation, ball

from numpy import polyfit, poly1d

from skimage.measure import label

import dijkstra3d

import scipy.ndimage as nd

from scipy import ndimage

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

import math 
import GraphConstruction.utils_base as butils
import GraphConstruction.utils_graph as gutils
import utils_multicl as mcutils
import utils_segcl as scutils
import json 

import csv

if __name__ == '__main__':

    data_path = './SampleData'
    patients = sorted(os.listdir(data_path))

    head_list = [0,5,6,11,17]  # head label 
    neck_list = [13, 14, 15, 16, 7, 12, 4, 10, 3, 9, 8, 2] # neck label
    patients=['002']

    
    for patient in patients:
        csv_file = os.path.join(data_path, patient, 'connection_paths.csv')
        headers = []
        headers.append('patient_name')
        with open(csv_file, 'w', newline='') as fp:
            writer = csv.DictWriter(fp, fieldnames=headers)
            writer.writeheader()
        content = []
        print(patient)
        start_time = datetime.datetime.now()
        connection_pair_intra_name = os.path.join(data_path, patient, 'connection_pair_intra')
        connection_pair_inter_name = os.path.join(data_path, patient, 'connection_pair_inter')
        connection_pairs = []
        if os.path.exists(connection_pair_intra_name):
            connection_pair_intra = butils.load_pairs(connection_pair_intra_name) # intra pairs
            connection_pairs.extend(connection_pair_intra)
        
        if os.path.exists(connection_pair_inter_name):
            connection_pair_inter = butils.load_pairs(connection_pair_inter_name) # inter pairs
            connection_pairs.extend(connection_pair_inter)
        
        if len(connection_pairs) != 0:
            content.append(patient)
            ori_img_path = os.path.join(data_path, patient, 'CTA.nii.gz')  # original image

            graph_edges = os.path.join(data_path, patient, 'CenterlineGraph_new')
            edges = butils.load_pairs(graph_edges)
            
            # labeled centerline from TaG-Net
            labeled_cl_name = os.path.join(data_path, patient, 'labeled_cl_new.txt')
            pc = np.loadtxt(labeled_cl_name)
            pc_label = pc[:,-1]
            label_list = np.unique(pc_label)

            # coordinates of start and end nodes
            for pair in connection_pairs:

                coordinates_pairs = scutils.gen_coordinates_pairs(pair, pc)
             
                connection_path_label = scutils.gen_start_end_label(pair, pc)
               
                # crop region
                distance_map = scutils.gen_crop_distance_map(ori_img_path, coordinates_pairs, pc)
             
                # connetion_path
                start_point = coordinates_pairs[1]
                end_point = coordinates_pairs[2]
                connection_path = dijkstra3d.dijkstra(distance_map, start_point, end_point)

                connection_path = [[path_i[0], path_i[1], path_i[2], connection_path_label] for path_i in connection_path]
                print(connection_path)
                coordinates_pairs.append(connection_path)

                content.append(coordinates_pairs)

            with open(csv_file, 'a', newline='') as fp:
                    writer = csv.writer(fp)
                    writer.writerow(content)
            end_time = datetime.datetime.now()
            print('time is {}'.format(end_time - start_time))
                




                   


        


   
