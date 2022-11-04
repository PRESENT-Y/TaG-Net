from networkx.algorithms.shortest_paths.generic import shortest_path_length
from networkx.classes.function import degree
import numpy  as  np
import pickle
import os
import sys
import SimpleITK as sitk
import networkx as nx 

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import GraphConstruction.utils_graph as gutils
import utils_segcl as scutils

import GraphConstruction.utils_base as butils
import itertools
from collections import Counter
from itertools import combinations


def gen_anatomical_graph(label_list):
    # prior
    # BCT(1),R-CCA(3),R-ICA(4),R-VA(7),BA(8),L-CCA(9), L-ICA(10),L-VA(12),R-SCA(13),L-SCA(14),L-ECA(15), R-ECA(16)

    # easy to hard 
    # 1, 8, 13, 14, 15, 16, 7, 12, 4, 10, 3, 9, 2
    # 5, 0, 17, 6, 11
    # L-VA(12) -> L-SCA(14) & BA(8)   
    # R-VA(7) -> R-SCA(13) & BA(8)

    # L-CCA(9) --> L-ICA(10) & L-ECA(15)
    # R-CCA(3) --> R-ICA(4) & R-ECA(16)

    # L-ECA(15) -->  L-CCA(9) 
    # R-ECA(16) -->  R-CCA(3)

    # L-ICA(10) --> L-PCA(17) & L-MCA(11) & ACA(5)
    # R-ICA(4) --> R-PCA(0) & R-MCA(6) & ACA(5)

    # L-PCA(17) --> BA(8) & L-ICA(10)
    # R-PCA(0) --> BA(8) & R-ICA(4)

    # ACA(5) --> L-ICA(10) & R-ICA(4)
    
    # R-MCA(6) --> R-ICA(4) 
    # L-MCA(11) --> L-ICA(10)

    # BCT(1) --> AO(2) & L-SCA(14) & L-CCA(9)
    # AO(2) --> R-SCA(13) & R-CCA(3) & BCT(1) & L-VA(12) (special)
    label_list_edges = [(0,4),(0,8),(1,2), (1,9), (1,14), (1,12),\
                        (2,3), (2,13), (3,4), (3,16), (4,5),(4,6),\
                        (5,10), (7,8), (7,13), (8,12), (8,17),\
                        (9,10),(9,15), (10,11), (10,17), (12,14)]
    
    gt_label_graph =  butils.gen_G_nx(len(label_list),label_list_edges)
    return gt_label_graph


def gen_wrong_connected_exist_label(label_list, label_pc, G_nx):
    flag = 0
    lack_flag = 0
    label_pairs = []
    for label in label_list:
        idx_label = np.nonzero(label_pc == label)
        idx_label = [int(idx) for idx in idx_label[0]]
        idx_neighbor = butils.gen_neighbors_exclude(idx_label, G_nx)
        idx_neighbor_label = label_pc[idx_neighbor]
        for idx in idx_neighbor_label:
            label_pair = [label, idx]
            label_pair_reverse = [idx, label]
            if (label_pair not in label_pairs) and (label_pair_reverse not in label_pairs):
                label_pairs.append(label_pair)

    anatomical_graph = gen_anatomical_graph(label_list)
    right_label_pair = anatomical_graph.edges()
    right_label_pair = [pair for pair in right_label_pair]
   
    head_list = [5, 6, 11, 0, 17]
    lack_pairs = []
   
    check_pairs_exist = []
    for pair in label_pairs:
        pair = (int(pair[0]), int(pair[1]))
        pair_reverse = (int(pair[1]), int(pair[0]))
        if (pair not in right_label_pair) and (pair_reverse not in right_label_pair):
            if (pair[1] not in head_list) and (pair[0] not in head_list):
                check_pairs_exist.append(pair)
                flag = 1
    if (1,12) in right_label_pair:
       right_label_pair.remove((1,12))
    for pair in right_label_pair:
        pair = [int(pair[0]), int(pair[1])]
        pair_reverse = [int(pair[1]), int(pair[0])]
        if (pair not in label_pairs) and (pair_reverse not in label_pairs):
            if (pair[1] not in head_list) and (pair[0] not in head_list):
                lack_pairs.append(pair)
                lack_flag = 1
    return flag, check_pairs_exist, lack_flag, lack_pairs, label_pairs


def gen_connected_components(point_idx_label, G_nx):

    selected_edges = gen_selected_point_graph(point_idx_label, G_nx)
    # sub graph 
    G_nx_label = butils.gen_G_nx(len(point_idx_label),selected_edges)
    connected_components = list(nx.connected_components(G_nx_label))

    return connected_components, G_nx_label


    
def gen_selected_point_graph(point_idx_label, G_nx):

    edge_list_label = []
    for i, idx in enumerate(point_idx_label):
        # print(G_nx.edges(idx))
        for edge in G_nx.edges(idx):
            if edge[1] in point_idx_label:
                edge_list_label.append(edge)

    idx_map= {j: i for i, j in enumerate(point_idx_label)}
    edge_unordered = np.array(edge_list_label)
    edges = np.array(list(map(idx_map.get, edge_unordered.flatten())),
            dtype=np.int32).reshape(edge_unordered.shape)

    return edges


def gen_degree_one(idx_label, G_nx_label, degree_list):

    G_G_label_map = {j:i for i, j in enumerate(idx_label)}
    G_label_G_map = {i:j for i, j in enumerate(idx_label)}
    idx_label_mapped_to_G_label = [G_G_label_map.get(i) for i in idx_label]
    degree_list_G_label = gutils.gen_degree_list(G_nx_label.edges(), len(idx_label))
    idx_degree_one = [i for i, degree in enumerate(degree_list_G_label[0]) if degree == 1]
    idx_in_G_label = [G_label_G_map.get(i) for i in idx_degree_one]
    degree_G_one = [idx for idx in  idx_in_G_label if degree_list[idx] == 1]
    degree_G_one = sorted(degree_G_one)
    return degree_G_one, G_label_G_map


def gen_all_idx_to_check(connected_components, G_label_G_map, degree_list, G_nx):
    degree_one_list = []
    same_region_pairs = []
    for connected_i in connected_components:
        connected_i = [i for i in connected_i]
        dx_in_G_label = [G_label_G_map.get(i) for i in connected_i]
        degree_one_connected_i = [i for i in dx_in_G_label if degree_list[i] == 1]
        if len(connected_i) == 1:
            if dx_in_G_label[0] not in degree_one_list:
               degree_one_list.append(dx_in_G_label[0])
        if len(degree_one_connected_i) == 1:
            for idx in degree_one_connected_i:
                if idx not in degree_one_list:
                   degree_one_list.append(idx)
        if len(degree_one_connected_i) >= 2:
            connected_i_pairs = list(combinations(degree_one_connected_i,2))
            paths_length = []
            for pair in connected_i_pairs:
                if nx.has_path(G_nx, pair[0], pair[1]):
                    path_length = nx.shortest_path_length(G_nx, pair[0], pair[1])
                    paths_length.append(path_length)
            max_idx = [idx for idx, length in enumerate(paths_length) if length == np.max(paths_length)]
            rest_connected_i_pairs = connected_i_pairs[max_idx[0]]
            same_region_pairs.append(rest_connected_i_pairs)
            for idx in rest_connected_i_pairs:
                if idx not in degree_one_list:
                   degree_one_list.append(idx)
    
    return degree_one_list, same_region_pairs
