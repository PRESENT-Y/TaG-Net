import numpy as np
import GraphConstruction.utils_graph as gutils
import GraphConstruction.utils_base as butils
import utils_multicl as mcutils
from itertools import combinations 
from collections import Counter
import networkx as nx

import warnings
warnings.filterwarnings('ignore')


def add_addition_edges(se_pairs_intra):
    
    if len(se_pairs_intra.shape)== 1:
        se_pairs_intra = [(se_pairs_intra[0], se_pairs_intra[1])]
    edges_addition = [[pair[0],pair[1]] for pair in se_pairs_intra]
    edges_addition = np.array( edges_addition)

    edge_list_merge = []
    if len(list(edges_addition)) != 0: 
        if (len(edges_addition.shape)== 1):
            edges_list = [(edges_addition[0],edges_addition[1])]
            edge_list_merge.append(edges_list)
        elif  len(edges_addition.shape) > 1:
            edges_list = [(edge[0], edge[1]) for edge in edges_addition]
            edge_list_merge.append(edges_list)
    if len(edge_list_merge) != 0:  
        edge_list_merge.append(edge_list)
        edge_list_merge = np.concatenate(edge_list_merge)
        nodes_degrees_array, G_nx= gutils.gen_degree_list_vis(edge_list_merge, len(pc)) 
    return nodes_degrees_array, G_nx

def find_start_end_nodes(label_pc, pair_to_check, G_nx):
    BA_label = [8,3,9]
    SA_label = [12]
    AO_label = [2]
    degree_one_list_all = []
    flag_1214 = 0
    degree_list = gutils.gen_degree_list(G_nx.edges(), len(label_pc))[0]
    for label in pair_to_check:
        idx_label = np.nonzero(label_pc == label)
        idx_label = list(idx_label[0])

        degree_one_list = [idx for idx in idx_label if degree_list[int(idx)] == 1]
        if label in BA_label:
            idxs_diff_label = butils.gen_idx_with_diff_label(G_nx, idx_label)
            if len(idxs_diff_label) != 0:
                degree_one_list = [idx for idx in idxs_diff_label + degree_one_list]
        
        if label in SA_label:
            if len(list(set(pair_to_check).intersection(set([12, 14]))))==2:
                print('{} has 12 and 14'.format(patient))
                if [1,12] in label_pairs:
                    flag_1214 = 1
                    break
                else:
                    idx_label_one = np.nonzero(label_pc == 1)
                    idx_label_one = list(idx_label_one[0])
                    label_one_one_list = [idx for idx in idx_label_one if degree_list[int(idx)] == 1]
                    degree_one_list = [idx for idx in (degree_one_list + label_one_one_list)]

        if label in AO_label:
            idx_label = np.nonzero(label_pc == label)
            idx_label = list(idx_label[0])   
            neighbors, nei_ori = butils.gen_neighbors_exclude_ori(idx_label, G_nx)
            degree_one_list = [idx for idx in nei_ori + degree_one_list] 

        degree_one_list_all.append(degree_one_list)  
    return degree_one_list_all, flag_1214   

def gen_start_end_pairs(degree_one_list_all, label_pc):
    # print(degree_one_list_all)
    degree_one_list_pairs = list(combinations(degree_one_list_all, 2))
    # print(degree_one_list_pairs)
    degree_one_list_pairs_temp =  degree_one_list_pairs.copy()
    for pair in degree_one_list_pairs_temp:
        # print(pair)
        label_one = label_pc[int(pair[0])]
        label_two = label_pc[int(pair[1])]
        if label_one == label_two:
            degree_one_list_pairs.remove(pair)
    return degree_one_list_pairs


def gen_connection_inter_pairs(degree_one_list_pairs, pc, connection_inter_pairs):
    # distance 
    sqr_dis_matrix = butils.cpt_sqr_dis_mat(pc[:,0:3])
    geo_distance_matrix = butils.cpt_geo_dis_mat(pc[:,0:3])
    geo_distance_matrix = sqr_dis_matrix 
    label_pair = degree_one_list_pairs.copy()
    if len(degree_one_list_pairs) != 0:
        if len(label_pair) >= 2:
            distance_pairs = []
            for label_pair_i in label_pair:
                distance_pairs.append(geo_distance_matrix[int(label_pair_i[0]), int(label_pair_i[1])])
            distance_pairs_min = np.min(distance_pairs)
            for label_pair_i in label_pair:
                if  geo_distance_matrix[int(label_pair_i[0]), int(label_pair_i[1])] == distance_pairs_min:
                    connection_inter_pairs.append(label_pair_i)
        else:
             connection_inter_pairs.append(label_pair)
    return connection_inter_pairs
    

def gen_label_to_check(label_list,pc, G_nx):
    label_to_check_list =[]
    for label in label_list:
        idx_label = np.nonzero(pc[:,-1] == label)
        idx_label = [int(idx) for idx in idx_label[0]]

        connected_components, _ = mcutils.gen_connected_components(idx_label, G_nx)
        connected_num = len(connected_components)
        if connected_num>1:
           label_to_check_list.append(label)
     
    label_to_check_list = [int(idx) for idx in label_to_check_list]
    return label_to_check_list

def gen_pair_min_distance(degree_one_list_pairs, geo_distance_matrix):

    distances = [geo_distance_matrix[pair[0], pair[1]] for pair in degree_one_list_pairs]
    distance_small_idx =  [i for i, distance in enumerate(distances) if distance == np.min(distances)]

    return distance_small_idx, np.min(distances)

def gen_region_pairs_most_common(degree_one_list_pairs,connected_num):
    degree_one_list_pairs_list =[]
    for pair in degree_one_list_pairs:
        for pair_i in pair:
            degree_one_list_pairs_list.append(pair_i)
    c = Counter(degree_one_list_pairs_list)
    c_most = Counter.most_common(c,2)
    if c_most[0][1] == c_most[1][1]:
        region_pairs = [degree_one_list_pairs]
    else:
        c_most = Counter.most_common(c,1)
        region_pairs  = []
        for c_most_i in c_most:
            region_pair = []
            c_most_i = c_most_i[0]
            for pair in degree_one_list_pairs:
                if c_most_i in pair:
                    region_pair.append(pair)
            region_pairs.append(region_pair)
    return region_pairs

def dul_remove(start_end_pair_sure):
    start_end_pair_sure_final = []
    for pair in start_end_pair_sure:
        pair = [pair[0], pair[1]]
        pair_reverse = [pair[1], pair[0]]
        if (pair not in start_end_pair_sure_final):
            if (pair_reverse not in start_end_pair_sure_final):
                start_end_pair_sure_final.append(pair)
        
        start_end_pair_sure = start_end_pair_sure_final.copy()
    return start_end_pair_sure

def gen_connection_intra_pairs(label_list, G_nx, pc):

    # degree
    degree_list = gutils.gen_degree_list(G_nx.edges(), len(pc))[0]

    # distance
    sqr_dis_matrix = butils.cpt_sqr_dis_mat(pc[:,0:3])
    geo_distance_matrix = butils.cpt_geo_dis_mat(pc[:,0:3])
    geo_distance_matrix = sqr_dis_matrix


    start_end_pair_sure = []
    for label in label_list:
        start_end_pair_sure_temp = []
        idx_label = list(np.nonzero(pc[:,-1] == label)[0])
        connected_components, G_nx_label = mcutils.gen_connected_components(idx_label, G_nx)
        connected_num = len(connected_components) ## segments number 
        if connected_num > 1:
            degree_one_list, G_label_G_map =  mcutils.gen_degree_one(idx_label, G_nx_label, degree_list)
            degree_one_list, same_region_pairs = mcutils.gen_all_idx_to_check(connected_components, G_label_G_map, degree_list, G_nx)
            degree_one_list_pairs = list(combinations(degree_one_list, 2))
            for pair in same_region_pairs:
                if pair in degree_one_list_pairs:
                    degree_one_list_pairs.remove(pair)
                pair_reverse = (pair[1], pair[0])
                if pair_reverse in degree_one_list_pairs:
                    degree_one_list_pairs.remove( pair_reverse)
                        
            if len(degree_one_list_pairs) == 1:
                # two region
                for pair in degree_one_list_pairs:
                    geo_min_distance =  geo_distance_matrix[pair[0], pair[1]]
                    sqr_min_distance = sqr_dis_matrix[pair[0], pair[1]]
                    # if min_distance < sum_label:
                    start_end_pair_sure_temp.append(degree_one_list_pairs[0])
            elif len(degree_one_list_pairs) == 2:
                distance_small_idx, min_distance = gen_pair_min_distance(degree_one_list_pairs, geo_distance_matrix)
                start_end_pair_sure_temp = [degree_one_list_pairs[idx] for idx in distance_small_idx]

                
            elif len(degree_one_list_pairs) > 2:
                # # 留下距离近的pair, 一旦找到一个连接，则删除此区域的其它点的连接
                
                iter_num = 0
                while (iter_num != connected_num-1) and len(degree_one_list_pairs) != 0:
                    
                    region_pairs = gen_region_pairs_most_common(degree_one_list_pairs,connected_num)
                    distance_small_idx = []
                    for region_pair in region_pairs:
                        region_min_idx, _ = gen_pair_min_distance(region_pair, geo_distance_matrix)
                    
                    region_min_pair = region_pair[region_min_idx[0]]
                    region_min_idx_in_all = [i for i, pair in enumerate(degree_one_list_pairs) if pair == region_min_pair]
                    distance_small_idx.append(region_min_idx_in_all)                           
                    if len(distance_small_idx) >= 1:
                        distance_small_idx = np.concatenate(distance_small_idx)
                    start_end_pair_sure_i = [degree_one_list_pairs[idx]for idx in distance_small_idx]
                    degree_one_list_pairs_cp = degree_one_list_pairs.copy()
                    for pair in start_end_pair_sure_i:
                        for pair_i in  degree_one_list_pairs_cp:
                            if pair[0] in pair_i:
                                if pair_i in degree_one_list_pairs:
                                    degree_one_list_pairs.remove(pair_i)
                            if pair[1] in pair_i:
                                if pair_i in degree_one_list_pairs:
                                    degree_one_list_pairs.remove(pair_i)
                    if start_end_pair_sure_i != []:
                        start_end_pair_sure_temp.append(start_end_pair_sure_i[0])
                    iter_num = iter_num + 1

                
        if start_end_pair_sure_temp != []:
            start_end_pair_sure.append(start_end_pair_sure_temp)
    if len(start_end_pair_sure) != 0:
        if len(start_end_pair_sure) > 1:
            start_end_pair_sure = np.concatenate(start_end_pair_sure)
        else:
            start_end_pair_sure = np.array(start_end_pair_sure)[0]   
    start_end_pair_sure  = dul_remove(start_end_pair_sure)

    return start_end_pair_sure

def gen_nodes_to_remove(wrong_pair, pc_label, G_nx, nodes_to_remove_all):
    pairs_to_del = []
    degree_list = gutils.gen_degree_list(G_nx.edges(), len(pc_label))[0]

    wrong_label = wrong_pair[0]
    idx_label = np.nonzero(pc_label == wrong_label)[0]
    idx_neighbors, nei_ori = butils.gen_neighbors_exclude_ori(idx_label, G_nx)
    for i, idx_neighbor in enumerate(idx_neighbors):
        if pc_label[idx_neighbor] == wrong_pair[1]:
            pair_to_del = (idx_neighbor, nei_ori[i])
            # G_nx.remove_edge(idx_neighbor, nei_ori[i])
            pairs_to_del.append(pair_to_del)
            print(pair_to_del)
    
            degree_list_three_idx = [idx_i for idx_i in idx_label if degree_list[idx_i]==3]
            path_pairs = [(idx_i, nei_ori[i]) for idx_i in degree_list_three_idx]
            for path_pair in path_pairs:
                print(path_pair)
                if nx.has_path(G_nx, path_pair[0], path_pair[1]):
                    pair_path = nx.shortest_path(G_nx, path_pair[0], path_pair[1])
                    if len(pair_path)/len(idx_label) < 1/10:
                        node_to_remove = [node for node in pair_path if node != path_pair[0]]
                        nodes_to_remove_all.append(node_to_remove)
    return nodes_to_remove_all