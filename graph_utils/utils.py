import os 
import SimpleITK as sitk 
import numpy as np
import random
import glob
import SimpleITK as sitk

import dgl

import networkx as nx
import datetime
import torch
import pickle

from collections import Counter
from sklearn.manifold import Isomap
from scipy.spatial.distance import cdist

# import pointnet2_utils 

import itertools
import graph_utils.utils_sampling as sampleutils
import GraphConstruction.utils_base as butils
# from mayavi import mlab


def  vis_graph_colorful(new_points, edges_list):

    color_map = np.array([
        [160, 32, 240],  # 1 # purple
        [0, 255, 0],  # 2 # green
        [46, 139, 87],  # 9 # see green
        [0, 0, 255],  # 3 # blue
        [255, 255, 0],  # 4 # Yellow
        [0, 255, 255],  # 5 # Cyan
        [255, 0, 0],  # 6 # red
        [255, 127, 0],  # 7 # dark orange
        [255, 0, 255],  # 8  # Magenta
        [46, 139, 87],  # 9 # see green
        [0, 255, 0],  # 2 # green
        [205, 205, 0],  # 10 # yellow 4
        [0, 191, 255],  # 11 #deep sky blue
        [188, 143, 143],  # 12 # RosyBrown
        [255, 20, 147],  # 13 deep pink
        [255, 181, 197],  # 14 # pink
    ]) / 255.

    pc_num = len(new_points)
    g = dgl.DGLGraph()
    g.add_nodes(pc_num)
    src, dst = tuple(zip(*edges_list))
    g.add_edges(dst, src)
    nx_G = g.to_networkx().to_undirected()
    degrees = nx_G.degree()


    nodes_degrees_array = np.full((1, pc_num), -1, dtype=int)
    for degree in degrees:
        node = degree[0]
        degree = degree[1]
        nodes_degrees_array[0, node] = degree
    nodes_degrees_list = nodes_degrees_array.tolist()[0]
    degree_list = np.unique(nodes_degrees_list)
    print(degree_list)

    mlab.figure(1, bgcolor=(1, 1, 1))
    mlab.clf()
    points = new_points
    pts=[]
    for degree_i in degree_list:
        node_index = np.nonzero(nodes_degrees_array == degree_i)[1]
        print("degree {} has {} nodes".format(degree_i, len(node_index)))
        pts = mlab.points3d(points[node_index, 0], points[node_index, 1], points[node_index, 2], color=color_map(1), scale_factor=3)
    
    pts = mlab.points3d(points[:, 0], points[:, 1], points[:, 2], color=(1, 0, 0), scale_factor=3)
    # pts2 = mlab.points3d(point_set[:, 0], point_set[:, 1], point_set[:, 2], color=(0, 0, 1), scale_factor=1.5)
    pts.mlab_source.dataset.lines = np.array(nx_G.esssdges())
    tube = mlab.pipeline.tube(pts, tube_radius=0.08)
    mlab.pipeline.surface(tube, color=(0, 1, 0))
    points_corner_min = [0, 0, 0]
    points_corner_max = [700, 700, 700]
    points_corner_max = np.vstack([points_corner_max, points_corner_min])
    mlab.points3d(points_corner_max[:, 0], points_corner_max[:, 1], points_corner_max[:, 2], color=(1, 1, 1),
                  scale_factor=0.7)
    mlab.show()

def load_pairs(path):
    """
    
    :param path: load path
    :return: graph pairs [(),(),()]
    """
    with open(path, 'rb') as handle:
        pairs = pickle.load(handle)
    return pairs

def dump_pairs(path, pairs):
    """
    
    :param path: save path
    :param pairs: graph pairs to save
    :return: None
    """
    with open(path, 'wb') as handle:
        pickle.dump(pairs, handle)

def graph_construction(point_num, edge_list):

    g = dgl.DGLGraph()
    g.add_nodes(point_num)
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    g.add_edges(dst, src)
    return g

def edge_pair_to_nodes(edge_list_all):
    nodes = []
    edge_list_all
    for edge_pair in edge_list_all:
        nodes.append(edge_pair[0])
        nodes.append(edge_pair[1])
    nodes = np.unique(nodes)
    return nodes

def edge_pair_to_array(pairs):

    edge_pair_array = np.full((2,len(pairs)),-1, dtype=int)
    pairs = sorted(pairs)
    i_pair = 0
    for pair in pairs:
        if i_pair < len(pairs):
            edge_pair_array[0,i_pair] = pair[0]
            edge_pair_array[1,i_pair] = pair[1]
            i_pair = i_pair + 1
    return edge_pair_array

def gen_degree_list(edge_list, point_num):

    # graph construction
    g = graph_construction(point_num, edge_list)
    
    # compute degrees
    nx_G = g.to_networkx().to_undirected()
    degrees = nx_G.degree()

    nodes_degrees_array = np.full((1,point_num),-1, dtype=int)
    for degree in degrees:
        node = degree[0]
        degree = degree[1]
        nodes_degrees_array[0,node] = degree
    # nodes_degrees_list = nodes_degrees_array.tolist()[0]

    return nodes_degrees_array


def gen_non_two_connection_idx(point_set, nodes_degrees_array, degree_list):

    # degree is not 2
    nodes_non_2 = []
    point_non_2_part = []
    for degree_num in degree_list:
        if degree_num != 2:
            node_index = np.nonzero(nodes_degrees_array == degree_num)[1]
            point_non_2_part = point_set[node_index]  # 
            nodes_non_2.append(node_index.tolist())
      
    nodes_non_2_index = np.concatenate(nodes_non_2).tolist()

    return nodes_non_2_index, point_non_2_part

def gen_more_than_two_connection_idx(nodes_degrees_array, degree_list):

    # degree is not 2
    nodes_non_2 = []
    # point_non_2_part = []
    for degree_num in degree_list:
        if degree_num > 2:
            node_index = np.nonzero(nodes_degrees_array == degree_num)[1]
            # point_non_2_part = point_set[node_index]  # 
            nodes_non_2.append(node_index.tolist())
      
    nodes_non_2_index = np.concatenate(nodes_non_2).tolist()

    # return nodes_non_2_index, point_non_2_part
    return nodes_non_2_index

def add_one_connection_another_idx(nodes_degrees_array, edge_list, nodes_non_2_index):

    node_index_1 = np.nonzero(nodes_degrees_array == 1)[1]
    node_index_1 = node_index_1.tolist()

    for edge_i in edge_list:
        if edge_i[0] in node_index_1:
            if edge_i[1] not in nodes_non_2_index:
                nodes_non_2_index.append(edge_i[1])
          #  point_part_non_2.append(point_set[edge_i[1],:])

        if edge_i[1] in node_index_1:
            if edge_i[0] not in nodes_non_2_index:
                nodes_non_2_index.append(edge_i[0])

    return nodes_non_2_index

def gen_tri_pair(nodes_more_than_two, nodes_array, pairs):
    Tri_pair = []
    for node in nodes_more_than_two:
        next_nodes = []
        index_node = np.array(np.where(nodes_array[0]==node))
        index_node = index_node[0]
        # index_node = np.squeeze(np.array(np.where(nodes_array[0]==node)))  # index  
        index_node = index_node.tolist() 
        for index_node_i in index_node:
            next_node = nodes_array[1][index_node_i]   # next node
            next_nodes.append(next_node)

        pairs_to_check = list(itertools.combinations(next_nodes, 2))
        for pair_to_check in pairs_to_check:
            if pair_to_check in pairs:
                Tri_pair.append((node, pair_to_check[0]))
                Tri_pair.append((node, pair_to_check[1]))
                Tri_pair.append(pair_to_check)
    return Tri_pair

def nodes_need_to_replace(edge_list_array, nodes_certain):
    nodes_need_to_change = []
    edge_list_array_one_temp = np.transpose(edge_list_array)  # transe into Npoints X
    edge_list_certain_nodes = edge_list_array_one_temp[nodes_certain]
    edge_list_certain_nodes = np.transpose(edge_list_certain_nodes)
    nodes_in_edge_list_certain = []
    for i in edge_list_certain_nodes:
        nodes_in_edge_list_certain.append(i.tolist())
    nodes_in_edge_list_certain = np.concatenate(nodes_in_edge_list_certain)
    nodes_in_edge_list_certain = np.unique(nodes_in_edge_list_certain)

    for node_i in nodes_in_edge_list_certain:
        if (node_i != -1) and (node_i not in nodes_certain):
            nodes_need_to_change.append(node_i)
    return nodes_need_to_change

def gen_certein_edge_list(point_set_N, edge_list_all, nodes_certain):
    edge_list_all_ori = edge_list_all.copy()

    # edge_list_all_temp will change with edge_list_all 
    edge_list_all_temp = np.transpose((edge_list_all))  # edge_list_all (6, 8646); edge_list_all_temp (8646, 6)
    nodes_certain_temp = sorted(nodes_certain)  # 4096
    nodes_certain_temp = [int(i) for i in nodes_certain_temp]

    for i in range(point_set_N):
        if i in nodes_certain:
            pass
        else:
            node_i = i

            # print(node_i)
            row_num = len(edge_list_all) # 6
            nodes_connected_i = np.unique(edge_list_all_temp[node_i]).tolist()  # nodes connected list
            if -1 in nodes_connected_i:
                nodes_connected_i.remove(-1)  # remove  -1
            if nodes_connected_i != []:
                for row_i in range(0, row_num):
                    edge_list_i = edge_list_all[row_i, :].tolist()  # next line
                    if node_i in edge_list_i:  # if in then return index  (maybe not only one index)
                        i_next_idxs_i = [i_next_idx_i for i_next_idx_i,x in enumerate(edge_list_i) if x == node_i]  # indexs
                        # i_next_idx_i = edge_list_i.index(node_i)  # index of node_i

                        # how to replace is a question, make sure iteratively
                        for i_next_idx_i in i_next_idxs_i:
                            nodes_connected_i_temp = nodes_connected_i.copy()
                            # if i_next_idx_i in nodes_certain:
                            if i_next_idx_i in nodes_connected_i:   # remove  i_next_idx_i  (i_next_idx_i is same with the connected node)
                                nodes_connected_i_temp.remove(i_next_idx_i)
                            # if i_next_idx_i in nodes_certain:
                            if len(nodes_connected_i_temp ) == 1:
                                if nodes_connected_i_temp[0] in nodes_certain_temp:
                                   edge_list_all[row_i, i_next_idx_i] = nodes_connected_i_temp[0]
                                # else:
                                #     edge_list_all[row_i, i_next_idx_i] = -1
                            elif len(nodes_connected_i_temp ) >= 2:
                                if nodes_connected_i_temp[1] not in edge_list_all[:,i_next_idx_i]:
                                   edge_list_all[row_i, i_next_idx_i] = nodes_connected_i_temp[1]
                                if nodes_connected_i_temp[0] not in edge_list_all[:,i_next_idx_i]:
                                   edge_list_all[row_i, i_next_idx_i] = nodes_connected_i_temp[0]
    return edge_list_all

def gen_sp_nodes_idx(point_set, edge_list):

    point_num = len(point_set)
    # degree of each node 
    nodes_degrees_array = gen_degree_list(edge_list, point_num)

     # degree list
    degree_list = np.unique(nodes_degrees_array.tolist()[0])
    # print(degree_list)
    

    # gen_non_two_connection_idx(point_set, nodes_degrees_array, degree_list):
    nodes_non_2_index, _ = gen_non_two_connection_idx(point_set, nodes_degrees_array, degree_list)
    # add_one_connection_another_idx(nodes_degrees_array, edge_list, nodes_non_2_index)
    sp_nodes_idx = add_one_connection_another_idx(nodes_degrees_array, edge_list, nodes_non_2_index)

    return sp_nodes_idx


def gen_certain_number(point_set, edge_list, certain_num):
    sp_index =  gen_sp_nodes_idx(point_set, edge_list)
    if certain_num <= len(sp_index):
        point_set_sp = point_set[sp_index]
        idx_selected = sampleutils.furthest_point_sample(point_set_sp,certain_num)
        sp_map= {i: j for i,j in enumerate(sp_index)}
        idx_selected_all = np.array(list(map(sp_map.get, idx_selected)))
    else:
        sp_index = np.array(sp_index).astype(int)
        Npoint = len(point_set)
        node_index_all = np.arange(Npoint)
        node_index_rest = np.delete(node_index_all, sp_index, 0)
        sp_index = np.transpose(sp_index)
        point_set_rest = np.delete(point_set, sp_index, 0)

        Nrest = certain_num - len(sp_index)
        # print('sp_index has {} points'.format(len(sp_index)))
        idx_selected = sampleutils.furthest_point_sample(point_set_rest, Nrest)

        # mapping idx_selected--> node_index_rest
        rest_map= {i: j for i,j in enumerate(node_index_rest)}
        idx_selected_map = np.array(list(map(rest_map.get, idx_selected)))
        idx_selected_all = np.hstack((idx_selected_map, sp_index))
    return idx_selected_all

def gen_certain_graph_from_graph(pairs, certain_idx):

    start = datetime.datetime.now()
    nodes_all = edge_pair_to_nodes(pairs)
    point_num_all = len(nodes_all)
    nodes_certain = [int(i)  for i in certain_idx]
    edge_list_array = gen_edge_list_array_from_pair(point_num_all, pairs) 
    nodes_change = nodes_need_to_replace(edge_list_array, nodes_certain)
    nodes_change_num = len(nodes_change)
    nodes_change_num_update = -1
    while nodes_change_num != nodes_change_num_update:
        nodes_change_num_update = nodes_change_num 
        # gen_certain_num_graph
        edge_list_array_one = gen_certein_edge_list(point_num_all, edge_list_array, nodes_certain)
        edge_list_array = edge_list_array_one
        # nodes_change_list
        nodes_change = nodes_need_to_replace(edge_list_array_one, nodes_certain)
        nodes_change_num = len(nodes_change)

    graph_edge_list_array = gen_certein_edge_list(point_num_all, edge_list_array_one, nodes_certain)
    # reverse_array_to_edge_list
    graph_pairs_certain = reverse_array_to_edge_list(graph_edge_list_array, nodes_certain)
    # reindex
    graph_pairs = reindex(graph_pairs_certain, nodes_certain)
    end = datetime.datetime.now()
    # print('graph building time is {}'.format(end - start))

    return graph_pairs

def tps(edges, npoint, xyz):
    # edges = edges.cpu().numpy()
    xyz = xyz.cpu().numpy()[0]
    sp_idx = gen_sp_nodes_idx(xyz, edges)
    idx_selected = gen_certain_number(xyz, edges, npoint)
    edge_list_certain = gen_certain_graph_from_graph(edges, idx_selected)

    idx_selected = torch.from_numpy(np.expand_dims(idx_selected,axis = 0)).int() 

    device = torch.device("cuda")
    idx_selected  = idx_selected.to(device)
    
    return idx_selected, edge_list_certain


def topology_preserving_sampling(edges, npoint, xyz):
    
    idx_selected = gen_certain_number(xyz, edges, npoint)

    edge_list_certain = gen_certain_graph_from_graph(edges, idx_selected)

    return idx_selected, edge_list_certain


#   reverse edge_list to array  
def gen_edge_list_array_from_pair(point_set_N, edge_list):

    g = graph_construction(point_set_N, edge_list)
    nodes = g.nodes()
    nodes = sorted(nodes)
    edges_bidirectional = g.edges()

    edge_list_all = np.full((point_set_N, 20), -1, dtype=int) # 7278 x 20
    for i, node_i in enumerate(edges_bidirectional[0]):

        # line idx represent the node idx 
        line = edge_list_all[node_i,:] 
        if edges_bidirectional[1][i] not in line:
            flag = 0
            for j, node_j in enumerate(line):
                if edge_list_all[node_i,j] != -1:
                    pass
                else:
                    edge_list_all[node_i,j] = edges_bidirectional[1][i] 
                    flag = 1
                if flag == 1:
                    break
    edge_list_all = np.transpose(edge_list_all)  # 4 x 7278 

    edge_list_all_new = []
    for edge_list_i in edge_list_all:
        if len(np.unique(edge_list_i)) != 1:
            edge_list_all_new.append(edge_list_i)
    edge_list_all_new = np.array(edge_list_all_new)

    return edge_list_all_new


def gen_edge_list_array_exclude_tri_pair(point_set_N, pairs, tri_pair):

    
    g = graph_construction(point_set_N, pairs)
    edges_bidirectional = g.edges()

    edge_list_all = np.full((point_set_N, 20), -1, dtype=int)
    node_exist = []

     
    for i, node_i in enumerate(edges_bidirectional[0]):

        # tri_piar
        c_axis = (edges_bidirectional[0][i], edges_bidirectional[1][i])
        c_axis = np.array(c_axis).tolist()
        c_axis = (c_axis[0], c_axis[1])
        
        if c_axis not in tri_pair:
            # line idx represent the node idx 
            line = edge_list_all[node_i,:] 
            if edges_bidirectional[1][i] not in line:
                flag = 0
                for j, node_j in enumerate(line):
                    if edge_list_all[node_i,j] != -1:
                        pass
                    else:
                        edge_list_all[node_i,j] = edges_bidirectional[1][i] 
                        flag = 1
                    if flag == 1:
                        break
    
    # 
    edge_list_all_new = []
    edge_list_all = np.transpose(edge_list_all)  # 20 x 7278
    for edge_list_i in edge_list_all:
        if len(np.unique(edge_list_i)) != 1:
            edge_list_all_new.append(edge_list_i)
    edge_list_all_new = np.array(edge_list_all_new)


    return edge_list_all_new


def reverse_array_to_edge_list(edge_list_all, nodes_certain):

    nodes_certain_temp = sorted(nodes_certain)  # 4096
    nodes_certain_temp = [int(i) for i in nodes_certain_temp]
    edge_list_certain = []
    edge_list_all = np.transpose((edge_list_all))
    for node_i in nodes_certain_temp:
        nodes_connected_i = np.unique(edge_list_all[node_i]).tolist()
        if -1 in nodes_connected_i:
            nodes_connected_i.remove(-1)
        if nodes_connected_i != []:
            for node_connected_i in nodes_connected_i:
                if node_connected_i in nodes_certain:
                    if node_i > node_connected_i:
                        c_axis = (node_connected_i, node_i)
                    else:
                        c_axis = (node_i, node_connected_i)
                    if ((node_connected_i, node_i) not in edge_list_certain) & (
                            (node_i, node_connected_i) not in edge_list_certain):
                        edge_list_certain.append(c_axis)

    return edge_list_certain


def reverse_array_to_edge_list_all(edge_list_array):
    pair_new = []
    for line in edge_list_array:
        if len(np.unique(line)) != 1: 
            for node_i in range(1,len(line)-1):
                # print(node_i)
                if line[node_i] != -1:
                    pair_i = (line[0], line[node_i])
                    pair_i_re = (line[node_i], line[0])
                    if (pair_i not in pair_new) and (pair_i_re not in pair_new):
                        pair_new.append(pair_i)
    return pair_new


def add_tri_pair(pair_new, Tri_pair):
    Tri_pair_new = []
    i = 0
    for Tri_pair_i in Tri_pair:
        
        if (i % 3 == 0) or (i % 3 ==2):
            if (Tri_pair_i not in pair_new) and ((Tri_pair_i[1], Tri_pair_i[0]) not in pair_new):
                pair_new.append(Tri_pair_i)
            if (Tri_pair_i not in Tri_pair_new) and ((Tri_pair_i[1], Tri_pair_i[0]) not in Tri_pair_new):
                Tri_pair_new.append(Tri_pair_i)
        i = i+1
    return pair_new

def updata_edge_list_array(Tri_pair, edge_list_array):
    edge_list_array = np.transpose(edge_list_array)
    for Tri_pair_i in Tri_pair:
        
        edge_list_line = edge_list_array[Tri_pair_i[0],:]
        edge_list_line = edge_list_line.tolist()
        if len(np.unique(edge_list_line)) == 1:
               edge_list_array[Tri_pair_i[0],0] = Tri_pair_i[0]
               edge_list_array[Tri_pair_i[0],1] = Tri_pair_i[1]  
        else: 
            if Tri_pair_i[1] not in edge_list_line[1:-1]:
                for i in range(0,len(edge_list_line)-1):
                    if edge_list_line[i] == -1:
                       edge_list_array[Tri_pair_i[0],i] = Tri_pair_i[1]
                       break
    
    for Tri_pair_i in Tri_pair:
        
        edge_list_line = edge_list_array[Tri_pair_i[1],:]
        edge_list_line = edge_list_line.tolist()
        if len(np.unique(edge_list_line)) == 1:
               edge_list_array[Tri_pair_i[1],0] = Tri_pair_i[1]
               edge_list_array[Tri_pair_i[1],1] = Tri_pair_i[0]  
        else: 
            if Tri_pair_i[0] not in edge_list_line[1:-1]:
                for i in range(0,len(edge_list_line)-1):
                    if edge_list_line[i] == -1:
                       edge_list_array[Tri_pair_i[1],i] = Tri_pair_i[0]
                       break


    # for row_i in range(len(edge_list_array)):
    #     if len(np.unique(edge_list_array[row_i])) == 1:
    #         print(row_i)   
    edge_list_array = np.transpose(edge_list_array)
    return edge_list_array



def reindex(edge_list_certain,nodes_certain):
    # edge_list_certain = all_edge
    edges_exist_reindex = []
    edges_exist = []
    for edge_i in edge_list_certain:
        if (edge_i[0] in nodes_certain) & (edge_i[1] in nodes_certain):
            edge_i_0_index = [i_next_idx_i for i_next_idx_i,x in enumerate(nodes_certain) if x == edge_i[0]]
            edge_i_1_index = [i_next_idx_i for i_next_idx_i,x in enumerate(nodes_certain) if x == edge_i[1]]

            if edge_i_0_index > edge_i_1_index:
                edge_exist_i = (edge_i_1_index[0], edge_i_0_index[0])
            else:
                edge_exist_i = (edge_i_0_index[0], edge_i_1_index[0])

            edges_exist.append(edge_i)
            edges_exist_reindex.append(edge_exist_i)

    edges_list = edges_exist_reindex
    return edges_list

def gen_tfg_idx( G_all_nodes, idx_i, point_num):
    G_all_nodes_cp = G_all_nodes.copy()
    centroid_id = idx_i[0]
    # idx_i = idx_i.tolist()
    rest_index = [idx_i_i for idx_i_i in range(point_num) if idx_i_i not in idx_i]
        # print(rest_index)
    G_all_nodes_cp.remove_nodes_from(rest_index)
    connected_components = list(nx.connected_components(G_all_nodes_cp))
    for connected_component_i in  connected_components:
        if int(centroid_id) in connected_component_i:
            b_pair_component = connected_component_i
    idx_sp_ori = list(b_pair_component)

    if len(idx_i)>=len(idx_sp_ori):
       idx_i = random.sample(idx_sp_ori, len(idx_i))
     
    return idx_sp_ori


def topology_aware_feature_grouping(radius, nsample, xyz, new_xyz, fps_idx, edges):

    edges = np.array(edges) #for nx.connected_components
    G_all_nodes = butils.gen_G_nx(xyz.shape[1],edges)
    point_num = xyz.shape[1]
    xyz = xyz.squeeze(0)
    import GraphConstruction.utils_vis as vutils
    vutils.vis_graph_degree(xyz,edges)  # whole graph
    dist_euclidean = cdist(xyz, xyz, 'euclidean')
    tfg_idx_all=[]
    fps_idx = fps_idx.tolist()[0]
    for centroid in fps_idx:
        # print(centroid)
        idx = np.nonzero(dist_euclidean[centroid, :] < radius)
        idx_list =list(idx[0])
        rest_index = [idx_i for idx_i in range(point_num) if idx_i not in idx_list]
        G_all_nodes_cp = G_all_nodes.copy()
        G_all_nodes_cp.remove_nodes_from(rest_index)
        connected_components = list(nx.connected_components(G_all_nodes_cp))
        for connected_component_i in  connected_components:
            if centroid in connected_component_i:
                b_pair_component = connected_component_i

        idx_sp = list(b_pair_component)
        idx_sp.remove(centroid)
        # vutils.vis_sp_point(xyz, idx_sp_ori , i, edges)  # TFG regions
        if len(idx_sp) >= nsample:
           tfg_idx = random.sample(idx_sp, nsample)
        else:
           tfg_idx = np.random.choice(idx_sp,nsample,replace=True)
           tfg_idx  =   tfg_idx.tolist()
        tfg_idx.insert(centroid,0)
        tfg_idx_all.append(tfg_idx)

    tfg_idx_all = np.array(tfg_idx_all)
    tfg_idx_all = torch.from_numpy(tfg_idx_all)
    tfg_idx_all =  tfg_idx_all.unsqueeze(0)
    tfg_idx_all  = torch.tensor(tfg_idx_all, dtype=torch.int32).cuda()
    return  tfg_idx_all  #[B, Npoint, nsample+1]
    