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

import itertools
import scipy.spatial as spt
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import GraphConstruction.utils_base as butils

def gen_degree_list_vis(edge_list, point_num):
    # graph construction
    g = graph_construction(point_num, edge_list)

    # compute degrees
    nx_G = g.to_networkx().to_undirected()
    degrees = nx_G.degree()

    nodes_degrees_array = np.full((1, point_num), -1, dtype=int)
    for degree in degrees:
        node = degree[0]
        degree = degree[1]
        nodes_degrees_array[0, node] = degree

    return nodes_degrees_array, nx_G

def gen_pairs(points, r_thresh):

    Npoint = len(points)
    # kdtrees
    ckt = spt.cKDTree(points)
    pairs = ckt.query_pairs(r=r_thresh, p=2.0)  
    # triangle removal
    start = datetime.datetime.now()
    pairs_one = tri_process(pairs,Npoint)
    pairs_one = tri_process_reverse(pairs_one,Npoint)
    i = 1
    while (len(pairs) != len(pairs_one))  and  (i<10):
        i = i+1
        pairs_one = set(pairs_one)
        pairs_one = tri_process(pairs_one,Npoint)
        pairs_two = tri_process_reverse(pairs_one,Npoint)
        pairs = pairs_one
        pairs_one = pairs_two
    
    pair_new = pairs_one
    end = datetime.datetime.now()
    print('iteration times is {} and time is {}'.format(i,end - start))

    return pair_new


def tri_process(pairs, Npoint):

    # edge_pair --> edge_array
    nodes_array = pair_to_array(pairs)   # pairs 4102   --> 2 x 4102
    # get graph degree
    nodes_degrees_array = gen_degree_list(pairs, Npoint)
    nodes_degrees_list = nodes_degrees_array.tolist()[0]
    degree_list = np.unique(nodes_degrees_list)
    # two more connection nodes
    nodes_more_than_two = gen_more_than_two_connection_idx(nodes_degrees_array, degree_list)
    # if there exists tri pair
    Tri_pair = gen_tri_pair(nodes_more_than_two, nodes_array, pairs)
    pair_new = gen_pair_exclude_tri_pair(pairs, Tri_pair)
    # Tri pair delete  
    pair_new = add_tri_pair(pair_new, Tri_pair, pairs) 

    return pair_new

def tri_process_reverse(pairs, Npoint):

    # edge_pair --> edge_array
    nodes_array = pair_to_array(pairs)   # pairs 4102   --> 2 x 4102
    # get graph degree
    nodes_degrees_array = gen_degree_list(pairs, Npoint)
    nodes_degrees_list = nodes_degrees_array.tolist()[0]
    degree_list = np.unique(nodes_degrees_list)
    ## two more connection nodes
    nodes_more_than_two = gen_more_than_two_connection_idx(nodes_degrees_array, degree_list)
    # if there exists tri pair
    Tri_pair = gen_tri_pair_reverse(nodes_more_than_two, nodes_array, pairs)
    pair_new = gen_pair_exclude_tri_pair(pairs, Tri_pair)
    # Tri pair delete  
    pair_new = add_tri_pair_reverse(pair_new, Tri_pair, pairs) 

    return pair_new


def pair_to_array(pairs):
    """
    input: pairs {(),(),()}
    output: pair_array 2 x len(pairs)
    """

    pair_array = np.full((2,len(pairs)),-1, dtype=int)
    pairs = sorted(pairs)
    i_pair = 0
    for pair in pairs:
        if i_pair < len(pairs):
            pair_array[0,i_pair] = pair[0]
            pair_array[1,i_pair] = pair[1]
            i_pair = i_pair + 1
    return pair_array

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
    return nodes_degrees_array

def graph_construction(Npoints, edge_list):
    """
    for vis
    :param Npoints: Number of points
    :param edge_list: pairs [(), (), ()...]
    :return: G
    """
    G = dgl.DGLGraph()
    G.add_nodes(Npoints)
    src, dst = tuple(zip(*edge_list))
    G.add_edges(src, dst)
    G.add_edges(dst, src)
    return G


def gen_more_than_two_connection_idx(nodes_degrees_array, degree_list):

    # degree is not 2
    nodes_non_2 = []
    for degree_num in degree_list:
        if degree_num > 2:
            node_index = np.nonzero(nodes_degrees_array == degree_num)[1] 
            nodes_non_2.append(node_index.tolist())
    nodes_non_2_index = np.concatenate(nodes_non_2).tolist()

    return nodes_non_2_index

def gen_tri_pair(nodes_more_than_two, nodes_array, pairs):
    Tri_pair = []
    for node in nodes_more_than_two:
        next_nodes = []
        index_node = np.array(np.where(nodes_array[0]==node))
        index_node = index_node[0] 
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



def gen_tri_pair_reverse(nodes_more_than_two, nodes_array, pairs):
    Tri_pair = []
    for node in nodes_more_than_two:
        next_nodes = []
        index_node = np.array(np.where(nodes_array[1]==node))
        index_node = index_node[0] 
        index_node = index_node.tolist() 
        for index_node_i in index_node:
            next_node = nodes_array[0][index_node_i]   # next node
            next_nodes.append(next_node)

        pairs_to_check = list(itertools.combinations(next_nodes, 2))
        for pair_to_check in pairs_to_check:
            if pair_to_check in pairs:
                Tri_pair.append((pair_to_check[0], node))
                Tri_pair.append((pair_to_check[1], node))
                Tri_pair.append(pair_to_check)
    return Tri_pair

def gen_pair_exclude_tri_pair(pairs, Tri_pair):
    new_pair = []
    for pair in pairs:
        pair_revese = (pair[1], pair[0])
        if (pair not in Tri_pair) and (pair_revese not in Tri_pair):
            new_pair.append(pair)
    return new_pair



def add_tri_pair_reverse(pair_new, Tri_pair, pairs):
    Tri_pair_new = []
    i = 0
    for Tri_pair_i in Tri_pair:
        
        if (i % 3 == 1) or (i % 3 ==2):
            if (Tri_pair_i not in pair_new) and ((Tri_pair_i[1], Tri_pair_i[0]) not in pair_new) and(Tri_pair_i in pairs):
                pair_new.append(Tri_pair_i)
            if (Tri_pair_i not in Tri_pair_new) and ((Tri_pair_i[1], Tri_pair_i[0]) not in Tri_pair_new):
                Tri_pair_new.append(Tri_pair_i)
        i = i+1
    return pair_new

def add_tri_pair(pair_new, Tri_pair, pairs):
    Tri_pair_new = []
    i = 0
    for Tri_pair_i in Tri_pair:
        
        if (i % 3 == 0) or (i % 3 ==2):
            if (Tri_pair_i not in pair_new) and ((Tri_pair_i[1], Tri_pair_i[0]) not in pair_new) and(Tri_pair_i in pairs):
                pair_new.append(Tri_pair_i)
            if (Tri_pair_i not in Tri_pair_new) and ((Tri_pair_i[1], Tri_pair_i[0]) not in Tri_pair_new):
                Tri_pair_new.append(Tri_pair_i)
        i = i+1
    return pair_new

def reidx_edges(G):
    nodes = G.nodes()
    edge_list_new = G.edges()
    idx = np.array(range(len(nodes)), dtype=np.int32)
    idx_map = {j:i for i, j in enumerate(nodes)}
    edge_unordered = np.array(edge_list_new)
    edges = np.array(list(map(idx_map.get, edge_unordered.flatten())), dtype=np.int32).reshape(edge_unordered.shape)
    edges = [(edge[0],edge[1]) for edge in edges]

    return edges

def gen_isolate_removal(pc,edges,thresh):

    G_nx = butils.gen_G_nx(len(pc), edges)
    node_to_remove = []
    all_connected_components = list(nx.connected_components(G_nx))
    all_connected_components_temp = all_connected_components.copy()
    for i, connected_i in enumerate(all_connected_components):
        connected_i = [int(idx) for idx in connected_i]
        if len(connected_i) <= thresh:
            node_to_remove.append(connected_i)
            all_connected_components_temp.remove(all_connected_components[i])
    # print(node_to_remove)

    if len(node_to_remove)>1:
        node_to_remove = np.concatenate(node_to_remove)

    node_to_remove= list(node_to_remove)
    G_nx.remove_nodes_from(node_to_remove)       
    
    node_to_remove = [int(node) for node in list(node_to_remove)]
    new_pc = np.delete(pc, node_to_remove, axis=0)  # for plotting
    new_edges = reidx_edges(G_nx)
    
    return new_pc, new_edges
    


