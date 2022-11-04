import pickle
import SimpleITK as sitk
import numpy as np
import math
import dgl
import networkx as nx



def load_itk_image(filename):
    """
    
    :param filename: CTA name to be loaded
    :return: CTA image, CTA origin, CTA spacing
    """
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing


def save_itk(image, origin, spacing, filename):
    """
    :param image: images to be saved
    :param origin: CTA origin
    :param spacing: CTA spacing
    :param filename: save name
    :return: None
    """
    if type(origin) != tuple:
        if type(origin) == list:
            origin = tuple(reversed(origin))
        else:
            origin = tuple(reversed(origin.tolist()))
    if type(spacing) != tuple:
        if type(spacing) == list:
            spacing = tuple(reversed(spacing))
        else:
            spacing = tuple(reversed(spacing.tolist()))
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    sitk.WriteImage(itkimage, filename, True)


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


def gen_G_nx(Npoints, edge_list):
    """
    for process, it is easy to add edges and remove node  
    try to use one package
    :param Npoints: Number of points
    :param edge_list: pairs [(), (), ()...]
    :return: G
    """

    G = nx.Graph()
    G.add_nodes_from(range(Npoints))
    G.add_edges_from(edge_list)

    return G

def gen_neighbors(ori_idx, G_nx):
    """
    generate neighbors of ori_idx 
    """
    neighbors = []
    for edge in G_nx.edges(ori_idx):
        neighbors.append(edge[1])
    return neighbors


def gen_neighbors_exclude(ori_idx, G_nx):
    """
    generate neighbors of ori_idx exclude itself
    """
    neighbors = []
    edges = []
    for idx in ori_idx:
        for edge_idx in G_nx.edges(idx):
            edges.append(edge_idx)
    
    for edge in edges:
        if edge[1] not in ori_idx:
           neighbors.append(edge[1])
        if edge[0] not in ori_idx:
           neighbors.append(edge[0])
    return neighbors

def gen_neighbors_exclude_ori(ori_idx, G_nx):
    """
    generate neighbors of ori_idx exclude itself

    return neighbor, nei_ori
    """
    neighbors = []
    nei_ori = []
    for edge in G_nx.edges(ori_idx):
        if edge[1] not in ori_idx:
           neighbors.append(edge[1])
           nei_ori.append(edge[0])
    return neighbors, nei_ori


def gen_idx_with_diff_label(G_nx, idx_label):
    idxs = []
    idx_neigbors = gen_neighbors_exclude(idx_label, G_nx)
    for idx_neigbor in idx_neigbors:
        for edge in G_nx.edges(idx_neigbor):
           if edge[1] in idx_label:
               idxs.append(edge[1])
    return idxs


def gen_idx_with_diff_label_diff(G_nx, idx_label):
    idxs = []
    idx_neigbors = gen_neighbors_exclude(idx_label, G_nx)
    for idx_neigbor in idx_neigbors:
        for edge in G_nx.edges(idx_neigbor):
           if edge[0] not in idx_label:
               idxs.append(edge[1])
    return idxs



import scipy.spatial as spt
from sklearn.manifold import Isomap

def cpt_geo_dis_mat(data):
    """
    geometric distance 
    """
    ckt = spt.cKDTree(data)
    isomap = Isomap(n_components=2, n_neighbors=2, path_method='auto')
    data_3d = isomap.fit_transform(data)
    geo_distance_matrix = isomap.dist_matrix_
    return geo_distance_matrix

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

def cpt_sqr_dis_mat(data):
    """
    square distance
    """
    square_distance_matrix = squareform(pdist(data, metric='euclidean'))

    return square_distance_matrix

def gen_neighbors_label(idx_neigbors, seg_data):
    seg_label = []
    for idx in idx_neigbors:
        seg_label.append(seg_data[idx])
    seg_label = np.unique(seg_label)

    return seg_label