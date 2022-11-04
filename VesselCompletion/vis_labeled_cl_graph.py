import os 
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import GraphConstruction.utils_vis as vutils
import GraphConstruction.utils_base as butils

if __name__ == '__main__':
   
    data_path = './SampleData'
    patients = sorted(os.listdir(data_path))
    patients = ['002']
    patients = ['003']
    for patient in patients:
        graph_path = os.path.join(data_path,patient,'CenterlineGraph_new')
        # graph_path = os.path.join(data_path,patient,'CenterlineGraph_removal')
        connection_pair_intra_path = os.path.join(data_path, patient, 'connection_pair_intra')
        connection_pair_inter_path = os.path.join(data_path, patient, 'connection_pair_inter')
        point_cloud_path = os.path.join(data_path,patient,'labeled_cl_new.txt')
        # point_cloud_path = os.path.join(data_path,patient,'labeled_cl_removal.txt')

        data = np.loadtxt(point_cloud_path).astype(np.float32)
        edge_list = butils.load_pairs(graph_path)

        if os.path.exists(connection_pair_intra_path):
            intra_pairs = butils.load_pairs(connection_pair_intra_path)
            edge_list.extend(intra_pairs)

        if os.path.exists(connection_pair_inter_path):
            inter_pairs = butils.load_pairs(connection_pair_inter_path)
            edge_list.extend(inter_pairs)

        # visualize labeled centerline graph
        vutils.vis_multi_graph(data, edge_list)
