from re import S
import SimpleITK as sitk 
import os 
import datetime
from networkx.classes.function import degree
import numpy as np
import pickle
import sys

import math

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import GraphConstruction.utils_base as butils
import GraphConstruction.utils_graph as gutils 
import utils_multicl as mcutils
import utils_completion as cutils

if __name__ == '__main__':
    
    
    data_path = './SampleData'
    patients = sorted(os.listdir(data_path))

    head_list = [0,5,6,11,17]  # head label 
    neck_list = [13, 14, 15, 16, 7, 12, 4, 10, 3, 9, 8, 2] # neck label
    patients=['002','003']
    for patient in patients:
        print(patient)
        start_time = datetime.datetime.now()

       

        graph_edges = os.path.join(data_path, patient, 'CenterlineGraph')
        edges = butils.load_pairs(graph_edges)
        
        # labeled centerline from TaG-Net
        labeled_cl_name = os.path.join(data_path, patient, 'labeled_cl.txt')
        pc = np.loadtxt(labeled_cl_name)
        pc_label = pc[:,-1]
        label_list = np.unique(pc_label)

        # graph 
        G_nx = butils.gen_G_nx(len(pc),edges)
        
        # remove noises
        node_to_remove_all = []
        for label in label_list:

            if label in head_list:
                thresh = 15
            if label in neck_list:
                thresh = 30
            if label == 1:
                thresh = 50  
            idx_label = np.nonzero(pc_label == label)[0]
            num_idx_label = len(idx_label)

            # sub graph
            connected_components, G_nx_label = mcutils.gen_connected_components(idx_label, G_nx)

            connected_num = len(connected_components)
            components_area = []
            for connected_i in connected_components:
                components_area.append(len(connected_i))
            components_area = sorted(components_area)

            node_to_remove = []   
            for connected_i in connected_components:
                idx_map_reverse= {i: j for i, j in enumerate(idx_label)}
                ori_idx = [idx_map_reverse.get(idx) for idx in list(connected_i)]
               
                idx_neigbors = butils.gen_neighbors_exclude(ori_idx , G_nx)
                # neigbor label
                seg_label = butils.gen_neighbors_label(idx_neigbors, pc_label)

                seg_label = [label_i for label_i in seg_label]
                if label in seg_label:
                    seg_label.remove(label)
                if len(seg_label) == 0:
                    # it is isolate
                    num_connected_i = len(connected_i)
                    if (num_connected_i/num_idx_label <= 1/10) and (num_connected_i<thresh):
                        node_to_remove.append(ori_idx)

            if np.array(node_to_remove).shape[0] >= 1:
                node_to_remove = np.concatenate(node_to_remove)

            node_to_remove_all.append(node_to_remove)

        if np.array(node_to_remove_all).shape[0] >= 1:
            node_to_remove_all = np.concatenate(node_to_remove_all)
        
        G_nx.remove_nodes_from(node_to_remove_all)   
        new_pc = np.delete(pc, node_to_remove_all, axis=0) 
        new_edges = gutils.reidx_edges(G_nx)
        
        new_graph_edges = os.path.join(data_path, patient, 'CenterlineGraph_new')
        butils.dump_pairs(new_graph_edges, new_edges)
        
        # labeled centerline from TaG-Net
        new_labeled_cl_name = os.path.join(data_path, patient, 'labeled_cl_new.txt')
        np.savetxt(new_labeled_cl_name, new_pc)
        

        end_time = datetime.datetime.now()
        print('time is {}'.format(end_time - start_time))

            
            
         


            
          




        
         





       