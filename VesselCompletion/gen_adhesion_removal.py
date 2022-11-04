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
    patients=['003']
    for patient in patients:
        print(patient)
        start_time = datetime.datetime.now()

        # intra connection pairs (within)
        connection_pair_intra_name = os.path.join(data_path, patient, 'adhesion_removal.txt')
        if not os.path.exists(connection_pair_intra_name):
            graph_edges = os.path.join(data_path, patient, 'CenterlineGraph_new')
            # graph_edges = os.path.join(data_path, patient, 'CenterlineGraph')
            edges = butils.load_pairs(graph_edges)
            
            # labeled centerline from TaG-Net
            labeled_cl_name = os.path.join(data_path, patient, 'labeled_cl_new.txt')
            # labeled_cl_name = os.path.join(data_path, patient, 'labeled_cl.txt')
            pc = np.loadtxt(labeled_cl_name)
            pc_label = pc[:,-1]
            label_list = np.unique(pc_label)

            # graph 
            G_nx = butils.gen_G_nx(len(pc),edges)
            
            # gen wrong connection label position
            flag_wrong, wrong_pairs, flag_lack, lack_pairs, label_pairs = mcutils.gen_wrong_connected_exist_label(label_list, pc_label, G_nx)
            
            pairs_to_del = []
            node_to_remove_all = []
            if flag_wrong == 1:
                # wrong_pair[0]
                # wrong_pair
                for wrong_pair in wrong_pairs:
                    node_to_remove_all = cutils.gen_nodes_to_remove(wrong_pair, pc_label, G_nx, node_to_remove_all)
                    wrong_pair = [wrong_pair[1], wrong_pair[0]]
                    node_to_remove_all = cutils.gen_nodes_to_remove(wrong_pair, pc_label, G_nx, node_to_remove_all)
                    print(node_to_remove_all)                  
                                        
                
                if np.array(node_to_remove_all).shape[0] >= 1:
                   node_to_remove_all = np.concatenate(node_to_remove_all)
        
                G_nx.remove_nodes_from(node_to_remove_all)   
                new_pc = np.delete(pc, node_to_remove_all, axis=0) 
                new_edges = gutils.reidx_edges(G_nx)
        
                new_graph_edges = os.path.join(data_path, patient, 'CenterlineGraph_removal')
                butils.dump_pairs(new_graph_edges, new_edges)
        
                # labeled centerline from TaG-Net
                new_labeled_cl_name = os.path.join(data_path, patient, 'labeled_cl_removal.txt')
                np.savetxt(new_labeled_cl_name, new_pc)
                

                end_time = datetime.datetime.now()
                print('time is {}'.format(end_time - start_time))
               
                            

        

                    
                
                

            
        #     # segment number of one label
        #     components_more_than_one_label_list = cutils.gen_label_to_check(label_list,pc, G_nx)
        #     neck_label_to_connect = [idx for idx in components_more_than_one_label_list if idx in neck_list]
        #     # head_label_to_connect = [idx for idx in components_more_than_one_label_list if idx in head_list
        #     connection_intra_pairs =  cutils.gen_connection_intra_pairs(neck_label_to_connect, G_nx, pc)

        #     # save connection pairs
        #     if len(connection_intra_pairs) >= 1:
        #         butils.dump_pairs(connection_pair_intra_name, connection_intra_pairs) 
           
        # else: 
        #     connection_intra_pairs = butils.load_pairs(connection_pair_intra_name)
        #     print(connection_intra_pairs)   
        

        # # inter connection pairs (between)
        # connection_pair_inter_name = os.path.join(data_path, patient, 'connection_pair_inter')
        # if not os.path.exists(connection_pair_inter_name):
        #     # initial constructed graph
        #     graph_edges = os.path.join(data_path, patient, 'CenterlineGraph_new')
        #     # graph_edges = os.path.join(data_path, patient, 'CenterlineGraph')
        #     edges = butils.load_pairs(graph_edges)
            
        #     # labeled centerline from TaG-Net
        #     labeled_cl_name = os.path.join(data_path, patient, 'labeled_cl_new.txt')
        #     # labeled_cl_name = os.path.join(data_path, patient, 'labeled_cl.txt')
        #     pc = np.loadtxt(labeled_cl_name)
        #     pc_label = pc[:,-1]
        #     label_list = np.unique(pc_label)
            
        #     # graph 
        #     if os.path.exists(connection_pair_intra_name):
        #         intra_pairs = butils.load_pairs(connection_pair_intra_name)
        #         edges.extend(intra_pairs)
          
        #     G_nx = butils.gen_G_nx(len(pc),edges)

        #     # degree
        #     degree_list = gutils.gen_degree_list(G_nx.edges(), len(pc))[0]
            
        #     # if there be an interruption/adhesion on labeled graph
        #     flag_wrong, wrong_pairs, flag_lack, lack_pairs, label_pairs = mcutils.gen_wrong_connected_exist_label(label_list, pc_label, G_nx)
            
        #     connection_inter_pairs = []
        #     for pair_to_check in lack_pairs:

        #         # find start and end nodes (degree being one)
        #         degree_one_list_all, flag_1214 = cutils.find_start_end_nodes(pc_label, pair_to_check, G_nx)
        #         if flag_1214 == 1:
        #             break
        #         if len(degree_one_list_all) >= 1:
        #             degree_one_list_all = np.concatenate(degree_one_list_all)
                
        #         # pairs (node pairs from a same segment are excluded)
        #         all_start_end_pairs = cutils.gen_start_end_pairs(degree_one_list_all, pc_label)

        #         # connection pairs 
        #         connection_inter_pairs = cutils.gen_connection_inter_pairs(all_start_end_pairs, pc, connection_inter_pairs)

        #         # save connection pairs
        #         if len(connection_inter_pairs) >= 1:
        #             # connection_pairs = np.concatenate(connection_pairs)
        #             butils.dump_pairs(connection_pair_inter_name, connection_inter_pairs)
        # else:
        #     connection_inter_pairs = butils.load_pairs(connection_pair_inter_name)
        #     print(connection_inter_pairs)     
        # end_time = datetime.datetime.now()
        # print('time is {}'.format(end_time - start_time))

            
            
         


            
          




        
         





       