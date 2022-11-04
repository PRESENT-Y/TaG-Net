import os 
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import GraphConstruction.utils_vis as vutils
import GraphConstruction.utils_base as butils

import csv

if __name__ == '__main__':
   
    data_path = './SampleData'
    patients = sorted(os.listdir(data_path))
    patients = ['002']
    for patient in patients:
        csv_file = os.path.join(data_path, patient, 'connection_paths.csv')
        content = []
        with open(csv_file, 'r') as fp:
            lines = csv.reader(fp)
            for line in lines:
                content.append(list(line))
        content = content[1]
        
        path_points_all = []
        for i, content_i in enumerate(content):
            # each connection path
            if i >= 1:
                content_i = eval(content_i.replace('array',''))
                # start_point
                sp = [int(idx) for idx in content_i[1]]
                # end_point
                ep = [int(idx) for idx in content_i[2]]

                path_points = content_i[3]
                for point in path_points:
                    point =  [int(point_i) for point_i in point]
                    path_points_all.append(point)
        path_points_all = np.array(path_points_all)
        print(path_points_all)

        graph_edges = os.path.join(data_path, patient, 'CenterlineGraph_new')
        edges = butils.load_pairs(graph_edges)
            
        # labeled centerline from TaG-Net
        labeled_cl_name = os.path.join(data_path, patient, 'labeled_cl_new.txt')
        pc = np.loadtxt(labeled_cl_name)
        pc = np.vstack((pc, path_points_all))

        # # visualize labeled centerline graph
        vutils.vis_multi_graph(pc, edges)
