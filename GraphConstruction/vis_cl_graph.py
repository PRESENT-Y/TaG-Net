import os 
import numpy as np
import utils_vis as vutils
import utils_base as butils

if __name__ == '__main__':
   
    data_path = './SampleData'
    patients = sorted(os.listdir(data_path))
    patients = ['001']
    # patients = ['Tr0006']
    for patient in patients:
        graph_path = os.path.join(data_path,patient,'CenterlineGraph')
        point_cloud_path = os.path.join(data_path,patient,'cl.txt')
        
        data = np.loadtxt(point_cloud_path).astype(np.float32)
        edge_list = butils.load_pairs(graph_path)
        
        # visualize centerline points
        # vutils.vis_ori_points(data[:,0:3])

        # visualize centerline graph
        vutils.vis_graph_degree(data[:,0:3], edge_list)