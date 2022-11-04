import numpy as np
from mayavi import mlab
import GraphConstruction.utils_graph as gutils
# import utils_graph as gutils

color_map_eighteen = np.array([
    [255, 0, 0],  # 1 # red
    [46, 139, 87],  # 2 # see green
    [0, 0, 255],  # 3 # blue
    [255, 255, 0],  # 4 # Yellow
    [0, 255, 255],  # 5 # Cyan
    [255, 0, 255],  # 6  # Magenta
    [255, 165, 0],  # 7 # Orange1
    [147, 112, 219],  # 8 # MediumPurple
    [50, 205, 50],  # 9 # LimeGreen
    [255, 215, 0],  # 10 # Gold1
    [102, 205, 170],  # 11 # Aquamarine3
    [255, 127, 0],  # 12 # SpringGreen
    [160, 32, 240],  # 13 # Purple
    [30, 144, 255],  # 14 # DodgerBlue
    [0, 191, 255],  # 15 # DeepSkyBlue1
    [255, 105, 180],  # 16 # HotPink
    [255, 192, 203],  # 17 # Pink
    [205, 92, 92],  # 18 #  IndianRed

]) / 255.


color_map = np.array([
    # [255, 0, 0],  # 1 # red
    [46, 139, 87],  # 2 # see green
    [0, 0, 255],  # 3 # blue
    [255, 127, 0],  # 12 # SpringGreen
    # [160, 32, 240],  # 13 # Purple
    # [0, 255, 255],  # 5 # Cyan
    [255, 0, 255],  # 6  # Magenta
    [255, 165, 0],  # 7 # Orange1
    [147, 112, 219],  # 8 # MediumPurple
    [50, 205, 50],  # 9 # LimeGreen
    [255, 215, 0],  # 10 # Gold1
    [102, 205, 170],  # 11 # Aquamarine3

    [160, 32, 240],  # 13 # Purple
    [30, 144, 255],  # 14 # DodgerBlue
    [0, 191, 255],  # 15 # DeepSkyBlue1
    [255, 105, 180],  # 16 # HotPink
    [255, 192, 203],  # 17 # Pink
    [205, 92, 92],  # 18 #  IndianRed

]) / 255.

def gen_neighbors(ori_idx, G_nx):
    neighbors = []
    for edge in G_nx.edges(ori_idx):
        neighbors.append(edge[1])
    return neighbors


def vis_graph_degree(data, edges_list):
    pc_num = len(data)
    nodes_degrees_array, nx_G = gutils.gen_degree_list_vis(edges_list, pc_num)
    nodes_degrees_list = nodes_degrees_array.tolist()[0]
    degree_list = np.unique(nodes_degrees_list)
    print(degree_list)

    mlab.figure(1, bgcolor=(1, 1, 1))
    mlab.clf()
    points = data[:,0:3]
    for degree_i in degree_list:
        if degree_i != 2:
            node_index = np.nonzero(nodes_degrees_array == degree_i)[1]
            print("degree {} has {} nodes".format(degree_i, len(node_index)))
            color_i = color_map[degree_i]
            pts = mlab.points3d(points[node_index, 2], points[node_index, 1], points[node_index, 0], \
                                # color=(color_i[0], color_i[1], color_i[2]), scale_factor=4)
                                color=(color_i[0], color_i[1], color_i[2]), scale_factor=0.02)
    # pts = mlab.points3d(points[:, 2], points[:, 1], points[:, 0], color=(1, 0, 0), scale_factor=1.5)
    pts = mlab.points3d(points[:, 2], points[:, 1], points[:, 0], color=(1, 0, 0), scale_factor=0.005)
    pts.mlab_source.dataset.lines = np.array(nx_G.edges())
    # tube = mlab.pipeline.tube(pts, tube_radius=0.05)
    tube = mlab.pipeline.tube(pts, tube_radius=0.001)
    mlab.pipeline.surface(tube, color=(0, 1, 0))
    # mlab.outline(color=(223 / 255, 223 / 255, 223 / 255), line_width=0.001)  # color value [0,1]
    mlab.show()


def vis_sp_point(data, idx, sp_idx, pairs):
    mlab.figure(1, bgcolor=(1, 1, 1))
    mlab.clf()
    points = data[:, 0:3]
    pc_num = len(points[:, -1])
    nodes_degrees_array, nx_G = gutils.gen_degree_list_vis(pairs, pc_num)

    pts = mlab.points3d(points[:, 2], points[:, 1], points[:, 0], color=(1, 0, 0), scale_factor=0.00001)
    pts.mlab_source.dataset.lines = np.array(nx_G.edges())
    # tube = mlab.pipeline.tube(pts, tube_radius=0.06)
    # tube = mlab.pipeline.tube(pts, tube_radius=0.2)
    tube = mlab.pipeline.tube(pts, tube_radius=0.0002)

    # mlab.points3d(points[node_index, 2], points[node_index, 1], points[node_index, 0], \
    #     color=(color_i[0],color_i[1],color_i[2]), scale_factor=0.005)
    mlab.pipeline.surface(tube, color=(0, 1, 0))
    mlab.points3d(points[:, 2], points[:, 1], points[:, 0], \
                  color=(1, 0, 0), scale_factor=0.00001)

    mlab.points3d(points[idx, 2], points[idx, 1], points[idx, 0], \
                  # color=(0.574,0.438, 0.855), scale_factor=2)
                  color=(0.574, 0.438, 0.855), scale_factor=0.005)

    mlab.points3d(points[sp_idx, 2], points[sp_idx, 1], points[sp_idx, 0], \
                  # color= (0.625,0.125, 0.9375), name=1, scale_factor=4)
                  color=(0.625, 0.125, 0.9375), scale_factor=0.01)
    # mlab.show()
    # (0.625,0.125, 0.9375)
    # (0.574, 0.438, 0.855)
    # pts2 = mlab.points3d(point_set[:, 0], point_set[:, 1], point_set[:, 2], color=(0, 0, 1), scale_factor=1.5)
    # tube = mlab.pipeline.tube(pts, tube_radius=0.008)
    # mlab.pipeline.surface(tube, color=(0, 1, 0))
    # points_corner_min = [0, 0, 0]
    # points_corner_max = [1, 1, 1]
    # points_corner_max = np.vstack([points_corner_max, points_corner_min])
    # mlab.points3d(points_corner_max[:, 0], points_corner_max[:, 1], points_corner_max[:, 2], color=(1, 1, 1),
    #               scale_factor=0.005)
    # mlab.pipeline.surface(tube, color=(0, 1, 0))
    # mlab.points3d(points[:, 2], points[:, 1], points[:, 0], \
    #               color=(1,0,0), scale_factor=0.00005)
    # mlab.outline(color=(223 / 255, 223 / 255, 223 / 255), line_width=0.001)  # color value [0,1]
    mlab.show()
    # mlab.show()

def vis_multi_graph(data, edges_list):

    pc_num = len(data)
    nodes_degrees_array, nx_G = gutils.gen_degree_list_vis(edges_list, pc_num)
    nodes_degrees_list = nodes_degrees_array.tolist()[0]
    degree_list = np.unique(nodes_degrees_list)
    print(degree_list)

    mlab.figure(1, bgcolor=(1, 1, 1))
    mlab.clf()
    points = data[:,0:3]
    label_list = np.unique(data[:,-1])
    for label in label_list:
        node_index = np.nonzero(data[:,-1] == label)
        color_i=color_map_eighteen[int(label)]
        pts = mlab.points3d(points[node_index, 2], points[node_index, 1], points[node_index, 0], \
            # color=(color_i[0], color_i[1], color_i[2]), scale_factor=2)
            color=(color_i[0], color_i[1], color_i[2]), scale_factor=0.02)

    # pts = mlab.points3d(points[:, 2], points[:, 1], points[:, 0], color=(1, 0, 0), scale_factor=0.5)
    pts = mlab.points3d(points[:, 2], points[:, 1], points[:, 0], color=(1, 0, 0), scale_factor=0.001)

    pts.mlab_source.dataset.lines = np.array(nx_G.edges())
    # tube = mlab.pipeline.tube(pts, tube_radius=0.05)
    tube = mlab.pipeline.tube(pts, tube_radius=0.01)
    mlab.pipeline.surface(tube, color=(0, 1, 0))
    mlab.show()



def vis_ori_points(data):
    mlab.figure(1, bgcolor=(1, 1, 1))
    mlab.clf()
    points = data[:,0:3]
    pts = mlab.points3d(points[:, 2], points[:, 1], points[:, 0], color=(1, 0, 0), scale_factor=2)
    tube = mlab.pipeline.tube(pts, tube_radius=0.1)
    mlab.pipeline.surface(tube, color=(0, 1, 0))
    mlab.outline(color=(223 / 255, 223 / 255, 223 / 255), line_width=0.001)  # color value [0,1]
    mlab.show()