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

import sys



def index_points(points, idx):
    """
    Input:
        points: input points data, [N, C]
        idx: sample index data, [D1,...DN]
    Return:
        new_points:, indexed points data, [D1,...DN, C]
    """
    idx = [int(i) for i in idx]
    new_points = points[idx, :]
    return new_points


def square_distance(src, dst):
    """
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
	     = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [N, C]
        dst: target points, [M, C]
    Output:
        dist: per-point square distance, [N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * np.matmul(src, dst.permute(0, 2, 1)) # 2*(xn * xm + yn * ym + zn * zm)
    dist += np.sum(src ** 2, -1).view(B, N, 1) # xn*xn + yn*yn + zn*zn
    dist += np.sum(dst ** 2, -1).view(B, 1, M) # xm*xm + ym*ym + zm*zm
    return dist

def furthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    
    N, C = xyz.shape
    centroids = np.zeros(npoint)
    distance = np.ones(N) * 1e10
    farthest = int(random.randint(0, N))
    for i in range(npoint):
    	# 更新第i个最远点
        centroids[i] = farthest
        # 取出这个最远点的xyz坐标
        centroid = xyz[farthest, :]
        # 计算点集中的所有点到这个最远点的欧式距离
        dist = np.sum((xyz - centroid) ** 2, -1)
        # 更新distances，记录样本中每个点距离所有已出现的采样点的最小距离
        mask = dist < distance
        distance[mask] = dist[mask]
        # 从更新后的distances矩阵中找出距离最远的点，作为最远点用于下一轮迭代
        farthest = int(np.where(distance == (np.max(distance)))[0][0])
    return centroids

