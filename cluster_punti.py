#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.cluster.vq import *


def create_clusters(num_clusters=2, size=10):

    cluster1_center = {'x': 4,'y': 4}
    cluster2_center = {'x': 10,'y': 10}
    distance = 4

    cluster1_points_x = []
    cluster1_points_y = []

    cluster2_points_x = []
    cluster2_points_y = []

    # clusters = []

    # for n in xrange(0,num_clusters):
    #     clusters[n]['x'] = 3

    print size
    for item in np.random.random_sample(size):
        item = item * distance - distance/2
        cluster1_points_x.append(cluster1_center['x'] + item)
        cluster2_points_x.append(cluster2_center['x'] + item)

    for item in np.random.random_sample(size):
        item = item * distance - distance/2
        cluster1_points_y.append(cluster1_center['y'] + item)
        cluster2_points_y.append(cluster2_center['y'] + item)

    return {
        'x_1': cluster1_points_x,
        'y_1': cluster1_points_y,
        'x_2': cluster2_points_x,
        'y_2': cluster2_points_y,
    }


clusters = create_clusters(2, 30)
xs = clusters['x_1'] + clusters['x_2']
ys = clusters['y_1'] + clusters['y_2']

# zip => da 2 array prende gli elementi a coppie in base alla posizione, e crea una tupla

centroids, idx = kmeans2(np.array(zip(xs, ys)),2)
colors = ([([0.4,1,0.4],[1,0.4,0.4],[0.1,0.8,1])[i] for i in idx])

plt.scatter(xs, ys, c=colors)
plt.scatter(centroids[:,0],centroids[:,1], marker='*', s = 300, linewidths=2, c=colors)
plt.show()