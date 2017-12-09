# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 14:11:04 2017

@author: Andrea
"""

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d as plt3d
from skimage.io import imread
from sklearn import neighbors
from sklearn.cluster import DBSCAN

import data_processing as data
import cluster_analysis

latitude, longitude, fault = data.read_csv();
x, y, z = data.transform_coordinates(latitude, longitude);
#plot_cartesian_coordinates(x, y, z);
X= data.preprocess_data(x, y, z)


############### DBSCAN ########################################################
#First of all, I have to set the eps parameter of the classifier. To do so,
#I create a fictitious output vector, filled with ones and then applied the
#kNN classifier.

dbscan= sklearn.cluster.DBSCAN(300, 4, n_jobs=-1)
dbscan.fit(X)
pred_labels = dbscan.labels_
n_clusters_ = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
dbscan_evaluate_paper= cluster_analysis.evaluate_cluster(X, fault, dbscan.labels_)

noise_index= pred_labels != -1
print(noise_index)
dbscan_evaluate_noise= cluster_analysis.evaluate_cluster(X[noise_index,:], fault[noise_index], dbscan.labels_[noise_index])

print('With noise')
print('Number of clusters: %d' % n_clusters_)
print("Precision: %0.3f" % dbscan_evaluate_paper[0])
print("Recall: %0.3f" % dbscan_evaluate_paper[1])
print("F1: %0.3f" % dbscan_evaluate_paper[2])
print("Rand Index: %0.3f" % dbscan_evaluate_paper[3])
print("Adjusted Rand Index: %0.3f" % dbscan_evaluate_paper[4])
print("Silhouette: %0.3f" % dbscan_evaluate_paper[5])

print('\nWithout noise')
print("Precision: %0.3f" % dbscan_evaluate_noise[0])
print("Recall: %0.3f" % dbscan_evaluate_noise[1])
print("F1: %0.3f" % dbscan_evaluate_noise[2])
print("Rand Index: %0.3f" % dbscan_evaluate_noise[3])
print("Adjusted Rand Index: %0.3f" % dbscan_evaluate_noise[4])
print("Silhouette: %0.3f" % dbscan_evaluate_noise[5])
#data.plot_classes(pred_labels, longitude, latitude, alpha=0.5, edge='k')