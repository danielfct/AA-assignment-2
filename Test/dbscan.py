import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d as plt3d
from skimage.io import imread
import sklearn

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
aux_label= np.zeros(fault.shape[0])
neigh= sklearn.neighbors.KNeighborsClassifier(n_neighbors= 4)
neigh.fit(X, aux_label)

distances= neigh.kneighbors()
k_dist= np.zeros(fault.shape[0])
for i in range(0, fault.shape[0]):
    k_dist[i]= distances[0][i,3]

print(k_dist)
k_dist.sort()
k_dist= k_dist[::-1]
print(k_dist)

ax_2 = plt.figure().add_subplot(111)
plt.plot(range(0, fault.shape[0]), k_dist, label= "4-th distance")
plt.xlabel("Points")
plt.ylabel("Distance")
plt.title("4-th Distance")
plt.legend()
plt.show()
plt.close()

#We set epsilon to the distance we have at point 300
eps= k_dist[500]
dbscan= sklearn.cluster.DBSCAN(eps, 4, n_jobs=-1)
dbscan.fit(X)

pred_labels = dbscan.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
print('Number of clusters: %d' % n_clusters_)
print("Precision: %0.3f" % cluster_analysis.precision(fault, pred_labels))
print("Recall: %0.3f" % cluster_analysis.recall(fault, pred_labels))
print("F1: %0.3f" % cluster_analysis.f1_score(fault, pred_labels))
print("Rand Index: %0.3f" % cluster_analysis.rand_index(fault, pred_labels))
print("Adjusted Rand Index: %0.3f" % cluster_analysis.adj_rand_index(fault, pred_labels))
print("Silhouette: %0.3f" % cluster_analysis.silhouette(X, pred_labels))
data.plot_classes(pred_labels, longitude, latitude, alpha=0.5, edge='k')