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

def k_distance(X, k= 4):
    aux_label= np.zeros(X.shape[0])
    neigh= sklearn.neighbors.KNeighborsClassifier(n_neighbors= 4)
    neigh.fit(X, aux_label)
    distances= neigh.kneighbors()
    k_dist= np.zeros(X.shape[0])
    for i in range(0, X.shape[0]):
        k_dist[i]= distances[0][i,3]
    k_dist.sort()
    k_dist= k_dist[::-1]
    return k_dist

def plot_k_distance(k_dist, k= 4):
    fig= plt.figure()
    plt.plot(range(0, k_dist.shape[0]), k_dist, label= str(k) + "-th distance")
    plt.xlabel("Points")
    plt.ylabel("Distance")
    plt.title(str(k) + "-th Distance")
    plt.legend()
    plt.show()
    fig.savefig('k_dist_dbscan.pdf')
    plt.close()


def dbscan_tuning(X, labels_true, min_eps, max_eps, pace):
    '''This function computes the DBSCAN algorithm for the values comprised
    between [eps-delta, eps+delta] if possible: if eps-delta is lesser than
    10 it sets 10, if eps+delta is greater than the biggest k-th distance it takes
    the biggest k-th distance as maximum value.'''
    values= np.arange(min_eps, max_eps, pace)
    dbscan_indices= np.zeros((values.shape[0], 6))
    dbscan_n_clusters= np.zeros(values.shape[0], dtype= np.int)
    i= 0
    for curr_eps in np.arange(min_eps, max_eps, pace):
        print(curr_eps)
        dbscan= sklearn.cluster.DBSCAN(curr_eps, 4, n_jobs= -1)
        dbscan.fit(X)
        pred_labels = dbscan.labels_
        #print(len(set(pred_labels)) - (1 if -1 in pred_labels else 0))
        n_clusters_ = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
        #print(n_clusters_)
        dbscan_n_clusters[i]= n_clusters_
        #print(dbscan_n_clusters[i])
        dbscan_indices[i]= cluster_analysis.evaluate_cluster(X, labels_true, pred_labels)
        i+= 1
    return dbscan_indices, dbscan_n_clusters
        

def dbscan_plot(dbscan_indices, min_eps, max_eps, pace):
    index_name= ['Precision', 'Recall', 'F1Score', 'Rand Index', 'Adjusted Rand Index', 'Silhouette']
    x_axis= np.arange(min_eps, max_eps, pace)
    fig= plt.figure(figsize=(20,20))
    for i in range(0,6):
        plt.subplot(3, 2, i+1)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.title('DBSCAN ' + index_name[i])
        plt.plot(x_axis, dbscan_indices[:,i])
        plt.ylabel(index_name[i])
        plt.xlabel('Value of Epsilon')
    plt.show()
    fig.savefig('dbscan_indeces.pdf')
    plt.close()

def dbscan_plot_cluster(dbscan_n_clusters, min_eps, max_eps, pace):
    print(dbscan_n_clusters)
    x_axis= np.arange(min_eps, max_eps, pace)
    fig= plt.figure()
    plt.title('DBSCAN - Number of clusters')
    plt.plot(x_axis, dbscan_n_clusters.ravel())
    plt.ylabel('N. Cluster')
    plt.xlabel('Value of Epsilon')
    plt.show()
    fig.savefig('dbscan_num_cluster.pdf')
    plt.close()
    
def dbscan(X, labels_true, eps, delta, pace= 1):
    k_dist= k_distance(X)
    plot_k_distance(k_dist)
    #We set epsilon to the distance we have at point 500
    min_eps= max(10, eps-delta)
    max_eps= min(k_dist.max(), eps+delta)
    dbscan_indices, dbscan_n_clusters= dbscan_tuning(X, fault, min_eps, max_eps, pace)
    
    
    dbscan_plot(dbscan_indices, min_eps, max_eps, pace)
    dbscan_plot_cluster(dbscan_n_clusters, min_eps, max_eps, pace)
    eps_paper= k_dist[500]
    dbscan= sklearn.cluster.DBSCAN(eps_paper, 4, n_jobs=-1)
    dbscan.fit(X)
    pred_labels = dbscan.labels_
    n_clusters_ = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
    dbscan_evaluate_paper= cluster_analysis.evaluate_cluster(X, labels_true, dbscan.labels_)
    print('Number of clusters: %d' % n_clusters_)
    print("Precision: %0.3f" % dbscan_evaluate_paper[0])
    print("Recall: %0.3f" % dbscan_evaluate_paper[1])
    print("F1: %0.3f" % dbscan_evaluate_paper[2])
    print("Rand Index: %0.3f" % dbscan_evaluate_paper[3])
    print("Adjusted Rand Index: %0.3f" % dbscan_evaluate_paper[4])
    print("Silhouette: %0.3f" % dbscan_evaluate_paper[5])
    data.plot_classes(pred_labels, longitude, latitude, alpha=0.5, edge='k')
    
#dbscan(X, fault, 300, 300, 1)
dbscan(X, fault, 300, 300, 1)
#TODO: compute the indices excluding the noise!