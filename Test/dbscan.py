import numpy as np
from sklearn import neighbors
from sklearn import cluster
import matplotlib.pyplot as plt

import cluster_analysis

def k_distance(X, k=4):
    """First of all, We have to set the eps parameter of the classifier. To do so,
    we create a fictitious output vector, filled with zeros and then applied the
    kNN classifier."""
    
    aux_label = np.zeros(X.shape[0])
    knn = neighbors.KNeighborsClassifier(n_neighbors=4)
    knn.fit(X, aux_label)
    distances = knn.kneighbors()
    k_dist = np.zeros(X.shape[0])
    for i in range(0, X.shape[0]):
        k_dist[i] = distances[0][i,3]
    k_dist.sort()
    k_dist = k_dist[::-1]
    return k_dist

def plot_k_distance(k_dist, k=4):
    fig = plt.figure()
    plt.plot(range(0, k_dist.shape[0]), k_dist, label=str(k) + "-th distance")
    plt.xlabel("Points")
    plt.ylabel("Distance")
    plt.title(str(k) + "-th Distance")
    plt.legend()
    plt.show()
    fig.savefig('k_dist_dbscan.pdf')
    plt.close()


def dbscan_tuning(X, labels_true, min_eps, max_eps, pace):
    """This function computes the DBSCAN algorithm for the values comprised
    between [eps-delta, eps+delta] if possible: if eps-delta is lesser than
    10 it sets 10, if eps+delta is greater than the biggest k-th distance it takes
    the biggest k-th distance as maximum value."""
    
    values = np.arange(min_eps, max_eps, pace)
    indices = np.zeros((values.shape[0], 6))
    n_clusters = np.zeros(values.shape[0], dtype=np.int)
    i = 0
    for curr_eps in np.arange(min_eps, max_eps, pace):
        print("Current epsilon " + str(curr_eps))
        dbscan_model = cluster.DBSCAN(curr_eps, 4, n_jobs= -1)
        dbscan_model.fit(X)
        pred_labels = dbscan_model.labels_
        n_clusters_ = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
        n_clusters[i]= n_clusters_
        #print(n_clusters_)
        indices[i]= cluster_analysis.evaluate_cluster(X, labels_true, pred_labels)
        i+= 1
    return indices, n_clusters
        

def plot_indices(indices, min_eps, max_eps, pace):
    index_name = ['Precision', 'Recall', 'F1Score', 'Rand Index', 'Adjusted Rand Index', 'Silhouette']
    x_axis = np.arange(min_eps, max_eps, pace)
    fig = plt.figure(figsize=(20,20))
    for i in range(0,6):
        plt.subplot(3, 2, i+1)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.title('DBSCAN ' + index_name[i])
        plt.plot(x_axis, indices[:,i])
        plt.ylabel(index_name[i])
        plt.xlabel('Value of Epsilon')
    plt.show()
    fig.savefig('dbscan_indeces.pdf')
    plt.close()

def plot_cluster(n_clusters, min_eps, max_eps, pace):
    #print(n_clusters)
    x_axis = np.arange(min_eps, max_eps, pace)
    fig = plt.figure()
    plt.title('DBSCAN - Number of clusters')
    plt.plot(x_axis, n_clusters.ravel())
    plt.ylabel('N. Cluster')
    plt.xlabel('Value of Epsilon')
    plt.show()
    fig.savefig('dbscan_num_cluster.pdf')
    plt.close() 