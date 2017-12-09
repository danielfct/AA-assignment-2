import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import cluster_analysis

def kmeans_tuning(X, max_cluster, labels_true, seed):
    """The function takes as input the dataset X,
    the maximum number of clusters we are willing to have,
    the true labelling of the data, the random seed.
    It computes the clusters applying the kmeans algorithm on the dataset using
    the given seed, from 2 to max_cluster. It outputs quality of indeces
    of the clustering: precision, recall, f1score, rand index, adjusted
    rand index, silhouette"""
    kmeans_eval= np.zeros((max_cluster - 1, 6))
    i= 0
    for k in range(2, max_cluster + 1):
        print("Current k " + str(k))
        kmeans_model = KMeans(k, random_state= seed)
        kmeans_model.fit(X)
        labels_pred= kmeans_model.labels_
        current_eval= cluster_analysis.evaluate_cluster(X, labels_true, labels_pred)
        kmeans_eval[i,:]= current_eval
        i += 1
    return kmeans_eval

def plot_cluster(max_cluster, kmeans_eval):
    index_name= ['Precision', 'Recall', 'F1Score', 'Rand Index', 'Adjusted Rand Index', 'Silhouette']
    x_axis= range(2, max_cluster + 1)
    fig= plt.figure(figsize=(20,20))
    for i in range(0, 6):
        plt.subplot(3, 2, i+1)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.title('KMeans ' + index_name[i])
        plt.plot(x_axis, kmeans_eval[:,i])
        plt.ylabel(index_name[i])
        plt.xlabel('Number of clusters')
    plt.show()
    fig.savefig('kmeans_indeces.pdf')
    plt.close()