import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sklearn
from skimage.io import imread

import data_processing
import cluster_analysis
import k_means
import dbscan
import gaussian
    
def plot_cartesian_coordinates(x, y, z):
    """Plot Cartesian coordinates of seismic events"""
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.set_title('Cartesian coordinates of Seismic Events', {'fontsize':14, 'fontweight':'bold'})
    ax.scatter3D(x, y, z, '.', s=10, c='green')
    plt.savefig('Seismic_events_cartesian_coordinates.png', bbox_inches='tight');

def plot_classes(plot_name, labels, longitude, latitude, alpha=0.5, edge='k'):
    """Plot seismic events using Mollweide projection.
    Arguments are the cluster labels and the longitude and latitude
    vectors of the events"""
    img = imread("Mollweide_projection_SW.jpg")        
    plt.figure(figsize=(10,5),frameon=False)    
    x = longitude/180*np.pi
    y = latitude/180*np.pi
    ax = plt.subplot(111, projection="mollweide")
    t = ax.transData.transform(np.vstack((x,y)).T)
    clims = np.array([(-np.pi,0),(np.pi,0),(0,-np.pi/2),(0,np.pi/2)])
    lims = ax.transData.transform(clims)
    plt.close()
    fig = plt.figure(figsize=(10,5),frameon=False)    
    plt.subplot(111)
    plt.imshow(img,zorder=0,extent=[lims[0,0],lims[1,0],lims[2,1],lims[3,1]],aspect=1)        
    x = t[:,0]
    y= t[:,1]
    nots = np.zeros(len(labels)).astype(bool)
    diffs = np.unique(labels)    
    ix = 0   
    for lab in diffs[diffs>=0]:        
        mask = labels==lab
        nots = np.logical_or(nots,mask)        
        plt.plot(x[mask], y[mask],'o', markersize=4, mew=1,zorder=1,alpha=alpha, markeredgecolor=edge)
        ix = ix+1                    
    mask = np.logical_not(nots)    
    if np.sum(mask)>0:
        plt.plot(x[mask], y[mask], '.', markersize=1, mew=1,markerfacecolor='w', markeredgecolor=edge)
    plt.axis('off')
    plt.title(plot_name)
    fig.savefig(plot_name)

def kmeans_performance(X, labels_true, longitude, latitude, max_cluster):
    kmeans_eval = k_means.kmeans_tuning(X, max_cluster, labels_true, 205)
    k_means.plot_cluster(max_cluster, kmeans_eval)

    index_name = ['Precision', 'Recall', 'F1-Score', 'Rand Index', 'Adjusted Rand Index', 'Silhouette']
    for i in range(0, len(index_name)):
        best_k = kmeans_eval[:,i].argmax() + 2
        best_kmeans_model = sklearn.cluster.KMeans(best_k, random_state=205)
        best_kmeans_model.fit(X)
        labels_pred = best_kmeans_model.labels_
        best_kmeans_eval = cluster_analysis.evaluate_cluster(X, labels_true, labels_pred)
        print('\nMaximising ' + index_name[i])
        print('Number of clusters: %d' % best_k)
        print("Precision: %0.3f" % best_kmeans_eval[0])
        print("Recall: %0.3f" % best_kmeans_eval[1])
        print("F1: %0.3f" % best_kmeans_eval[2])
        print("Rand Index: %0.3f" % best_kmeans_eval[3])
        print("Adjusted Rand Index: %0.3f" % best_kmeans_eval[4])
        print("Silhouette: %0.3f" % best_kmeans_eval[5])
    # Plot classes when maximizing Silhouette
    plot_classes("KMeans", labels_pred, longitude, latitude, alpha=0.5, edge='k')

def dbscan_performance(X, labels_true, longitude, latitude, epsilon, delta, pace= 1):
    #Calculate k_distances
    k_dist = dbscan.k_distance(X)
    dbscan.plot_k_distance(k_dist)
    #We set epsilon to the distance we have at point 500
    min_eps = max(10, epsilon-delta)
    max_eps = min(k_dist.max(), epsilon+delta)
    indices, n_clusters = dbscan.dbscan_tuning(X, labels_true, min_eps, max_eps, pace)
    dbscan.plot_indices(indices, min_eps, max_eps, pace)
    dbscan.plot_cluster(n_clusters, min_eps, max_eps, pace)
    eps_paper = k_dist[500]
    dbscan_model = sklearn.cluster.DBSCAN(eps_paper, 4, n_jobs=-1)
    dbscan_model.fit(X)
    labels_pred = dbscan_model.labels_
   
    dbscan_eval_paper = cluster_analysis.evaluate_cluster(X, labels_true, labels_pred)
    n_clusters_ = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)
    print('Number of clusters: %d' % n_clusters_)
    print("Precision: %0.3f" % dbscan_eval_paper[0])
    print("Recall: %0.3f" % dbscan_eval_paper[1])
    print("F1: %0.3f" % dbscan_eval_paper[2])
    print("Rand Index: %0.3f" % dbscan_eval_paper[3])
    print("Adjusted Rand Index: %0.3f" % dbscan_eval_paper[4])
    print("Silhouette: %0.3f" % dbscan_eval_paper[5])
    plot_classes("DBSCAN", labels_pred, longitude, latitude, alpha=0.5, edge='k')

def gmm_performance(X, labels_true, longitude, latitude, max_range):
    gmm_indices, gmm = gaussian.gmm_tuning(X, labels_true, max_range)
    gaussian.gmm_plot(gmm_indices, max_range)
    gmm.fit(X)
    labels_pred= gmm.predict(X)
    
    gmm_evaluate= cluster_analysis.evaluate_cluster(X, labels_true, labels_pred)
    n_clusters = gmm.n_components
    print('Number of components: %d' % n_clusters)
    print("Precision: %0.3f" % gmm_evaluate[0])
    print("Recall: %0.3f" % gmm_evaluate[1])
    print("F1: %0.3f" % gmm_evaluate[2])
    print("Rand Index: %0.3f" % gmm_evaluate[3])
    print("Adjusted Rand Index: %0.3f" % gmm_evaluate[4])
    print("Silhouette: %0.3f" % gmm_evaluate[5])
    plot_classes("GMM", labels_pred, longitude, latitude, alpha=0.5, edge='k')

def main():
    # Get the data
    filename = 'tp2_data.csv'
    latitude, longitude, fault = data_processing.read_csv(filename);
    x, y, z = data_processing.transform_coordinates(latitude, longitude);
    plot_cartesian_coordinates(x, y, z);
    X= data_processing.preprocess_data(x, y, z)
    
    plot_classes("Faults", fault, longitude, latitude)
    
    # KMEANS
    print("Testing KMeans Model")
    max_cluster= 150
    kmeans_performance(X, fault, longitude, latitude, max_cluster)
    
    # DBSCAN
    print("Testing DBSCAN Model")
    epsilon = 300
    delta = 300
    dbscan_performance(X, fault, longitude, latitude, epsilon, delta)
    
    # GMM
    print("Testing Gaussian Mixture Model")
    max_range = 200
    gmm_performance(X, fault, longitude, latitude, max_range)
    
main()