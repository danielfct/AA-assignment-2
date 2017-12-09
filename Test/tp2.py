import numpy as np
import matplotlib.pyplot as plt
import sklearn
from skimage.io import imread

import data_processing
import cluster_analysis
import dbscan
    
def plot_cartesian_coordinates(x, y, z):
    """Plot Cartesian coordinates of seismic events"""
    
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.set_title('Cartesian coordinates of Seismic Events', {'fontsize':14, 'fontweight':'bold'})
    ax.scatter3D(x, y, z, '.', s=10, c='green')
    plt.savefig('Seismic_events_cartesian_coordinates.png', bbox_inches='tight');

def plot_classes(labels, longitude, latitude, alpha=0.5, edge='k'):
    """Plot seismic events using Mollweide projection.
    Arguments are the cluster labels and the longitude and latitude
    vectors of the events"""

    img = imread("Mollweide_projection_SW.jpg")        
    plt.figure(figsize=(10,5), frameon=False)    
    x = longitude/180 * np.pi
    y = latitude/180 * np.pi
    ax = plt.subplot(111, projection="mollweide")
    #print(ax.get_xlim(), ax.get_ylim())
    t = ax.transData.transform(np.vstack((x, y)).T)
    #print(np.min(np.vstack((x, y)).T, axis=0))
    #print(np.min(t, axis=0))
    clims = np.array([(-np.pi, 0), (np.pi, 0), (0, -np.pi/2), (0, np.pi/2)])
    lims = ax.transData.transform(clims)
    plt.close()
    plt.figure(figsize=(10,5), frameon=False)    
    plt.subplot(111)
    plt.imshow(img, zorder=0, extent=[lims[0,0], lims[1,0], lims[2,1], lims[3,1]], aspect=1)        
    x = t[:,0]
    y= t[:,1]
    nots = np.zeros(len(labels)).astype(bool)
    diffs = np.unique(labels)    
    ix = 0   
    for lab in diffs[diffs >= 0]:        
        mask = labels == lab
        nots = np.logical_or(nots, mask)        
        plt.plot(x[mask], y[mask], 'o', markersize=4, mew=1, zorder=1, alpha=alpha, markeredgecolor=edge)
        ix = ix + 1                    
    mask = np.logical_not(nots)    
    if np.sum(mask) > 0:
        plt.plot(x[mask], y[mask], '.', markersize=1, mew=1, markerfacecolor='w', markeredgecolor=edge)
    plt.axis('off') 

def test_dbscan(X, labels_true, longitude, latitude, eps, delta, pace= 1):
    k_dist= dbscan.k_distance(X)
    dbscan.plot_k_distance(k_dist)
    #We set epsilon to the distance we have at point 500
    min_eps= max(10, eps-delta)
    max_eps= min(k_dist.max(), eps+delta)
    dbscan_indices, dbscan_n_clusters= dbscan.dbscan_tuning(X, labels_true, min_eps, max_eps, pace)
    dbscan.dbscan_plot(dbscan_indices, min_eps, max_eps, pace)
    dbscan.dbscan_plot_cluster(dbscan_n_clusters, min_eps, max_eps, pace)
    eps_paper= k_dist[500]
    dbscan_model = sklearn.cluster.DBSCAN(eps_paper, 4, n_jobs=-1)
    dbscan_model.fit(X)
    pred_labels = dbscan.labels_
    n_clusters_ = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
    dbscan_evaluate_paper= cluster_analysis.evaluate_cluster(X, labels_true, pred_labels)
    print('Number of clusters: %d' % n_clusters_)
    print("Precision: %0.3f" % dbscan_evaluate_paper[0])
    print("Recall: %0.3f" % dbscan_evaluate_paper[1])
    print("F1: %0.3f" % dbscan_evaluate_paper[2])
    print("Rand Index: %0.3f" % dbscan_evaluate_paper[3])
    print("Adjusted Rand Index: %0.3f" % dbscan_evaluate_paper[4])
    print("Silhouette: %0.3f" % dbscan_evaluate_paper[5])
    plot_classes(pred_labels, longitude, latitude, alpha=0.5, edge='k')

def main():
    # Get the data
    latitude, longitude, fault = data_processing.read_csv();
    x, y, z = data_processing.transform_coordinates(latitude, longitude);
    #plot_cartesian_coordinates(x, y, z);
    X= data_processing.preprocess_data(x, y, z)
    
    # DBSCAN
    test_dbscan(X, fault, longitude, latitude, 300, 300, 1)
    #TODO: compute the indices excluding the noise!
    
     ## TODO do the same for kmeans and gaussian