# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:41:16 2017

@author: Andrea
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d as plt3d
from skimage.io import imread
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist, pdist
from scipy import sparse as sp
import sklearn
from sklearn import mixture
import itertools

import cluster_analysis

FILENAME = 'tp2_data.csv'
RADIUS = 6371

def read_csv():
    """Reads the data from the csv file and returns the latitude, longitude and fault"""
    
    data = pd.read_csv(FILENAME)
    latitude = data.iloc[:,2]
    longitude = data.iloc[:,3]
    fault = data.iloc[:,-1]
    return latitude, longitude, fault

def transform_coordinates(latitude, longitude):
    """Transforms the latitude and longitude values into Earth-centered, Earth-fixed coordinates (x,y,z)"""
    
    x = RADIUS * np.cos(latitude * np.pi/180) * np.cos(longitude * np.pi/180)
    y = RADIUS * np.cos(latitude * np.pi/180) * np.sin(longitude * np.pi/180)
    z = RADIUS * np.sin(latitude * np.pi/180)
    return x, y, z	

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

def preprocess_data(x, y, z):
    num_row= x.shape[0]
    X= np.empty((num_row,3)) #I create an empty matrix with num_row rows and 3 columns
    for i in range(num_row):
        X[i,]= [x[i], y[i], z[i]]
    return X

latitude, longitude, fault = read_csv();
x, y, z = transform_coordinates(latitude, longitude);
plot_cartesian_coordinates(x, y, z);
X= preprocess_data(x,y,z)

############### GAUSSIAN MIXTURE ##############################################
lowest_bic = np.infty
bic = []
max_range= 100
n_components_range = range(1, max_range + 1)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    print('\n'+cv_type)
    iter= 0
    for n_components in n_components_range:
        print(iter/max_range)
        iter+= 1
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)


##BEST GMM
clf.fit(X)
pred_labels = clf.predict(X)

# Number of clusters in labels, ignoring noise if present.
n_clusters_= clf.n_components
print('Number of clusters: %d' % n_clusters_)
print("Precision: %0.3f" % cluster_analysis.precision(fault, pred_labels))
print("Recall: %0.3f" % cluster_analysis.recall(fault, pred_labels))
print("F1: %0.3f" % cluster_analysis.f1_score(fault, pred_labels))
print("Rand Index: %0.3f" % cluster_analysis.rand_index(fault, pred_labels))
print("Adjusted Rand Index: %0.3f" % cluster_analysis.adj_rand_index(fault, pred_labels))
print("Silhouette: %0.3f" % cluster_analysis.silhouette(X, pred_labels))
plot_classes(pred_labels, longitude, latitude, alpha=0.5, edge='k')