import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn import mixture
import itertools

import cluster_analysis

import data_processing as data

latitude, longitude, fault = data.read_csv();
x, y, z = data.transform_coordinates(latitude, longitude);
#plot_cartesian_coordinates(x, y, z);
X= data.preprocess_data(x,y,z)

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
data.plot_classes(pred_labels, longitude, latitude, alpha=0.5, edge='k')