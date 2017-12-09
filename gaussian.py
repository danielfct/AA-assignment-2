import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture

import cluster_analysis

def gmm_tuning(X, labels_true, max_range):
    n_components_range = range(2, max_range + 1)
    lowest_bic = np.infty
    bic = []
    gmm_indices= np.zeros((max_range - 1, 6))
    i= 0
    for n_components in n_components_range:
        print(n_components)
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components= n_components)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
        labels_pred = gmm.predict(X)
        gmm_indices[i]= cluster_analysis.evaluate_cluster(X, labels_true, labels_pred)
        i+= 1
    return gmm_indices, best_gmm

def gmm_plot(gmm_indices, max_range):
    index_name= ['Precision', 'Recall', 'F1Score', 'Rand Index', 'Adjusted Rand Index', 'Silhouette']
    x_axis= range(2, max_range + 1)
    fig= plt.figure(figsize=(20,20))
    for i in range(0,6):
        plt.subplot(3, 2, i+1)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.title('Gaussian Mixture Model ' + index_name[i])
        plt.plot(x_axis, gmm_indices[:,i])
        plt.ylabel(index_name[i])
        plt.xlabel('Number of Components')
    plt.show()
    fig.savefig('gmm_indices')
    plt.close()