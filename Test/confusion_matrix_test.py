# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:33:13 2017

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
from sklearn.metrics import confusion_matrix

def check_clusterings(labels_true, labels_pred):
    """Check that the two clusterings matching 1D integer arrays."""
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    # input checks
    if labels_true.ndim != 1:
        raise ValueError(
            "labels_true must be 1D: shape is %r" % (labels_true.shape,))
    if labels_pred.ndim != 1:
        raise ValueError(
            "labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
    if labels_true.shape != labels_pred.shape:
        raise ValueError(
            "labels_true and labels_pred must have same size, got %d and %d"
            % (labels_true.shape[0], labels_pred.shape[0]))
    return labels_true, labels_pred

def contingency_matrix(labels_true, labels_pred):
    """Build a contingency matrix describing the relationship between labels.
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference
    labels_pred : array, shape = [n_samples]
        Cluster labels to evaluate
    Returns
    -------
    contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`. If
        ``eps is None``, the dtype of this array will be integer. If ``eps`` is
        given, the dtype will be float.
        Will be a ``scipy.sparse.csr_matrix`` if ``sparse=True``.
    """

    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
                                 (class_idx, cluster_idx)),
                                shape=(n_classes, n_clusters),
                                dtype=np.int64)
    contingency = contingency.toarray()
    return contingency

def positive_negative(labels_true, labels_pred):
    labels_true, labels_pred= check_clusterings(labels_true, labels_pred)
    contingency_table= contingency_matrix(labels_true, labels_pred)
        #checking dimensions match
    n= contingency_table.sum()
    #if (labels_true.shape[0] != n):
    #    raise ValueError("Labels and Contingency Table dimension do NOT match!")
    #computing the quantities that will be required to find the values
    squared_table_sum= np.square(contingency_table).sum()
    print(type(squared_table_sum)) #check if it is np.int64
    squared_partition_sum= np.square(contingency_table.sum(axis= 0)).sum()
    print(type(squared_partition_sum)) #check if it is np.int64
    squared_cluster_sum= np.square(contingency_table.sum(axis= 1)).sum()
    print(type(squared_cluster_sum)) #check if it is np.int64
    squared_n= np.square(n)
    print(type(squared_n)) #check if it is np.int64
    
    tp= 0.5 * (squared_table_sum - n)
    print(type(tp))
    print(tp)
    fn= 0.5 * (squared_partition_sum - squared_table_sum)
    print(type(fn))
    print(fn)
    fp= 0.5 * (squared_cluster_sum - squared_table_sum)
    print(type(fp))
    print(fp)
    tn= 0.5 * (squared_n - squared_cluster_sum - squared_partition_sum + squared_table_sum)
    print(type(fn))
    print(fn)
    
    print('n is ' + str(n))
    if (tp+fn+fp+tn) != (n * (n - 1) / 2):
        raise ValueError('Something is wrong with tp, fn, fp, fn')
    return tp, fn, fp, fn

def positive_negative_confusion_test(contingency_table):
    #checking dimensions match
    n= contingency_table.sum()
    #if (labels_true.shape[0] != n):
    #    raise ValueError("Labels and Contingency Table dimension do NOT match!")
    #computing the quantities that will be required to find the values
    squared_table_sum= np.square(contingency_table).sum()
    print(type(squared_table_sum)) #check if it is np.int64
    squared_partition_sum= np.square(contingency_table.sum(axis= 0)).sum()
    print(type(squared_partition_sum)) #check if it is np.int64
    squared_cluster_sum= np.square(contingency_table.sum(axis= 1)).sum()
    print(type(squared_cluster_sum)) #check if it is np.int64
    squared_n= np.square(n)
    print(type(squared_n)) #check if it is np.int64
    
    tp= 0.5 * (squared_table_sum - n)
    print(type(tp))
    print(tp)
    fn= 0.5 * (squared_partition_sum - squared_table_sum)
    print(type(fn))
    print(fn)
    fp= 0.5 * (squared_cluster_sum - squared_table_sum)
    print(type(fp))
    print(fp)
    tn= 0.5 * (squared_n - squared_cluster_sum - squared_partition_sum + squared_table_sum)
    print(type(fn))
    print(fn)
    
    print('n is ' + str(n))
    if (tp+fn+fp+tn) != (n * (n - 1) / 2):
        raise ValueError('Something is wrong with tp, fn, fp, fn')
    return tp, fn, fp, fn

def precision(tp, fp):
    return tp/(tp+fp)
    
def recall(tp, fn):
    return tp/(tp+fn)

labels_true= np.array([1, 3, 2, 1])
labels_pred= np.array([0, 1, 0, 1])
test= contingency_matrix(labels_true, labels_pred)
print(test)

#true positive
print('\nTrue Positive')
tp= 0.5 * (np.square(test).sum() - labels_true.shape[0])
print(tp)

#false negative
print('\nFalse Negative')
fn= 0.5 * ( np.square(test.sum(axis= 0)).sum() - np.square(test).sum())
print(fn)

#false positive
print('\nFalse Positive')
fp= 0.5 * ( np.square(test.sum(axis= 1)).sum() - np.square(test).sum())
print(fp)

#true negative
print('\nTrue Negative')
tn= 0.5 * (np.square(labels_true.shape[0]) - np.square(test.sum(axis= 1)).sum()
-  np.square(test.sum(axis= 0)).sum() + np.square(test).sum())
print(tn)

print("CHECK")
print(tn+fp+fn+tp)
print(labels_true.shape[0] * (labels_true.shape[0]-1) * 0.5)


#test 2
test_new= np.array([[0, 50, 0], [47, 0, 3], [14, 0, 36]], dtype= np.int64)
print(test_new)

tp, fn, fp, fn= positive_negative_confusion_test(test_new)
print("TP")
print(tp)
print("FN")
print(fn)
print("FP")
print(fp)
print("FN")
print(fn)