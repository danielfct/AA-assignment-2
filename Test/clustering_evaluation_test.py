# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:22:50 2017

@author: Andrea
"""
import numpy as np
from scipy import sparse as sp
from sklearn import metrics

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

def contingency_matrix(labels_true, labels_pred, eps=None, sparse=False):
    """Build a contingency matrix describing the relationship between labels.
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference
    labels_pred : array, shape = [n_samples]
        Cluster labels to evaluate
    eps : None or float, optional.
        If a float, that value is added to all values in the contingency
        matrix. This helps to stop NaN propagation.
        If ``None``, nothing is adjusted.
    sparse : boolean, optional.
        If True, return a sparse CSR continency matrix. If ``eps is not None``,
        and ``sparse is True``, will throw ValueError.
        .. versionadded:: 0.18
    Returns
    -------
    contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`. If
        ``eps is None``, the dtype of this array will be integer. If ``eps`` is
        given, the dtype will be float.
        Will be a ``scipy.sparse.csr_matrix`` if ``sparse=True``.
    """

    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

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
                                dtype=np.int)
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
    return contingency

def clustering_evaluation(labels_true, labels_pred):
    labels_true, labels_pred= check_clusterings(labels_true, labels_pred)
    c= contingency_matrix(labels_true, labels_pred, sparse= True)
    n_samples, = labels_true.shape
    tk = np.dot(c.data, c.data) - n_samples
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples
    N= labels_true.shape[0] * (labels_true.shape[0]-1) / 2
    TP= np.int64(tk)
    FP= np.int64(pk) - np.int64(tk)
    FN= np.int64(qk) - np.int64(tk)
    TN= np.int64(N) - TP - FP - FN
    return TP, TN, FP, FN

def precision(labels_true, labels_pred):
    TP, TN, FP, FN= clustering_evaluation(labels_true, labels_pred)
    if TP == 0 and FP == 0:
        print("Both TP and FP are null!")
        return -1
    return TP / (TP + FP)

def recall(labels_true, labels_pred):
    TP, TN, FP, FN= clustering_evaluation(labels_true, labels_pred)
    return TP / (FN + TP)

def F1Score(labels_true, labels_pred):
    prec= precision(labels_true, labels_pred)
    rec= recall(labels_true, labels_pred)
    return 2 *(prec * rec) / (prec + rec)


def FMI(labels_true, labels_pred):
    TP, TN, FP, FN= clustering_evaluation(labels_true, labels_pred)
    if TP != 0:
        fmi= np.float64(TP / np.sqrt((TP + FP) * (TP + FN)))
    else:
        fmi = 0
    return fmi

def test(labels_true, labels_pred):
    fmi= FMI(labels_true, labels_pred)
    fmi_true= np.float64(metrics.fowlkes_mallows_score(labels_true, labels_pred))
    print("Testing FMI: %0.3f" % fmi)
    print("True FMI: %0.3f" % fmi_true)
#I try to obtain the FP, TP, TN, FN by means of the functions already
#contained in scikit-learn. (https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/metrics/cluster/supervised.py#L787)
#I still have to check whether the maths is understable

#TEST 1
print("\ntest 1")
test([0, 0, 1, 1], [0, 0, 1, 1])
print("Precision: %0.3f" % precision([0, 0, 1, 1], [0, 0, 1, 1]))
print("Recall: %0.3f" % recall([0, 0, 1, 1], [0, 0, 1, 1]))
print("F1: %0.3f" % F1Score([0, 0, 1, 1], [0, 0, 1, 1]))

#TEST 2
print("\ntest 2")
test([0, 0, 1, 1], [1, 1, 0, 0])
print("Precision: %0.3f" % precision([0, 0, 1, 1], [1, 1, 0, 0]))
print("Recall: %0.3f" % recall([0, 0, 1, 1], [1, 1, 0, 0]))
print("F1: %0.3f" % F1Score([0, 0, 1, 1], [1, 1, 0, 0]))

#TEST 3
print("\ntest 3")
test([0, 0, 0, 0], [0, 1, 2, 3])
print("Precision: %0.3f" % precision([0, 0, 0, 0], [0, 1, 2, 3]))
print("Recall: %0.3f" % recall([0, 0, 0, 0], [0, 1, 2, 3]))
print("F1: %0.3f" % F1Score([0, 0, 0, 0], [0, 1, 2, 3]))

#TEST 4
print("\ntest 4")
test([1, 1, 2, 2], [0, 0, 3, 3])
print("Precision: %0.3f" % precision([1, 1, 2, 2], [0, 0, 3, 3]))
print("Recall: %0.3f" % recall([1, 1, 2, 2], [0, 0, 3, 3]))
print("F1: %0.3f" % F1Score([1, 1, 2, 2], [0, 0, 3, 3]))
