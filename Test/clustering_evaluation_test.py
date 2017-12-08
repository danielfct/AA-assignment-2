                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       # -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:22:50 2017

@author: Andrea
"""

import numpy as np
import cluster_analysis
from sklearn import metrics


def FMI(labels_true, labels_pred):
    TP, TN, FP, FN= cluster_analysis.clustering_evaluation(labels_true, labels_pred)
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
print("Precision: %0.3f" % cluster_analysis.precision([0, 0, 1, 1], [0, 0, 1, 1]))
print("Recall: %0.3f" % cluster_analysis.recall([0, 0, 1, 1], [0, 0, 1, 1]))
print("F1: %0.3f" % cluster_analysis.f1_score([0, 0, 1, 1], [0, 0, 1, 1]))

#TEST 2
print("\ntest 2")
test([0, 0, 1, 1], [1, 1, 0, 0])
print("Precision: %0.3f" % cluster_analysis.precision([0, 0, 1, 1], [1, 1, 0, 0]))
print("Recall: %0.3f" % cluster_analysis.recall([0, 0, 1, 1], [1, 1, 0, 0]))
print("F1: %0.3f" % cluster_analysis.f1_score([0, 0, 1, 1], [1, 1, 0, 0]))

#TEST 3
print("\ntest 3")
test([0, 0, 0, 0], [0, 1, 2, 3])
print("Precision: %0.3f" % cluster_analysis.precision([0, 0, 0, 0], [0, 1, 2, 3]))
print("Recall: %0.3f" % cluster_analysis.recall([0, 0, 0, 0], [0, 1, 2, 3]))
print("F1: %0.3f" % cluster_analysis.f1_score([0, 0, 0, 0], [0, 1, 2, 3]))

#TEST 4
print("\ntest 4")
test([1, 1, 2, 2], [0, 0, 3, 3])
print("Precision: %0.3f" % cluster_analysis.precision([1, 1, 2, 2], [0, 0, 3, 3]))
print("Recall: %0.3f" % cluster_analysis.recall([1, 1, 2, 2], [0, 0, 3, 3]))
print("F1: %0.3f" % cluster_analysis.f1_score([1, 1, 2, 2], [0, 0, 3, 3]))