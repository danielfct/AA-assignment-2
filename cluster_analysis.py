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
    contingency : {array-like}, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`.
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
    #print(contingency_table)
        #checking dimensions match
    n= contingency_table.sum()
    if (labels_true.shape[0] != n):
        raise ValueError("Labels and Contingency Table dimension do NOT match!")
    #computing the quantities that will be required to find the values
    squared_table_sum= np.square(contingency_table).sum()
    #print(type(squared_table_sum)) #check if it is np.int64
    squared_partition_sum= np.square(contingency_table.sum(axis= 1)).sum()
    #print(type(squared_partition_sum)) #check if it is np.int64
    squared_cluster_sum= np.square(contingency_table.sum(axis= 0)).sum()
    #print(type(squared_cluster_sum)) #check if it is np.int64
    squared_n= np.square(n)
    #print(type(squared_n)) #check if it is np.int64
    
    tp= 0.5 * (squared_table_sum - n)
    #print(type(tp))
    #print(tp)
    fn= 0.5 * (squared_partition_sum - squared_table_sum)
    #print(type(fn))
    #print(fn)
    fp= 0.5 * (squared_cluster_sum - squared_table_sum)
    #print(type(fp))
    #print(fp)
    tn= 0.5 * (squared_n - squared_cluster_sum - squared_partition_sum + squared_table_sum)
    #print(type(fn))
    #print(fn)
    
    #print('n is ' + str(n))
    if (tp+fn+fp+tn) != (n * (n - 1) / 2):
        raise ValueError('Something is wrong with tp, fn, fp, fn')
    return tp, fn, fp, tn

def precision(tp, fp):
    return tp/(tp+fp)
    
def recall(tp, fn):
    return tp/(tp+fn)

def f1_score(tp, fn, fp):
    prec= precision(tp, fp)
    rec= recall(tp, fn)
    return 2 *(prec * rec) / (prec + rec)

def rand_index(tp, fn, fp, tn):
    return (tp + tn) / (tp + tn + fp + tn)

def adj_rand_index(labels_true, labels_pred):
    return metrics.adjusted_rand_score(labels_true, labels_pred)

def silhouette(X, labels_pred):
    return metrics.silhouette_score(X, labels_pred)

def evaluate_cluster(X, labels_true, labels_pred):
    tp, fn, fp, tn= positive_negative(labels_true, labels_pred)
    return np.array([
            precision(tp, fp), 
            recall(tp, fn), 
            f1_score(tp, fn, fp), 
            rand_index(tp, fn, fp, tn), 
            adj_rand_index(labels_true, labels_pred), 
            silhouette(X, labels_pred)
            ])