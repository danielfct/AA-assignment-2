import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d as plt3d
from skimage.io import imread
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist, pdist
from scipy import sparse as sp

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
    print(ax.get_xlim(), ax.get_ylim())
    t = ax.transData.transform(np.vstack((x, y)).T)
    print(np.min(np.vstack((x, y)).T, axis=0))
    print(np.min(t, axis=0))
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


################ CLUSTER EVALUATION ###########################################
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

def f1_score(labels_true, labels_pred):
    prec= precision(labels_true, labels_pred)
    rec= recall(labels_true, labels_pred)
    return 2 *(prec * rec) / (prec + rec)

def rand_index(labels_true, labels_pred):
    TP, TN, FP, FN= clustering_evaluation(labels_true, labels_pred)
    return (TP + TN) / (TP + TN + FP + FN)

def adj_rand_index(labels_true, labels_pred):
    return metrics.adjusted_rand_score(labels_true, labels)

def silhouette(X, labels_pred):
    return metrics.silhouette_score(X, labels_pred)


################### KMEANS ####################################################
def kmeans_tuning(X, max_cluster, labels_true, random_state):
    num_clusters= np.empty(max_cluster - 1, dtype= int)
    precision_kmeans= np.empty(max_cluster - 1)
    recall_kmeans= np.empty(max_cluster - 1)
    f1_score_kmeans= np.empty(max_cluster - 1)
    rand_index_kmeans= np.empty(max_cluster - 1)
    adj_rand_index_kmeans= np.empty(max_cluster - 1)
    silhouette_kmeans= np.empty(max_cluster - 1)
    
    for k in range(2, max_cluster):
        kmeans= KMeans(n_clusters= k, random_state= random_state).fit(X)
        labels_pred = kmeans.labels_
        num_clusters[k - 1]= k
        precision_kmeans[k - 1]= precision(labels_true, labels_pred)
        recall_kmeans[k - 1]= recall(labels_true, labels_pred)
        f1_score_kmeans[k - 1]= f1_score(labels_true, labels_pred)
        rand_index_kmeans[k - 1]= rand_index(labels_true, labels_pred)
        adj_rand_index_kmeans[k - 1]= adj_rand_index(labels_true, labels_pred)
        silhouette_kmeans[k - 1]= silhouette(X, labels_pred)
    
    #plt.plot(num_clusters.ravel(), adj_rand_index.ravel())
    return num_clusters, precision_kmeans, recall_kmeans, f1_score_kmeans, rand_index_kmeans, adj_rand_index_kmeans, silhouette_kmeans
 

a, b, c, d, e, f, g= kmeans_tuning(X, 100, fault, 20171205)
       
k= 3
kmeans= KMeans(n_clusters= k, random_state= 20171205).fit(X)
labels = kmeans.labels_
labels_true= fault

print('Estimated number of clusters: %d' % k)
print('Precision %0.3f' % precision(labels_true, labels))
print('Recall %0.3f' % recall(labels_true, labels))
print('F1 Score %0.3f' % f1_score(labels_true, labels))
print('Rand Index %0.3f' % rand_index(labels_true, labels))
print('Adjusted Rand Index %0.3f' % adj_rand_index(labels_true, labels))
print('Silhouette %0.3f' % silhouette(X, labels))


plot_classes(labels, longitude, latitude, alpha=0.5, edge='k')
