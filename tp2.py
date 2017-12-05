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

################### KMEANS ####################################################
#test
#kmeans= KMeans(n_clusters= 100, random_state= 0).fit(X)
#NOT SURE THIS IS ACTUALLY USEFUL!!
def KneeElbowAnalysis(x, max_k, seed):
    '''Given data x, a maximum number of clusters max_k and a random seed seed,
    the functions plots the WCSS and BCSS for k ranging from 1 to max_k'''
    plot("WARNING: This function might need some tuning!")
    k_values = range(1, max_k)
    clusterings = [KMeans(n_clusters=k, random_state=seed).fit(x) for k in k_values]
    centroids = [clustering.cluster_centers_ for clustering in clusterings]

    #scipy.spatial.distance.cdist(XA, XB, metric='euclidean', *args, **kwargs)[source]
    #Computes distance between each pair of the two collections of inputs.
    D_k = [cdist(x, cent, 'euclidean') for cent in centroids]
    cIdx = [np.argmin(D,axis=1) for D in D_k]
    dist = [np.min(D,axis=1) for D in D_k]
    avgWithinSS = [sum(d)/x.shape[0] for d in dist]

    # Total with-in sum of square
    wcss = [sum(d**2) for d in dist]

    #scipy.spatial.distance.pdist(X, metric='euclidean', *args, **kwargs)[source]
    #Pairwise distances between observations in n-dimensional space.
    tss = sum(pdist(x)**2)/x.shape[0]
    bss = tss-wcss

    kIdx = 10-1
    
    #
    # elbow curve
    #
    fig = plt.figure()
    font = {'family' : 'sans', 'size'   : 12}
    plt.rc('font', **font)
    plt.plot(k_values, wcss, 'bo-', color='red', label='WCSS')
    plt.plot(k_values, bss, 'bo-', color='blue', label='BCSS')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.legend()
    plt.title('Knee for KMeans clustering');
    return 

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
#def kmeans(X, max_num_clusters, seed)
k= 5
kmeans= KMeans(n_clusters= k, random_state= 20171205).fit(X)
labels = kmeans.labels_
labels_true= fault

print('Estimated number of clusters: %d' % k)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))


#I try to obtain the FP, TP, TN, FN by means of the functions already
#contained in scikit-learn. (https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/metrics/cluster/supervised.py#L787)
#I still have to check whether the maths is understable
labels_true, labels= check_clusterings(labels_true, labels)
c= contingency_matrix(labels_true, labels, sparse= True)
n_samples, = labels.shape
tk = np.dot(c.data, c.data) - n_samples
pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples
qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples
N= labels_true.shape[0] * (labels_true.shape[0]-1) / 2
TP= np.int64(tk)
FP= np.int64(pk) - np.int64(tk)
FN= np.int64(qk) - np.int64(tk)
TN= np.int64(N) - TP - FP - FN
FMI= np.float64(TP / np.sqrt((TP + FP) * (TP + FN)))
FMI_true= np.float64(metrics.fowlkes_mallows_score(labels_true, labels))
print("Testing FMI: %0.3f" % FMI)
#WARNING: THE FUNCTION IMPLEMENTED IN SCIKIT-LEARN DOES NOT WORK AS IT WORKS
#WITH INT32 AND THEREFORE OVERFLOWS
print("True FMI: %0.3f" % FMI_true)


#def main():

