try:
    import numpy as np
    from sklearn.metrics import pairwise_distances
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
except Exception as e:
    print('Import error: {}'.format(str(e)))

def scatterplot(x, c, label, square=False):
    """2D Scatter plotting using t-SNE and PCA in sklearn
    
    Parameters
    ----------
    x: numpy.ndarray, data vector, shape = (n, d) 
    
    c: numpy.ndarray, centroids, shape = (k, d)
    
    label: list of int/str, clustering labels. Its length is n
    
    square: Boolean, ensuring scatterplot has square-like space
        It is the argument of PCA(whiten)

    Examples
    --------
    >>> from clustervis import scatterplot
    >>> z = scatterplot(x, c)
    >>> 
    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(z[:,0], z[:,1])
    
    Returns
    -------
    z_x: numpy.ndarray, 2d scatterplot vector of data point, shape = (n, 2)

    z_c: numpy.ndarray, 2d scatterplot vectors of centroids, shape = (k, 2)
    """
    
    z_c = _scatter_centroids(c, square)
    z_x, zrmax = _scatter_x_near_centroid(x, c, z_c, label)
    return z_x, z_c, zrmax

def _scatter_centroids(c, whiten):
    tsne = TSNE(n_components=2)
    z_c = tsne.fit_transform(c)
    
    x_range = max(z_c[:,0]) - min(z_c[:,0])
    y_range = max(z_c[:,1]) - min(z_c[:,1])
    factor = 1 / max(x_range, y_range)
    z_c *= factor
        
    pca = PCA(n_components=2, whiten=whiten)
    z_c = pca.fit_transform(z_c)
    return z_c

def _scatter_x_near_centroid(x, c, z_c, label):
    xcdist = pairwise_distances(x, c)
    xrmin = [100000] * xcdist.shape[1]
    xrmax = [0] * xcdist.shape[1]
    for li, di in zip(label, xcdist):
        d_x2c = di.min()
        xrmin[li] = min(xrmin[li], d_x2c)
        xrmax[li] = max(xrmax[li], d_x2c)
    xrmin = [v * 0.95 for v in xrmin]

    xfactor = []
    for li, di in zip(label, xcdist):
        d_x2c = di.min()
        factor = (di.min() - xrmin[li]) / (xrmax[li] - xrmin[li])
        xfactor.append(factor)
    
    cdist = pairwise_distances(z_c, z_c)
    zrmax = [sorted(row)[1] for row in cdist]
    zrmax = [v / 2 for v in zrmax]

    z_x = []
    for li, factor in zip(label, xfactor):
        smoothing = 0.1 * (np.random.rand() * 0.5 + 0.5)
        radius = max(factor, smoothing) * zrmax[li]
        pert = _random_pertubation(radius)
        zi = (z_c[li][0] + pert[0], z_c[li][1] + pert[1])
        z_x.append(zi)
    
    z_x = np.asarray(z_x)
    return z_x, zrmax

def _random_pertubation(radius):
    ang = np.random.rand() * 2 * np.pi
    xy = np.asarray((np.cos(ang), np.sin(ang)))
    xy *= radius
    return xy