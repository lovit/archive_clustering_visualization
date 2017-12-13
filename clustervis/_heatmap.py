from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt

def visualize_heatmap(centroids, metric='cosine', sort=None, segmentor=None, **kargs):    
    n = centroids.shape[0]
    pdist = pairwise_distances(centroids, metric=metric)

    if sort == 'row_sum':
        pdist = _row_sum_sorting(pdist)
    elif sort == 'dist_pole':
        pdist = _dist_pole_sorting(centroids, pdist, metric, **kargs)

    figure = _draw_figure(pdist, **kargs)

    if segmentor == 'reverse_band':
        points = _reversed_band_segmentation(pdist, **kargs)

    if segmentor and points and len(points) > 2:
        plt.hold()
        _plot_segmented_heatmap(pdist, points, kargs)

    return figure

def _draw_figure(pdist, **kargs):
    figsize = kargs.get('figsize', (15, 15))
    dpi = kargs.get('dpi', 50)
    facecolor = kargs.get('facecolor', None)
    edgecolor = kargs.get('edgecolor', None)
    frameon = kargs.get('frameon', True)
    title = kargs.get('title', None)
    cmap = kargs.get('cmap', 'gray')

    n,m = pdist.shape
    figure = plt.figure(figsize=figsize, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor, frameon=frameon)
    #ax = figure.add_subplot(1,1,1)
    #ax.set_xlim([0, n])
    #ax.set_ylim([0, m])
    plt.imshow(pdist, cmap=cmap)
    if title:
        plt.title(title)

    return figure

def _row_sum_sorting(pdist):
    n = pdist.shape[0]
    sim_order = pdist.sum(axis=0).argsort()

    indices_orig = list(range(n))
    indices_revised = np.ix_(sim_order,sim_order)

    pdist_revised = np.empty_like(pdist)
    pdist_revised[np.ix_(indices_orig,indices_orig)] = pdist[indices_revised]
    return pdist_revised

def _dist_pole_sorting(x, pdist, metric, **kargs):

    df = kargs.get('document_frequency', None)
    max_dist = kargs.get('dist_pole_max_dist', 0.5)

    if df:
        assert x.shape[0] == len(df)
        sorted_indices, _ = zip(*sorted(enumerate(df), lambda x:-x[1]))
    else:
        sorted_indices = pdist.sum(axis=0).argsort()

    n = x.shape[0]
    groups = []

    for i, idx in enumerate(sorted_indices):
        if i == 0:
            groups.append([idx])
            continue
        found_cluster = None
        for g, idxs in enumerate(groups):
            dist = pdist[idxs,idx].mean()
            if dist > max_dist:
                continue
            found_cluster = g
            break
        if found_cluster == None:
            groups.append([idx])
        else:
            groups[found_cluster].append(idx)

    group_order = [idx for group in groups for idx in group]

    indices_orig = list(range(n))
    indices_revised = np.ix_(group_order,group_order)

    pdist_revised = np.empty_like(pdist)
    pdist_revised[np.ix_(indices_orig,indices_orig)] = pdist[indices_revised]

    return pdist_revised

def _reversed_band_segmentation(pdist, **kargs):
    def reversed_band_matrix(n, k):
        A = np.zeros((n,n))
        for i in range(-k,k):
            A += np.eye(n,k=i)
        A = np.flip(A,0)
        return A
    
    n = kargs.get('reverse_band_filter_width', int(pdist.shape[0]*0.5))
    k = kargs.get('reverse_band_band_width', min(int(n/2), 10))
    
    A = reversed_band_matrix(n, k)
    means = []
    for i in range(1,pdist.shape[0]-n):
        block = pdist[i+0:i+n,i+0:i+n]
        means.append(np.mean(A * block))

    tangents = []
    for ix in range(1, len(means)):
        tangents.append(means[ix]-means[ix-1])

    points = []
    for ix, val in enumerate(tangents[:-1]):
        if val < 0:
            pass
        if val*tangents[ix+1] < 0:
            points.append(ix+n//2)
            
    return [0] + points + [pdist.shape[0]]

def _plot_segmented_heatmap(pdist, points, kargs):
    def add_lines(k, xmin, xmax):
        plt.hlines(k, xmin, xmax, colors=line_color, linewidth=line_width)
        plt.vlines(k, xmin, xmax, colors=line_color, linewidth=line_width)
    
    line_width = kargs.get('line_width', 3)    
    line_color = kargs.get('boundary_line_color', 'red')
        
    for ix, point in enumerate(points):
        if ix == 0 or ix == len(points)-1:
            continue
        add_lines(point, points[ix-1], point)
        add_lines(point, point, points[ix+1])