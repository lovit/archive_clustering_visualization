from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt

def visualize_heatmap(centroids, metric='cosine', sort=None, segmentor=None, **kargs):    
    n = centroids.shape[0]
    pdist = pairwise_distances(centroids, metric=metric)

    indices_revised = None
    points = None
    
    if sort == 'row_sum':
        pdist, indices_revised = _row_sum_sorting(pdist)
    elif sort == 'dist_pole':
        pdist, indices_revised, points= _dist_pole_sorting(centroids, pdist, metric, **kargs)

    figure = _draw_figure(pdist, **kargs)

    if not points:
        if segmentor == 'reverse_band':
            points = _reversed_band_segmentation(pdist, **kargs)
        elif segmentor == 'gaussian_filter':
            points, _ = _gaussian_filter_segmentation(pdist, **kargs)

    if points and len(points) > 2:
        plt.hold()
        _plot_segmented_heatmap(pdist, points, kargs)
    
    if type(indices_revised) == np.ndarray:
        indices_revised = indices_revised.tolist()

    if not points or len(points) <= 2:
        segments = [0] * n
    else:
        segments = [group for group, (b, e) in enumerate(zip(points, points[1:])) for _ in range(e-b)]

    return figure, indices_revised, segments

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
    return pdist_revised, sim_order

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
    boundaries = [0]
    for group in groups:
        boundaries.append(boundaries[-1] + len(group))
    boundaries.append(n)
    
    indices_orig = list(range(n))
    indices_revised = np.ix_(group_order,group_order)

    pdist_revised = np.empty_like(pdist)
    pdist_revised[np.ix_(indices_orig,indices_orig)] = pdist[indices_revised]

    return pdist_revised, group_order, boundaries

def _reversed_band_segmentation(pdist, **kargs):
    def reversed_band_matrix(n, k):
        A = np.zeros((n,n))
        for i in range(-k,k):
            A += np.eye(n,k=i)
        A = np.flip(A,0)
        return A
    
    n = kargs.get('filter_width', int(pdist.shape[0]*0.5))
    k = kargs.get('band_width', min(int(n/2), 10))
    
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

def _generate_gaussian_filter(s):
    f = np.zeros(shape=(s*2, s*2))
    for i in range(0, s):
        for j in range(0, s):
            d = np.exp(-((i^2)+(j^2)) / s)
            f[s-i-1, s-j-1] = -d
            f[s+i, s+j]     = -d
            f[s-i-1, s+j]   = d
            f[s+i, s-j-1]   = d
    return f / sum(sum(abs(f)))

def _gaussian_filter_segmentation(pdist, **kargs):
    t = kargs.get('threshold', 0.3)
    width = kargs.get('filter_width', 5)
    f = _generate_gaussian_filter(width)
    n, w = pdist.shape[0], int(f.shape[0]/2)
    s, d = np.zeros(n), np.zeros(n)
    for i in range(0, n-2*w+1):
        for j in range(0, 2*w):
            for k in range(0, 2*w):
                #d[i+w] += distance(y[i+j,:], y[i+k,:]) * f[j, k]
                d[i+w] += pdist[i+j,i+k] * f[j, k]
        if d[i+w] > t: s[i+w] = 1
    s[0] = 1
    s[n-1] = 1
    points = [i for i, si in enumerate(s) if si == 1]
    return points, d

def _plot_segmented_heatmap(pdist, points, kargs):
    def add_lines(k, xmin, xmax):
        plt.hlines(k, xmin, xmax, colors=line_color, linewidth=line_width)
        plt.vlines(k, xmin, xmax, colors=line_color, linewidth=line_width)
    
    line_width = kargs.get('line_width', 3)    
    line_color = kargs.get('line_color', 'red')
        
    for ix, point in enumerate(points):
        if ix == 0 or ix == len(points)-1:
            continue
        add_lines(point, points[ix-1], point)
        add_lines(point, point, points[ix+1])