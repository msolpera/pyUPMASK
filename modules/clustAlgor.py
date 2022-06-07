
import numpy as np
import sklearn.cluster as skclust
import sklearn.mixture as skmixture
from scipy.spatial.distance import cdist
from scipy import spatial
from .voronoiVols import voronoi_volumes


def DDhisto(clust_data, N_membs, n_clusters):
    """
    Testing simple N-dimensional histogram
    """
    import pandas as pd
    N_stars, N_dim = clust_data.shape

    N_bins = N_stars / N_membs
    if N_bins**N_dim > n_clusters:
        N_bins = int(n_clusters**(1 / N_dim))

    H, edges = np.histogramdd(clust_data, N_bins)

    # Find indexes of points within edges
    labels_dim = []
    for dim in range(N_dim):
        labels_dim.append(list(map(str, np.digitize(
            clust_data.T[dim], edges[dim], right=True))))

    # Convert the indexes of edges to a unique index
    ids = ["".join(_) for _ in zip(*labels_dim)]
    # Convert indexes to unique integers
    labels = pd.factorize(ids)[0]

    return labels


def voronoi(clust_data, N_membs, n_clusters, N_st_max):
    """
    Voronoi assignation. Not really a clustering method.
    """
    N_stars = clust_data.shape[0]

    step = N_membs
    if N_stars / n_clusters > N_membs:
        step = int(N_stars / n_clusters)

    # Obtain Voronoi volumes
    vol_v = voronoi_volumes(clust_data)
    # Convert to densities
    dens = 1. / vol_v

    # Indexes for clusters
    idxs = np.argsort(-dens)
    cl_idx = idxs[::step]
    # Cap at n_clusters
    cl_idx = cl_idx[:n_clusters]

    dist = None
    if N_stars < N_st_max:
        # Find the distances to all stars, for all stars
        dist = cdist(clust_data, clust_data)

    labels = densLabeling(N_st_max, N_stars, clust_data, dist, cl_idx)

    return labels


def kNNdens(clust_data, cl_method_pars, N_membs, n_clusters, N_st_max):
    """
    Adapted from: 'Clustering by fast search and find of density peaks',
    Rodriguez and Laio (2014)
    """
    N_stars = clust_data.shape[0]

    try:
        NN_dd = cl_method_pars['NN_dd']
    except KeyError:
        NN_dd = N_membs

    # Find NN_dd nearest neighbors.
    tree = spatial.cKDTree(clust_data)
    inx = tree.query(clust_data, k=NN_dd + 1)
    # Mean distance to the NN_dd neighbors.
    NN_dist = inx[0].mean(1)
    # Convert to densities
    dens = 1. / NN_dist

    # For each star, find the distance to the *closest* star that has a
    # larger density (stored in 'delta'). For the star with largest
    # density, assign the distance to the most distant star.
    delta = np.zeros(dens.size)

    # Only use for arrays with less than 'Nmax' stars. Otherwise too much
    # memory is required.
    if N_stars < N_st_max:
        # Find the distances to all stars, for all stars
        dist = cdist(clust_data, clust_data)
        for i, st_dens in enumerate(dens):
            msk = dens > st_dens
            # Store the index of the star with the largest density.
            if msk.sum() == 0:
                idx_max = i
            else:
                delta[i] = dist[i][msk].min()
        # For this star, assign the largest distance.
        delta[idx_max] = delta.max()

    else:
        for i, st_dens in enumerate(dens):
            # Distance from 'st' to all other stars
            dist = cdist([clust_data[i]], clust_data)
            msk = dens > st_dens
            # Store the index of the star with the largest density.
            if msk.sum() == 0:
                idx_max = i
            else:
                delta[i] = dist[0][msk].min()
        # For this star, assign the largest distance.
        delta[idx_max] = delta.max()

    # Density times delta
    mult = dens * delta
    # Indexes that sort 'mult' in descending order
    idx_s = np.argsort(-mult)

    # Indexes for clusters
    cl_idx = idx_s[:n_clusters]
    labels = densLabeling(N_st_max, N_stars, clust_data, dist, cl_idx)

    return labels


def densLabeling(Nmax, N_stars, clust_data, dist, cl_idx):
    """
    Assign to each star a label corresponding to the cluster that is
    closest to it.
    """
    if N_stars < Nmax:
        labels = np.argmin(dist[cl_idx, :], 0)
    else:
        # Assign the 'label' given the closest cluster to each star.
        tree = spatial.cKDTree(clust_data[cl_idx])
        _, labels = tree.query(clust_data)

    return labels


def RKmeans(clust_data, n_clusters):
    """
    Use R's K-means method.
    """
    from rpy2.robjects import r
    nr, nc = clust_data.shape
    ocdata_px = r.matrix(clust_data, nrow=nr, ncol=nc)
    r.assign('ocdata_px', ocdata_px)
    r.assign('nclust', n_clusters)

    # This line apparently serves no purpose in the original code
    # aggregate(ocdata_px, by=list(fit$cluster), FUN=mean)

    r('fit <- kmeans(ocdata_px, nclust, nstart=50, iter.max=100)')
    r('ocdata_px <- data.frame(ocdata_px, resMclust.class=fit$cluster)')
    # r('labels_R <- ocdata_px$resMclust.class')
    # labels = np.array(list(r('labels_R')))
    labels = np.array(list(r('fit$cluster')))

    return labels


def sklearnMethods(clust_method, cl_method_pars, clust_data, n_clusters):
    """
    Find clusters in the 'clust_data' array using the selected algorithm.
    """

    if clust_method == 'KMeans':
        model = skclust.KMeans()

    elif clust_method == 'MiniBatchKMeans':
        model = skclust.MiniBatchKMeans()

    elif clust_method == 'AffinityPropagation':
        model = skclust.AffinityPropagation()

    elif clust_method == 'SpectralClustering':
        model = skclust.SpectralClustering()

    elif clust_method == 'AgglomerativeClustering':
        model = skclust.AgglomerativeClustering()

    elif clust_method == 'GaussianMixture':
        model = skmixture.GaussianMixture()

    elif clust_method == 'BayesianGaussianMixture':
        model = skmixture.BayesianGaussianMixture()

    elif clust_method == 'DBSCAN':
        model = skclust.DBSCAN()

    elif clust_method == 'OPTICS':
        model = skclust.OPTICS()

    elif clust_method == 'MeanShift':
        model = skclust.MeanShift()

    elif clust_method == 'Birch':
        model = skclust.Birch()

    elif clust_method == 'HDBSCAN':
        import hdbscan
        model = hdbscan.HDBSCAN()

    # Set parameters for the method (if any)
    if cl_method_pars:
        model.set_params(**cl_method_pars)

    # Only these methods require the number of clusters to be set
    if clust_method in (
            'KMeans', 'MiniBatchKMeans', 'SpectralClustering',
            'AgglomerativeClustering', 'GaussianMixture',
            'BayesianGaussianMixture'):

        if clust_method in ('GaussianMixture', 'BayesianGaussianMixture'):
            model.n_components = n_clusters
        else:
            model.n_clusters = n_clusters

    # Fit the model
    model.fit(clust_data)

    # Extract the labels
    if clust_method in ('GaussianMixture', 'BayesianGaussianMixture'):
        labels = model.predict(clust_data)
        # probs_gmm = model.predict_proba(clust_data).max(1)
    else:
        labels = model.labels_

    return labels


def pycl(clust_method, clust_data, n_clusters):
    """
    """
    if clust_method == 'pyclKmeans':
        from pyclustering.cluster.kmeans import kmeans
        from pyclustering.cluster.center_initializer import\
            kmeans_plusplus_initializer

        initial_centers = kmeans_plusplus_initializer(
            clust_data, n_clusters).initialize()
        model = kmeans(clust_data, initial_centers)
        # final_centers = model.get_centers()

    elif clust_method == 'pyclGA':
        from pyclustering.cluster.ga import genetic_algorithm
        # Create instance of observer that will collect all information:
        # observer_instance = ga_observer(True, True, True)
        model = genetic_algorithm(
            clust_data, count_clusters=n_clusters, chromosome_count=100,
            population_count=20, coeff_mutation_count=.5)

    # Fit the model
    model.process()

    if clust_method[4:] == 'Kmeans':
        labels = model.predict(clust_data)
    elif clust_method[4:] == 'GA':
        labels = np.zeros(clust_data.shape[0])
        for i, clust in enumerate(model.get_clusters()):
            labels[clust] = i
        labels = labels.astype(int)

    return labels

# The 'HDBSCAN' method is taken from https://hdbscan.readthedocs.io/. Here's
# a nice article explaining it: https://towardsdatascience.com/
# understanding-hdbscan-and-density-based-clustering-121dbee1320e
    # elif clust_method == 'HDBSCAN':
    #     import hdbscan
    #     model = hdbscan.HDBSCAN()

# The 'KMeansSwap' method is adapted from the article
# 'Efficiency of random swap clustering', Franti (2018)
    # elif clust_method == 'KMeansSwap':
    #     model = skclust.KMeans()
    #     model.n_clusters = n_clusters
    #     model.fit(clust_data)

    #     inertia_old = model.inertia_
    #     centers = model.cluster_centers_
    #     for _ in range(cl_method_pars['n_runs']):
    #         centers2 = np.array(centers)

    #         idx_1 = np.random.choice(n_clusters)
    #         idx_2 = np.random.choice(clust_data.shape[0])
    #         centers2[idx_1] = clust_data[idx_2]

    #         model = skclust.KMeans(
    #             init=centers2, n_clusters=n_clusters, n_init=1, max_iter=2)
    #         model.fit(clust_data)
    #         if model.inertia_ < inertia_old:
    #             centers = model.cluster_centers_
    #             inertia_old = model.inertia_

    #     # Reset this parameter
    #     model.max_iter = 300
