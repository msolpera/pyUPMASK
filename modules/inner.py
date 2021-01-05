
import numpy as np
from . import clustAlgor
from . import rjctRandField


def loop(
    clust_xy, clust_data, N_membs, clust_method, clRjctMethod,
    KDE_vals, Kest, C_thresh, cl_method_pars, prfl, N_cl_max=10000,
        N_st_max=20000, minStars=3):
    """
    Perform the inner loop: cluster --> reject

    Iterate over all the clusters defined by the clustering algorithm, storing
    those masks that point to clusters that survived the test with uniform
    random 2D fields in (x, y).

    N_cl_max : int
      maximum number of clusters allowed
    N_st_max : int
      maximum numbers of stars for the Voronoi and kNNdens algorithms.
      Above this threshold, the 'cdist' function is avoided.
    minStars: int
      If the cluster contains less than this many stars, skip the test and
      mark its stars as field stars.
    """

    print("  Performing clustering on array of shape ({}, {})".format(
        *clust_data.shape), file=prfl)

    # Number of clusters: min is 2, max is N_cl_max
    n_clusters = max(2, int(clust_data.shape[0] / N_membs))

    if n_clusters > N_cl_max:
        print("  Too many clusters. Capping at {}".format(N_cl_max), file=prfl)
        n_clusters = N_cl_max

    # Obtain all the clusters in the non-spatial data
    if clust_method == 'Voronoi':
        labels = clustAlgor.voronoi(clust_data, N_membs, n_clusters, N_st_max)
    elif clust_method == 'kNNdens':
        labels = clustAlgor.kNNdens(
            clust_data, cl_method_pars, N_membs, n_clusters, N_st_max)
    elif clust_method == 'rkmeans':
        labels = clustAlgor.RKmeans(clust_data, n_clusters)
    elif clust_method[:4] != 'pycl':
        # scikit-learn methods
        labels = clustAlgor.sklearnMethods(
            clust_method, cl_method_pars, clust_data, n_clusters)
    else:
        # TODO Not fully implemented yet
        labels = clustAlgor.pycl(clust_data, n_clusters)

    # import matplotlib.pyplot as plt
    # for cl in clusts_msk:
    #     plt.scatter(*clust_data[cl].T[:2], alpha=.25)

    N_clusts = len(set(labels))
    print("  Identified {} clusters".format(N_clusts), file=prfl)

    msk_all, N_survived = np.array([False for _ in range(labels.shape[0])]), 0
    # For each cluster found by the clustering method, check if it is composed
    # of field stars or actual cluster members, using their (x, y)
    # distribution.
    for i in range(labels.min(), labels.max() + 1):
        # Separate stars assigned to this label
        cl_msk = labels == i

        # Smaller C_s values point to samples that come from a uniform
        # distribution in (x, y), i.e., a "cluster" made up of field
        # stars. Hence, we keep as "true" clusters those with C_s values
        # larger than C_thresh.

        # Not enough elements, skip this cluster
        if cl_msk.sum() < minStars:
            continue

        # Test how similar this cluster's (x, y) distribution is compared
        # to a uniform random distribution.
        if clRjctMethod == 'rkfunc':
            C_s = rjctRandField.rkfunc(clust_xy[cl_msk], Kest)
        if clRjctMethod == 'kdetest':
            C_s, KDE_vals = rjctRandField.kdetest(
                clust_xy[cl_msk], KDE_vals)
        if clRjctMethod == 'kdetestpy':
            C_s, KDE_vals = rjctRandField.kdetestpy(
                clust_xy[cl_msk], KDE_vals)

        # C_thresh : Any cluster with a smaller value will be classified as
        # being composed of field stars and discarded.

        # 1% critical value. From Dixon (2001), 'Ripley's K function'
        if clRjctMethod == 'rkfunc':
            C_thresh = 1.68 / cl_msk.sum()

        if C_s >= C_thresh:
            # Store mask that points to stars that should be *kept*
            print("   Cluster {} survived (C_s={:.2f}), N={}".format(
                i, C_s, cl_msk.sum()), file=prfl)
            # Combine all the masks using a logical OR
            msk_all = np.logical_or.reduce([msk_all, cl_msk])
            N_survived += 1

    return N_clusts, msk_all, N_survived, KDE_vals
