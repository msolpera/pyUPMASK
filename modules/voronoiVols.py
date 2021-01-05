
import numpy as np
# from scipy import integrate
from astropy.stats import sigma_clipped_stats
from scipy.spatial import Voronoi, ConvexHull


def voronoi_volumes(points, Nsigma=3.):
    """
    For an D-dimensional dataset obtain its Voronoi diagram, and calculate
    the volume associated to each region. Unbounded regions are assigned the
    95th percentile volume. Outlier regions (with large volumes) are clipped to
    median volumes.
    """

    Ndim = points.shape[1]
    v = Voronoi(points)

    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):

        # Indices of the Voronoi vertices forming each Voronoi region.
        # -1 indicates vertex outside the Voronoi diagram.
        indices = v.regions[reg_num]

        # If the region is not bounded, assign a 'nan' volume.
        if -1 in indices:
            vol[i] = np.nan
        else:
            # # Clip vertexes outside of the frame's boundaries
            # vertxs = np.clip(v.vertices[indices], a_min=0., a_max=1.)

            # Coordinates of the Voronoi vertices.
            vertxs = v.vertices[indices]
            # Obtain volume for this region
            vol[i] = area_of_polygon(vertxs, Ndim)

    # For points with unbounded regions, assign the 95th percentile volume.
    vol[np.isnan(vol)] = np.nanpercentile(vol, 95)

    # Clip volumes of N-sigma outliers to the median volume of the dataset.
    mean, median, std = sigma_clipped_stats(vol)
    vol[vol > mean + Nsigma * std] = median

    return vol


def area_of_polygon(points, Ndim):
    """
    Calculates the area of an arbitrary polygon given its vertices

    Source: http://stackoverflow.com/a/4682656/1391441
    """
    if Ndim > 2:
        # N-dimensional approach (slower)
        p_area = ConvexHull(points).volume
    else:
        # For Ndim=2 this is faster than using ConvexHull()
        x, y = zip(*points)
        area = 0.0
        for i in range(-1, len(x) - 1):
            area += x[i] * (y[i + 1] - y[i - 1])
        p_area = abs(area) / 2.0

    return p_area


# def voronoi_vols2CDF(vols, cummul):
#     """
#     Return the value of the CDF for the 2D Voronoi area PDF (normalized)
#     for every 'vols' value.

#     Used by the Kolmogorov-Smirnov test.
#     """
#     N_CDF = cummul.shape[1]
#     # Normalize: vols = vols / <vols> = vols / (1/N) * vols * N
#     vols *= vols.size
#     idxs = np.searchsorted(cummul[0], vols)
#     idxs[idxs == N_CDF] = N_CDF - 1
#     CDF_vals = cummul[1][idxs]

#     return CDF_vals


# def voronoi_CDF2vols(xy, cummul):
#     """
#     Returned randomly sampled sampled Voronoi 2D volumes
#     """
#     N = xy.shape[0]
#     A = np.ptp(xy.T[0]) * np.ptp(xy.T[1])
#     # Assuming an area of 1*1=1
#     # A = 1.

#     probs = np.random.uniform(0., 1., N)
#     idxs = np.searchsorted(cummul[1], probs)

#     N_CDF = cummul.shape[1]
#     idxs[idxs == N_CDF] = N_CDF - 1
#     vols = cummul[0][idxs]

#     # Multiply by the mean volume
#     vols *= A / N

#     return vols


# def vor_2d_cummltv(vol_max=5., N_CDF=10000):
#     """
#     Estimate the CDF of the 'vor_2d_poisson()' PDF between 0. and 'vol_max',
#     with a step of 'N_CDF'. This is the CDF for the *normalized* 2D Voronoi
#     areas.
#     """
#     print("Generating Voronoi 2D CDF")
#     xx = np.linspace(0., vol_max, N_CDF)
#     cummul = []
#     for v in xx:
#         cummul.append([v, integrate.quad(vor_2d_poisson, 0., v)[0]])
#     return np.array(cummul).T


# def vor_2d_poisson(y):
#     """
#     Function that fits the normalized area of Voronoi cells for a 2D Poisson
#     distribution.
#     From: On the size-distribution of Poisson Voronoi cells,
#     Jarai-Szabo & Zoltan (2007); (Eq. 10)
#     https://doi.org/10.1016/j.physa.2007.07.063
#     """
#     # y = vor_cell_area / <vor_cell_area>
#     a = (343. / 15.) * np.sqrt(7. / (2. * np.pi))
#     b, c = 5. / 2., -7. / 2.

#     return a * (y ** b) * np.exp(c * y)
