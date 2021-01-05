
import warnings
import numpy as np
from scipy.stats import gaussian_kde


def rkfunc(xy, Kest):
    """
    Test how similar this cluster's (x, y) distribution is compared
    to a uniform random distribution using Ripley's K.
    https://stats.stackexchange.com/a/122816/10416
    """
    # Avoid large memory consumption if the data array is too big
    if xy.shape[0] > 5000:
        mode = "none"
    else:
        mode = 'translation'

    # https://rdrr.io/cran/spatstat/man/Kest.html
    # "Users are advised *not to* to specify this argument; there is a sensible
    # default"
    # "For a rectangular window it is prudent to restrict the r values to a
    # maximum of 1/4 of the smaller side length of the rectangle
    # (Ripley, 1977, 1988; Diggle, 1983)"
    rad = np.linspace(.01, .25, 50)

    # Hide RunTimeWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        L_t = Kest.Lfunction(xy, rad, mode=mode)

    # Catch all-nans
    if np.isnan(L_t).all():
        C_s = -np.inf
    else:
        C_s = np.nanmax(abs(L_t - rad))

    return C_s


def kdetest(xy, KDE_vals):
    """
    """
    from rpy2.robjects import r

    rx = r.matrix(xy.T[0])
    ry = r.matrix(xy.T[1])
    r.assign('dataX', rx)
    r.assign('dataY', ry)
    r("""
    kde2dmap <- kde2d(dataX, dataY, n=nKde,lims=c(0, 1, 0, 1))
    dist_d <- ((max(as.vector(kde2dmap$z))-mean(as.vector(kde2dmap$z)))/
    sd(as.vector(kde2dmap$z)))
    """)
    dist_d = r('dist_d')[0]

    N = xy.shape[0]
    # Read stored value from table.
    try:
        mean, std = KDE_vals[N]
    except KeyError:
        r.assign('nstars', N)

        r("""
        maxDistStats <- vector("double", nruns)
        for(i in 1:nruns) {
          dataX <- runif(nstars, 0, 1)
          dataY <- runif(nstars, 0, 1)
          kde2dmap <- kde2d(dataX, dataY, n=nKde, lims=c(0, 1, 0, 1))
          maxDistStats[i] <- ((max(as.vector(kde2dmap$z))-
          mean(as.vector(kde2dmap$z)))/sd(as.vector(kde2dmap$z)))
        }
        retStat <- data.frame(mean=mean(maxDistStats), sd=sd(maxDistStats))
        mean <- retStat$mean
        sd <- retStat$sd
        """)
        mean, std = list(r('mean'))[0], list(r('sd'))[0]
        KDE_vals[N] = mean, std

    C_s = (dist_d - mean) / std

    return C_s, KDE_vals


def kdetestpy(xy, KDE_vals, Nfields=500):
    """
    """
    N = xy.shape[0]
    xyminmax = (0, 1, 0, 1)
    xmin, xmax, ymin, ymax = xyminmax
    xx, yy = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    # Read stored value from table.
    try:
        mean, std = KDE_vals[N]
    except KeyError:
        dist_u = []
        for _ in range(Nfields):
            # Generate random uniform 2D distribution
            xy_u = np.random.uniform((xmin, ymin), (xmax, ymax), (N, 2))
            kde = gaussian_kde(xy_u.T)
            kde2dmap = kde.evaluate(positions)
            dist_u.append(
                (kde2dmap.max() - kde2dmap.mean()) / kde2dmap.std())

        mean, std = np.mean(dist_u), np.std(dist_u)
        KDE_vals[N] = mean, std

    # Evaluate subset
    # from scipy.stats import iqr
    # bw = 1.06 * np.min([xy.std(None), iqr(xy, None) / 1.34], None) *\
    #     N**(-1. / 5.)
    # kde = gaussian_kde(xy.T, bw_method=bw / xy.T.std(ddof=1))
    kde = gaussian_kde(xy.T)
    kde2dmap = kde.evaluate(positions)
    dist_d = (kde2dmap.max() - kde2dmap.mean()) / kde2dmap.std()

    C_s = (dist_d - mean) / std

    # import matplotlib.pyplot as plt
    # plt.subplot(121)
    # z = np.reshape(kde2dmap.T, xx.shape)
    # plt.contour(xx, yy, z, N, linewidths=0.8, colors='k')
    # plt.contourf(xx, yy, z, N, cmap="RdBu_r")
    # plt.plot(xy[0], xy[1], 'ok', ms=3)

    # # KDEpy. It's actually slightly slower
    # from KDEpy import FFTKDE
    # s = t.time()
    # # Scott's rule: N**(-1. / (dims + 4))
    # bw = N**(-1. / 6.) * xy.std()

    # dist_u = []
    # for _ in range(200):
    #     # Generate random uniform 2D distribution
    #     xy_u = np.random.uniform((xmin, ymin), (xmax, ymax), (N, 2))
    #     points = FFTKDE(bw=bw, kernel='gaussian').fit(xy_u).evaluate(
    #         positions.T)
    #     dist_u.append((points.max() - points.mean()) / points.std())

    # mean, std = np.mean(dist_u), np.std(dist_u)
    # KDE_vals[N] = mean, std

    # points = FFTKDE(bw=bw, kernel='gaussian').fit(xy).evaluate(positions.T)
    # dist_d = (points.max() - points.mean()) / points.std()

    # C_s = (dist_d - mean) / std

    # # Statsmodels
    # import statsmodels.api as sm
    # points = sm.nonparametric.KDEMultivariate(
    #     data=xy, var_type='cc').pdf(positions)

    return C_s, KDE_vals
