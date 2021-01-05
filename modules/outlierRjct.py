
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


def stdRegion(data, nstd):
    """
    """
    msk_all = []
    # Process each dimension separately
    for arr in data.T:
        # Mask outliers (np.nan).
        med, std = np.nanmedian(arr), np.nanstd(arr)
        dmin, dmax = med - nstd * std, med + nstd * std
        msk = (arr > dmin) & (arr < dmax)
        msk_all.append(msk.data)
    # Combine into a single mask
    msk_data = np.logical_and.reduce(msk_all)

    return msk_data


def sklearnMethod(data, oultr_method, n_neighbors=50):
    """
    """
    # Predict outliers
    if oultr_method == 'local':
        y_pred = LocalOutlierFactor(n_neighbors=n_neighbors).fit_predict(
            data)
    elif oultr_method == 'forest':
        y_pred = IsolationForest().fit_predict(data)

    # This elements are kept
    msk_data = y_pred > 0

    return msk_data
