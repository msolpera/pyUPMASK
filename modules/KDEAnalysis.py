
import warnings
import numpy as np
from scipy.stats import gaussian_kde


def probs(xy, data, cl_probs, Nst_max=5000):
    """
    Assign probabilities to all stars after generating the KDEs for field and
    member stars. The Cluster probability is obtained applying the formula for
    two mutually exclusive and exhaustive hypotheses.
    """

    # Combine coordinates with the rest of the features.
    all_data = np.concatenate([xy.T, data.T]).T
    # Split into the two populations.
    field_stars = all_data[cl_probs == 0.]
    membs_stars = all_data[cl_probs == 1.]

    # To improve the performance, cap the number of stars using a random
    # selection of 'Nf_max' elements.
    if field_stars.shape[0] > Nst_max:
        idxs = np.arange(field_stars.shape[0])
        np.random.shuffle(idxs)
        field_stars = field_stars[idxs[:Nst_max]]

    # Evaluate all stars in both KDEs
    try:
        kd_field = gaussian_kde(field_stars.T)
        kd_memb = gaussian_kde(membs_stars.T)

        L_memb = kd_memb.evaluate(all_data.T)
        L_field = kd_field.evaluate(all_data.T)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Probabilities for mutually exclusive and exhaustive hypotheses
            cl_probs = 1. / (1. + (L_field / L_memb))

    except (np.linalg.LinAlgError, ValueError):
        print("WARNING: Could not perform KDE probabilities estimation")

    return cl_probs
