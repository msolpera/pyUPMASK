
import numpy as np
from .GUMM import GUMMProbs


def GUMMProbCut(GUMM_perc, gumm_p):
    """
    """
    # Select the probability cut.
    if GUMM_perc == 'auto':
        # Create the percentiles (/100.) vs provabilities array.
        percentiles = np.arange(.01, .99, .01)
        perc_probs = np.array([
            percentiles, np.percentile(gumm_p, percentiles * 100.)]).T

        # Find 'elbow' where the probabilities start climbing from ~0.
        prob_cut = rotate(perc_probs)

        # import matplotlib.pyplot as plt
        # from scipy.spatial import distance
        # plt.subplot(221)
        # plt.title("P<{:.4f}".format(prob_cut))
        # plt.scatter(*perc_probs.T)
        # plt.subplot(222)
        # msk = gumm_p < prob_cut
        # plt.scatter(*clust_xy[~msk].T, c=gumm_p[~msk])

        # cl_index = []
        # probs_lst, probs_lst2 = np.arange(.01, .99, .01), []
        # for p in probs_lst:
        #     # Mask for stars that are *rejected*
        #     msk = gumm_p < p
        #     if (~msk).sum() == 0:
        #         break
        #     dmean = distance.cdist(np.array([cl_cent]), clust_xy[~msk]).mean()
        #     cl_index.append([msk.sum(), dmean])
        #     probs_lst2.append(p)

        # if (np.array(cl_index).T[0] == 0).all():
        #     prob_cut = 1.
        # else:
        #     cl_index /= np.array(cl_index).T.max(1)
        #     idx = np.argmin(abs(cl_index.T[0] - cl_index.T[1]))
        #     prob_cut = probs_lst2[idx]

        # plt.subplot(223)
        # plt.title("P<{:.4f}".format(probs_lst2[idx]))
        # plt.plot(probs_lst2, cl_index.T[0], label="N")
        # plt.plot(probs_lst2, cl_index.T[1], label="d")
        # plt.legend()

        # plt.subplot(224)
        # msk = gumm_p < probs_lst[idx]
        # plt.scatter(*clust_xy[~msk].T, c=gumm_p[~msk])

    else:
        prob_cut = np.percentile(gumm_p, GUMM_perc)

    return prob_cut


def rotate(data):
    """
    Rotate a 2d vector.

    (Very) Stripped down version of the great 'kneebow' package, by Georg
    Unterholzner. Source:

    https://github.com/georg-un/kneebow

    data   : 2d numpy array. The data that should be rotated.
    return : probability corresponding to the elbow.
    """
    # The angle of rotation in radians.
    theta = np.arctan2(
        data[-1, 1] - data[0, 1], data[-1, 0] - data[0, 0])

    # Rotation matrix
    co, si = np.cos(theta), np.sin(theta)
    rot_matrix = np.array(((co, -si), (si, co)))

    # Rotate data vector
    rot_data = data.dot(rot_matrix)

    # Find elbow index
    elbow_idx = np.where(rot_data == rot_data.min())[0][0]

    # Adding a small percentage to the selected probability improves the
    # results by making the cut slightly stricter.
    prob_cut = data[elbow_idx][1] + 0.05

    return prob_cut


def lowCIGUMMClean(N_membs, GUMM_perc, ID, cl_probs, clust_ID, clust_xy, prfl):
    """
    Remove stars marked as members if their GUMM probability is below
    the 'pob_cut' threshold.
    """
    gumm_p = GUMMProbs(clust_xy)
    prob_cut = GUMMProbCut(GUMM_perc, gumm_p)

    # Don't overwrite
    probs_cl = list(cl_probs)

    for i, st_id in enumerate(ID):
        p = probs_cl[i]
        # If this was marked as a cluster star
        if p == 1.:
            # Find its GUMM probability
            j = np.where(clust_ID == st_id)[0][0]
            g_p = gumm_p[j]
            # If its GUMM probability is below the threshold
            if g_p <= prob_cut:
                # Mark star as non-member
                probs_cl[i] = 0.
            # else:
            #     # Replace the 1. with the GUMM probability
            #     probs_cl[i] = g_p

    # Replace the final probabilities list with the cleaner one.
    msk = np.array(probs_cl) > 0.
    if msk.sum() > N_membs:
        cl_probs = probs_cl
        print(" \nGUMM analysis: reject {} stars as non-members".format(
            msk.sum()), file=prfl)

    return cl_probs
