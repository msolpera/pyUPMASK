
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from . import inner
from .GUMM import GUMMProbs
from .GUMMExtras import GUMMProbCut, lowCIGUMMClean
from . import KDEAnalysis


def loop(
    ID, xy, data, data_err, resampleFlag, PCAflag, PCAdims, GUMM_flag,
    GUMM_perc, KDEP_flag, IL_runs, N_membs, N_cl_max, clust_method,
    clRjctMethod, Kest, C_thresh, cl_method_pars, prfl, KDE_vals,
        standard_scale=True):
    """
    Perform the outer loop: inner loop until all "fake" clusters are rejected
    """

    # Make a copy of the original data to avoid over-writing it
    clust_ID, clust_xy = np.array(list(ID)), np.array(list(xy))

    # Re-sample the data using its uncertainties?
    clust_data = reSampleData(
        resampleFlag, data, data_err, prfl, standard_scale)

    # Apply PCA and features reduction
    clust_data = dimReduc(clust_data, PCAflag, PCAdims, prfl)

    # Call the inner loop until all the "fake clusters" are rejected
    for _iter in range(IL_runs):
        print("\n IL iteration {}".format(_iter + 1), file=prfl)

        # Call the Inner Loop (IL)
        N_clusts, msk_all, N_survived, KDE_vals = inner.loop(
            clust_xy, clust_data, N_membs, N_cl_max, clust_method,
            clRjctMethod, KDE_vals, Kest, C_thresh, cl_method_pars, prfl)

        # No clusters were rejected in this iteration of the IL. This means
        # that the method converged. Break
        if N_clusts == N_survived:
            print(" All clusters survived, N={}".format(
                clust_xy.shape[0]), file=prfl)
            break

        # Applying 'msk_all' results in too few stars. Break
        if msk_all.sum() < N_membs:
            print(" N_stars<{:.0f} Breaking".format(N_membs), file=prfl)
            break
        print(" A total of {} stars survived in {} clusters".format(
            msk_all.sum(), N_survived), file=prfl)

        # Keep only stars identified as members and move on to the next
        # iteration
        clust_ID, clust_xy, clust_data = clust_ID[msk_all],\
            clust_xy[msk_all], clust_data[msk_all]

        # Clean using GUMM
        if GUMM_flag:
            print(" Performing GUMM analysis...", file=prfl)
            gumm_p = GUMMProbs(clust_xy)
            prob_cut = GUMMProbCut(GUMM_perc, gumm_p)
            # Mark all stars as members
            probs_cl = np.ones(len(clust_xy))
            # Mark as non-members those below 'prob_cut'
            probs_cl[gumm_p <= prob_cut] = 0.

            # Keep only member stars for the next run (if enough stars remain
            # in the list)
            msk = probs_cl > 0.
            if msk.sum() > N_membs:
                clust_ID, clust_xy, clust_data = clust_ID[msk], clust_xy[msk],\
                    clust_data[msk]
                print(" Rejected {} stars as non-members".format(
                    len(probs_cl) - msk.sum()), file=prfl)

    if _iter + 1 == IL_runs:
        print("Maximum number of IL runs reached. Breaking", file=prfl)

    # Mark all the stars that survived in 'clust_ID' as members assigning
    # a probability of '1'. All others are field stars and are assigned
    # a probability of '0'.
    cl_probs = np.zeros(len(ID))
    for i, st in enumerate(ID):
        if st in clust_ID:
            cl_probs[i] = 1.

    # Perform a final cleaning on the list of stars selected as members.
    # Use the last list of coordinates and IDs from the inner loop.
    # This is only ever used for *very* low contaminated clusters.
    if GUMM_flag:
        print("Performing final GUMM analysis...", file=prfl)
        cl_probs = lowCIGUMMClean(
            N_membs, GUMM_perc, ID, cl_probs, clust_ID, clust_xy, prfl)

    # Estimate probabilities using KDEs for the field stars, and assigned
    # true members.
    if KDEP_flag:
        print("Performing KDE analysis...", file=prfl)
        cl_probs = KDEAnalysis.probs(xy, data, cl_probs)

    return list(cl_probs), KDE_vals


def reSampleData(resampleFlag, data, data_err, prfl, standard_scale=True):
    """
    Re-sample the data given its uncertainties using a normal distribution
    """
    if resampleFlag:
        # Gaussian random sample
        grs = np.random.normal(0., 1., data.shape[0])
        sampled_data = data + grs[:, np.newaxis] * data_err
    else:
        sampled_data = np.array(list(data))

    if standard_scale:
        print(
            "Standard scale: removed mean and scaled to unit variance",
            file=prfl)
        sampled_data = StandardScaler().fit(sampled_data).transform(
            sampled_data)

    return sampled_data


def dimReduc(cl_data, PCAflag, PCAdims, prfl):
    """
    Perform PCA and feature reduction
    """
    if PCAflag:
        pca = PCA(n_components=PCAdims)
        cl_data_pca = pca.fit(cl_data).transform(cl_data)
        print(" Selected N={} PCA features".format(PCAdims), file=prfl)
        var_r = ["{:.2f}".format(_) for _ in pca.explained_variance_ratio_]
        print(" Variance ratio: ", ", ".join(var_r), file=prfl)
    else:
        cl_data_pca = cl_data

    return cl_data_pca
