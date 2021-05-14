
import os
from pathlib import Path
import numpy as np
from astropy.stats import RipleysKEstimator
import time as t
from modules import outer
from modules.dataIO import readINI, dread, dmask, dxynorm, dwrite
import multiprocessing as mp


def main(
    parallel_flag, parallel_procs, rnd_seed, verbose, ID_c, x_c, y_c,
    data_cols, data_errs, oultr_method, stdRegion_nstd, OL_runs, resampleFlag,
    PCAflag, PCAdims, GUMM_flag, GUMM_perc, KDEP_flag, IL_runs, N_membs,
        N_cl_max, clust_method, clRjctMethod, C_thresh, cl_method_pars):
    """
    """
    out_folder = "output"
    # Create 'output' folder if it does not exist
    Path('./{}'.format(out_folder)).mkdir(parents=True, exist_ok=True)

    # Process all files inside the '/input' folder
    inputfiles = readFiles()
    for file_path in inputfiles:

        print("\n")
        print("===========================================================")
        print("Processing         : {}".format(file_path.name))
        # Set a random seed for reproducibility
        if rnd_seed == 'None':
            seed = np.random.randint(100000)
        else:
            seed = int(rnd_seed)
        print("Random seed        : {}".format(seed))
        np.random.seed(seed)

        # Original data
        full_data, cl_ID, cl_xy, cl_data, cl_errs, data_rjct = dread(
            file_path, ID_c, x_c, y_c, data_cols, data_errs)

        # Remove outliers
        msk_data, ID, xy, data, data_err = dmask(
            cl_ID, cl_xy, cl_data, cl_errs, oultr_method, stdRegion_nstd)

        # Normalize (x, y) data to [0, 1]
        xy01 = dxynorm(xy)

        probs_all = dataProcess(
            ID, xy01, data, data_err, verbose, OL_runs,
            parallel_flag, parallel_procs, resampleFlag, PCAflag, PCAdims,
            GUMM_flag, GUMM_perc, KDEP_flag, IL_runs, N_membs, N_cl_max,
            clust_method, clRjctMethod, C_thresh, cl_method_pars)

        if OL_runs > 1:
            # Obtain the mean of all runs. This are the final probabilities
            # assigned to each star in the frame
            probs_mean = np.mean(probs_all, 0)
        else:
            probs_mean = probs_all[0]

        # Write final data to file
        dwrite(
            out_folder, file_path, full_data, msk_data, probs_all, probs_mean)
        # Write rejected data (if any)
        if len(data_rjct) > 0:
            dwrite(out_folder, file_path, data_rjct, None, [], [])


def dataProcess(
    ID, xy, data, data_err, verbose, OL_runs, parallel_flag,
    parallel_procs, resampleFlag, PCAflag, PCAdims, GUMM_flag, GUMM_perc,
    KDEP_flag, IL_runs, N_membs, N_cl_max, clust_method, clRjctMethod,
        C_thresh, cl_method_pars):
    """
    """
    start_t = t.time()

    # TODO this should be handled by the logging() module
    # Set print() according to the 'verbose' parameter
    if verbose == 0:
        prfl = open(os.devnull, 'w')
    else:
        prfl = None

    # Print input parameters to screen
    if parallel_flag:
        print("Parallel runs      : {}".format(parallel_flag))
        print("Processes          : {}".format(parallel_procs))
    print("Outer loop runs    : {}".format(OL_runs))
    if PCAflag:
        print("Apply PCA          : {}".format(PCAflag))
        print(" PCA N_dims        : {}".format(PCAdims))
    if GUMM_flag:
        print("Apply GUMM         : {}".format(GUMM_flag))
        print(" GUMM percentile   : {}".format(GUMM_perc))
    if KDEP_flag:
        print("Obtain KDE probs   : {}".format(KDEP_flag))

    print("Inner loop runs    : {}".format(IL_runs))
    print("Stars per cluster  : {}".format(N_membs))
    print("Maximum clusters   : {}".format(N_cl_max))
    print("Clustering method  : {}".format(clust_method))
    if cl_method_pars:
        for key, val in cl_method_pars.items():
            print(" {:<17} : {}".format(key, val))
    print("")
    # print("Rejection method   : {}".format(clRjctMethod))
    # if clRjctMethod != 'rkfunc':
    #     print("Threshold          : {:.2f}".format(C_thresh))

    # Define RK test with an area of 1.
    # Kest = None
    # if clRjctMethod == 'rkfunc':
    Kest = RipleysKEstimator(area=1, x_max=1, y_max=1, x_min=0, y_min=0)
    # if clRjctMethod == 'kdetest' or clust_method == 'rkmeans':
    #     from rpy2.robjects import r
    #     from rpy2.robjects import numpy2ri
    #     from rpy2.robjects.packages import importr
    #     # cat(paste("R version: ",R.version.string,"\n"))
    #     importr('MASS')
    #     r("""
    #     set.seed(12345)
    #     """)
    #     numpy2ri.activate()
    #     r.assign('nruns', 2000)
    #     r.assign('nKde', 50)

    # Arguments for the Outer Loop
    OLargs = (
        ID, xy, data, data_err, resampleFlag, PCAflag, PCAdims, GUMM_flag,
        GUMM_perc, KDEP_flag, IL_runs, N_membs, N_cl_max, clust_method,
        clRjctMethod, Kest, C_thresh, cl_method_pars, prfl)

    # TODO: Breaks if verbose=0
    if parallel_flag is True:
        if parallel_procs == 'None':
            # Use *almost* all the cores
            N_cpu = mp.cpu_count() - 1
        else:
            N_cpu = int(parallel_procs)
        with mp.Pool(processes=N_cpu) as p:
            manager = mp.Manager()
            KDE_vals = manager.dict({})
            probs_all = p.starmap(
                OLfunc, [(OLargs, KDE_vals) for _ in range(OL_runs)])

    else:
        KDE_vals = {}
        probs_all = []
        for _ in range(OL_runs):
            print("\n--------------------------------------------------------")
            print("OL run {}".format(_ + 1))
            # The KDE_vals dictionary is updated after each OL run
            probs, KDE_vals = outer.loop(*OLargs, KDE_vals)
            probs_all.append(probs)

            p_dist = [
                (np.mean(probs_all, 0) > _).sum() for _ in
                (.1, .3, .5, .7, .9)]
            print("\nP>(.1, .3, .5, .7, .9): {}, {}, {}, {}, {}".format(
                *p_dist), file=prfl)

    elapsed = t.time() - start_t
    if elapsed > 60.:
        elapsed, ms_id = elapsed / 60., "minutes"
    else:
        ms_id = "seconds"
    print("\nTime consumed: {:.1f} {}".format(elapsed, ms_id))

    return probs_all


def OLfunc(args, KDE_vals):
    """
    Here to handle the parallel runs.
    """
    probs, _ = outer.loop(*args, KDE_vals)
    return probs


def readFiles():
    """
    Read files from the input folder
    """
    files = []
    for pp in Path('input').iterdir():
        if pp.is_file():
            files += [pp]
        else:
            files += [arch for arch in pp.iterdir()]

    return files


if __name__ == '__main__':

    # # Limit numpy's cores used to 1
    # # Source: https://stackoverflow.com/a/58195413/1391441, also
    # # https://stackoverflow.com/q/17053671/1391441

    # parallel_flag, parallel_procs = params[:2]
    # if parallel_flag:
    #     if parallel_procs == 'None':
    #         # Use *almost* all the cores
    #         parallel_procs = mp.cpu_count() - 1
    #     else:
    #         # Never use more than these cores
    #         parallel_procs = min(int(parallel_procs), mp.cpu_count() - 1)
    # else:
    #     parallel_procs = 1

    # Read input parameters.
    params = readINI()

    if params[0] is False:
        # Disable numpy's multithreading
        parallel_procs = str(1)
        os.environ["OMP_NUM_THREADS"] = parallel_procs
        os.environ["MKL_NUM_THREADS"] = parallel_procs
        os.environ["OPENBLAS_NUM_THREADS"] = parallel_procs
        os.environ["VECLIB_MAXIMUM_THREADS"] = parallel_procs
        os.environ["NUMEXPR_NUM_THREADS"] = parallel_procs
    else:
        # If numpy is allowed to multithread, disable the parallel run
        params[1] = False

    main(*params[1:])
