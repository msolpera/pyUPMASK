# pyUPMASK

`pyUPMASK` is an unsupervised clustering method for stellar clusters that builds upon the original [UPMASK](https://cran.r-project.org/web/packages/UPMASK/) package. Its general approach makes it plausible to be applied to analyses that deal with binary classes of any kind, as long as the fundamental hypotheses are met.

The core of the algorithm follows the method developed in `UPMASK` but introducing several key enhancements that make it not only more general, they also improve its performance considerably.



## Installation

We recommend that the packages are installed inside a [conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment. To do this, first follow the steps to install [Miniconda](https://docs.conda.io/en/latest/miniconda.html). After this,  install the required packages into a `conda` environment using the following command in a terminal:

    $ conda create -n pyupmask numpy scikit-learn scipy astropy

To activate the environment (called `pyupmask`), use:

    $ conda activate pyupmask

The `(pyupmask)` before the `$` symbol in the terminal indicates that the environment is activated.

Alternatively you can install the packages in your system directly using `pip`, but this method is not recommended.

Once you have downloaded the pyUPMASK compressed file from GitHub, simply extract its contents anywhere in your system.


## Running

Once the package is uncompressed and the environment activated, the user needs to set the desired input parameters in the `params.ini` file. The cluster data files that will be processed must to be stored in the `input/` sub-folder.

The code is run simply with:

    (pyupmask) $ python pyUPMASK.py

Notice that you need to activate the environment *before* running `pyUPMASK`, every time you want to run it.

The code comes with a synthetic cluster to test it. The results will be stored in the `output/` sub-folder.


## Referencing

The accompanying article describing the code in detail can be accessed
[via A&A](https://www.aanda.org/articles/aa/pdf/2021/06/aa40252-20.pdf), and referenced using the following BibTeX entry:

````
@ARTICLE{2021A&A...650A.109P,
       author = {{Pera}, M.~S. and {Perren}, G.~I. and {Moitinho}, A. and {Navone}, H.~D. and {Vazquez}, R.~A.},
        title = "{pyUPMASK: an improved unsupervised clustering algorithm}",
      journal = {\aap},
     keywords = {open clusters and associations: general, methods: data analysis, open clusters and associations: individual: NGC 2516, methods: statistical, Astrophysics - Astrophysics of Galaxies},
         year = 2021,
        month = jun,
       volume = {650},
          eid = {A109},
        pages = {A109},
          doi = {10.1051/0004-6361/202040252},
archivePrefix = {arXiv},
       eprint = {2101.01660},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021A&A...650A.109P},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
````

