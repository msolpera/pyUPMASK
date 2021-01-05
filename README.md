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

The latest version of the code can be obtained as a compressed file in [Releases](https://github.com/asteca/ASteCA/releases). Once downloaded, simply extract its contents anywhere in your system.


## Running

Once the package is uncompressed and the environment activated, the user needs to set the desired input parameters in the `params.ini` file. The cluster data files that will be processed must to be stored in the `input/` sub-folder.

The code is run simply with:

    (pyupmask) $ python pyUPMASK.py

Notice that you need to activate the environment *before* running `pyUPMASK`, every time you want to run it.

The code comes with a synthetic cluster to test it. The results will be stored in the `output/` sub-folder.


## Referencing

The accompanying article describing the code in detail can be accessed
[via A&A][xxxx], and referenced using the following BibTeX entry:

````
@article{Pera_2021,
    author = {{Pera, M. S.}and {Perren, G. I.} and {Moitinho, A.} and {Navone, H. D.} and {V\'azquez, R. A.}},
    title = {pyUPMASK: an improved unsupervised clustering algorithm},
    DOI= "xxx",
    url= "http://dx.doi.org/xxx",
    journal = {A\&A},
    year = 2021,
    volume = ???,
    pages = "??",
    month = "??"
}
````

