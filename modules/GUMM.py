
import numpy as np
from scipy.stats import multivariate_normal


def GUMMProbs(clust_xy, n_epochs=1000, stable_per=.1):
    """
    Fit a model composed of a 2D Gaussian and a 2D uniform distribution in a
    square region with [0., 1.] range.

    Based on the GMM model implementation shown in:
    https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95
    """
    cluster = initialize_cluster(clust_xy)

    lkl_old, nstable = -np.inf, 0
    for i in range(n_epochs):

        expectation_step(clust_xy, cluster)
        maximization_step(clust_xy, cluster)

        likelihood = cluster['likelihood']

        # Convergence check 1%
        if abs(likelihood - lkl_old) / likelihood < .1:
            nstable += 1
        if likelihood > lkl_old:
            lkl_old = likelihood
        if nstable == int(stable_per * n_epochs):
            # Converged. Breaking
            break

    # Extract probabilities associated to the 2D Gaussian
    gumm_p = np.array(list(cluster['gamma_g'].flatten()))

    # cl_cent = cluster['mu']

    return gumm_p


def initialize_cluster(data):
    """
    Initialize the 2D Gaussian parameters, and the weights for both
    distributions.
    """
    mu = np.random.uniform(.1, .9, (2,))
    cov = np.eye(2) * np.random.uniform(.1, .9, (2, 2))

    cluster = {'pi_u': .5, 'pi_g': .5, 'mu': mu, 'cov': cov}

    return cluster


def expectation_step(X, cluster):
    """
    """
    try:
        # Evaluate Gaussian distribution
        gamma_g = cluster['pi_g'] * multivariate_normal(
            mean=cluster['mu'], cov=cluster['cov']).pdf(X)

        # Evaluate uniform distribution (just a constant)
        gamma_u = cluster['pi_u'] * np.ones(X.shape[0])

        # Normalizing constant
        gammas_sum = gamma_g + gamma_u

        # Probabilities for each element
        cluster['gamma_g'] = gamma_g / gammas_sum
        cluster['gamma_u'] = gamma_u / gammas_sum

        # Save for breaking out
        cluster['likelihood'] = np.sum(np.log(gammas_sum))

    except np.linalg.LinAlgError:
        pass


def maximization_step(X, cluster):
    """
    """
    gamma_g = cluster['gamma_g']
    N_k = gamma_g.sum(0)

    # Mean
    mu = (gamma_g[:, np.newaxis] * X).sum(0) / N_k
    # Covariance
    cov = np.zeros((X.shape[1], X.shape[1]))
    for j in range(X.shape[0]):
        diff = (X[j] - mu).reshape(-1, 1)
        cov += gamma_g[j] * np.dot(diff, diff.T)
    cov /= N_k

    # Weight for the Gaussian distribution
    N = float(X.shape[0])
    pi_g = N_k / N

    # Weight for the uniform distribution
    pi_u = np.sum(cluster['gamma_u'], axis=0) / N

    # Update parameters
    cluster.update({'pi_u': pi_u, 'pi_g': pi_g, 'mu': mu, 'cov': cov})
