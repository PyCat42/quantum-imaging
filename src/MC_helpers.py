import numpy as np
from scipy.stats import norm

def sample_sinc2(sobol_2):
    """
    Sampling from a sinc2 function that is approximated by the sum of three Gaussians:
    N(x + mu, sigma_1), N(x, sigma_0), N(x − mu, sigma_1)
    :param sobol_2: array containing 2 Sobol sequences
    :return: np.array of the same shape with samples
    """
    c0 = 0.920
    c1 = 0.021
    mu = 4.493
    sigma_0 = 1.130
    sigma_1 = 0.527

    total = c0 + 2 * c1
    p0 = c0 / total # probability for a point to be in the central peak
    p1 = c1 / total # probability for a point to be in one of the side peaks

    peak = sobol_2[..., 0]  # picking which peak to sample from
    val = sobol_2[..., 1]  # sampling a value from the peak
    samples = np.zeros_like(val)

    # choose which peak to sample from
    # and then perform standard sampling from a Gaussian
    mask_left = (peak < p1)
    samples[mask_left] = norm.ppf(val[mask_left], loc=-mu, scale=sigma_1)

    mask_central = (peak >= p1) & (peak < p1 + p0)
    samples[mask_central] = norm.ppf(val[mask_central], loc=0, scale=sigma_0)

    mask_right = (peak >= p1 + p0)
    samples[mask_right] = norm.ppf(val[mask_right], loc=mu, scale=sigma_1)

    return samples

def sinc2_approximation(x):
    """
    Calculating sinc2 function that is approximated by the sum of three Gaussians:
    N(x + mu, sigma_1), N(x, sigma_0), N(x − mu, sigma_1)
    """
    c0 = 0.920
    c1 = 0.021
    total = c0 + 2 * c1

    mu = 4.493
    sigma_0 = 1.130
    sigma_1 = 0.527
    return ((c1 * norm.pdf(x, loc=-mu, scale=sigma_1)
            + c0 * norm.pdf(x, loc=0, scale=sigma_0)
            + c1 * norm.pdf(x, loc=mu, scale=sigma_1))
            / total)

def sinc_phys(x):
    """
    Sinc in np is defined as sin(pi * x) / (pi * x),
    while in the thesis it's sin(x) / x
    :param x:
    :return:
    """
    return np.sinc(x / np.pi)
