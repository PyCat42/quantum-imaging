import numpy as np
from scipy.optimize import brentq
from scipy.constants import c

from src.sellmeier import n_eff


def spherical_to_cartesian(k, theta, phi):
    """
    Transfers wavevector from spherical to cartesian coordinates.
    :param k: wavevector
    :param theta: spherical angle
    :param phi: azimuthal angle
    :return:
    """
    k_x = k * np.sin(theta) * np.cos(phi)
    k_y = k * np.sin(theta) * np.sin(phi)
    k_z = k * np.cos(theta) * np.ones_like(phi)
    return k_x, k_y, k_z


def cartesian_to_spherical(k_x, k_y, k_z):
    """
    Transfers wavevector from cartesian to spherical coordinates.
    :param k_x: x component of wavevector
    :param k_y: y component of wavevector
    :param k_z: z component of wavevector
    :return:
    """
    k = np.sqrt(k_x**2 + k_y**2 + k_z**2)
    theta = np.arccos(k_z / k)
    phi = np.arctan2(k_y, k_x)

    return k, theta, phi

def lambda_i_from_lambda_s(lambda_p, lambda_s):
    """
    Calculates idler wavelength from energy conservation law (ω_p = ω_i + ω_s)
    :param lambda_p: pump wavelength
    :param lambda_s: signal wavelength
    :return: idler wavelength
    """
    return lambda_p * lambda_s / (lambda_s - lambda_p)

def get_theta_i(k_i_normal, lambda_i, n_i):
    """
    Numerically solving for idler wavelength when its wavevector is known.
    :param k_i_normal: idler wavevector component normal to the beam propagation direction
    :param lambda_i: idler wavelength
    :param n_i: idler refractive index function (from Sellmeier.py)
    :return: idler spherical angle
    """
    def solve_single_k(single_k_i_normal):
        objective = lambda theta_i: 2 * np.pi * n_i(lambda_i, theta_i) * np.sin(theta_i) / lambda_i - single_k_i_normal
        a = 0
        b = np.pi / 2

        f_a = objective(a)
        f_b = objective(b)

        if f_a * f_b > 0:
            return np.nan
        try:
            return brentq(objective, a, b)
        except ValueError:
            return np.nan

    vec_solution = np.vectorize(solve_single_k)
    return vec_solution(k_i_normal)

def get_theta_i_vectorized(k_i_normal, lambda_i, n_i, n_iter=35):
    """
    Vectorized solution of

        2*pi*n(lambda_i, theta_i)*sin(theta_i)/lambda_i
            = k_i_normal

    using simultaneous bisection for all array elements.
    """

    k_i_normal = np.asarray(k_i_normal)
    lambda_i = np.asarray(lambda_i)

    theta_lo = np.zeros_like(k_i_normal, dtype=np.float64)
    theta_hi = np.full_like(k_i_normal, np.pi / 2, dtype=np.float64)

    # First check which points actually have a solution.
    f_lo = (
        2 * np.pi
        * n_i(lambda_i, theta_lo)
        * np.sin(theta_lo)
        / lambda_i
        - k_i_normal
    )

    f_hi = (
        2 * np.pi
        * n_i(lambda_i, theta_hi)
        * np.sin(theta_hi)
        / lambda_i
        - k_i_normal
    )

    valid = (f_lo <= 0) & (f_hi >= 0)

    # Bisection
    for _ in range(n_iter):
        theta_mid = 0.5 * (theta_lo + theta_hi)

        f_mid = (
            2 * np.pi
            * n_i(lambda_i, theta_mid)
            * np.sin(theta_mid)
            / lambda_i
            - k_i_normal
        )

        below = f_mid < 0

        theta_lo = np.where(below, theta_mid, theta_lo)
        theta_hi = np.where(below, theta_hi, theta_mid)

    theta = 0.5 * (theta_lo + theta_hi)
    theta[~valid] = np.nan

    return theta

def solve_single_k_theta(single_k_i, single_theta_i, n_i_1, n_i_2, temp, lambda_p):
    """
    Numerically solving for idler wavelength when its wavevector and polar angle are known.
    :param single_k_i: idler wavevector component normal to the beam propagation direction (single value)
    :param single_theta_i: idler polar angle (single value)
    :param n_i_1: function for the first component of idler effective refractive index
    :param n_i_2: function for the second component of idler effective refractive index
    :param temp: crystal temperature (in deg C)
    :param lambda_p: pump wavelength
    :return: idler wavelength
    """
    objective = lambda lambda_i: (
        2 * np.pi * n_eff(n_i_1, n_i_2, lambda_i, single_theta_i, temp ) / lambda_i
        - single_k_i
    )

    a = lambda_p
    b = 10 * lambda_p

    f_a = objective(a)
    f_b = objective(b)

    if f_a * f_b > 0:
        return np.nan

    try:
        return brentq(objective, a, b)
    except ValueError:
        return np.nan

def get_lambda_i(k_i, theta_i, n_i_1, n_i_2, temp, lambda_p):
    """
    Numerically solving for idler wavelength when its wavevector and polar angle are known.
    This functions runs calculation simultaneously for multiple idler samples.
    :param k_i: idler wavevector component normal to the beam propagation direction (vector of values)
    :param theta_i: idler polar angle (vector of values)
    :param n_i_1: function for the first component of idler effective refractive index
    :param n_i_2: function for the second component of idler effective refractive index
    :param temp: crystal temperature (in deg C)
    :param lambda_p: pump wavelength
    :return: vector containing idler wavelengths
    """
    return np.array([
        solve_single_k_theta(k, theta, n_i_1, n_i_2, temp, lambda_p)
        for k, theta in zip(k_i, theta_i)
    ])

def get_lambda_i_vectorized(
    k_i,
    theta_i,
    n_i_1,
    n_i_2,
    temp,
    lambda_p,
    n_iter=40):
    """
    Vectorized solution of

        2*pi*n_eff(lambda_i, theta_i)/lambda_i = k_i

    using simultaneous bisection.
    """

    k_i = np.asarray(k_i)
    theta_i = np.asarray(theta_i)

    k_i, theta_i = np.broadcast_arrays(k_i, theta_i)

    lo = np.full_like(k_i, lambda_p, dtype=np.float64)
    hi = np.full_like(k_i, 10 * lambda_p, dtype=np.float64)

    def k_from_lambda(wavelength):
        n = n_eff(
            n_i_1,
            n_i_2,
            wavelength,
            theta_i,
            temp
        )
        return 2 * np.pi * n / wavelength

    k_lo = k_from_lambda(lo)
    k_hi = k_from_lambda(hi)

    # k(lambda) should decrease with lambda.
    valid = (k_lo >= k_i) & (k_hi <= k_i)

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        k_mid = k_from_lambda(mid)

        # k_mid > target -> wavelength needs to increase
        too_large_k = k_mid > k_i

        lo = np.where(too_large_k, mid, lo)
        hi = np.where(too_large_k, hi, mid)

    lambda_i = 0.5 * (lo + hi)
    lambda_i[~valid] = np.nan

    return lambda_i


def w_from_k_components(k_ix, k_iy, k_iz, n_i_1, n_i_2, temp, lambda_p):
    """
    Calculates angular frequency when wavevector components are known.
    :param k_ix: component of wavevector along x-axis
    :param k_iy: component of wavevector along y-axis
    :param k_iz: component of wavevector along z-axis
    :param n_i_1: function for the first component of idler effective refractive index
    :param n_i_2: function for the second component of idler effective refractive index
    :param temp: crystal temperature (in deg C)
    :param lambda_p: pump wavelength
    :return: angular frequency
    """
    k_i = np.sqrt(k_ix ** 2 + k_iy ** 2 + k_iz ** 2)

    theta_i = np.arctan2(
        np.sqrt(k_ix ** 2 + k_iy ** 2),
        k_iz
    )

    lambda_i = get_lambda_i_vectorized(k_i, theta_i, n_i_1, n_i_2, temp, lambda_p)

    return 2 * np.pi * c / lambda_i
