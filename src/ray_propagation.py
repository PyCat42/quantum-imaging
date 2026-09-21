import numpy as np
from scipy.optimize import brentq


def M_free_space(L):
    """
    Transfer matrix for propagation in a medium with const refractive index.
    :param L: propagation distance
    :return: 2x2 np.array
    """
    return np.array([[1, L], [0, 1]])


def M_mirror(theta=0):
    """
    Transfer matrix for a regular mirror.
    :param theta: mirror tilt angle
    :return: 2x2 np.array
    """
    return np.array([[1, 0], [0, 1]])


def M_lens(f):
    """
    Transfer matrix for a thin lens.
    :param f: focal length of the lens
    :return: 2x2 np.array
    """
    return np.array([[1, 0], [- 1 / f, 1]])

def medium_to_air(lambda_vac, theta_med, n):
    """
    Modeling transition from medium (crystal) to air.
    :param lambda_med: ray wavelength in medium
    :param theta_med: ray angle in medium
    :param n: refractive index function (from Sellmeier.py)
    :return: ray angle in vacuum (theta_vac_solution)
            ray wavelength in medium (lambda_vac_solution)
    """

    lambda_vac = np.asarray(lambda_vac, dtype=float)
    theta_med = np.asarray(theta_med, dtype=float)

    # Make input arrays the same shape before boolean masking.
    lambda_vac, theta_med = np.broadcast_arrays(lambda_vac, theta_med)
    n_med = n(lambda_vac, theta_med)
    n_med = np.asarray(n_med, dtype=float)

    # The index function should also broadcast to the common shape.
    n_med = np.broadcast_to(n_med, lambda_vac.shape)

    sin_theta_vac = n_med * np.sin(theta_med)

    valid = (
        np.isfinite(lambda_vac)
        & np.isfinite(theta_med)
        & np.isfinite(n_med)
        & (np.abs(sin_theta_vac) <= 1.0)
    )

    lambda_vac_out = np.where(valid, lambda_vac, np.nan)
    theta_vac = np.where(
        valid,
        np.arcsin(np.clip(sin_theta_vac, -1.0, 1.0)),
        np.nan,
    )

    return lambda_vac_out, theta_vac

def propagate(r, alpha, M):
    """
    Propagate created beam using transfer matrices
    :param r: beam transverse position
    :param alpha: beam angle (equals to spherical theta angle)
    :param M: transfer matrix
    :return: [r_out, alpha_out]
    """
    vector_in = np.stack([r, alpha], axis=1)
    vector_out = np.matmul(vector_in, M.T)
    r_out = vector_out[:, 0]
    alpha_out = vector_out[:, 1]
    return r_out, alpha_out

def get_ray_coordinates(r, psi):
    """
    Calculate x and y coordinates for a propagated beam.
    :param r: [transverse position, angle]
    :param psi: azimuthal angle (randomly selected)
    :return: x, y coordinates
    """
    x = r * np.cos(psi)
    y = r * np.sin(psi)
    return x, y
