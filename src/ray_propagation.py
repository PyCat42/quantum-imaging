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


def medium_to_air(lambda_med, theta_med, n):
    """
    Modeling transition from medium (crystal) to air.
    Section 4.3.2, equation 4.10, Riexinger
    :param lambda_med: ray wavelength in medium
    :param theta_med: ray angle in medium
    :param n: refractive index function (from Sellmeier.py)
    :return: ray angle in vacuum (theta_vac_solution)
            ray wavelength in medium (lambda_vac_solution)
    """
    def medium_to_air_single(lambda_med_single, theta_med_single):
        objective = lambda lambda_vac: lambda_vac / n(lambda_vac, theta_med_single) - lambda_med_single

        # n > 1 => lambda_vac > lambda_med
        lambda_vac_solution = brentq(objective, lambda_med_single, 10 * lambda_med_single)

        # use Snell's law to retrieve theta_vac
        theta_vac_solution = np.arcsin(n(lambda_vac_solution, theta_med_single) * np.sin(theta_med_single))

        return lambda_vac_solution, theta_vac_solution

    vec_solutions = np.vectorize(medium_to_air_single)
    return vec_solutions(lambda_med, theta_med)

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
