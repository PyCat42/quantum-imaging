import numpy as np

def object_phase_bar(x, y, width=1e-4, center_y=0.0):
    """
    Infinite horizontal phase bar.
    :param x: x coordinate in the object plane
    :param y: y coordinate in the object plane
    :param width: width of the object
    :param center_y: bar center position in metres.
    :return: t_o - transmission coefficient matrix of the object
            phi_o - matrix of phases added to the beam after interaction with the object
    """
    t_o = np.ones_like(x, dtype=float)
    phi_o = np.zeros_like(x, dtype=float)

    mask = np.abs(y - center_y) <= width / 2
    phi_o[mask] = np.pi

    return t_o, phi_o
