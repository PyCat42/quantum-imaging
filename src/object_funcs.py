import numpy as np

def no_object(x, y):
    t_o = np.ones_like(x, dtype=float)
    phi_o = np.zeros_like(x, dtype=float)

    return t_o, phi_o

def object_phase_bar(x, y, width=1e-3, center_y=0.0):
    """
    Infinite horizontal phase bar.
    The bar is parallel to the x-axis.
    Outside bar:  t_o = 1, phi_o = 0
    Inside bar: t_o = 1, phi_o = pi
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

def object_absorbing_bar(x, y, width=5e-3, center_y=0.0):
    """
    Infinite horizontal opaque bar.
    The bar is parallel to the x-axis.
    Outside bar:  t_o = 1, phi_o = 0
    Inside bar: t_o = 0, phi_o = 0
    :param x: x coordinate of a point
    :param y: y coordinate of a point
    :param width: width of a bar (y direction)
    :param center_y: y coordinate of bar center
    :return: t_o - transmission coefficient matrix of the object
            phi_o - matrix of phases added to the beam after interaction with the object
    """
    t_o = np.ones_like(x, dtype=float)
    phi_o = np.zeros_like(x, dtype=float)

    mask = np.abs(y - center_y) <= width / 2

    # Completely block the idler field inside the bar.
    t_o[mask] = 0.0

    return t_o, phi_o

def object_two_absorbing_bars(x, y, bar_width=2e-3, gap=1.1e-3, center_y=0.0):
    """
    Two infinite horizontal opaque bars.
    The bars are parallel to the x-axis and separated along y.
    Outside bars:  t_o = 1, phi_o = 0
    Inside bars: t_o = 0, phi_o = 0
    :param x: x coordinate of a point
    :param y: y coordinate of a point
    :param bar_width: width of each of the bars (y direction)
    :param gap: width of the gap between two bars
    :param center_y: y coordinate of bar centers
    :return: t_o - transmission coefficient matrix of the object
            phi_o - matrix of phases added to the beam after interaction with the object
    """
    if bar_width <= 0:
        raise ValueError("bar_width must be positive.")
    if gap < 0:
        raise ValueError("gap must be non-negative.")

    t_o = np.ones_like(x, dtype=float)
    phi_o = np.zeros_like(x, dtype=float)

    # Bar centers are separated by bar_width + gap!
    center_offset = 0.5 * (bar_width + gap)
    center_y_lower = center_y - center_offset
    center_y_upper = center_y + center_offset

    lower_bar_mask = (
        np.abs(y - center_y_lower)
        <= 0.5 * bar_width
    )

    upper_bar_mask = (
        np.abs(y - center_y_upper)
        <= 0.5 * bar_width
    )

    t_o[lower_bar_mask | upper_bar_mask] = 0.0

    return t_o, phi_o
