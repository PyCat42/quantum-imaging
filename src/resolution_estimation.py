import numpy as np

def rayleigh_criterion(v_background, v_bar, v_mid):
    """
    Calculates Rayleigh-like criterion for resolution of the setup.
    :param v_background: visibility somewhere far from the bars
    :param v_bar: visibility at the center of the bar (darkest point)
    :param v_mid: visibility at the center of the space between the bars
    :return: dip
            1: means full midpoint recovery to background.
            0.735: threshold value
            0: means no recovery; midpoint equals the bar minimum.
    """
    full_contrast = v_background - v_bar
    if (
        not np.isfinite(full_contrast)
        or np.abs(full_contrast) < 1e-14
    ):
        return np.nan

    return (v_mid - v_bar) / full_contrast

def rayleigh_like_two_bar_metric(
    visibility,
    x_centers,
    y_centers,
    bar_width,
    separation,
    center_y=0.0,
):
    """
    Compute a Rayleigh-like midpoint-recovery metric for two
    symmetric horizontal opaque bars, using exactly one pixel
    for each of: background, bar, and midpoint.
    The bars have centers at:
        y = center_y - separation/2
        y = center_y + separation/2
    :param visibility: visibility image indexed as visibility[y_index, x_index]
    :param x_centers: x coordinate centers corresponding to the visibility array, in metres
    :param y_centers: y coordinate centers corresponding to the visibility array, in metres
    :param bar_width: width of each bar in metres
    :param separation: center-to-center separation between the bars in metres
    :param center_y: midpoint between the bar centers in metres (default is 0.0)
    :return:
    """
    visibility = np.asarray(visibility, dtype=float)
    x_centers = np.asarray(x_centers, dtype=float)
    y_centers = np.asarray(y_centers, dtype=float)

    if visibility.shape != (y_centers.size, x_centers.size):
        raise ValueError(
            f"Expected visibility.shape == ({len(y_centers)}, {len(x_centers)});"
            f"got {visibility.shape}."
        )

    if bar_width <= 0:
        raise ValueError("bar_width must be positive.")
    if separation < bar_width:
        raise ValueError("separation must be >= bar_width.")

    ny, nx = visibility.shape

    # Column closest to x = 0
    x0_index = int(np.argmin(np.abs(x_centers)))

    # Known bar-center locations.
    y_bar_lower = center_y - 0.5 * separation
    y_bar_upper = center_y + 0.5 * separation

    # The midpoint is at x=0, y=center_y.
    y_mid = center_y

    # Background: take the furthest pixel along y at x = 0.
    # Here we just take the top edge.
    y_bg = y_centers[-1]  # or y_centers[0] if you prefer the bottom edge
    y_bg_index = ny - 1   # or 0 for bottom edge

    # Helper: find nearest pixel index along y for a target y-coordinate.
    def nearest_y_index(y_target):
        return int(np.argmin(np.abs(y_centers - y_target)))

    # Indices for bar and midpoint at x = 0.
    y_bar_lower_index = nearest_y_index(y_bar_lower)
    y_bar_upper_index = nearest_y_index(y_bar_upper)
    y_mid_index = nearest_y_index(y_mid)

    # Extract single-pixel visibility values at x = 0.
    V_bar_lower = visibility[y_bar_lower_index, x0_index]
    V_bar_upper = visibility[y_bar_upper_index, x0_index]
    V_bar = 0.5 * (V_bar_lower + V_bar_upper)

    V_mid = visibility[y_mid_index, x0_index]
    V_bg = visibility[y_bg_index, x0_index]

    rayleigh_metric_experimental = rayleigh_criterion(V_bg, V_bar, V_mid)
    rayleigh_metric_threshold = 0.735

    resolved = (
        np.isfinite(rayleigh_metric_experimental)
        and rayleigh_metric_experimental >= rayleigh_metric_threshold
    )

    return rayleigh_metric_experimental, resolved
