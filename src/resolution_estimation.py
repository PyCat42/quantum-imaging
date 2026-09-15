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
    visibility, x_centers, y_centers,
    bar_width, separation,
    center_y=0.0,
    sample_half_width=None, background_offset=None,
):
    """
    Compute a Rayleigh-like midpoint-recovery metric for two
    symmetric horizontal opaque bars.

    The bars have centers at:
        y = center_y - separation/2
        y = center_y + separation/2

    The pair midpoint is:
        y = center_y.

    Parameters
    ----------
    visibility : ndarray, shape (Ny, Nx)
        Visibility image indexed as visibility[y_index, x_index].

    x_centers, y_centers : ndarray
        Coordinate centers corresponding to the visibility array,
        in metres.

    bar_width : float
        Width of each bar in metres.

    separation : float
        Center-to-center separation between the bars in metres.

    center_y : float
        Midpoint between the bar centers in metres.

    sample_half_width : float or None
        Half-width of square local sampling windows in metres.
        If None, use min(0.2 * bar_width, 0.25 * separation).

    background_offset : float or None
        Distance from each bar center to the background-sampling
        positions. If None, use separation/2 + bar_width.

    Returns
    -------
    result : dict
        Contains V_bar, V_mid, V_bg, recovery, residual_dip,
        pass/fail under the Rayleigh-like threshold, and sampled
        positions.
    """
    visibility = np.asarray(visibility, dtype=float)
    x_centers = np.asarray(x_centers, dtype=float)
    y_centers = np.asarray(y_centers, dtype=float)

    if visibility.shape != (y_centers.size, x_centers.size):
        raise ValueError(
            "Expected visibility.shape == "
            "(len(y_centers), len(x_centers)); got "
            f"{visibility.shape}, "
            f"({y_centers.size}, {x_centers.size})."
        )
    if bar_width <= 0:
        raise ValueError("bar_width must be positive.")
    if separation < bar_width:
        raise ValueError("separation must be >= bar_width.")

    if sample_half_width is None:
        sample_half_width = min(
            0.20 * bar_width,
            0.25 * separation,
        )
    if sample_half_width <= 0:
        raise ValueError("sample_half_width must be positive.")

    if background_offset is None:
        background_offset = (
            0.5 * separation
            + bar_width
        )

    x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing="xy")

    # Known bar-center locations.
    y_bar_lower = center_y - 0.5 * separation
    y_bar_upper = center_y + 0.5 * separation

    # The midpoint is at x=0, y=center_y.
    x_mid = 0.0
    y_mid = center_y

    # Take background samples beyond the outer bar edges.
    y_bg_lower = center_y - background_offset
    y_bg_upper = center_y + background_offset

    def local_mean(x0, y0):
        region = (
            (np.abs(x_grid - x0) <= sample_half_width)
            & (np.abs(y_grid - y0) <= sample_half_width)
            & np.isfinite(visibility)
        )

        if not np.any(region):
            return np.nan

        return np.nanmean(
            visibility[region]
        )

    V_bar_lower = local_mean(0.0, y_bar_lower)
    V_bar_upper = local_mean(0.0, y_bar_upper)
    V_bar = np.nanmean([V_bar_lower, V_bar_upper])

    V_mid = local_mean(x_mid, y_mid)

    V_bg_lower = local_mean(0.0, y_bg_lower)
    V_bg_upper = local_mean(0.0, y_bg_upper)
    V_bg = np.nanmean([V_bg_lower, V_bg_upper])

    rayleigh_metric_experimental = rayleigh_criterion(V_bg, V_bar, V_mid)
    rayleigh_metric_threshold = 0.735

    resolved = (
        np.isfinite(rayleigh_metric_experimental)
        and rayleigh_metric_experimental >= rayleigh_metric_threshold
    )

    return (rayleigh_metric_experimental, resolved,
            {
                "V_bar_lower": V_bar_lower,
                "V_bar_upper": V_bar_upper,
                "V_bar": V_bar,
                "V_mid": V_mid,
                "V_bg_lower": V_bg_lower,
                "V_bg_upper": V_bg_upper,
                "V_bg": V_bg,
                "rayleigh_metric_experimental": rayleigh_metric_experimental,
                "rayleigh_metric_threshold": rayleigh_metric_threshold,
                "resolved": resolved,
                "bar_width": bar_width,
                "separation": separation,
                "gap": separation - bar_width,
                "sample_half_width": sample_half_width,
                "bar_center_lower": y_bar_lower,
                "bar_center_upper": y_bar_upper,
                "midpoint_y": y_mid,
                "background_y_lower": y_bg_lower,
                "background_y_upper": y_bg_upper,
            }
            )