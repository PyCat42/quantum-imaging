import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import ticker
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
import numpy as np

from src.imaging_systems import Michaelson

def plot_spectrum(spdc, vals):
    """
    Plots SPDC spectrum with phase matching contour.
    :param spdc: instance of SPDC class
    :param vals: intensity grid
    :return:
    """
    # Plot spectrum
    plt.figure(figsize=(10, 6))
    spectrum = vals.reshape((
        spdc.theta_s_grid.size,
        spdc.lambda_s_grid.size,
    ))
    plt.pcolormesh(
        spdc.lambda_s_grid * 1e9, # convert to nm
        np.rad2deg(spdc.theta_s_grid), # convert theta to degrees
        spectrum,
        shading='auto',
        cmap='magma',
        vmin=0
    )
    plt.colorbar(label="Transition rate dR")

    # Plot phase matching contour
    lam, theta, dkz = spdc.get_phase_matching_contour()
    plt.contour(
        lam * 1e9, np.rad2deg(theta), dkz,
        levels=[0.0],
        colors='red', linewidths=0.6, linestyles='-',
    )
    contour_handle = mlines.Line2D(
        [], [],
        color='red',
        linewidth=2,
        linestyle='-',
        label=r'$\Delta k_z = 0$',
    )
    plt.legend(handles=[contour_handle])

    plt.title("Spectrum")
    plt.xlabel(r"$\lambda_s$ (nm)")
    plt.ylabel(r"$\theta_s$ internal (deg)")

    plt.show()

def plot_visibility_images(detector_size, detector_pixels,
                           plus_image, minus_image, detector_visibility):
    """
    Plots constructive and destructive interference, as well as visibility map side by side.
    :param detector_size: size of the detector in m
    :param detector_pixels: number of pixels in the detector
    :param plus_image: constructive interference image
    :param minus_image: destructive interference image
    :param detector_visibility: visibility map on the detector
    :return:
    """
    x_edges = np.linspace(-detector_size / 2, detector_size / 2, detector_pixels + 1)
    y_edges = np.linspace(-detector_size / 2, detector_size / 2, detector_pixels + 1)

    # In case that in some detector region we don't have any signal samples (NaN) it will be white
    img1 = np.ma.masked_invalid(plus_image)
    img2 = np.ma.masked_invalid(minus_image)
    img3 = np.ma.masked_invalid(detector_visibility)

    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(17, 5), sharey=True)

    cmap = plt.colormaps["magma"].copy()
    cmap.set_bad("white")

    # Use common normalization for constructive and destructive interference image
    vmax = np.nanmax(
        np.concatenate([plus_image.ravel(), minus_image.ravel()])
    )

    mesh0 = ax[0].pcolormesh(
        x_edges*1e3,
        y_edges*1e3,
        img1,
        shading="auto",
        cmap=cmap,
        vmin=0,
        vmax=vmax
    )
    ax[0].set_title("Constructive Interference")
    ax[0].set_xlabel(r"$x_{\mathrm{det}}$ (mm)")
    ax[0].set_ylabel(r"$y_{\mathrm{det}}$ (mm)")
    ax[0].set_aspect('equal')
    fig.colorbar(mesh0, ax=ax[0], label="Visibility", orientation="vertical")

    mesh1 = ax[1].pcolormesh(
        x_edges*1e3,
        y_edges*1e3,
        img2,
        shading="auto",
        cmap=cmap,
        vmin=0,
        vmax=vmax
    )
    ax[1].set_title("Destructive Interference")
    ax[1].set_xlabel(r"$x_{\mathrm{det}}$ (mm)")
    ax[1].set_aspect('equal')
    fig.colorbar(mesh1, ax=ax[1], label="Count rate (1/s)", orientation="vertical")

    mesh2 = ax[2].pcolormesh(
        x_edges*1e3,
        y_edges*1e3,
        img3,
        shading="auto",
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    ax[2].set_title("Detector Visibility")
    ax[2].set_xlabel(r"$x_{\mathrm{det}}$ (mm)")
    ax[2].set_aspect('equal')
    fig.colorbar(mesh2, ax=ax[2], label="Count rate (1/s)", orientation="vertical")

    plt.show()

def overlay_object_outline(ax,
    detector_size, detector_pixels,
    object_func,
    magnification,
    color="red", linewidth=1.5, linestyle="--", alpha=0.95,
    contour_level=0.01
):
    """
    Draw the geometrically magnified object boundary over a detector image.
    :param ax: existing axes containing a detector-plane image
    :param detector_size: physical detector width in metres
    :param detector_pixels: number of detector pixels along each dimension
    :param object_func: function with signature t_o, phi_o = object_func(x_object, y_object)
    :param magnification: signed detector/object magnification
    :param color, linewidth, linestyle, alpha: matplotlib contour style settings
    :param contour_level: level at which to draw object contour
    :return:
    """
    if not np.isfinite(magnification) or magnification == 0:
        raise ValueError(
            "magnification must be a finite nonzero number."
        )

    detector_edges = np.linspace(-detector_size / 2, detector_size / 2, detector_pixels + 1)
    detector_centers = 0.5 * (detector_edges[:-1] + detector_edges[1:])
    x_det, y_det = np.meshgrid(
        detector_centers,
        detector_centers,
        indexing="xy",
    )

    # A negative M automatically includes image inversion.
    x_obj = x_det / magnification
    y_obj = y_det / magnification

    t_o, phi_o = object_func(x_obj, y_obj)
    t_o = np.asarray(t_o, dtype=float)
    phi_o = np.asarray(phi_o, dtype=float)

    if t_o.shape != x_det.shape:
        raise ValueError(
            "object_func returned an array with shape "
            f"{t_o.shape}, but expected {x_det.shape}."
        )
    if phi_o.shape != x_det.shape:
        raise ValueError(
            "object_func returned an array with shape "
            f"{phi_o.shape}, but expected {x_det.shape}."
        )

    object_map = t_o
    finite = np.isfinite(object_map)
    if not np.any(finite):
        return None

    object_values = object_map[finite]
    object_min = np.min(object_values)
    object_max = np.max(object_values)

    # No boundary if the object map is constant.
    if np.isclose(
        object_min,
        object_max,
        rtol=1e-12,
        atol=1e-14,
    ):
        return None

    contour_set = ax.contour(
        x_det,
        y_det,
        object_map,
        levels=[contour_level],
        colors=[color],
        linewidths=linewidth,
        linestyles=linestyle,
        alpha=alpha,
    )

    return contour_set

def plot_visibility(
    detector_size,
    detector_pixels,
    detector_visibility,
    overlay_object_contour=False,
    object_func=None,
    magnification=None,
    outline_color="red",
    draw_x0=False
):
    """

    :param detector_size: physical detector width in metres
    :param detector_pixels: number of detector pixels along each dimension
    :param detector_visibility: measured visibility in detector plane
    :param overlay_object_contour: True or False (default)
    :param object_func: function with signature t_o, phi_o = object_func(x_obj, y_obj)
    :param magnification: imaging system magnification
    :param outline_color: object outline color
    :param draw_x0: draw vertical cut through x=0 (for resolution!)
    :return:
    """
    detector_visibility = np.asarray(detector_visibility, dtype=float)
    if detector_visibility.shape != (detector_pixels, detector_pixels):
        raise ValueError(
            "detector_visibility shape does not match detector_pixels: "
            f"got {detector_visibility.shape}, expected "
            f"({detector_pixels}, {detector_pixels})."
        )
    visibility_masked = np.ma.masked_invalid(detector_visibility)

    x_edges = np.linspace(-detector_size / 2, detector_size / 2, detector_pixels + 1)
    y_edges = np.linspace(-detector_size / 2, detector_size / 2, detector_pixels + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        visibility_masked,
        shading="flat",
        cmap="magma",
        vmin=0,
        vmax=1,
    )

    if overlay_object_contour:
        if object_func is None and magnification is None:
            raise ValueError("Magnification and object_func must be provided to overlay object contour.")
        else:
            contour_set = overlay_object_outline(
                ax=ax,
                detector_size=detector_size,
                detector_pixels=detector_pixels,
                object_func=object_func,
                magnification=magnification,
                color=outline_color,
                linewidth=1.5,
                linestyle="--",
            )

            if contour_set is not None:
                ax.plot(
                    [],
                    [],
                    color=outline_color,
                    linestyle="--",
                    linewidth=1.5,
                    label="Projected object edge",
                )

                ax.legend(
                    loc="upper right",
                    fontsize=9,
                )

    ax.set_title("Detector visibility")
    ax.set_xlabel(r"$x_{\mathrm{det}}$ (m)")
    ax.set_ylabel(r"$y_{\mathrm{det}}$ (m)")
    ax.set_aspect("equal")

    if draw_x0:
        ax.axvline(
            0,
            color="lime",
            linewidth=1,
            alpha=0.8,
        )

    fig.colorbar(
        mesh,
        ax=ax,
        label="Visibility",
    )

    plt.tight_layout()
    plt.show()

def draw_object(ax, plane_size, plane_pixels, object_func, quantity="transmission"):
    x = np.linspace(- plane_size / 2, plane_size / 2, plane_pixels + 1)
    y = np.linspace(- plane_size / 2, plane_size / 2, plane_pixels + 1)
    extent_mm = [
        x[0], x[-1],
        y[0], y[-1],
    ]

    x_mesh, y_mesh = np.meshgrid(
        x, y,
        indexing="xy",
    )

    t_o, phi_o = object_func(x_mesh, y_mesh)

    if quantity == "phase":
        values = phi_o
        vmin = -np.pi
        vmax = np.pi

    elif quantity == "transmission":
        values = t_o
        vmin = 0.0
        vmax = 1.0

    background = ax.imshow(
        values,
        origin="lower",
        extent=extent_mm,
        aspect="equal",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        alpha=0.3,
        interpolation="nearest",
        zorder=0,
    )

    return background

def plot_obj_plane(x_obj, y_obj, phi_obj,
                   plane_size, plane_pixels, object_func, quantity):
    fig, ax = plt.subplots(figsize=(7, 6))

    background = draw_object(ax, plane_size, plane_pixels, object_func, quantity)

    sampled = ax.scatter(
        x_obj.ravel(),
        y_obj.ravel(),
        c=phi_obj.ravel(),
        s=1,
        cmap="magma",
        vmin=0,
        vmax=np.pi,
    )

    ax.set_xlabel("$x_o$ (mm)")
    ax.set_ylabel("$y_o$ (mm)")
    ax.set_title("Idler samples in object plane")
    ax.set_aspect("equal")

    half_size_mm = plane_size / 2
    ax.set_xlim(-half_size_mm, half_size_mm)
    ax.set_ylim(-half_size_mm, half_size_mm)


    fig.colorbar(sampled, ax=ax, label="Object phase (rad)")

    plt.show()

def plot_absorbing_object(
    object_func,
    object_plane_size,
    object_plane_pixels,
    draw_markers=True,
    bar_width=None,
    gap=None,
    center_y=0.0,
    transmission_threshold=0.5,
    title="Absorbing object",
    unit="mm",
    ax=None,
):
    """
    Plot a binary absorbing object as black object / white background.
    :param object_func: function with signature t_o, phi_o = object_func(x, y)
    :param object_plane_size: object-plane physical size in metres
    :param object_plane_pixels: object-plane pixel number
    :param draw_markers: draw purple markers for bar width w and gap g
                        (assume two horizontal absorbing bar target)
    :param bar_width: marks width of each of the bars
    :param gap: transparent edge-to-edge distance g between bars
    :param center_y: midpoint between the two bars in m
    :param transmission_threshold:
            t_o <= transmission_threshold  -> black absorbing object
            t_o > transmission_threshold   -> white transparent background
    :param title: plot title
    :param unit: display unit
    :param ax: existing axes - if None, creates a new figure.
    :return:
    """
    if np.isscalar(object_plane_size):
        size_x = float(object_plane_size)
        size_y = float(object_plane_size)
    else:
        size_x, size_y = map(float, object_plane_size)

    if np.isscalar(object_plane_pixels):
        n_x = int(object_plane_pixels)
        n_y = int(object_plane_pixels)
    else:
        n_x, n_y = map(int, object_plane_pixels)

    if size_x <= 0 or size_y <= 0:
        raise ValueError("object_plane_size must contain positive values.")
    if n_x < 2 or n_y < 2:
        raise ValueError("object_plane_pixels must be at least 2 per axis.")

    unit_data = {
        "m": (1.0, "m"),
        "mm": (1e3, "mm"),
        "um": (1e6, r"$\mu$m"),
    }
    if unit not in unit_data:
        raise ValueError(
            "unit must be 'm', 'mm', or 'um'."
        )
    scale, unit_label = unit_data[unit]

    x_edges = np.linspace(-size_x / 2, size_x / 2, n_x + 1)
    y_edges = np.linspace(-size_y / 2, size_y / 2, n_y + 1)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_grid, y_grid = np.meshgrid(
        x_centers,
        y_centers,
        indexing="xy",
    )

    t_o, phi_o = object_func(x_grid, y_grid)
    t_o = np.asarray(t_o, dtype=float)
    phi_o = np.asarray(phi_o, dtype=float)

    expected_shape = (n_y, n_x)
    if t_o.shape != expected_shape:
        raise ValueError(
            "object_func returned transmission shape "
            f"{t_o.shape}, expected {expected_shape}."
        )
    if phi_o.shape != expected_shape:
        raise ValueError(
            "object_func returned phase shape "
            f"{phi_o.shape}, expected {expected_shape}."
        )
    if not np.all(np.isfinite(t_o)):
        raise ValueError(
            "Object transmission contains NaN or infinite values."
        )

    # True means opaque / black object.
    object_mask = t_o <= transmission_threshold

    # Image values:
    # 0 -> black opaque object
    # 1 -> white transparent background
    display_image = np.ones_like(
        t_o,
        dtype=float,
    )

    display_image[object_mask] = 0.0

    created_figure = False

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
        created_figure = True
    else:
        fig = ax.figure

    mesh = ax.pcolormesh(
        x_edges * scale,
        y_edges * scale,
        display_image,
        shading="flat",
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    ax.set_title(title)
    ax.set_xlabel(rf"$x_\mathrm{{obj}}$ ({unit_label})")
    ax.set_ylabel(rf"$y_\mathrm{{obj}}$ ({unit_label})")
    ax.set_aspect("equal")

    # Optional purple width/gap dimension markers.
    if draw_markers:
        if bar_width is None or gap is None:
            raise ValueError("When draw_markers=True, provide both bar_width and gap.")
        if bar_width <= 0:
            raise ValueError("bar_width must be positive.")
        if gap < 0:
            raise ValueError("gap must be non-negative.")

        # Two-bar geometry:
        upper_bar_lower_edge = center_y + 0.5 * gap
        upper_bar_upper_edge = center_y + 0.5 * gap + bar_width
        lower_bar_upper_edge = center_y - 0.5 * gap

        # Put arrows to the right of the object-plane plot.
        x_marker = 0.08 * size_x
        x_marker_text = 0.1 * size_x

        # Width arrow: spans upper-bar lower/upper edges.
        ax.annotate(
            "",
            xy=(
                x_marker * scale,
                upper_bar_lower_edge * scale,
            ),
            xytext=(
                x_marker * scale,
                upper_bar_upper_edge * scale,
            ),
            arrowprops={
                "arrowstyle": "<->",
                "color": "purple",
                "linewidth": 2.0,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            annotation_clip=False,
        )

        ax.text(
            x_marker_text * scale,
            0.5 * (
                upper_bar_lower_edge
                + upper_bar_upper_edge
            ) * scale,
            rf"$w={bar_width * scale:.3g}$ {unit_label}",
            color="purple",
            fontsize=11,
            va="center",
            ha="left",
            bbox={
                "facecolor": "white",
                "edgecolor": "purple",
                "alpha": 1,
                "boxstyle": "round,pad=0.2",
            },
        )

        # Gap arrow: spans the inner edges of the two bars.
        ax.annotate(
            "",
            xy=(
                x_marker * scale,
                lower_bar_upper_edge * scale,
            ),
            xytext=(
                x_marker * scale,
                upper_bar_lower_edge * scale,
            ),
            arrowprops={
                "arrowstyle": "<->",
                "color": "purple",
                "linewidth": 2.0,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            annotation_clip=False,
        )

        ax.text(
            x_marker_text * scale,
            center_y * scale,
            rf"$g={gap * scale:.3g}$ {unit_label}",
            color="purple",
            fontsize=11,
            va="center",
            ha="left",
            bbox={
                "facecolor": "white",
                "edgecolor": "purple",
                "alpha": 0.85,
                "boxstyle": "round,pad=0.2",
            },
        )

    if created_figure:
        plt.tight_layout()
        plt.show()

    return {
        "fig": fig,
        "ax": ax,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "transmission": t_o,
        "phase": phi_o,
        "object_mask": object_mask,
        "display_image": display_image,
        "mesh": mesh,
    }

def fwhm_from_curve(x, y):
    """
    Find the full width at half maximum of a sampled 1D curve.
    :param x: x coordinates
    :param y: curve values
    :return:
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Filter
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Find curve peak and it's x and y coords
    peak_index = np.argmax(y)
    y_peak = y[peak_index]
    x_peak = x[peak_index]

    if not np.isfinite(y_peak) or y_peak <= 0:
        return np.nan, np.nan, np.nan, x_peak, y_peak

    half_max = 0.5 * y_peak

    # Locate the last point below half-max to the left of the peak.
    left_indices = np.where(
        y[:peak_index] < half_max
    )[0]

    # Locate the first point below half-max to the right of the peak.
    right_indices = np.where(
        y[peak_index + 1:] < half_max
    )[0] + peak_index + 1

    if left_indices.size == 0 or right_indices.size == 0:
        return np.nan, np.nan, np.nan, x_peak, y_peak

    i_left = left_indices[-1]
    i_right = right_indices[0]

    # Find interpolated half-maximum crossing positions:
    # - Left: y[i_left] < half <= y[i_left + 1]
    x_left = np.interp(
        half_max,
        [y[i_left], y[i_left + 1]],
        [x[i_left], x[i_left + 1]],
    )
    # - Right: y[i_right - 1] >= half > y[i_right]
    x_right = np.interp(
        half_max,
        [y[i_right], y[i_right - 1]],
        [x[i_right], x[i_right - 1]],
    )

    # Calculate FWHM
    fwhm = x_right - x_left

    return fwhm, x_left, x_right, x_peak, y_peak

def add_fwhm_annotation(
    ax,
    x_left,
    x_right,
    y_peak,
    unit_scale=1e3,
    unit_label="mm",
    color="darkblue",
):
    """
    Draw an FWHM bracket and label on a 1D density plot.
    :param ax:
    :param x_left: left half-maximum position in m
    :param x_right: right half-maximum position in m
    :param y_peak: curve y peak position in m
    :param unit_scale: position-axis conversion from m, e.g. 1e3 for mm
    :param unit_label: unit label used in the text annotation
    :param color: color of the bracket/annotation.
    :return:
    """
    if not (
        np.isfinite(x_left)
        and np.isfinite(x_right)
        and np.isfinite(y_peak)
    ):
        return

    fwhm = x_right - x_left

    # The density y-axis will be converted from 1/m to 1/mm
    # outside this function. Use a fraction of y_peak for the
    # annotation vertical position.
    bracket_y = 0.55 * y_peak
    tick_height = 0.02 * y_peak

    x_left_plot = x_left * unit_scale
    x_right_plot = x_right * unit_scale

    ax.hlines(
        bracket_y,
        x_left_plot,
        x_right_plot,
        color=color,
        linewidth=2,
    )

    ax.vlines(
        [x_left_plot, x_right_plot],
        bracket_y - tick_height,
        bracket_y + tick_height,
        color=color,
        linewidth=2,
    )

    ax.text(
        0.5 * (x_left_plot + x_right_plot),
        bracket_y - 0.08 * y_peak,
        rf"FWHM = {fwhm * unit_scale:.3g} {unit_label}",
        ha="center",
        va="bottom",
        color=color,
        fontsize=10,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.75,
            "pad": 1.5,
        },
    )

def plot_psf(psf,
             object_x_edges, object_y_edges,
             object_x_centers=None, object_y_centers=None,
             cut_half_width=2e-3, normalize=True,
             n_kde_points=2000
    ):
    # Convert PSF to array and check dimensions and values
    psf = np.asarray(psf, dtype=float).copy()
    if psf.ndim != 2:
        raise ValueError(f"PSF must be a 2D array; got shape {psf.shape}.")

    psf[~np.isfinite(psf)] = 0.0
    if np.any(psf < 0):
        raise ValueError(
            "PSF contains negative values."
            "A transition-probability kernel should be non-negative."
        )

    psf_sum_before = psf.sum()
    if psf_sum_before <= 0:
        raise ValueError(
            "PSF has zero total weight and cannot be plotted."
        )

    n_y, n_x = psf.shape

    # Get grid coordinates
    object_x_edges = np.asarray(object_x_edges, dtype=float)
    object_y_edges = np.asarray(object_y_edges, dtype=float)
    if object_x_edges.size != n_x + 1:
        raise ValueError(
            "object_x_edges must contain one more value than the "
            f"number of PSF x pixels: got {object_x_edges.size} "
            f"edges for {n_x} columns."
        )
    if object_y_edges.size != n_y + 1:
        raise ValueError(
            "object_y_edges must contain one more value than the "
            f"number of PSF y pixels: got {object_y_edges.size} "
            f"edges for {n_y} rows."
        )
    dx = np.diff(object_x_edges)
    dy = np.diff(object_y_edges)

    dx = dx[0]
    dy = dy[0]

    if object_x_centers is None:
        object_x_centers = 0.5 * (object_x_edges[:-1] + object_x_edges[1:])
    else:
        object_x_centers = np.asarray(object_x_centers, dtype=float)
    if object_x_centers.size != n_x:
        raise ValueError("object_x_centers length must match psf.shape[1].")

    if object_y_centers is None:
        object_y_centers = 0.5 * (object_y_edges[:-1] + object_y_edges[1:])
    else:
        object_y_centers = np.asarray(object_y_centers, dtype=float)
    if object_y_centers.size != n_y:
        raise ValueError("object_y_centers length must match psf.shape[0].")

    # Normalize PSF
    if normalize:
        psf /= psf.sum()

    # Find marginal distributions
    px_mass = psf.sum(axis=0)
    py_mass = psf.sum(axis=1)

    px_mass /= px_mass.sum()
    py_mass /= py_mass.sum()

    px_density = px_mass / dx
    py_density = py_mass / dy

    # Get KDEs
    kde_x = gaussian_kde(
        dataset=object_x_centers,
        weights=px_mass,
        bw_method="scott",
    )

    kde_y = gaussian_kde(
        dataset=object_y_centers,
        weights=py_mass,
        bw_method="scott",
    )

    x_dense = np.linspace(
        object_x_edges[0],
        object_x_edges[-1],
        n_kde_points,
    )

    y_dense = np.linspace(
        object_y_edges[0],
        object_y_edges[-1],
        n_kde_points,
    )

    kde_x_density = kde_x(x_dense)
    kde_y_density = kde_y(y_dense)

    # Find PSF maximum value and it's coordinates
    y_max, x_max = np.unravel_index(
        np.argmax(psf),
        psf.shape,
    )
    x_peak = object_x_centers[x_max]
    y_peak = object_y_centers[y_max]

    fig, ax = plt.subplots(
        2,
        2,
        figsize=(12, 10),
    )

    ax = ax.ravel()

    cmap = plt.colormaps["magma"].copy()
    cmap.set_bad("white")

    # PSF plot
    mesh0 = ax[0].pcolormesh(
            object_x_edges * 1e3,
            object_y_edges * 1e3,
            psf,
            shading="flat",
            cmap=cmap,
            vmin=0,
            vmax=np.max(psf)
    )

    # Zoomed in PSF
    mesh1 = ax[1].pcolormesh(
        object_x_edges * 1e3,
        object_y_edges * 1e3,
        psf,
        shading="flat",
        cmap=cmap,
        vmin=0,
        vmax=np.max(psf)
    )
    ax[1].set_xlim(-cut_half_width * 1e3, cut_half_width * 1e3)
    ax[1].set_ylim(-cut_half_width * 1e3, cut_half_width * 1e3)

    for i in range(2):
        # Mark detector center
        ax[i].axhline(
            0,
            color="gray",
            linewidth=0.8,
            alpha=0.8,
        )
        ax[i].axvline(
            0,
            color="gray",
            linewidth=0.8,
            alpha=0.8,
        )
        ax[i].set_xlabel(r"$x_\mathrm{obj}$ (mm)")
        ax[i].set_ylabel(r"$y_\mathrm{obj}$ (mm)")
        ax[i].set_aspect("equal")

    zoom_half_width_mm = cut_half_width * 1e3

    zoom_box = Rectangle(
        xy=(
            -zoom_half_width_mm,
            -zoom_half_width_mm,
        ),
        width=2 * zoom_half_width_mm,
        height=2 * zoom_half_width_mm,
        fill=False,
        edgecolor="red",
        linewidth=2.0,
        linestyle="--",
        zorder=10,
        label="ROI",
    )

    ax[0].add_patch(zoom_box)

    ax[0].set_title("Object-plane PSF")
    cbar0 = fig.colorbar(mesh0, ax=ax[0], label="Normalized PSF weight per pixel")
    cbar0.formatter = ticker.ScalarFormatter()
    cbar0.formatter.set_powerlimits((0, 0))

    ax[1].set_title("Zoomed in object-plane PSF")
    cbar1 = fig.colorbar(mesh1, ax=ax[1], label="Normalized PSF weight per pixel")
    cbar1.formatter = ticker.ScalarFormatter()
    cbar1.formatter.set_powerlimits((0, 0))

    # PSF marginal distribution: x
    ax[2].hist(
        object_x_centers * 1e3,
        bins=object_x_edges * 1e3,
        weights=px_mass,
        density=True,
        histtype="stepfilled",
        linewidth=1.6,
        color="lightsalmon",
        edgecolor="tomato",
        alpha=0.5,
        label="Binned PSF marginal",
    )

    ax[2].plot(
        x_dense * 1e3,
        kde_x_density * 1e-3,
        color="orange",
        linewidth=2.2,
        label=f"Weighted KDE",
    )
    ax[2].axvline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Plane center",
    )
    fwhm_x, x_left, x_right, x_peak_kde, x_peak_density = (
        fwhm_from_curve(
            x_dense,
            kde_x_density,
        )
    )
    add_fwhm_annotation(
        ax=ax[2],
        x_left=x_left,
        x_right=x_right,
        y_peak=x_peak_density * 1e-3,
        unit_scale=1e3,
        unit_label="mm",
        color="orangered"
    )
    ax[2].axvline(
        x_peak_kde * 1e3,
        color="orangered",
        linestyle=":",
        linewidth=1.5,
        label="KDE peak",
    )
    ax[2].set_xlim(-cut_half_width * 1e3, cut_half_width * 1e3)
    ax[2].set_title(r"Horizontal marginal PSF")
    ax[2].set_xlabel(r"$x_\mathrm{obj}$ (mm)")
    ax[2].set_ylabel(r"Probability density ($mm^{-1}$)")
    ax[2].legend()

    ax[3].hist(
        object_y_centers * 1e3,
        bins=object_y_edges * 1e3,
        weights=py_mass,
        density=True,
        histtype="stepfilled",
        linewidth=1.6,
        color="mediumorchid",
        edgecolor="darkorchid",
        alpha=0.5,
        label="Binned PSF marginal",
    )
    ax[3].plot(
        y_dense * 1e3,
        kde_y_density * 1e-3,
        color="darkviolet",
        linewidth=2.2,
        label=f"Weighted KDE",
    )
    ax[3].axvline(
        0,
        color="gray",
        linewidth=1,
        label="Plane center",

    )
    fwhm_y, y_left, y_right, y_peak_kde, y_peak_density = (
        fwhm_from_curve(
            y_dense,
            kde_y_density,
        )
    )
    add_fwhm_annotation(
        ax=ax[3],
        x_left=y_left,
        x_right=y_right,
        y_peak=y_peak_density * 1e-3,
        unit_scale=1e3,
        unit_label="mm",
        color="purple",
    )
    ax[3].axvline(
        y_peak_kde * 1e3,
        color="purple",
        linestyle=":",
        linewidth=1.5,
        label="KDE peak",
    )
    ax[3].set_xlim(-cut_half_width * 1e3, cut_half_width * 1e3)
    ax[3].set_title(r"Vertical marginal PSF")
    ax[3].set_xlabel(r"$y_\mathrm{obj}$ (mm)")
    ax[3].set_ylabel(r"Probability density ($mm^{-1}$)")
    ax[3].legend()

    for axis in ax:
        axis.ticklabel_format(
            axis="both",
            style="sci",
            scilimits=(-3, 3),
            useMathText=True,
        )

    plt.tight_layout()
    plt.show()

    return psf
