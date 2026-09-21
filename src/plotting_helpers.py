import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from matplotlib import ticker
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import numpy as np


def plot_spectrum(spdc, vals):
    """
    Plots SPDC spectrum with phase matching contour.
    :param spdc: instance of SPDC class
    :param vals: intensity grid
    :return:
    """
    spectrum = vals.reshape((
        spdc.theta_s_grid.size,
        spdc.lambda_s_grid.size,
    ))

    if spdc.lambda_s_max < 2 * spdc.lambda_p:
        plt.figure(figsize=(8, 6))
    else:
        plt.figure(figsize=(10, 6))

    plt.pcolormesh(
        spdc.lambda_s_grid * 1e9,  # convert to nm
        np.rad2deg(spdc.theta_s_grid),  # convert theta to degrees
        spectrum,
        shading='auto',
        cmap='magma',
        vmin=0,
        vmax=5e18
    )
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(r"Transition rate dR ($(m \cdot deg \cdot s)^{-1}$)", fontsize=16)

    # Plot phase matching contour
    lam, theta, dkz = spdc.get_phase_matching_contour()
    plt.contour(
        lam * 1e9, np.rad2deg(theta), dkz,
        levels=[0.0],
        colors='springgreen', linewidths=0.4, linestyles='-',
    )
    # Base legend handles
    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color='springgreen',
            linewidth=2,
            linestyle='-',
            label=r'$\Delta k_z = 0$',
        )
    ]

    # Add SPDC metadata entries to legend
    legend_handles.extend([
        mlines.Line2D([], [], color='none', label=rf'$t = {spdc.t}^{{\circ}} C$'),
        mlines.Line2D([], [], color='none', label=rf'$L = {spdc.L * 1e3}mm$'),
        mlines.Line2D(
            [], [], color='none', label=rf'$\omega_0 = {spdc.omega_0 * 1e6} \mu m$'
        ),
    ])

    # Find intersections of contour with x-axis  (theta = 0)
    # - Take the row closest to theta = 0
    theta_idx = np.argmin(np.abs(theta))
    dkz_xaxis = dkz[theta_idx, :]
    lam_xaxis = lam[:,] if lam.ndim == 1 else lam[theta_idx, :]

    # - Find sign changes in dkz
    sign_changes = np.where(np.sign(dkz_xaxis[:-1]) != np.sign(dkz_xaxis[1:]))[0]
    intersections = []
    for i in sign_changes:
        # Linear interpolation between two neighboring points
        x1 = lam_xaxis[i]
        x2 = lam_xaxis[i + 1]

        y1 = dkz_xaxis[i]
        y2 = dkz_xaxis[i + 1]

        if y2 != y1:
            x_intersection = x1 - y1 * (x2 - x1) / (y2 - y1)

        intersections.append(x_intersection)

    # Convert to nm and sort by wavelength
    intersections_nm = np.sort(np.array(intersections) * 1e9)

    if len(intersections_nm) == 0:
        print("No colinear SPDC emission!")
    if len(intersections_nm) == 1:
        if intersections_nm[0] < 2 * spdc.lambda_p * 1e9:
            # Smaller wavelength = Signal = red star
            lambda_signal = intersections_nm[0]
            sc_signal = plt.scatter(lambda_signal, 0.0, color='red', marker='*', s=100, zorder=5, clip_on=False,
                                    label=rf'Signal ($\lambda_s={lambda_signal:.2f}$ nm)')
            legend_handles.extend([sc_signal])
        else:
            # Equal wavelengths = Degeneracy = green star
            lambda_deg = intersections_nm[0]
            sc_deg = plt.scatter(lambda_deg, 0.0, color='green', marker='*', s=100, zorder=5, clip_on=False,
                                    label=rf'Degeneracy ($\lambda_s=\lambda_i={lambda_deg:.2f}$ nm)')
            legend_handles.extend([sc_deg])

    elif len(intersections_nm) == 2:
        # Smaller wavelength = Signal = red star
        lambda_signal = intersections_nm[0]
        sc_signal = plt.scatter(lambda_signal, 0.0, color='red', marker='*', s=100, zorder=5, clip_on=False,
                                label=rf'Signal ($\lambda_s={lambda_signal:.2f}$ nm)')
        legend_handles.extend([sc_signal])

        # Larger wavelength = Idler = blue star
        lambda_idler = intersections_nm[1]
        sc_idler = plt.scatter(lambda_idler, 0.0, color='blue', marker='*', s=100, zorder=5, clip_on=False,
                    label=rf'Idler ($\lambda_i={lambda_idler:.2f}$ nm)')
        legend_handles.extend([sc_idler])

    plt.title("SPDC Spectrum", fontsize=20)
    plt.xlabel(r"$\lambda_s$ (nm)", fontsize=16)
    plt.ylabel(r"$\theta_{s, int}$ (deg)", fontsize=16)
    if spdc.lambda_s_max > 2 * spdc.lambda_p:
        plt.legend(handles=legend_handles, loc='upper center')
    else:
        plt.legend(handles=legend_handles, loc='upper right')

    plt.tick_params(axis='both', labelsize=14)
    plt.tight_layout()

    filename = f"spectrum_L{spdc.L}_t{spdc.t}_w{spdc.omega_0}.pdf"
    plt.savefig(filename, dpi=300)

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
        x_edges * 1e3,
        y_edges * 1e3,
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
    fig.colorbar(mesh0, ax=ax[0], label="Count rate (1/s)", orientation="vertical")

    mesh1 = ax[1].pcolormesh(
        x_edges * 1e3,
        y_edges * 1e3,
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
        x_edges * 1e3,
        y_edges * 1e3,
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

def overlay_object_outline(
        ax,
        plane_size, plane_pixels,
        object_func,
        magnification,
        color="red", linewidth=1.5, linestyle="--", alpha=0.95,
        contour_level=0.01, scale=1
):
    """
    Draw the geometrically magnified object boundary over a detector image.
    :param ax: existing axes containing a detector-plane image
    :param plane_size: physical detector width in metres
    :param plane_pixels: number of detector pixels along each dimension
    :param object_func: function with signature t_o, phi_o = object_func(x_object, y_object)
    :param magnification: signed detector/object magnification
    :param color, linewidth, linestyle, alpha: matplotlib contour style settings
    :param contour_level: level at which to draw object contour
    :param scale: physical scale of the plot (1 == m, 1e3 == mm, and so on)
    :return:
    """
    if not np.isfinite(magnification) or magnification == 0:
        raise ValueError(
            "magnification must be a finite nonzero number."
        )

    detector_edges = np.linspace(-plane_size / 2, plane_size / 2, plane_pixels + 1)
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
        x_det * scale,
        y_det * scale,
        object_map,
        levels=[contour_level],
        colors=[color],
        linewidths=linewidth,
        linestyles=linestyle,
        alpha=alpha,
    )

    return contour_set

def plot_interf(
        detector_size,
        detector_pixels,
        detector_interf,
        interf_type="Constructive"
):
    """
    Plots constructive and destructive interference plots
    :param detector_size: physical detector width in metres
    :param detector_pixels: number of detector pixels along each dimension
    :param detector_interf: measured visibility in detector plane
    :param interf_type: "Constructive" or "Destructive"
    :return:
    """
    detector_interf = np.asarray(detector_interf, dtype=float)
    if detector_interf.shape != (detector_pixels, detector_pixels):
        raise ValueError(
            "detector_interf shape does not match detector_pixels: "
            f"got {detector_interf.shape}, expected "
            f"({detector_pixels}, {detector_pixels})."
        )
    interf_masked = np.ma.masked_invalid(detector_interf)

    x_edges = np.linspace(-detector_size / 2, detector_size / 2, detector_pixels + 1)
    y_edges = np.linspace(-detector_size / 2, detector_size / 2, detector_pixels + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        interf_masked,
        shading="flat",
        cmap="magma",
        vmin=0
    )
    ax.set_title(f"{interf_type} interference")
    ax.set_xlabel(r"$x_{\mathrm{det}}$ (m)")
    ax.set_ylabel(r"$y_{\mathrm{det}}$ (m)")
    ax.set_aspect("equal")

    fig.colorbar(mesh, ax=ax, label="Count rate (1/s)")

    plt.tight_layout()
    plt.show()

def plot_visibility(
        plane_size,
        plane_pixels,
        plane_visibility,
        plane="Detector",
        overlay_object_contour=True,
        object_func=None,
        magnification=None,
        outline_color="cyan",
        draw_x0=False,
        filename="visibility.pdf"
):
    """
    Creates visibility colormap plot.
    :param plane_size: physical detector width in metres
    :param plane_pixels: number of detector pixels along each dimension
    :param plane_visibility: measured visibility in detector plane
    :param overlay_object_contour: True or False (default)
    :param object_func: function with signature t_o, phi_o = object_func(x_obj, y_obj)
    :param magnification: imaging system magnification
    :param outline_color: object outline color
    :param draw_x0: draw vertical cut through x=0 (for resolution!)
    :param filename: output filename
    :return:
    """
    plane_visibility = np.asarray(plane_visibility, dtype=float)
    if plane_visibility.shape != (plane_pixels, plane_pixels):
        raise ValueError(
            "plane_visibility shape does not match plane_pixels: "
            f"got {plane_visibility.shape}, expected "
            f"({plane_pixels}, {plane_pixels})."
        )
    visibility_masked = np.ma.masked_invalid(plane_visibility)

    x_edges = np.linspace(-plane_size / 2, plane_size / 2, plane_pixels + 1)
    y_edges = np.linspace(-plane_size / 2, plane_size / 2, plane_pixels + 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    mesh = ax.pcolormesh(
        x_edges*1e3,
        y_edges*1e3,
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
                plane_size=plane_size,
                plane_pixels=plane_pixels,
                object_func=object_func,
                magnification=magnification,
                color=outline_color,
                linewidth=3,
                linestyle="--",
                scale=1e3
            )

            if contour_set is not None:
                ax.plot(
                    [],
                    [],
                    color=outline_color,
                    linestyle="--",
                    linewidth=3,
                    label="Object edge (theory)"
                )
                ax.legend(loc="upper right", fontsize=24)

    ax.set_title(f"Visibility: {plane} plane", fontsize=24)
    ax.set_xlabel(r"$x_{\mathrm{det}}$ (mm)", fontsize=24)
    ax.set_ylabel(r"$y_{\mathrm{det}}$ (mm)", fontsize=24)
    ax.set_aspect("equal")
    ax.tick_params(axis="both", labelsize=24)

    if draw_x0:
        ax.axvline(
            0,
            color="limegreen",
            linewidth=3,
            alpha=0.8,
        )

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label(label="Visibility", fontsize=26)
    cbar.ax.yaxis.get_offset_text().set_fontsize(24)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def draw_object(ax, plane_size, plane_pixels, object_func, quantity="transmission"):
    x = np.linspace(- plane_size / 2, plane_size / 2, plane_pixels + 1)
    y = np.linspace(- plane_size / 2, plane_size / 2, plane_pixels + 1)
    extent_mm = [
        x[0], x[-1],
        y[0], y[-1],
    ]
    x_mesh, y_mesh = np.meshgrid(x, y, indexing="xy")
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


def plot_obj_plane(
        x_obj, y_obj, phi_obj,
        plane_size, plane_pixels, object_func, quantity
):
    """
    Plots object and idler samples in object plane.
    :param x_obj: x coordinate in object plane
    :param y_obj: y coordinate in object plane
    :param phi_obj: phase value for (x,y) coordinate in object plane
    :param plane_size: size of object plane
    :param plane_pixels: number of object plane pixels
    :param object_func: function defiing object transmission and phase
    :param quantity: quantity that we plot ("phase" or "transmission")
    :return:
    """
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
        raise ValueError("unit must be 'm', 'mm', or 'um'.")
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
            f"object_func returned transmission shape {t_o.shape}, expected {expected_shape}.")
    if phi_o.shape != expected_shape:
        raise ValueError(
            f"object_func returned phase shape {phi_o.shape}, expected {expected_shape}.")
    if not np.all(np.isfinite(t_o)):
        raise ValueError("Object transmission contains NaN or infinite values.")

    # True means opaque / black object.
    object_mask = t_o <= transmission_threshold

    # Image values:
    # 0 -> black opaque object
    # 1 -> white transparent background
    display_image = np.ones_like(t_o, dtype=float)
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

    ax.set_title(title, fontsize=26)
    ax.set_xlabel(rf"$x_\mathrm{{obj}}$ ({unit_label})", fontsize=24)
    ax.set_ylabel(rf"$y_\mathrm{{obj}}$ ({unit_label})", fontsize=24)
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
            0.5 * (upper_bar_lower_edge + upper_bar_upper_edge) * scale,
            rf"$w={bar_width * scale:.3g}$ {unit_label}",
            color="purple",
            fontsize=20,
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
            fontsize=20,
            va="center",
            ha="left",
            bbox={
                "facecolor": "white",
                "edgecolor": "purple",
                "alpha": 0.85,
                "boxstyle": "round,pad=0.2",
            },
        )
        ax.tick_params(axis="both", labelsize=20)

    if created_figure:
        plt.tight_layout()
        plt.savefig("res_target.pdf", dpi=300)
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
    left_indices = np.where(y[:peak_index] < half_max)[0]

    # Locate the first point below half-max to the right of the peak.
    right_indices = np.where(y[peak_index + 1:] < half_max)[0] + peak_index + 1

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
        linewidth=4,
    )

    ax.vlines(
        [x_left_plot, x_right_plot],
        bracket_y - tick_height,
        bracket_y + tick_height,
        color=color,
        linewidth=3,
    )

    fwhm_label = (
        f"FWHM = \n"
        f"{fwhm * unit_scale:.3g} "
        f"{unit_label}"
    )

    handle = Line2D(
        [],
        [],
        color=color,
        linewidth=3,
        marker="_",
        markersize=12,
        label=fwhm_label,
    )

    return handle

def crop_psf_center(
    psf,
    x_edges,
    y_edges,
    new_size,
    renormalize=False,
):
    """
    Crop a 2D PSF to a centered physical square.
    :param psf: PSF array
    :param x_edges: x pixel-edge coordinate in metres used for PSF creation
    :param y_edges: y pixel-edge coordinate in metres used for PSF creation
    :param new_size: desired square crop width in metres
    :param renormalize: if True, normalize the cropped PSF so its sum is one (default is False
    :return: psf_crop : cropped PSF
            x_edges_crop, y_edges_crop : matching edge coordinates
    """

    psf = np.asarray(psf, dtype=float)
    x_edges = np.asarray(x_edges, dtype=float)
    y_edges = np.asarray(y_edges, dtype=float)

    # Check PSF dimensions
    if psf.ndim != 2:
        raise ValueError(
            "psf must be a 2D array."
        )

    # Check PSF shape
    n_y, n_x = psf.shape
    if x_edges.size != n_x + 1:
        raise ValueError("x_edges length must equal psf.shape[1] + 1.")
    if y_edges.size != n_y + 1:
        raise ValueError("y_edges length must equal psf.shape[0] + 1.")

    if new_size <= 0:
        raise ValueError("new_size must be positive.")

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    # Determine which part of PSF to keep
    half_size = new_size / 2
    x_keep = (np.abs(x_centers) <= half_size)
    y_keep = (np.abs(y_centers) <= half_size)
    if not np.any(x_keep):
        raise ValueError(
            "Requested crop contains no x pixels. "
            "Increase new_size."
        )
    if not np.any(y_keep):
        raise ValueError(
            "Requested crop contains no y pixels. "
            "Increase new_size."
        )

    x_indices = np.flatnonzero(x_keep)
    x_start = x_indices[0]
    x_stop = x_indices[-1] + 1

    y_indices = np.flatnonzero(y_keep)
    y_start = y_indices[0]
    y_stop = y_indices[-1] + 1

    # Get cropped PSF and its edges
    psf_crop = psf[y_start:y_stop, x_start:x_stop].copy()
    x_edges_crop = x_edges[x_start:x_stop + 1]
    y_edges_crop = y_edges[y_start:y_stop + 1]

    if renormalize:
        total = np.nansum(psf_crop)

        if total <= 0:
            raise ValueError("Cropped PSF has zero total weight.")

        psf_crop /= total

    return psf_crop, x_edges_crop, y_edges_crop

def analyze_psf(
    psf,
    object_x_edges,
    object_y_edges,
    display_size=1e-3,
    normalize=True,
    normalize_display=True,
    n_kde_points=2000,
    kde_bandwidth="scott"
):
    """
    Plot a 2D object-plane PSF and its horizontal/vertical marginal distributions.
    :param psf: non-negative PSF/kernel
    :param object_x_edges: x pixel-edge coordinates in metres
    :param object_y_edges: y pixel-edge coordinates in metres
    :param display_size: width of square physical region shown and analyzed, in metres
    (this does NOT alter the returned full PSF or the kernel used by the convolution method)
    :param zoomed_size: half-width of zoom window and marginal plots in metres
    :param normalize: normalize the full PSF before cropping/plotting
    :param normalize_display: if True, renormalize the cropped display PSF so its displayed mass sums to one
    :param n_kde_points: number of points for one-dimensional KDE curves
    :param kde_bandwidth: passed to scipy.stats.gaussian_kde as bw_method
    :return:
    """
    psf_full = np.asarray(psf, dtype=float).copy()
    if psf_full.ndim != 2:
        raise ValueError(f"PSF must be 2D; got shape {psf_full.shape}.")

    psf_full[~np.isfinite(psf_full)] = 0.0

    if np.any(psf_full < 0):
        raise ValueError(
            "PSF contains negative values. "
            "A probability/intensity kernel must be non-negative."
        )

    psf_sum_before = np.sum(psf_full)

    if not np.isfinite(psf_sum_before) or psf_sum_before <= 0:
        raise ValueError("PSF has zero or non-finite total weight.")

    n_y, n_x = psf_full.shape

    object_x_edges = np.asarray(object_x_edges, dtype=float)
    object_y_edges = np.asarray(object_y_edges, dtype=float)
    if object_x_edges.ndim != 1:
        raise ValueError("object_x_edges must be one-dimensional.")
    if object_y_edges.ndim != 1:
        raise ValueError("object_y_edges must be one-dimensional.")
    if object_x_edges.size != n_x + 1:
        raise ValueError(
            "object_x_edges must contain n_x + 1 entries: "
            f"got {object_x_edges.size}, expected {n_x + 1}."
        )
    if object_y_edges.size != n_y + 1:
        raise ValueError(
            "object_y_edges must contain n_y + 1 entries: "
            f"got {object_y_edges.size}, expected {n_y + 1}."
        )

    object_x_centers = 0.5 * (object_x_edges[:-1] + object_x_edges[1:])
    object_y_centers = 0.5 * (object_y_edges[:-1] + object_y_edges[1:])

    dx_values = np.diff(object_x_edges)
    dy_values = np.diff(object_y_edges)

    dx = dx_values[0]
    dy = dy_values[0]

    if not np.allclose(dx_values, dx):
        raise ValueError(
            "This plotting function currently assumes uniformly "
            "spaced x pixels."
        )

    if not np.allclose(dy_values, dy):
        raise ValueError(
            "This plotting function currently assumes uniformly "
            "spaced y pixels."
        )

    if normalize:
        psf_full /= np.sum(psf_full)

    full_mass = np.sum(psf_full)

    # Crop only for display/marginal/FWHM analysis.
    if display_size is None:
        psf_crop = psf_full.copy()

        x_edges_crop = object_x_edges.copy()
        y_edges_crop = object_y_edges.copy()

        x_centers_crop = object_x_centers.copy()
        y_centers_crop = object_y_centers.copy()

    else:
        if display_size <= 0:
            raise ValueError("display_size must be positive.")

        psf_crop, x_edges_crop, y_edges_crop = crop_psf_center(
            psf=psf_full,
            x_edges=object_x_edges,
            y_edges=object_y_edges,
            new_size=display_size,
            renormalize=False,
        )
        x_centers_crop = 0.5 * (x_edges_crop[:-1] + x_edges_crop[1:])
        y_centers_crop = 0.5 * (y_edges_crop[:-1] + y_edges_crop[1:])

    crop_mass_before_display_normalization = np.sum(psf_crop)

    if (not np.isfinite(crop_mass_before_display_normalization)
        or crop_mass_before_display_normalization <= 0):
        raise ValueError("Selected display crop has zero or non-finite PSF mass.")

    if normalize_display:
        psf_plot = (psf_crop / crop_mass_before_display_normalization)
    else:
        psf_plot = psf_crop.copy()

    # Crop diagnostics.
    peak_y_crop, peak_x_crop = np.unravel_index(
        np.argmax(psf_plot),
        psf_plot.shape,
    )

    x_peak_crop = x_centers_crop[peak_x_crop]
    y_peak_crop = y_centers_crop[peak_y_crop]

    # Marginals of plotted PSF.
    px_mass = np.sum(psf_plot, axis=0)
    py_mass = np.sum(psf_plot, axis=1)

    px_mass_sum = np.sum(px_mass)
    py_mass_sum = np.sum(py_mass)

    if px_mass_sum <= 0 or py_mass_sum <= 0:
        raise ValueError("PSF marginal has zero mass.")
    px_mass /= px_mass_sum
    py_mass /= py_mass_sum

    x_dense = np.linspace(
        x_edges_crop[0],
        x_edges_crop[-1],
        n_kde_points,
    )

    y_dense = np.linspace(
        y_edges_crop[0],
        y_edges_crop[-1],
        n_kde_points,
    )

    kde_x = gaussian_kde(
        dataset=x_centers_crop,
        weights=px_mass,
        bw_method=kde_bandwidth,
    )
    kde_x_density = kde_x(x_dense)

    kde_y = gaussian_kde(
        dataset=y_centers_crop,
        weights=py_mass,
        bw_method=kde_bandwidth,
    )
    kde_y_density = kde_y(y_dense)

    fig = plt.figure(
        figsize=(22, 20),
        layout="constrained",
    )

    gs = fig.add_gridspec(
        nrows=2,
        ncols=4,
        height_ratios=[1.15, 1.0],
        width_ratios=[1.0, 1.0, 1.0, 1.0],
    )

    # Upper row: PSF takes up 2 central columns.
    ax_psf = fig.add_subplot(gs[0, 1:3])
    # Bottom row: Two marginal distributions.
    ax_x = fig.add_subplot(gs[1, 0:2])
    ax_y = fig.add_subplot(gs[1, 2:4], sharey=ax_x)

    ax_y.tick_params(
        axis="y",
        which="both",
        left=False,
        labelleft=False,
    )

    cmap = plt.colormaps["magma"].copy()
    cmap.set_bad("white")

    vmax = np.max(psf_plot)

    # Full selected display region.
    mesh0 = ax_psf.pcolormesh(
        x_edges_crop * 1e3,
        y_edges_crop * 1e3,
        psf_plot,
        shading="flat",
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
    )

    ax_psf.axhline(
        0.0,
        color="gray",
        linewidth=2,
        alpha=0.7
    )

    ax_psf.axvline(
        0.0,
        color="gray",
        linewidth=2,
        alpha=0.7
    )

    ax_psf.scatter(
        x_peak_crop * 1e3,
        y_peak_crop * 1e3,
        marker="x",
        s=100,
        linewidths=2,
        color="lime",
        label="PSF max",
        zorder=20,
    )

    ax_psf.set_xlabel(r"$x_\mathrm{obj}$ (mm)", fontsize=22)
    ax_psf.set_ylabel(r"$y_\mathrm{obj}$ (mm)", fontsize=22)
    ax_psf.set_aspect("equal")
    ax_psf.tick_params(axis="both", labelsize=18)
    ax_psf.legend(loc="upper right", fontsize=16)

    ax_psf.set_title("PSF", fontsize=24)
    cbar0 = fig.colorbar(mesh0, ax=ax_psf)
    cbar0.ax.tick_params(labelsize=18)
    cbar0.set_label(label="Normalized weight", fontsize=22)
    cbar0.ax.yaxis.get_offset_text().set_fontsize(18)
    cbar0.formatter = ticker.ScalarFormatter()
    cbar0.formatter.set_powerlimits((0, 0))

    # Horizontal marginal.
    ax_x.hist(
        x_centers_crop * 1e3,
        bins=x_edges_crop * 1e3,
        weights=px_mass,
        density=True,
        histtype="stepfilled",
        linewidth=1.4,
        color="lightsalmon",
        edgecolor="tomato",
        alpha=0.55,
        label="Histogram",
    )

    ax_x.plot(
        x_dense * 1e3,
        kde_x_density * 1e-3,
        color="orangered",
        linewidth=2.0,
        label="KDE",
    )
    fwhm_x, x_left, x_right, x_peak_kde, x_peak_density = fwhm_from_curve(x_dense, kde_x_density)

    fwhm_handle_x = add_fwhm_annotation(
        ax=ax_x,
        x_left=x_left,
        x_right=x_right,
        y_peak=x_peak_density * 1e-3,
        unit_scale=1e3,
        unit_label="mm",
        color="red",
    )

    ax_x.axvline(
        x_peak_kde * 1e3,
        color="orangered",
        linestyle=":",
        linewidth=3,
        label="KDE max",
    )

    ax_x.axvline(
        0.0,
        color="gray",
        linestyle="-",
        linewidth=2,
        label="Centar ravni",
    )

    ax_x.set_xlim(-display_size * 1e3 / 2, display_size * 1e3 / 2)
    ax_x.set_title("Horizontal marginal PSF", fontsize=24)
    ax_x.set_xlabel(r"$x_\mathrm{obj}$ (mm)", fontsize=22)
    ax_x.set_ylabel(r"Probability density (mm$^{-1}$)", fontsize=22)
    ax_x.tick_params(axis="both", labelsize=18)
    handles_x, labels_x = ax_x.get_legend_handles_labels()
    handles_x.append(fwhm_handle_x)
    ax_x.legend(handles=handles_x, fontsize=16)

    # Vertical marginal.
    ax_y.hist(
        y_centers_crop * 1e3,
        bins=y_edges_crop * 1e3,
        weights=py_mass,
        density=True,
        histtype="stepfilled",
        linewidth=2,
        color="mediumorchid",
        edgecolor="darkorchid",
        alpha=0.55,
        label="Histogram",
    )

    ax_y.plot(
        y_dense * 1e3,
        kde_y_density * 1e-3,
        color="darkviolet",
        linewidth=2.0,
        label="KDE",
    )

    fwhm_y, y_left, y_right, y_peak_kde, y_peak_density = fwhm_from_curve(y_dense, kde_y_density)
    fwhm_handle_y = add_fwhm_annotation(
        ax=ax_y,
        x_left=y_left,
        x_right=y_right,
        y_peak=y_peak_density * 1e-3,
        unit_scale=1e3,
        unit_label="mm",
        color="purple",
    )

    ax_y.axvline(
        y_peak_kde * 1e3,
        color="darkviolet",
        linestyle=":",
        linewidth=3,
        label="KDE max",
    )

    ax_y.axvline(
        0.0,
        color="gray",
        linestyle="-",
        linewidth=2,
        label="Plane center",
    )
    ax_y.set_xlim(-display_size * 1e3 / 2, display_size * 1e3 / 2)
    ax_y.set_title("Vertical marginal PSF", fontsize=24)
    ax_y.set_xlabel(r"$y_\mathrm{obj}$ (mm)", fontsize=22)
    ax_y.tick_params(axis="x", labelsize=18)
    handles_y, labels_y = ax_y.get_legend_handles_labels()
    handles_y.append(fwhm_handle_y)
    ax_y.legend(handles=handles_y, fontsize=16)

    for axis in [ax_psf, ax_x]:
        axis.ticklabel_format(
            axis="both",
            style="sci",
            scilimits=(-3, 3),
            useMathText=True,
        )
    ax_y.ticklabel_format(
        axis="x",
        style="sci",
        scilimits=(-3, 3),
        useMathText=True,
    )

    plt.savefig("psf_analysis.pdf", dpi=300)
    plt.show()

    return max(fwhm_x, fwhm_y)

def plot_vertical_visibility_profile(
        detector_visibility,
        detector_size,
        detector_pixels=None,
        x_centers=None,
        y_centers=None,
        expected_bar_width=None,
        expected_separation=None,
        center_y=0.0,
        magnification=1.0,
        prominence=0.02,
        distance_pixels=None,
        x_det=0.0,
        unit="mm",
        ax=None,
        filename="visibility_cut.pdf"
):
    """
    Plot a vertical cut through detector visibility at x_det.
    :param detector_visibility: 2D detector visibility matrix (uses image convention visibility[y_index, x_index])
    :param detector_size: detector physical dimensions in metres
    :param detector_pixels: detector pixel count
    :param x_centers, y_centers: detector coordinate centers in metres
           (if supplied, they are used directly and detector_pixels can be omitted)
    :param expected_bar_width: width of one bar in object-plane in metres;
           used only for drawing expected projected bar positions.
    :param expected_separation: center-to-center bar separation in object-plane in metres
    :param center_y: oject-plane midpoint between the two bars in metres
    :param magnification: signed detector/object magnification
    :param prominence: minimum prominence used by scipy.signal.find_peaks
        (visibility normally lies in [-1, 1], so 0.02 is a reasonable
        starting value.)
    :param distance_pixels: minimum separation in samples between detected extrema
    :param x_det: x coordinate on which we do the cut (default 0.0)
    :param unit: unit used for horizontal axis display {"m", "mm", "um"}.
    :param ax: existing axes. If None, a new figure is created.
    :param filename: name of the file to which created figure is saved
    :return:
    """
    visibility = np.asarray(detector_visibility, dtype=float)

    if visibility.ndim != 2:
        raise ValueError("detector_visibility must be a 2D array with shape (Ny, Nx).")

    n_y, n_x = visibility.shape

    if x_centers is None or y_centers is None:
        if detector_pixels is None:
            raise ValueError("Provide either x_centers/y_centers or detector_pixels.")

        if np.isscalar(detector_size):
            detector_size_x = float(detector_size)
            detector_size_y = float(detector_size)
        else:
            detector_size_x, detector_size_y = map(float, detector_size)

        if np.isscalar(detector_pixels):
            detector_pixels_x = int(detector_pixels)
            detector_pixels_y = int(detector_pixels)
        else:
            detector_pixels_x, detector_pixels_y = map(int, detector_pixels)

        if (
            detector_pixels_x != n_x
            or detector_pixels_y != n_y
        ):
            raise ValueError(
                "detector_pixels does not match visibility shape: "
                f"visibility has shape {visibility.shape}, while "
                f"detector_pixels implies ({detector_pixels_y}, {detector_pixels_x})."
            )

        x_edges = np.linspace(-detector_size_x / 2, detector_size_x / 2, n_x + 1)
        y_edges = np.linspace(-detector_size_y / 2, detector_size_y / 2, n_y + 1)

        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    else:
        x_centers = np.asarray(x_centers, dtype=float)
        y_centers = np.asarray(y_centers, dtype=float)

        if x_centers.size != n_x:
            raise ValueError("x_centers length must equal visibility.shape[1].")
        if y_centers.size != n_y:
            raise ValueError("y_centers length must equal visibility.shape[0].")

    unit_map = {
        "m": (1.0, "m"),
        "mm": (1e3, "mm"),
        "um": (1e6, r"$\mu$m"),
    }
    if unit not in unit_map:
        raise ValueError("unit must be 'm', 'mm', or 'um'.")
    coordinate_scale, unit_label = unit_map[unit]

    # Find detector column closest to x_det = 0.
    x0_index = int(np.argmin(np.abs(x_centers - x_det)))
    x0_used = x_centers[x0_index]

    y_profile = visibility[:, x0_index]
    valid = np.isfinite(y_profile)

    y_valid = y_centers[valid]
    profile_valid = y_profile[valid]

    if distance_pixels is None:
        distance_pixels = max(1, int(0.02 * profile_valid.size))

    # Find local maxima in V(y).
    maxima_local, maxima_properties = find_peaks(
        profile_valid,
        prominence=prominence,
        distance=distance_pixels,
    )

    # Find local minima by finding maxima of -V(y).
    minima_local, minima_properties = find_peaks(
        -profile_valid,
        prominence=prominence,
        distance=distance_pixels,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
        created_figure = True
    else:
        fig = ax.figure
        created_figure = False

    ax.plot(
        y_valid * coordinate_scale,
        profile_valid,
        color="indigo",
        linewidth=4,
        label=rf"$V(0, y_\mathrm{{det}})$"
    )

    ax.axvline(
        0,
        color="gray",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label="Detector center"
    )

    # Optional projected two-bar geometry.
    if expected_bar_width is not None and expected_separation is not None:
        if magnification == 0:
            raise ValueError("magnification must be nonzero.")

        # Object-plane bar centers.
        y_bar_lower_obj = center_y - 0.5 * expected_separation
        y_bar_upper_obj = center_y + 0.5 * expected_separation

        # Object-plane bar edges.
        lower_bar_bottom_obj = y_bar_lower_obj - 0.5 * expected_bar_width
        lower_bar_top_obj = y_bar_lower_obj + 0.5 * expected_bar_width

        upper_bar_bottom_obj = y_bar_upper_obj - 0.5 * expected_bar_width
        upper_bar_top_obj = y_bar_upper_obj + 0.5 * expected_bar_width

        # Map object-plane locations to detector plane.
        projected_edges = np.array([
            lower_bar_bottom_obj,
            lower_bar_top_obj,
            upper_bar_bottom_obj,
            upper_bar_top_obj,
        ]) * magnification

        projected_centers = np.array([
            y_bar_lower_obj,
            y_bar_upper_obj,
        ]) * magnification

        midpoint_detector = center_y * magnification

        theory_legend_handle = Line2D(
            [],
            [],
            color="none",
            linewidth=0,
            label="Theory:",
        )
        ax.add_line(theory_legend_handle)

        for y_edge in projected_edges:
            ax.axvline(
                y_edge * coordinate_scale,
                color="cyan",
                linestyle="--",
                linewidth=2,
                alpha=1,
                label="Object edges"
            )

        for y_bar_center in projected_centers:
            ax.axvline(
                y_bar_center * coordinate_scale,
                color="blue",
                linestyle=":",
                linewidth=2,
                alpha=0.75,
                label="Object centers"
            )

        ax.axvline(
            midpoint_detector * coordinate_scale,
            color="purple",
            linestyle="-.",
            linewidth=2,
            alpha=0.8,
            label="Midpoint",
        )

    ax.set_title(rf"Vertical visibility cut: $x_{{det}} \approx {x0_used}$", fontsize=26)
    ax.set_xlabel(rf"$y_\mathrm{{det}}$ ({unit_label})", fontsize=24)
    ax.set_ylabel("Visibility", fontsize=24)
    ax.set_ylim(0,1.2)

    handles, labels = ax.get_legend_handles_labels()

    unique_handles = []
    unique_labels = []

    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)

    ax.legend(
        unique_handles,
        unique_labels,
        loc="lower left",
        fontsize=24,
    )
    ax.tick_params(axis='both', labelsize=20)

    if created_figure:
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.show()
