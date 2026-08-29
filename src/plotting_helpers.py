import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines


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

    mesh0 = ax[0].pcolormesh(
        x_edges,
        y_edges,
        img1,
        shading="auto",
        cmap=cmap
    )
    ax[0].set_title("Constructive Interference")
    ax[0].set_xlabel(r"$x_{\mathrm{det}}$ (m)")
    ax[0].set_ylabel(r"$y_{\mathrm{det}}$ (m)")
    ax[0].set_aspect('equal')
    fig.colorbar(mesh0, ax=ax[0], label="Visibility", orientation="vertical")

    mesh1 = ax[1].pcolormesh(
        x_edges,
        y_edges,
        img2,
        shading="auto",
        cmap=cmap,
    )
    ax[1].set_title("Destructive Interference")
    ax[1].set_xlabel(r"$x_{\mathrm{det}}$ (m)")
    ax[1].set_aspect('equal')
    fig.colorbar(mesh1, ax=ax[1], label="Visibility", orientation="vertical")

    mesh2 = ax[2].pcolormesh(
        x_edges,
        y_edges,
        img3,
        shading="auto",
        cmap=cmap,
        vmin=-1,
        vmax=1,
    )
    ax[2].set_title("Detector Visibility")
    ax[2].set_xlabel(r"$x_{\mathrm{det}}$ (m)")
    ax[2].set_aspect('equal')
    fig.colorbar(mesh2, ax=ax[2], label="Visibility", orientation="vertical")

    plt.show()

def plot_visibility(detector_size, detector_pixels, detector_visibility):
    """
    Plots visibility map at the detector plane.
    :param detector_size: size of the detector in m
    :param detector_pixels: number of pixels in the detector
    :param detector_visibility: visibility map on the detector
    :return:
    """
    x_edges = np.linspace(-detector_size / 2, detector_size / 2, detector_pixels + 1)
    y_edges = np.linspace(-detector_size / 2, detector_size / 2, detector_pixels + 1)

    # Plot visibility
    plt.figure(figsize=(10, 6))

    # In case that in some detector region we don't have any signal samples (NaN) it will be white
    cmap = plt.colormaps["magma"].copy()
    cmap.set_bad("white")
    plt.pcolormesh(
        x_edges,
        y_edges,
        detector_visibility,
        shading="auto",
        cmap=cmap,
        vmin=-1,
        vmax=1,
    )
    plt.title("Detector visibility")
    plt.xlabel(r"$x_{\mathrm{det}}$ (m)")
    plt.ylabel(r"$y_{\mathrm{det}}$ (m)")
    plt.colorbar(label="Visibility")
    plt.show()
