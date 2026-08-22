"""
Calculates temperature dependent refractive indices for different crystals using Sellmeier equations.
"""
import numpy as np

# --------------------------------------------------------------------------------------
# LiNbO3 Sellmeier equation params
# Gayer, O., Sacks, Z., Galun, E. et al.
# Temperature and wavelength dependent refractive index equations for MgO-doped congruent and stoichiometric LiNbO3 .
# Appl. Phys. B 91, 343–348 (2008). https://doi.org/10.1007/s00340-008-2998-2

# LiNbO3 is biaxial crystal so we have two refractive indices - ordinary and extraordinary
# --------------------------------------------------------------------------------------

def f(temp=25):
    """
    Calculates temperature dependent parameter f.
    Needed because in Gayer et al. refractive indices are temperature dependent
    :param temp: crystal temperature in deg C (default is 25 deg C - room temperature)
    :return: f
    """
    return (temp - 24.5) * (temp + 570.82)


def n_e_LiNbO3(wavelength, temp=25):
    """
    Calculates extraordinary refractive index using Sellmeier equation.
    :param wavelength: in m
    :param temp: temperature of the crystal in deg C
    :return: n_e
    """
    # extraordinary params
    a1_e = 5.756
    a2_e = 0.0983
    a3_e = 0.2020
    a4_e = 189.32
    a5_e = 12.52
    a6_e = 1.32e-2

    b1_e = 2.860e-6
    b2_e = 4.700e-8
    b3_e = 6.113e-8
    b4_e = 1.516e-4

    wavelength2 = (wavelength * 1e6)**2

    return np.sqrt(a1_e + b1_e * f(temp)
                   + (a2_e + b2_e * f(temp))/(wavelength2 - (a3_e + b3_e * f(temp))**2)
                   + (a4_e + b4_e * f(temp))/(wavelength2 - a5_e**2)
                   - a6_e * wavelength2)


def n_o_LiNbO3(wavelength, temp=25):
    """
    Calculates ordinary refractive index using Sellmeier equation.
    :param wavelength: in m
    :param temp: temperature of the crystal in deg C
    :return: n_o
    """
    # ordinary params
    a1_o = 5.653
    a2_o = 0.1185
    a3_o = 0.2091
    a4_o = 89.61
    a5_o = 10.85
    a6_o = 1.97e-2

    b1_o = 7.941e-7
    b2_o = 3.134e-8
    b3_o = -4.641e-9
    b4_o = -2.188e-6

    wavelength2 = (wavelength * 1e6)**2

    return np.sqrt(a1_o + b1_o * f(temp) +
                   (a2_o + b2_o * f(temp))/(wavelength2 - (a3_o + b3_o * f(temp))**2)
                   + (a4_o + b4_o * f(temp))/(wavelength2 - a5_o**2)
                   - a6_o * wavelength2)

# --------------------------------------------------------------------------------------
# KTP Sellmeier equation params and temperature dependency
# Kiyoshi Kato and Eiko Takaoka
# "Sellmeier and thermo-optic dispersion formulas for KTP,"
# Appl. Opt. 41, 5040-504 (2002)

# KTP is biaxial so we have three refractive indices - along x, y and z axis
# --------------------------------------------------------------------------------------

def n_x_KTP(wavelength, temp=25):
    """
    Calculates refractive index along x direction.
    :param wavelength: in m
    :param temp: crystal temperature in deg C
    (KTP indices from Kato and Takaoka are temperature independent, but we want to match function calls)
    :return:
    """
    wavelength_microm = wavelength * 1e6 # equations use wavelength in micrometers

    A_x = 3.29100
    B_x = 0.04140
    C_x = 0.03978
    D_x = 9.35522
    E_x = 31.45571
    n_x = np.sqrt(
        A_x + B_x / (wavelength_microm**2 - C_x)
        + D_x / (wavelength_microm**2 - E_x)
    )

    # Temperature dependence
    A_T_x = 0.1717
    B_T_x = - 0.5353
    C_T_x = 0.8416
    D_T_x = 0.1627
    dnx = (
        A_T_x / wavelength_microm**3
        + B_T_x / wavelength_microm**2
        + C_T_x / wavelength_microm
        + D_T_x
    ) * 1e-5
    # temperature dependence is valid in range 0.43 <= wavelength_microm <= 1.3

    return n_x + temp * dnx

def n_y_KTP(wavelength, temp=25):
    """
    Calculates refractive index along y direction.
    :param wavelength: in m
    :param temp: crystal temperature in deg C
    (KTP indices from Kato and Takaoka are temperature independent, but we want to match function calls)
    :return:
    """
    wavelength_microm = wavelength * 1e6

    A_y = 3.45018
    B_y = 0.04341
    C_y = 0.04597
    D_y = 16.98825
    E_y = 39.43799
    n_y = np.sqrt(
        A_y + B_y / (wavelength_microm**2 - C_y)
        + D_y / (wavelength_microm**2 - E_y))

    # Temperature dependence
    A_T_y = 0.1997
    B_T_y = - 0.4063
    C_T_y = 0.5154
    D_T_y = 0.5425
    dny = (
        A_T_y / wavelength_microm ** 3
        + B_T_y / wavelength_microm ** 2
        + C_T_y / wavelength_microm
        + D_T_y
    ) * 1e-5
    # temperature dependence is valid in range 0.43 <= wavelength_microm <= 1.3

    return n_y + temp * dny

def n_z_KTP(wavelength, temp=25):
    """
    Calculates refractive index along z direction.
    :param wavelength: in m
    :param temp: crystal temperature in deg C
    (KTP indices from Kato and Takaoka are temperature independent, but we want to match function calls)
    :return:
    """
    wavelength_microm = wavelength * 1e6

    A_z = 4.59423
    B_z = 0.06206
    C_z = 0.04763
    D_z = 110.80672
    E_z = 86.12171
    n_z = np.sqrt(
        A_z + B_z / (wavelength_microm**2 - C_z)
        + D_z / (wavelength_microm**2 - E_z)
    )

    if wavelength_microm <= 1.5:
        A_T_z = 0.9221
        B_T_z = - 2.9220
        C_T_z = 3.6677
        D_T_z = - 0.1897
        dnz = (
            A_T_z / wavelength_microm ** 3
            + B_T_z / wavelength_microm ** 2
            + C_T_z / wavelength_microm
            + D_T_z
        ) * 1e-5
    else:
        A_T_z = - 0.5523
        B_T_z = 3.3920
        C_T_z = - 1.7101
        D_T_z = 0.3424
        dnz = (
            A_T_z / wavelength_microm
            + B_T_z
            + C_T_z * wavelength_microm
            + D_T_z * wavelength_microm**2
        ) * 1e-5
    # temperature dependence is valid in range 0.53 <= wavelength_microm <= 3.53

    return n_z + temp * dnz


# --------------------------------------------------------------------------------------

def n_eff(n_1, n_2, wavelength, theta, temp=25):
    """
    Calculates effective refractive index for LiNbO3 crystal.
    :param n_1: function used for the calculation of the ordinary refraction index equivalent
    :param n_2: function used for the calculation of the extraordinary refraction index equivalent
    :param wavelength: in m
    :param theta: angle between the optical axis and propagation direction
    :param temp: crystal temperature
    :return: n_eff
    """
    return 1 / np.sqrt((np.sin(theta)**2 / n_1(wavelength, temp)**2)
                       + np.cos(theta)**2 / n_2(wavelength, temp)**2)
