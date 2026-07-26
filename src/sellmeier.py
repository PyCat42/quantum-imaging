"""
Calculates refractive indices for different crystals using Sellmeier equations.
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
    :param temp: crystal temperature (default is 25 deg C - room temperature)
    :return: f
    """
    return (temp - 24.5) * (temp + 570.82)


def n_e_LiNbO3(wavelength, temp=25):
    """
    Calculates extraordinary refractive index using Sellmeier equation.
    :param wavelength: in microm
    :param temp: temperature of the crystal
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

    wavelength2 = wavelength**2

    return np.sqrt(a1_e + b1_e * f(temp)
                   + (a2_e + b2_e * f(temp))/(wavelength2 - (a3_e + b3_e * f(temp))**2)
                   + (a4_e + b4_e * f(temp))/(wavelength2 - a5_e**2)
                   - a6_e * wavelength2)


def n_o_LiNbO3(wavelength, temp=25):
    """
    Calculates ordinary refractive index using Sellmeier equation.
    :param wavelength: in microm
    :param temp: temperature of the crystal
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

    wavelength2 = (wavelength * 10**6)**2

    return np.sqrt(a1_o + b1_o * f(temp) +
                   (a2_o + b2_o * f(temp))/(wavelength2 - (a3_o + b3_o * f(temp))**2)
                   + (a4_o + b4_o * f(temp))/(wavelength2 - a5_o**2)
                   - a6_o * wavelength2)

# --------------------------------------------------------------------------------------
# KTP Sellmeier equation params
# Kiyoshi Kato and Eiko Takaoka
# "Sellmeier and thermo-optic dispersion formulas for KTP,"
# Appl. Opt. 41, 5040-504 (2002)

# KTP is biaxial so we have three refractive indices - along x, y and z axis
# --------------------------------------------------------------------------------------

def n_x_KTP(wavelength, temp=25):
    """
    Calculates refractive index along x direction.
    :param wavelength: in microm
    :param temp: crystal temperature
    (KTP indices from Kato and Takaoka are temperature independent, but we want to match function calls)
    :return:
    """
    A_x = 3.29100
    B_x = 0.04140
    C_x = 0.03978
    D_x = 9.35522
    E_x = 31.45571

    wavelength2 = wavelength**2

    return np.sqrt(A_x + B_x / (wavelength2 - C_x) + D_x / (wavelength2 - E_x))


def n_y_KTP(wavelength, temp=25):
    """
    Calculates refractive index along y direction.
    :param wavelength: in microm
    :param temp: crystal temperature
    (KTP indices from Kato and Takaoka are temperature independent, but we want to match function calls)
    :return:
    """
    A_y = 3.45018
    B_y = 0.04341
    C_y = 0.04597
    D_y = 16.98825
    E_y = 39.43799

    wavelength2 = wavelength**2

    return np.sqrt(A_y + B_y / (wavelength2 - C_y) + D_y / (wavelength2 - E_y))


def n_z_KTP(wavelength, temp=25):
    """
    Calculates refractive index along z direction.
    :param wavelength: in microm
    :param temp: crystal temperature
    (KTP indices from Kato and Takaoka are temperature independent, but we want to match function calls)
    :return:
    """
    A_z = 4.59423
    B_z = 0.06206
    C_z = 0.04763
    D_z = 110.80672
    E_z = 86.12171

    wavelength2 = wavelength**2

    return np.sqrt(A_z + B_z / (wavelength2 - C_z) + D_z / (wavelength2 - E_z))


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
