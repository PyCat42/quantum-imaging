import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from numpy.lib._stride_tricks_impl import sliding_window_view
from scipy.constants import c, pi, epsilon_0
from scipy.signal import fftconvolve
from scipy.stats import norm, qmc
from scipy.special import ndtri
from tqdm.auto import tqdm
import multiprocessing as mp

from src.object_funcs import object_phase_bar
from src.ray_propagation import *
from src.sellmeier import *
from src.calculation_helpers import *
from src.MC_helpers import sample_sinc2, sinc_phys, sinc2_approximation

class SPDC():
    """
    Class for simulation of SPDC source.
    """
    def __init__(self, L=1e-3, t=25, period=5.335e-6, m=-1, chi_eff=6e-12,
                 lambda_p=405e-9, n_p=n_x_KTP,
                 P=0.5, omega_0=60e-6, T_I=0,
                 lambda_s_min=625e-9, lambda_s_max=825e-9,
                 theta_s_min=0, theta_s_max=0.06,
                 phi_s_min=0, phi_s_max=2*pi,
                 n_s_func=n_eff, n_s_1=n_x_KTP, n_s_2=n_x_KTP,
                 n_i_1=n_x_KTP, n_i_2=n_x_KTP,
                 dn_i_1_dlambda=dn_x_KTP_dlambda, dn_i_2_dlambda=dn_x_KTP_dlambda,
                 min_N_i=int(2**5), max_N_i=int(2**10), N_phi=int(2**5),
                 grid_size=int(100), seed=None,
                 eps=1e-12, conv_check=100, min_rel_err=1e-2, min_abs_error=1e-21,
                 signal_batch_size=24, n_conv=16, n_processes=9):
        #TODO: Make parameter database

        # -------- CRYSTAL --------
        self.L = L  # crystal length in meters
        self.t = t  # crystal temperature in deg C
        self.period = period  # period of periodically poled crystal in m

        self.m = m  # quasi-phase matching order
        # - calculate QPM momentum if crystal is periodically poled
        if self.period == 0:
            self.k_m = 0
        else:
            self.k_m = self.m * 2 * pi / self.period

        self.chi_eff = chi_eff  # effective nonlinear susceptibility in m/V

        # -------- PUMP --------
        self.lambda_p = lambda_p  # pump wavelength in m
        self.n_p = n_p(self.lambda_p, temp=self.t)  # function for the calculation of the refractive index
        self.k_p = 2 * pi * self.n_p / self.lambda_p  # pump wavevector
        self.w_p = 2 * pi * c / self.lambda_p  # pump angular frequency
        self.P = P  # pump power in W
        self.omega_0 = omega_0  # pump waist in m
        self.gauss_scale = 1 / self.omega_0
        self.T_I = T_I # interaction time in s

        # -------- SIGNAL --------
        self.grid_size = grid_size # signals are sampled on a grid of this size
        self.N_s = self.grid_size ** 2 # number of signal photons to simulate
        self.lambda_s_min = lambda_s_min # minimal value of signal wavelength
        self.lambda_s_max = lambda_s_max # maximal value of signal wavelength
        self.theta_s_min = theta_s_min # minimal value of signal polar angle
        self.theta_s_max = theta_s_max # maximal value of signal polar angle
        self.N_phi = N_phi  # number of phi angles sampled for each signal sample
        self.phi_s_min = phi_s_min  # minimal value of signal azimuthal angle
        self.phi_s_max = phi_s_max  # maximal value of signal azimuthal angle
        self.n_s_func = n_s_func # function for calculating signal refractive index
        self.n_s_1 = n_s_1 # first refractive index that is contained in effective signal refractive index
        self.n_s_2 = n_s_2 # second refractive index that is contained in effective signal refractive index

        # -------- IDLER --------
        self.min_N_i = min_N_i  # minimal number of idler photons to simulate per signal photon
        self.max_N_i = max_N_i  # maximal number of idler photons to simulate per signal photon
        self.n_i_1 = n_i_1 # first refractive index that is contained in effective idler refractive index
        self.n_i_2 = n_i_2 # second refractive index that is contained in effective idler refractive index
        self.dn_i_1_dlambda = dn_i_1_dlambda  # derivative of the first refractive index that is contained in effective idler refractive index
        self.dn_i_2_dlambda = dn_i_2_dlambda  # derivative of the second refractive index that is contained in effective idler refractive index

        # -------- SIMULATION --------
        # Generator seed
        self.seed = seed
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Create sampling grid
        self.lambda_s_grid = np.linspace(self.lambda_s_min, self.lambda_s_max, self.grid_size)
        self.theta_s_grid = np.linspace(self.theta_s_min, self.theta_s_max, self.grid_size)
        self.lambda_mesh, self.theta_mesh = np.meshgrid(self.lambda_s_grid, self.theta_s_grid)
        self.lambda_s = self.lambda_mesh.flatten()
        self.theta_s = self.theta_mesh.flatten()

        # Calculate signal params on grid
        self.n_s = self.n_s_func(self.n_s_1, self.n_s_2, self.lambda_s, self.theta_s) # signal refractive index
        self.k_s = 2 * np.pi * self.n_s / self.lambda_s # signal wavevector
        self.w_s = 2 * np.pi * c / self.lambda_s # signal angular frequency

        # Optimization: Precalculate variables!
        # - integral prefactor
        self.const_rate_prefactor = (16 * self.omega_0 ** 2 * self.P * self.L ** 2 * self.chi_eff ** 2
                                     / (epsilon_0 * c * self.n_p * (2 * pi) ** 8))
        if self.m != 0:  # QPM exists
            self.const_rate_prefactor /= self.m ** 2
        if self.T_I == 0:  # CW pump case
            self.const_rate_prefactor *= (2 * np.pi)
        else:
            self.const_rate_prefactor *= self.T_I

        # - signal prefactor
        self.signal_prefactor = self.k_s ** 4 * self.w_s * np.sin(self.theta_s) / self.n_s ** 2

        # - phi
        self.phi_s = np.linspace(self.phi_s_min, self.phi_s_max, self.N_phi, endpoint=False)
        self.sin_phi = np.sin(self.phi_s)
        self.cos_phi = np.cos(self.phi_s)

        # Early stopping
        self.eps = eps  # numerical tolerance
        self.conv_check = conv_check  # check for convergence at each conv_check iterations
        self.n_conv = n_conv
        self.min_rel_err = min_rel_err  # minimal change in relative dR error that is considered as such
        self.min_abs_error = min_abs_error  # minimal absolute change in dR that is considered as such

        # Paralelization
        self.signal_batch_size = signal_batch_size # number of signal samples that are run on the same process
        self.signal_batches = [
            (start, min(start + self.signal_batch_size, self.N_s))
            for start in range(0, self.N_s, self.signal_batch_size)
        ]
        self.n_processes = n_processes

    def get_transition_rate(self):
        # Loop over signals and perform integration over phi and then also idlers
        mc_val = []
        for (start, end) in tqdm(self.signal_batches):
            k_s_batch = self.k_s[start:end]
            w_s_batch = self.w_s[start:end]
            theta_batch = self.theta_s[start:end]
            sin_theta = np.sin(theta_batch)[:, None]
            signal_prefactor_batch = self.signal_prefactor[start:end]

            # Spherical to cartesian - faster than calling function lots of times
            k_sx = k_s_batch[:, None] * sin_theta * self.cos_phi[None, :]  # (signal_batch_size, N_phi)
            k_sy = k_s_batch[:, None] * sin_theta * self.sin_phi[None, :]  # (signal_batch_size, N_phi)
            k_sz = k_s_batch * np.cos(theta_batch)  # (signal_batch_size, ) - we don't store repeated values of k_sz!

            # Sample maximal numer of idlers
            if self.seed is not None:
                sobol_seed = int(self.rng.integers(0, 2 ** 32 - 1))
            else:
                sobol_seed = None
            idler_sampler = qmc.Sobol(
                d=4,
                scramble=True,
                seed=sobol_seed
            )
            idler_points = idler_sampler.random_base2(
                m=int(np.log2(self.max_N_i))
            )

            # Calculate transversal momentum mismatch
            # These two are equivalent:
            # x = norm.ppf(val, loc=mu, scale=sigma)
            # x = mu + sigma * ndtri(val)
            # ndtri is the inverse standard normal CDF and is much faster to calculate
            delta_k_x = self.gauss_scale * ndtri(idler_points[:, 0])[None, None, :]  # (1, 1, max_N_i)
            delta_k_y = self.gauss_scale * ndtri(idler_points[:, 1])[None, None, :]  # (1, 1, max_N_i)

            # Calculate transversal idler momentum components
            k_ix = - delta_k_x - k_sx[:, :, None]
            k_iy = - delta_k_y - k_sy[:, :, None]
            k_i_normal = np.sqrt(k_ix ** 2 + k_iy ** 2)

            # Retain adaptive stopping without processing one batch at a time (i.e. eliminate idler loop)
            # Generate all samples and calculate the weighted values for all 512 samples in one vectorized operation.
            # Idler refraction index function
            n_i_func = lambda wavelength, theta_i: n_eff(self.n_i_1, self.n_i_2, wavelength, theta_i, self.t)
            d_n_i_dlambda_func = lambda wavelength, theta_i: (
                dn_eff_dlambda(self.n_i_1, self.dn_i_1_dlambda,
                               self.n_i_2, self.dn_i_2_dlambda,
                               wavelength, theta_i, self.t)
            )

            if self.T_I == 0:
                # CW pump => force energy conservation
                w_i = (self.w_p - w_s_batch)[:, None, None]
                delta_w = 0

                # Calculate idler wavelength
                lambda_i = 2 * pi * c / w_i

                # Solve for idler polar angle
                theta_i = get_theta_i_vectorized(k_i_normal, lambda_i, n_i_func)
                # ...and get idler refractive index and momentum
                n_i = n_i_func(lambda_i, theta_i)
                k_i = 2 * np.pi * n_i / lambda_i

                # Calculate longitudinal momentum from magnitude and transverse momenta
                k_iz2 = k_i ** 2 - k_ix ** 2 - k_iy ** 2
                # ...and check if this squared value is valid
                valid = (
                        np.isfinite(k_iz2)
                        & (k_iz2 >= 0)
                        & np.isfinite(theta_i)
                        & np.isfinite(n_i)
                )
                k_iz = np.full_like(k_iz2, np.nan)
                k_iz[valid] = np.sqrt(k_iz2[valid])
                # ...then get longitudinal momentum mismatch
                delta_k_z = np.full_like(k_iz2, np.nan)
                delta_k_z = (
                        self.k_p
                        - k_sz[:, None, None]
                        - k_iz
                        + self.k_m
                )

                delta_k_z[~valid] = np.nan

                # calculate integrand function value
                # - calculate Jacobian = 1 / (d(omega) / d(k_iz)) analytically
                # k = 2 * pi * n(lambda, theta) / lambda
                # fixed theta: dk/dlambda = 2 * pi * ((1 / lambda) * (dn/dlambda) - n / lambda**2)
                # w = 2 * pi * c / lambda
                # dw/dlambda = - 2 * pi * c / lambda**2
                # => dw/dk = c / (n - lambda * dn/dlambda)
                jacobian = (n_i - lambda_i * d_n_i_dlambda_func(lambda_i, theta_i) * (1 / np.abs(np.cos(theta_i)))) / c
                # - calculate other prefactors
                idler_prefactor = jacobian * w_i / n_i ** 2
                gauss_term = np.exp(- self.omega_0 ** 2 * (delta_k_x ** 2 + delta_k_y ** 2) / 2)
                sinc_k_z_term = (sinc_phys(delta_k_z * self.L / 2)) ** 2
                f_val = (self.const_rate_prefactor * (signal_prefactor_batch)[:, None, None] * idler_prefactor
                         * sinc_k_z_term * gauss_term)

                # get MC weight
                p_val = (
                        norm.pdf(delta_k_x, loc=0, scale=self.gauss_scale)
                        * norm.pdf(delta_k_y, loc=0, scale=self.gauss_scale)
                )

                valid &= (
                        np.isfinite(f_val)
                        & np.isfinite(p_val)
                        & (p_val > 1e-300)
                )

            else:
                # Obtain longitudinal momentum mismatch by sampling sinc2 function
                sinc_arg_sampled = sample_sinc2(idler_points[:, 2:])
                delta_k_z = sinc_arg_sampled * 2 / self.L
                # ...and obtain longitudinal momentum component
                k_iz = self.k_p - k_sz[:, None, None] - delta_k_z[None, None, :] + self.k_m

                # Transform idler components in spherical coordinates
                k_i, theta_i, phi_i = cartesian_to_spherical(k_ix, k_iy, k_iz)

                # Calculate idler wavelength, refraction index and angular frequency
                lambda_i = get_lambda_i_vectorized(
                    k_i,
                    theta_i,
                    self.n_i_1,
                    self.n_i_2,
                    self.t,
                    self.lambda_p
                )
                n_i = n_i_func(lambda_i, theta_i)
                w_i = 2 * np.pi * c / lambda_i

                # Get energy mismatch
                delta_w = self.w_p - w_s_batch[:, None, None] - w_i

                # Calculate other prefactors
                idler_prefactor = w_i / n_i ** 2
                gauss_term = np.exp(- self.omega_0 ** 2 * (delta_k_x ** 2 + delta_k_y ** 2) / 2)
                sinc_k_z_term = (sinc_phys(sinc_arg_sampled)) ** 2
                sinc_omega_term = (sinc_phys(delta_w * self.T_I / 2)) ** 2
                f_val = (self.const_rate_prefactor * signal_prefactor_batch[:, None, None] * idler_prefactor
                         * sinc_k_z_term * gauss_term * sinc_omega_term)

                # Get MC weight
                p_val = (norm.pdf(delta_k_x, loc=0, scale=self.gauss_scale)
                         * norm.pdf(delta_k_y, loc=0, scale=self.gauss_scale)
                         * sinc2_approximation(sinc_arg_sampled) * self.L / 2)

                valid = (
                        np.isfinite(f_val)
                        & np.isfinite(p_val)
                        & (p_val > 1e-300)
                )

            # Add to the MC sum
            weighted_val = np.zeros_like(f_val)
            weighted_val = np.divide(
                f_val,
                p_val,
                out=weighted_val,
                where=valid
            )

            # Adaptive stopping criteria
            cumsum = np.cumsum(weighted_val, axis=-1)
            cumsum2 = np.cumsum(weighted_val ** 2, axis=-1)

            # We have minimal number of samples we look at
            n = np.arange(self.min_N_i, self.max_N_i + 1)
            mean = cumsum[:, :, self.min_N_i-1:] / n
            variance = np.zeros_like(mean)
            np.divide(
                cumsum2[:, :, self.min_N_i-1:] - n * mean ** 2,
                n - 1,
                out=variance,
                where=n > 1
            )
            variance = np.maximum(variance, 0.0)

            abs_err = np.sqrt(variance / n)
            rel_err = abs_err / (mean + self.eps)

            # When either absolute or relative error drops low enough we have convergence
            converged = (
                (abs_err < self.min_abs_error)
                | (rel_err < self.min_rel_err)
            )

            # We look for group of n_conv consecutively converged samples
            runs = sliding_window_view(
                converged,
                window_shape=self.n_conv,
                axis=-1
            ).all(axis=-1)
            has_convergence = np.any(runs, axis=-1)
            first_group = np.argmax(runs, axis=-1)
            max_index = mean.shape[-1] - 1
            conv_index = np.where(
                has_convergence,
                np.minimum(
                    first_group + self.n_conv,
                    max_index
                ), # if it has converged we chose first index from the group we found
                max_index # if it hasn't we just take maximum index
            )
            phi_integrals = np.take_along_axis(
                mean,
                conv_index[..., None],
                axis=-1
            )[..., 0]

            signal_integrals = np.mean(phi_integrals, axis=1)

            mc_val.append(signal_integrals)

        return np.concatenate(mc_val)

    def sampler(self, args):
        batch_index, start, end, sobol_seed = args

        # One worker works on one signal batch
        k_s_batch = self.k_s[start:end]
        w_s_batch = self.w_s[start:end]
        theta_batch = self.theta_s[start:end]
        sin_theta = np.sin(theta_batch)[:, None]
        signal_prefactor_batch = self.signal_prefactor[start:end]

        # Spherical to cartesian - faster than calling function lots of times
        k_sx = k_s_batch[:, None] * sin_theta * self.cos_phi[None, :]  # (signal_batch_size, N_phi)
        k_sy = k_s_batch[:, None] * sin_theta * self.sin_phi[None, :]  # (signal_batch_size, N_phi)
        k_sz = k_s_batch * np.cos(theta_batch)  # (signal_batch_size, ) - we don't store repeated values of k_sz!

        # Sample maximal numer of idlers
        idler_sampler = qmc.Sobol(
            d=4,
            scramble=True,
            seed=sobol_seed
        )
        idler_points = idler_sampler.random_base2(
            m=int(np.log2(self.max_N_i))
        )

        # Calculate transversal momentum mismatch
        # These two are equivalent:
        # x = norm.ppf(val, loc=mu, scale=sigma)
        # x = mu + sigma * ndtri(val)
        # ndtri is the inverse standard normal CDF and is much faster to calculate
        delta_k_x = self.gauss_scale * ndtri(idler_points[:, 0])[None, None, :]  # (1, 1, max_N_i)
        delta_k_y = self.gauss_scale * ndtri(idler_points[:, 1])[None, None, :]  # (1, 1, max_N_i)

        # Calculate transversal idler momentum components
        k_ix = - delta_k_x - k_sx[:, :, None]
        k_iy = - delta_k_y - k_sy[:, :, None]
        k_i_normal = np.sqrt(k_ix ** 2 + k_iy ** 2)

        # Retain adaptive stopping without processing one batch at a time (i.e. eliminate idler loop)
        # Generate all samples and calculate the weighted values for all 512 samples in one vectorized operation.
        # Idler refraction index function
        n_i_func = lambda wavelength, theta_i: n_eff(self.n_i_1, self.n_i_2, wavelength, theta_i, self.t)
        d_n_i_dlambda_func = lambda wavelength, theta_i: (
            dn_eff_dlambda(self.n_i_1, self.dn_i_1_dlambda,
                           self.n_i_2, self.dn_i_2_dlambda,
                           wavelength, theta_i, self.t)
        )

        if self.T_I == 0:
            # CW pump => force energy conservation
            w_i = (self.w_p - w_s_batch)[:, None, None]
            delta_w = 0

            # Calculate idler wavelength
            lambda_i = 2 * pi * c / w_i

            # Solve for idler polar angle
            theta_i = get_theta_i_vectorized(k_i_normal, lambda_i, n_i_func)
            # ...and get idler refractive index and momentum
            n_i = n_i_func(lambda_i, theta_i)
            k_i = 2 * np.pi * n_i / lambda_i

            # Calculate longitudinal momentum from magnitude and transverse momenta
            k_iz2 = k_i ** 2 - k_ix ** 2 - k_iy ** 2
            # ...and check if this squared value is valid
            valid = (
                    np.isfinite(k_iz2)
                    & (k_iz2 >= 0)
                    & np.isfinite(theta_i)
                    & np.isfinite(n_i)
            )
            k_iz = np.full_like(k_iz2, np.nan)
            k_iz[valid] = np.sqrt(k_iz2[valid])
            # ...then get longitudinal momentum mismatch
            delta_k_z = np.full_like(k_iz2, np.nan)
            delta_k_z = (
                    self.k_p
                    - k_sz[:, None, None]
                    - k_iz
                    + self.k_m
            )

            delta_k_z[~valid] = np.nan

            # calculate integrand function value
            # - calculate Jacobian = 1 / (d(omega) / d(k_iz)) analytically
            # k = 2 * pi * n(lambda, theta) / lambda
            # fixed theta: dk/dlambda = 2 * pi * ((1 / lambda) * (dn/dlambda) - n / lambda**2)
            # w = 2 * pi * c / lambda
            # dw/dlambda = - 2 * pi * c / lambda**2
            # => dw/dk = c / (n - lambda * dn/dlambda)
            jacobian = (n_i - lambda_i * d_n_i_dlambda_func(lambda_i, theta_i) * (1 / np.abs(np.cos(theta_i)))) / c
            # - calculate other prefactors
            idler_prefactor = jacobian * w_i / n_i ** 2
            gauss_term = np.exp(- self.omega_0 ** 2 * (delta_k_x ** 2 + delta_k_y ** 2) / 2)
            sinc_k_z_term = (sinc_phys(delta_k_z * self.L / 2)) ** 2
            f_val = (self.const_rate_prefactor * (signal_prefactor_batch)[:, None, None] * idler_prefactor
                     * sinc_k_z_term * gauss_term)

            # get MC weight
            p_val = (
                    norm.pdf(delta_k_x, loc=0, scale=self.gauss_scale)
                    * norm.pdf(delta_k_y, loc=0, scale=self.gauss_scale)
            )

            valid &= (
                    np.isfinite(f_val)
                    & np.isfinite(p_val)
                    & (p_val > 1e-300)
            )

        else:
            # Obtain longitudinal momentum mismatch by sampling sinc2 function
            sinc_arg_sampled = sample_sinc2(idler_points[:, 2:])
            delta_k_z = sinc_arg_sampled * 2 / self.L
            # ...and obtain longitudinal momentum component
            k_iz = self.k_p - k_sz[:, None, None] - delta_k_z[None, None, :] + self.k_m

            # Transform idler components in spherical coordinates
            k_i, theta_i, phi_i = cartesian_to_spherical(k_ix, k_iy, k_iz)

            # Calculate idler wavelength, refraction index and angular frequency
            lambda_i = get_lambda_i_vectorized(
                k_i,
                theta_i,
                self.n_i_1,
                self.n_i_2,
                self.t,
                self.lambda_p
            )
            n_i = n_i_func(lambda_i, theta_i)
            w_i = 2 * np.pi * c / lambda_i

            # Get energy mismatch
            delta_w = self.w_p - w_s_batch[:, None, None] - w_i

            # Calculate other prefactors
            idler_prefactor = w_i / n_i ** 2
            gauss_term = np.exp(- self.omega_0 ** 2 * (delta_k_x ** 2 + delta_k_y ** 2) / 2)
            sinc_k_z_term = (sinc_phys(sinc_arg_sampled)) ** 2
            sinc_omega_term = (sinc_phys(delta_w * self.T_I / 2)) ** 2
            f_val = (self.const_rate_prefactor * signal_prefactor_batch[:, None, None] * idler_prefactor
                     * sinc_k_z_term * gauss_term * sinc_omega_term)

            # Get MC weight
            p_val = (norm.pdf(delta_k_x, loc=0, scale=self.gauss_scale)
                     * norm.pdf(delta_k_y, loc=0, scale=self.gauss_scale)
                     * sinc2_approximation(sinc_arg_sampled) * self.L / 2)

            valid = (
                    np.isfinite(f_val)
                    & np.isfinite(p_val)
                    & (p_val > 1e-300)
            )

        return f_val, p_val, valid, lambda_i, theta_i

    def adaptive_stopping(self, weighted_val):
        # Adaptive stopping criteria
        cumsum = np.cumsum(weighted_val, axis=-1)
        cumsum2 = np.cumsum(weighted_val ** 2, axis=-1)

        # We have minimal number of samples we look at
        n = np.arange(self.min_N_i, self.max_N_i + 1)
        mean = cumsum[:, :, self.min_N_i - 1:] / n
        variance = np.zeros_like(mean)
        np.divide(
            cumsum2[:, :, self.min_N_i - 1:] - n * mean ** 2,
            n - 1,
            out=variance,
            where=n > 1
        )
        variance = np.maximum(variance, 0.0)

        abs_err = np.sqrt(variance / n)
        rel_err = abs_err / (mean + self.eps)

        # When either absolute or relative error drops low enough we have convergence
        converged = (
                (abs_err < self.min_abs_error)
                | (rel_err < self.min_rel_err)
        )

        # We look for group of n_conv consecutively converged samples
        runs = sliding_window_view(
            converged,
            window_shape=self.n_conv,
            axis=-1
        ).all(axis=-1)
        has_convergence = np.any(runs, axis=-1)
        first_group = np.argmax(runs, axis=-1)
        max_index = mean.shape[-1] - 1
        conv_index = np.where(
            has_convergence,
            np.minimum(
                first_group + self.n_conv,
                max_index
            ),  # if it has converged we chose first index from the group we found
            max_index  # if it hasn't we just take maximum index
        )

        return mean, conv_index

    def get_transition_rate_batch(self, args):
        batch_index, start, end, sobol_seed = args

        # Sample idlers over signal grid
        f_val, p_val, valid, _, _ = self.sampler(args)

        # Add to the MC sum
        weighted_val = np.zeros_like(f_val)
        weighted_val = np.divide(
            f_val,
            p_val,
            out=weighted_val,
            where=valid
        )

        # Adaptive stopping
        mean, conv_index = self.adaptive_stopping(weighted_val)

        # Integrate
        phi_integrals = np.take_along_axis(
            mean,
            conv_index[..., None],
            axis=-1
        )[..., 0]
        signal_integrals = np.mean(phi_integrals, axis=1)

        return batch_index, signal_integrals

    def get_transition_rate_parallel(self):
        n_batches = len(self.signal_batches)

        # Generate seeds ONLY in the parent process.
        if self.seed is not None:
            sobol_seeds = [
                int(self.rng.integers(0, 2 ** 32 - 1))
                for _ in range(n_batches)
            ]
        else:
            sobol_seeds = [None] * n_batches

        tasks = [
            (
                batch_index,
                start,
                end,
                sobol_seeds[batch_index],
            )
            for batch_index, (start, end)
            in enumerate(self.signal_batches)
        ]

        results = [None] * n_batches

        with mp.Pool(processes=self.n_processes) as pool:

            for batch_index, signal_integrals in tqdm(
                    pool.imap_unordered(
                        self.get_transition_rate_batch,
                        tasks,
                    ),
                    total=n_batches,
                    desc="Signal batches",
            ):
                results[batch_index] = signal_integrals

        return np.concatenate(results)

    def get_phase_matching_contour(self):
        """
        Compute transverse phase matching grid.
        :return:
        """
        # Initialize grid
        contour_grid_size = 500 # contour has more point by default, no matter how fine spectrum grid is
        contour_lambda_grid = np.linspace(self.lambda_s_min, self.lambda_s_max, contour_grid_size)
        contour_theta_grid = np.linspace(self.theta_s_min, self.theta_s_max, contour_grid_size)
        l = np.asarray(contour_lambda_grid).ravel()
        t = np.asarray(contour_theta_grid).ravel()
        lam, theta = np.meshgrid(l, t, indexing="xy")

        # Signal refractive index and wavevector (vectorized in the right shape)
        n_s = self.n_s_func(self.n_s_1, self.n_s_2, lam, theta)
        k_s = 2 * np.pi * n_s / lam

        # Energy conservation
        denom = lam - self.lambda_p
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        lambda_i = self.lambda_p * lam / denom

        # Transverse momentum conservation: k_s sin(theta_s) = k_i sin(theta_i)
        # => sin (theta_i) = (k_s / k_i) sin(theta_s)
        # k_i itself also depends on theta_i - both sides depend on it!

        # Apply iterative approach (here only 5 steps as the dependence of n_i on theta_i
        # is usually smooth and not extremely strong, this fixed-point iteration converges quickly
        # Step 1: Initial guess - theta is 0 everywhere
        theta_i = np.zeros_like(lambda_i)
        for _ in range(30):
            # Step 2: Compute idler refractive index with the current estimate...
            n_i = n_eff(self.n_i_1, self.n_i_2, lambda_i, theta_i, self.t)
            # ... and from it the idler vector magnitude
            k_i = 2 * np.pi * n_i / lambda_i

            # Step 3: Get new estimate for theta from transverse momentum conservation
            sin_theta = k_s * np.sin(theta) / k_i
            valid = np.abs(sin_theta) <= 1

            # Step 4: Update theta
            theta_i = np.full_like(theta, np.nan)
            theta_i[valid] = np.arcsin(sin_theta[valid])

        # Now recompute n_i and k_i with the final estimate...
        n_i = n_eff(self.n_i_1, self.n_i_2, lambda_i, theta_i, self.t)
        k_i = 2 * np.pi * n_i / lambda_i
        # ...and evaluate longitudinal mismatch
        dkz = self.k_p + self.k_m - k_s * np.cos(theta) - k_i * np.cos(theta_i)

        return lam, theta, dkz

    def plot_spectrum(self):
        if self.n_processes is not None:
            vals = self.get_transition_rate_parallel()
        else:
            vals = self.get_transition_rate()
        theta_deg = np.rad2deg(self.theta_s_grid) # convert theta to degrees

        # Plot spectrum
        plt.figure(figsize=(10, 6))
        spectrum = vals.reshape(self.lambda_mesh.shape)
        plt.pcolormesh(
            self.lambda_s_grid * 1e9,  # convert to nm to match the figure
            theta_deg,
            spectrum,
            shading='auto',
            cmap='magma',
            vmin=0
        )
        plt.colorbar(label="Transition rate dR")

        # Plot phase matching contour
        lam, theta, dkz = self.get_phase_matching_contour()
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


class Imaging():
    """
    Class for simulating the quantum imaging with undetected light setup.
    """
    def __init__(self, source=SPDC(),
                 detector_size=512e-3, detector_pixels=1000,
                 object_func=object_phase_bar):
        # ------ SPDC source -------
        self.source = source # SPDC class instance

        # ------ Detector ------
        self.detector_size = detector_size # size of detector in m
        self.detector_pixels = detector_pixels # number of detector pixels

        # ------ Object -------
        self.object_func = object_func # function that mathematically represents object properties

    def full_simulation(self):
        src = self.source

        # Loop over signals and perform integration over phi and then also idlers
        mc_val_plus = []
        mc_val_minus = []
        for (start, end) in tqdm(self.signal_batches):
            k_s_batch = self.k_s[start:end]
            w_s_batch = self.w_s[start:end]
            theta_batch = self.theta_s[start:end]
            sin_theta = np.sin(theta_batch)[:, None]
            signal_prefactor_batch = self.signal_prefactor[start:end]

            # Spherical to cartesian - faster than calling function lots of times
            k_sx = k_s_batch[:, None] * sin_theta * self.cos_phi[None, :]  # (signal_batch_size, N_phi)
            k_sy = k_s_batch[:, None] * sin_theta * self.sin_phi[None, :]  # (signal_batch_size, N_phi)
            k_sz = k_s_batch * np.cos(theta_batch)  # (signal_batch_size, ) - we don't store repeated values of k_sz!

            # Sample maximal numer of idlers
            if self.seed is not None:
                sobol_seed = int(self.rng.integers(0, 2 ** 32 - 1))
            else:
                sobol_seed = None
            idler_sampler = qmc.Sobol(
                d=4,
                scramble=True,
                seed=sobol_seed
            )
            idler_points = idler_sampler.random_base2(
                m=int(np.log2(self.max_N_i))
            )

            # Calculate transversal momentum mismatch
            # These two are equivalent:
            # x = norm.ppf(val, loc=mu, scale=sigma)
            # x = mu + sigma * ndtri(val)
            # ndtri is the inverse standard normal CDF and is much faster to calculate
            delta_k_x = self.gauss_scale * ndtri(idler_points[:, 0])[None, None, :]  # (1, 1, max_N_i)
            delta_k_y = self.gauss_scale * ndtri(idler_points[:, 1])[None, None, :]  # (1, 1, max_N_i)

            # Calculate transversal idler momentum components
            k_ix = - delta_k_x - k_sx[:, :, None]
            k_iy = - delta_k_y - k_sy[:, :, None]
            k_i_normal = np.sqrt(k_ix ** 2 + k_iy ** 2)

            # Retain adaptive stopping without processing one batch at a time (i.e. eliminate idler loop)
            # Generate all samples and calculate the weighted values for all 512 samples in one vectorized operation.
            # Idler refraction index function
            n_i_func = lambda wavelength, theta_i: n_eff(self.n_i_1, self.n_i_2, wavelength, theta_i, self.t)
            d_n_i_dlambda_func = lambda wavelength, theta_i: (
                dn_eff_dlambda(self.n_i_1, self.dn_i_1_dlambda,
                               self.n_i_2, self.dn_i_2_dlambda,
                               wavelength, theta_i, self.t)
            )

            if self.T_I == 0:
                # CW pump => force energy conservation
                w_i = (self.w_p - w_s_batch)[:, None, None]
                delta_w = 0

                # Calculate idler wavelength
                lambda_i = 2 * pi * c / w_i

                # Solve for idler polar angle
                theta_i = get_theta_i_vectorized(k_i_normal, lambda_i, n_i_func)
                # ...and get idler refractive index and momentum
                n_i = n_i_func(lambda_i, theta_i)
                k_i = 2 * np.pi * n_i / lambda_i

                # Calculate longitudinal momentum from magnitude and transverse momenta
                k_iz2 = k_i ** 2 - k_ix ** 2 - k_iy ** 2
                # ...and check if this squared value is valid
                valid = (
                        np.isfinite(k_iz2)
                        & (k_iz2 >= 0)
                        & np.isfinite(theta_i)
                        & np.isfinite(n_i)
                )
                k_iz = np.full_like(k_iz2, np.nan)
                k_iz[valid] = np.sqrt(k_iz2[valid])
                # ...then get longitudinal momentum mismatch
                delta_k_z = np.full_like(k_iz2, np.nan)
                delta_k_z = (
                        self.k_p
                        - k_sz[:, None, None]
                        - k_iz
                        + self.k_m
                )

                delta_k_z[~valid] = np.nan

                # calculate integrand function value
                # - calculate Jacobian = 1 / (d(omega) / d(k_iz)) analytically
                # k = 2 * pi * n(lambda, theta) / lambda
                # fixed theta: dk/dlambda = 2 * pi * ((1 / lambda) * (dn/dlambda) - n / lambda**2)
                # w = 2 * pi * c / lambda
                # dw/dlambda = - 2 * pi * c / lambda**2
                # => dw/dk = c / (n - lambda * dn/dlambda)
                jacobian = (n_i - lambda_i * d_n_i_dlambda_func(lambda_i, theta_i) * (1 / np.abs(np.cos(theta_i)))) / c
                # - calculate other prefactors
                idler_prefactor = jacobian * w_i / n_i ** 2
                gauss_term = np.exp(- self.omega_0 ** 2 * (delta_k_x ** 2 + delta_k_y ** 2) / 2)
                sinc_k_z_term = (sinc_phys(delta_k_z * self.L / 2)) ** 2
                f_val = (self.const_rate_prefactor * (signal_prefactor_batch)[:, None, None] * idler_prefactor
                         * sinc_k_z_term * gauss_term)

                # get MC weight
                p_val = (
                        norm.pdf(delta_k_x, loc=0, scale=self.gauss_scale)
                        * norm.pdf(delta_k_y, loc=0, scale=self.gauss_scale)
                )

                valid &= (
                        np.isfinite(f_val)
                        & np.isfinite(p_val)
                        & (p_val > 1e-300)
                )

            else:
                # Obtain longitudinal momentum mismatch by sampling sinc2 function
                sinc_arg_sampled = sample_sinc2(idler_points[:, 2:])
                delta_k_z = sinc_arg_sampled * 2 / self.L
                # ...and obtain longitudinal momentum component
                k_iz = self.k_p - k_sz[:, None, None] - delta_k_z[None, None, :] + self.k_m

                # Transform idler components in spherical coordinates
                k_i, theta_i, phi_i = cartesian_to_spherical(k_ix, k_iy, k_iz)

                # Calculate idler wavelength, refraction index and angular frequency
                lambda_i = get_lambda_i_vectorized(
                    k_i,
                    theta_i,
                    self.n_i_1,
                    self.n_i_2,
                    self.t,
                    self.lambda_p
                )
                n_i = n_i_func(lambda_i, theta_i)
                w_i = 2 * np.pi * c / lambda_i

                # Get energy mismatch
                delta_w = self.w_p - w_s_batch[:, None, None] - w_i

                # Calculate other prefactors
                idler_prefactor = w_i / n_i ** 2
                gauss_term = np.exp(- self.omega_0 ** 2 * (delta_k_x ** 2 + delta_k_y ** 2) / 2)
                sinc_k_z_term = (sinc_phys(sinc_arg_sampled)) ** 2
                sinc_omega_term = (sinc_phys(delta_w * self.T_I / 2)) ** 2
                f_val = (self.const_rate_prefactor * signal_prefactor_batch[:, None, None] * idler_prefactor
                         * sinc_k_z_term * gauss_term * sinc_omega_term)

                # Get MC weight
                p_val = (norm.pdf(delta_k_x, loc=0, scale=self.gauss_scale)
                         * norm.pdf(delta_k_y, loc=0, scale=self.gauss_scale)
                         * sinc2_approximation(sinc_arg_sampled) * self.L / 2)

                valid = (
                        np.isfinite(f_val)
                        & np.isfinite(p_val)
                        & (p_val > 1e-300)
                )

            # apply transition from crystal (medium) to vacuum and propagate idler to the object
            lambda_i_vac, theta_i_vac = medium_to_air(lambda_i, theta_i, n_i_func)
            M_idler = M_free_space(f_i) @ M_lens(f_i) @ M_free_space(f_i)

            # calculate coordinates after the propagation
            r_i_initial = np.zeros_like(theta_i_vac)
            r_o, theta_o = propagate(r_i_initial, theta_i_vac, M_idler)
            x_o, y_o = get_ray_coordinates(r_o, theta_o)

            # get object transmission coefficient and phase from object function
            t_o, phi_o = self.object_func(x_o, y_o)

            # Add to the MC sum
            weighted_val = np.zeros_like(f_val)
            weighted_val = np.divide(
                f_val,
                p_val,
                out=weighted_val,
                where=valid
            )

            weighted_val_plus = weighted_val * (1 + (t_o * np.cos(phi_o)))
            weighted_val_minus = weighted_val * (1 - (t_o * np.cos(phi_o)))

            # Adaptive stopping criteria
            for arr in [weighted_val_plus, weighted_val_minus]:
                cumsum = np.cumsum(weighted_val, axis=-1)
                cumsum2 = np.cumsum(weighted_val ** 2, axis=-1)

                # We have minimal number of samples we look at
                n = np.arange(self.min_N_i, self.max_N_i + 1)
                mean = cumsum[:, :, self.min_N_i-1:] / n
                variance = np.zeros_like(mean)
                np.divide(
                    cumsum2[:, :, self.min_N_i-1:] - n * mean ** 2,
                    n - 1,
                    out=variance,
                    where=n > 1
                )
                variance = np.maximum(variance, 0.0)

                abs_err = np.sqrt(variance / n)
                rel_err = abs_err / (mean + self.eps)

                # When either absolute or relative error drops low enough we have convergence
                converged = (
                    (abs_err < self.min_abs_error)
                    | (rel_err < self.min_rel_err)
                )

                # We look for group of n_conv consecutively converged samples
                runs = sliding_window_view(
                    converged,
                    window_shape=self.n_conv,
                    axis=-1
                ).all(axis=-1)
                has_convergence = np.any(runs, axis=-1)
                first_group = np.argmax(runs, axis=-1)
                max_index = mean.shape[-1] - 1
                conv_index = np.where(
                    has_convergence,
                    np.minimum(
                        first_group + self.n_conv,
                        max_index
                    ), # if it has converged we chose first index from the group we found
                    max_index # if it hasn't we just take maximum index
                )
                phi_integrals = np.take_along_axis(
                    mean,
                    conv_index[..., None],
                    axis=-1
                )[..., 0]

                signal_integrals = np.mean(phi_integrals, axis=1)

                if arr == weighted_val_plus:
                    mc_val_plus.append(signal_integrals)
                else:
                    mc_val_minus.append(signal_integrals)

        mean_plus = np.mean(np.concatenate(mc_val_plus))
        mean_minus = np.mean(np.concatenate(mc_val_minus))

        visibility = (mean_plus - mean_minus) / (mean_plus + mean_minus + 1e-20)

        return visibility

    def get_kernel(self):
        k_s_central = np.array([2 * np.pi * n_p * lambda_s_central / lambda_s_central])
        theta_s = np.array([0.0])
        phi_s = np.array([0.0])

        for (start, end) in tqdm(self.signal_batches):
            k_s_batch = self.k_s[start:end]
            w_s_batch = self.w_s[start:end]
            theta_batch = self.theta_s[start:end]
            sin_theta = np.sin(theta_batch)[:, None]
            signal_prefactor_batch = self.signal_prefactor[start:end]

            # Spherical to cartesian - faster than calling function lots of times
            k_sx = k_s_batch[:, None] * sin_theta * self.cos_phi[None, :]  # (signal_batch_size, N_phi)
            k_sy = k_s_batch[:, None] * sin_theta * self.sin_phi[None, :]  # (signal_batch_size, N_phi)
            k_sz = k_s_batch * np.cos(theta_batch)  # (signal_batch_size, ) - we don't store repeated values of k_sz!

            # Sample maximal numer of idlers
            if self.seed is not None:
                sobol_seed = int(self.rng.integers(0, 2 ** 32 - 1))
            else:
                sobol_seed = None
            idler_sampler = qmc.Sobol(
                d=4,
                scramble=True,
                seed=sobol_seed
            )
            idler_points = idler_sampler.random_base2(
                m=int(np.log2(self.max_N_i))
            )

            # Calculate transversal momentum mismatch
            # These two are equivalent:
            # x = norm.ppf(val, loc=mu, scale=sigma)
            # x = mu + sigma * ndtri(val)
            # ndtri is the inverse standard normal CDF and is much faster to calculate
            delta_k_x = self.gauss_scale * ndtri(idler_points[:, 0])[None, None, :]  # (1, 1, max_N_i)
            delta_k_y = self.gauss_scale * ndtri(idler_points[:, 1])[None, None, :]  # (1, 1, max_N_i)

            # Calculate transversal idler momentum components
            k_ix = - delta_k_x - k_sx[:, :, None]
            k_iy = - delta_k_y - k_sy[:, :, None]
            k_i_normal = np.sqrt(k_ix ** 2 + k_iy ** 2)

            # Retain adaptive stopping without processing one batch at a time (i.e. eliminate idler loop)
            # Generate all samples and calculate the weighted values for all 512 samples in one vectorized operation.
            # Idler refraction index function
            n_i_func = lambda wavelength, theta_i: n_eff(self.n_i_1, self.n_i_2, wavelength, theta_i, self.t)
            d_n_i_dlambda_func = lambda wavelength, theta_i: (
                dn_eff_dlambda(self.n_i_1, self.dn_i_1_dlambda,
                               self.n_i_2, self.dn_i_2_dlambda,
                               wavelength, theta_i, self.t)
            )

            if self.T_I == 0:
                # CW pump => force energy conservation
                w_i = (self.w_p - w_s_batch)[:, None, None]
                delta_w = 0

                # Calculate idler wavelength
                lambda_i = 2 * pi * c / w_i

                # Solve for idler polar angle
                theta_i = get_theta_i_vectorized(k_i_normal, lambda_i, n_i_func)
                # ...and get idler refractive index and momentum
                n_i = n_i_func(lambda_i, theta_i)
                k_i = 2 * np.pi * n_i / lambda_i

                # Calculate longitudinal momentum from magnitude and transverse momenta
                k_iz2 = k_i ** 2 - k_ix ** 2 - k_iy ** 2
                # ...and check if this squared value is valid
                valid = (
                        np.isfinite(k_iz2)
                        & (k_iz2 >= 0)
                        & np.isfinite(theta_i)
                        & np.isfinite(n_i)
                )
                k_iz = np.full_like(k_iz2, np.nan)
                k_iz[valid] = np.sqrt(k_iz2[valid])
                # ...then get longitudinal momentum mismatch
                delta_k_z = np.full_like(k_iz2, np.nan)
                delta_k_z = (
                        self.k_p
                        - k_sz[:, None, None]
                        - k_iz
                        + self.k_m
                )

                delta_k_z[~valid] = np.nan

                # calculate integrand function value
                # - calculate Jacobian = 1 / (d(omega) / d(k_iz)) analytically
                # k = 2 * pi * n(lambda, theta) / lambda
                # fixed theta: dk/dlambda = 2 * pi * ((1 / lambda) * (dn/dlambda) - n / lambda**2)
                # w = 2 * pi * c / lambda
                # dw/dlambda = - 2 * pi * c / lambda**2
                # => dw/dk = c / (n - lambda * dn/dlambda)
                jacobian = (n_i - lambda_i * d_n_i_dlambda_func(lambda_i, theta_i) * (1 / np.abs(np.cos(theta_i)))) / c
                # - calculate other prefactors
                idler_prefactor = jacobian * w_i / n_i ** 2
                gauss_term = np.exp(- self.omega_0 ** 2 * (delta_k_x ** 2 + delta_k_y ** 2) / 2)
                sinc_k_z_term = (sinc_phys(delta_k_z * self.L / 2)) ** 2
                f_val = (self.const_rate_prefactor * (signal_prefactor_batch)[:, None, None] * idler_prefactor
                         * sinc_k_z_term * gauss_term)

                # get MC weight
                p_val = (
                        norm.pdf(delta_k_x, loc=0, scale=self.gauss_scale)
                        * norm.pdf(delta_k_y, loc=0, scale=self.gauss_scale)
                )

                valid &= (
                        np.isfinite(f_val)
                        & np.isfinite(p_val)
                        & (p_val > 1e-300)
                )

            else:
                # Obtain longitudinal momentum mismatch by sampling sinc2 function
                sinc_arg_sampled = sample_sinc2(idler_points[:, 2:])
                delta_k_z = sinc_arg_sampled * 2 / self.L
                # ...and obtain longitudinal momentum component
                k_iz = self.k_p - k_sz[:, None, None] - delta_k_z[None, None, :] + self.k_m

                # Transform idler components in spherical coordinates
                k_i, theta_i, phi_i = cartesian_to_spherical(k_ix, k_iy, k_iz)

                # Calculate idler wavelength, refraction index and angular frequency
                lambda_i = get_lambda_i_vectorized(
                    k_i,
                    theta_i,
                    self.n_i_1,
                    self.n_i_2,
                    self.t,
                    self.lambda_p
                )
                n_i = n_i_func(lambda_i, theta_i)
                w_i = 2 * np.pi * c / lambda_i

                # Get energy mismatch
                delta_w = self.w_p - w_s_batch[:, None, None] - w_i

                # Calculate other prefactors
                idler_prefactor = w_i / n_i ** 2
                gauss_term = np.exp(- self.omega_0 ** 2 * (delta_k_x ** 2 + delta_k_y ** 2) / 2)
                sinc_k_z_term = (sinc_phys(sinc_arg_sampled)) ** 2
                sinc_omega_term = (sinc_phys(delta_w * self.T_I / 2)) ** 2
                f_val = (self.const_rate_prefactor * signal_prefactor_batch[:, None, None] * idler_prefactor
                         * sinc_k_z_term * gauss_term * sinc_omega_term)

                # Get MC weight
                p_val = (norm.pdf(delta_k_x, loc=0, scale=self.gauss_scale)
                         * norm.pdf(delta_k_y, loc=0, scale=self.gauss_scale)
                         * sinc2_approximation(sinc_arg_sampled) * self.L / 2)

                valid = (
                        np.isfinite(f_val)
                        & np.isfinite(p_val)
                        & (p_val > 1e-300)
                )

        # apply transition from crystal (medium) to vacuum and propagate idler to the object
        lambda_i_vac, theta_i_vac = medium_to_air(lambda_i, theta_i, n_i_func)
        M_idler = M_free_space(f_i) @ M_lens(f_i) @ M_free_space(f_i)

        # calculate coordinates after the propagation
        r_i_initial = np.zeros_like(theta_i_vac)
        r_o, theta_o = propagate(r_i_initial, theta_i_vac, M_idler)
        x_o, y_o = get_ray_coordinates(r_o, theta_o)

        # create bins corresponding to the detector pixels
        bins = np.linspace(-detector_size / 2.0, detector_size / 2.0, pixel_num + 1)

        # create and normalize kernel
        kernel, _, _ = np.histogram2d(x_o.flatten(), y_o.flatten(), bins=[bins, bins], weights=weighted_val)
        kernel /= np.sum(kernel)

        return kernel, bins

    def convolution_method_imaging(self):
        kernel, bins = self.get_kernel()
        centers = (bins[:-1] + bins[1]) / 2
        x_grid, y_grid = np.meshgrid(centers, centers)
        t_o, phi_o = self.object_func(x_grid, y_grid)
        obj = t_o * np.cos(phi_o)
        o_plane_img = fftconvolve(obj, kernel, mode='same')
        return o_plane_img

    def plot_image(self):
        detector_size = 512e-4  # m
        pixel_num = 500
        x_edges = np.linspace(-detector_size / 2, detector_size / 2, pixel_num + 1)
        y_edges = np.linspace(-detector_size / 2, detector_size / 2, pixel_num + 1)
        histo, x_edges, y_edges = np.histogram2d(x_detector.flatten(), y_detector.flatten(),
                                                 bins=[x_edges, y_edges],
                                                 weights=integral_vals.flatten())
        plt.figure(figsize=(10, 8))
        plt.imshow(histo.T, extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                   aspect="auto", origin='lower')
        plt.title("Detector counts")
        plt.xlabel("x_detector")
        plt.ylabel("y_detector")
        plt.colorbar(label="Counts")
        plt.show()
