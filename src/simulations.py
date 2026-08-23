import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from numpy.lib._stride_tricks_impl import sliding_window_view
from scipy.constants import c, pi, epsilon_0
from scipy.stats import norm, qmc
from scipy.special import ndtri
from tqdm.auto import tqdm

from src.sellmeier import *
from src.calculation_helpers import *
from src.MC_helpers import sample_sinc2, sinc_phys, sinc2_approximation

class SPDC():
    """
    Class for simulation of SPDC source.
    """
    def __init__(self, L=2e-3, t=25, period=5.827e-6, m=-1, chi_eff=16.6e-12,
                 lambda_p=405e-9, n_p=n_x_KTP,
                 P=0.5, omega_0=60e-6, T_I=0,
                 lambda_s_min=600e-9, lambda_s_max=1400e-9,
                 theta_s_min=0, theta_s_max=0.07,
                 phi_s_min=0, phi_s_max=2*pi,
                 n_s_func=n_eff, n_s_1=n_x_KTP, n_s_2=n_x_KTP,
                 n_i=n_eff, n_i_1=n_y_KTP, n_i_2=n_y_KTP,
                 min_N_i=int(2**5), max_N_i=int(2**9), N_phi=int(2**5), grid_size=int(200),
                 eps=1e-12, conv_check=100, min_rel_err=1e-2, min_abs_error=1e-21,
                 signal_batch_size=20, n_conv=16):
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
        self.n_i = n_i # function for calculating idler refractive index
        self.n_i_1 = n_i_1 # first refractive index that is contained in effective idler refractive index
        self.n_i_2 = n_i_2 # second refractive index that is contained in effective idler refractive index

        # -------- SIMULATION --------
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

    def get_transition_rate(self):
        """
        To calculate the transition rate of the SPDC source equation 2.67 from
        is used.
        :return:
        """
        # Calculate integral prefactor
        const_rate_prefactor = (16 * self.omega_0**2 * self.P * self.L**2 * self.chi_eff**2
                                / (epsilon_0 * c * self.n_p * (2 * pi)**8))

        if self.m != 0:  # QPM exists
            const_rate_prefactor /= self.m ** 2

        if self.T_I == 0:  # CW pump case
            const_rate_prefactor *= (2 * np.pi)
        else:
            const_rate_prefactor *= self.T_I

        # Sample phi to completely define signal
        phi_s = np.linspace(0, 2 * np.pi, self.N_phi, endpoint=False)

        # Loop over signals and perform integration over phi and then also idlers
        sum_val = np.zeros(self.N_s)
        for s in tqdm(range(self.N_s)):
            # Integrate over phi
            phi_sum = 0.0
            for phi in phi_s:
                idler_sum = 0.0
                idler_sum2 = 0.0

                # Calculate signal components for this phi
                k_sx, k_sy, k_sz = spherical_to_cartesian(self.k_s[s], self.theta_s[s], phi)

                # Sample maximal numer of idlers
                idler_sampler = qmc.Sobol(d=4, scramble=True)
                idler_points = idler_sampler.random(self.max_N_i)

                # Take smaller batches of sampled idlers
                # (implementing adaptive stopping - we stop integration for this specific signal if
                # relative or absolute error don't change by more than predefined limits)
                max_loops = int((self.max_N_i - self.min_N_i) / self.conv_check) + 1
                for i in range(max_loops):
                    # Sample transversal momentum mismatch
                    min_i = 0 if i == 0 else self.min_N_i + (i - 1) * self.conv_check
                    max_i = self.min_N_i + i * self.conv_check
                    delta_k_x = norm.ppf(idler_points[min_i:max_i, 0], loc=0, scale=self.gauss_scale)
                    delta_k_y = norm.ppf(idler_points[min_i:max_i, 1], loc=0, scale=self.gauss_scale)

                    # Calculate transversal idler momentum components
                    k_ix = - delta_k_x - k_sx
                    k_iy = - delta_k_y - k_sy
                    k_i_normal = np.sqrt(k_ix ** 2 + k_iy ** 2)

                    # Idler refraction index function
                    n_i_func = lambda wavelength, theta_i: n_eff(self.n_i_1, self.n_i_2, wavelength, theta_i, self.t)

                    if self.T_I == 0:
                        # CW pump => force energy conservation
                        w_i = self.w_p - self.w_s[s]
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
                        delta_k_z[valid] = self.k_p - k_sz - k_iz[valid] + self.k_m

                        # calculate integrand function value
                        # - calculate Jacobian = 1 / (d(omega) / d(k_iz))
                        dk = 1e-5 * abs(k_iz)
                        omega_plus = w_from_k_components(k_ix, k_iy, k_iz + dk, self.n_i_1, self.n_i_2, self.t, self.lambda_p)
                        omega_minus = w_from_k_components(k_ix, k_iy, k_iz - dk, self.n_i_1, self.n_i_2, self.t, self.lambda_p)
                        domega_dkiz = (omega_plus - omega_minus) / (2 * dk)
                        jacobian = 1 / abs(domega_dkiz)
                        # - calculate other prefactors
                        signal_prefactor = (self.k_s[s] ** 4 * self.w_s[s] * np.sin(self.theta_s[s])) / (self.n_s[s] ** 2)
                        idler_prefactor = jacobian * w_i / n_i ** 2
                        gauss_term = np.exp(- self.omega_0 ** 2 * (delta_k_x ** 2 + delta_k_y ** 2) / 2)
                        sinc_k_z_term = (sinc_phys(delta_k_z * self.L / 2)) ** 2
                        f_val = (const_rate_prefactor * signal_prefactor * idler_prefactor
                                 * sinc_k_z_term * gauss_term)

                        # get MC weight
                        p_val = (norm.pdf(delta_k_x, loc=0, scale=self.gauss_scale)
                                 * norm.pdf(delta_k_y, loc=0, scale=self.gauss_scale))

                        valid &= (
                                np.isfinite(f_val)
                                & np.isfinite(p_val)
                                & (p_val > 1e-300)
                        )

                    else:
                        # Obtain longitudinal momentum mismatch by sampling sinc2 function
                        sinc_arg_sampled = sample_sinc2(idler_points[min_i:max_i, 2:])
                        delta_k_z = sinc_arg_sampled * 2 / self.L
                        # ...and obtain longitudinal momentum component
                        k_iz = self.k_p - k_sz - delta_k_z + self.k_m

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
                        delta_w = self.w_p - self.w_s[s] - w_i

                        # Calculate integrand function value
                        signal_prefactor = (self.k_s[s] ** 4 * self.w_s[s] * np.sin(self.theta_s[s])) / (self.n_s[s] ** 2)
                        idler_prefactor = w_i / n_i ** 2
                        gauss_term = np.exp(- self.omega_0 ** 2 * (delta_k_x ** 2 + delta_k_y ** 2) / 2)
                        sinc_k_z_term = (sinc_phys(sinc_arg_sampled)) ** 2
                        sinc_omega_term = (sinc_phys(delta_w * self.T_I / 2)) ** 2
                        f_val = (const_rate_prefactor * signal_prefactor * idler_prefactor
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
                    weighted_val[valid] = f_val[valid] / p_val[valid]
                    idler_sum += np.sum(weighted_val)
                    idler_sum2 += np.sum(weighted_val**2)

                    # Adaptive stopping criteria
                    mean = idler_sum / max_i
                    variance = np.maximum(0.0, (idler_sum2 / max_i - mean ** 2) / (max_i - 1))

                    abs_err = np.sqrt(variance)
                    rel_err = abs_err / (mean + self.eps)

                    if i > 0 and (abs_err < self.min_abs_error or rel_err < self.min_rel_err):
                        break

                phi_sum += idler_sum / max_i

            sum_val[s] += phi_sum

        return sum_val / self.N_phi

    def get_transition_rate_vectorized(self):
        # Loop over signals and perform integration over phi and then also idlers
        mc_val = []
        seed = 42
        rng = np.random.default_rng(seed)
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

            sobol_seed = rng.integers(0, 2 ** 32 - 1)

            idler_sampler = qmc.Sobol(
                d=4,
                scramble=True,
                seed=int(sobol_seed)
            )

            idler_points = idler_sampler.random(self.max_N_i)

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
                # - calculate Jacobian = 1 / (d(omega) / d(k_iz))
                dk = 1e-5 * abs(k_iz)
                omega_plus = w_from_k_components(k_ix, k_iy, k_iz + dk, self.n_i_1, self.n_i_2, self.t, self.lambda_p)
                omega_minus = w_from_k_components(k_ix, k_iy, k_iz - dk, self.n_i_1, self.n_i_2, self.t, self.lambda_p)
                domega_dkiz = (omega_plus - omega_minus) / (2 * dk)
                jacobian = 1 / abs(domega_dkiz)
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
                k_iz = self.k_p - k_sz - delta_k_z + self.k_m

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
                delta_w = self.w_p - w_s_batch - w_i

                # Calculate integrand function value
                idler_prefactor = w_i / n_i ** 2
                gauss_term = np.exp(- self.omega_0 ** 2 * (delta_k_x ** 2 + delta_k_y ** 2) / 2)
                sinc_k_z_term = (sinc_phys(sinc_arg_sampled)) ** 2
                sinc_omega_term = (sinc_phys(delta_w * self.T_I / 2)) ** 2
                f_val = (self.const_rate_prefactor * signal_prefactor_batch * idler_prefactor
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
            n = np.arange(1, self.max_N_i + 1)
            mean = cumsum / n
            variance = np.zeros_like(mean)
            np.divide(
                cumsum2 - n * mean ** 2,
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
            conv_index = np.where(
                has_convergence,
                np.minimum(
                    first_group + self.n_conv,
                    self.max_N_i - 1
                ), # if it has converged we chose first index from the group we found
                self.max_N_i - 1 # if it hasn't we just take maximum index
            )

            phi_integrals = np.take_along_axis(
                mean,
                conv_index[..., None],
                axis=-1
            )[..., 0]

            signal_integrals = np.mean(phi_integrals, axis=1)

            mc_val.append(signal_integrals)

        return np.concatenate(mc_val)

    def get_transition_rate_parallel(self):
        return

    def get_phase_matching_contour(self):
        """
        Compute transverse phase matching grid.
        :return:
        """
        # Initialize grid
        l = np.asarray(self.lambda_s_grid).ravel()
        t = np.asarray(self.theta_s_grid).ravel()
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
            n_i = self.n_i(self.n_i_1, self.n_i_2, lambda_i, theta_i, self.t)
            # ... and from it the idler vector magnitude
            k_i = 2 * np.pi * n_i / lambda_i

            # Step 3: Get new estimate for theta from transverse momentum conservation
            sin_theta = k_s * np.sin(theta) / k_i
            valid = np.abs(sin_theta) <= 1

            # Step 4: Update theta
            theta_i = np.full_like(theta, np.nan)
            theta_i[valid] = np.arcsin(sin_theta[valid])

        # Now recompute n_i and k_i with the final estimate...
        n_i = self.n_i(self.n_i_1, self.n_i_2, lambda_i, theta_i, self.t)
        k_i = 2 * np.pi * n_i / lambda_i
        # ...and evaluate longitudinal mismatch
        dkz = self.k_p + self.k_m - k_s * np.cos(theta) - k_i * np.cos(theta_i)

        return lam, theta, dkz

    def plot_spectrum(self):
        vals = self.get_transition_rate_vectorized()
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

    def compare_theta_i_solvers(self, n_fixed_1=5, n_fixed_2=30):
        """
        Compare fixed-point iterations with a bracketing root solver for theta_i.

        Plots the signal phase-matching contours Delta k_z = 0 obtained
        from each theta_i solution method.
        """

        lam, theta_s = np.meshgrid(
            self.lambda_s_grid,
            self.theta_s_grid,
            indexing="xy"
        )

        # Signal properties
        n_s = self.n_s_func(
            self.n_s_1,
            self.n_s_2,
            lam,
            theta_s
        )
        k_s = 2 * np.pi * n_s / lam

        # CW energy conservation:
        # 1/lambda_i = 1/lambda_p - 1/lambda_s
        lambda_i = np.full_like(lam, np.nan, dtype=float)

        valid_energy = lam > self.lambda_p
        lambda_i[valid_energy] = (
                self.lambda_p * lam[valid_energy]
                / (lam[valid_energy] - self.lambda_p)
        )

        def n_i_of(lambda_i_array, theta_i_array):
            return self.n_i(
                self.n_i_1,
                self.n_i_2,
                lambda_i_array,
                theta_i_array,
                self.t
            )

        def fixed_point_solution(n_iterations):
            theta_i = np.zeros_like(lam)

            for _ in range(n_iterations):
                n_i = n_i_of(lambda_i, theta_i)
                k_i = 2 * np.pi * n_i / lambda_i

                sin_theta_i = k_s * np.sin(theta_s) / k_i

                valid = (
                        valid_energy
                        & np.isfinite(sin_theta_i)
                        & (np.abs(sin_theta_i) <= 1)
                )

                theta_next = np.full_like(theta_i, np.nan)
                theta_next[valid] = np.arcsin(sin_theta_i[valid])

                theta_i = theta_next

            n_i = n_i_of(lambda_i, theta_i)
            k_i = 2 * np.pi * n_i / lambda_i

            dkz = self.k_p + self.k_m - (
                    k_s * np.cos(theta_s)
            ) - (
                          k_i * np.cos(theta_i)
                  )

            dkz[~np.isfinite(dkz)] = np.nan

            return theta_i, dkz

        def root_solution():
            theta_i = np.full_like(lam, np.nan, dtype=float)

            for index in np.ndindex(lam.shape):
                if not valid_energy[index]:
                    continue

                lambda_s_value = lam[index]
                theta_s_value = theta_s[index]
                lambda_i_value = lambda_i[index]
                k_s_value = k_s[index]

                def equation(theta_i_value):
                    n_i_value = self.n_i(
                        self.n_i_1,
                        self.n_i_2,
                        lambda_i_value,
                        theta_i_value,
                        self.t
                    )
                    k_i_value = 2 * np.pi * n_i_value / lambda_i_value

                    return (
                            k_i_value * np.sin(theta_i_value)
                            - k_s_value * np.sin(theta_s_value)
                    )

                # Search only the forward hemisphere.
                lower = 0.0
                upper = np.pi / 2 - 1e-10

                try:
                    f_lower = equation(lower)
                    f_upper = equation(upper)

                    # A root exists only if the function changes sign.
                    if np.isfinite(f_lower) and np.isfinite(f_upper):
                        if f_lower * f_upper <= 0:
                            theta_i[index] = brentq(
                                equation,
                                lower,
                                upper,
                                xtol=1e-12,
                                rtol=1e-12,
                                maxiter=100
                            )

                except (ValueError, RuntimeError, FloatingPointError):
                    continue

            n_i = n_i_of(lambda_i, theta_i)
            k_i = 2 * np.pi * n_i / lambda_i

            dkz = self.k_p + self.k_m - (
                    k_s * np.cos(theta_s)
            ) - (
                          k_i * np.cos(theta_i)
                  )

            dkz[~np.isfinite(dkz)] = np.nan

            return theta_i, dkz

        # Solve with all three methods.
        theta_i_fixed_1, dkz_fixed_1 = fixed_point_solution(n_fixed_1)
        theta_i_fixed_2, dkz_fixed_2 = fixed_point_solution(n_fixed_2)
        theta_i_root, dkz_root = root_solution()

        # Plot phase matching contours.
        plt.figure(figsize=(10, 6))



        plt.contour(
            lam * 1e9,
            np.rad2deg(theta_s),
            dkz_root,
            levels=[0],
            colors="lime",
            linewidths=2.5,
            linestyles="-"
        )

        plt.contour(
            lam * 1e9,
            np.rad2deg(theta_s),
            dkz_fixed_2,
            levels=[0],
            colors="tab:blue",
            linewidths=2,
            linestyles="-."
        )

        plt.contour(
            lam * 1e9,
            np.rad2deg(theta_s),
            dkz_fixed_1,
            levels=[0],
            colors="tab:red",
            linewidths=2,
            linestyles="--"
        )



        handles = [
            mlines.Line2D(
                [], [],
                color="tab:red",
                linestyle="--",
                linewidth=2,
                label=f"Fixed point: {n_fixed_1} iterations"
            ),
            mlines.Line2D(
                [], [],
                color="tab:blue",
                linestyle="-.",
                linewidth=2,
                label=f"Fixed point: {n_fixed_2} iterations"
            ),
            mlines.Line2D(
                [], [],
                color="lime",
                linestyle="-",
                linewidth=2.5,
                label="Brent root solver"
            )
        ]

        plt.legend(handles=handles)
        plt.xlabel(r"$\lambda_s$ (nm)")
        plt.ylabel(r"$\theta_s$ internal (deg)")
        plt.title(r"Comparison of $\theta_i$ solvers: $\Delta k_z = 0$")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()

        # Quantify agreement at pixels where both methods are valid.
        valid_5 = np.isfinite(theta_i_fixed_1) & np.isfinite(theta_i_root)
        valid_30 = np.isfinite(theta_i_fixed_2) & np.isfinite(theta_i_root)

        if np.any(valid_5):
            error_5 = np.abs(
                theta_i_fixed_1[valid_5]
                - theta_i_root[valid_5]
            )
            print(
                f"{n_fixed_1} iterations: "
                f"maximum |Δtheta_i| = {np.rad2deg(np.max(error_5)):.3e} deg, "
                f"mean |Δtheta_i| = {np.rad2deg(np.mean(error_5)):.3e} deg"
            )

        if np.any(valid_30):
            error_30 = np.abs(
                theta_i_fixed_2[valid_30]
                - theta_i_root[valid_30]
            )
            print(
                f"{n_fixed_2} iterations: "
                f"maximum |Δtheta_i| = {np.rad2deg(np.max(error_30)):.3e} deg, "
                f"mean |Δtheta_i| = {np.rad2deg(np.mean(error_30)):.3e} deg"
            )

        return {
            "lambda_s": lam,
            "theta_s": theta_s,
            "theta_i_fixed_5": theta_i_fixed_1,
            "theta_i_fixed_30": theta_i_fixed_2,
            "theta_i_root": theta_i_root,
            "dkz_fixed_5": dkz_fixed_1,
            "dkz_fixed_30": dkz_fixed_2,
            "dkz_root": dkz_root,
        }


class Imaging():
    """
    Class for simulating the quantum imaging with undetected light setup.
    """
    def __init__(self, lambda_p=532e-6):
        self.lambda_p = lambda_p # pump wavelength in microm

    def simulate(self):
        return