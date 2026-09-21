from numpy.lib._stride_tricks_impl import sliding_window_view
from scipy.constants import pi, epsilon_0
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import fftconvolve
from scipy.stats import norm, qmc
from scipy.special import ndtri
from tqdm.auto import tqdm
import multiprocessing as mp

from src.object_funcs import object_phase_bar
from src.sellmeier import *
from src.calculation_helpers import *
from src.MC_helpers import sample_sinc2, sinc_phys, sinc2_approximation
from src.imaging_systems import *

class SPDC():
    """
    Class for simulation of SPDC source.
    """
    def __init__(self, L=1e-3, t=25, period=5.335e-6, m=-1, chi_eff=6e-12,
                 lambda_p=405e-9, n_p=n_z_KTP,
                 P=0.5, omega_0=60e-6, T_I=0,
                 lambda_s_min=625e-9, lambda_s_max=845e-9,
                 theta_s_min=0, theta_s_max=0.06,
                 phi_s_min=0, phi_s_max=2*pi,
                 n_s_1=n_z_KTP, n_s_2=n_z_KTP,
                 n_i_1=n_z_KTP, n_i_2=n_z_KTP,
                 dn_i_1_dlambda=dn_z_KTP_dlambda, dn_i_2_dlambda=dn_z_KTP_dlambda,
                 min_N_i=int(2**5), max_N_i=int(2**10), N_phi=int(2**5),
                 grid_size=int(100), seed=None,
                 eps=1e-12, conv_check=100, min_rel_err=1e-2, min_abs_error=1e-21,
                 signal_batch_size=24, n_conv=2, n_processes=9):
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
        self.u_lambda = (np.arange(self.grid_size) + 0.5) / self.grid_size
        self.u_theta = (np.arange(self.grid_size) + 0.5) / self.grid_size

        self.lambda_s_grid = self.lambda_s_min + self.u_lambda * (self.lambda_s_max - self.lambda_s_min)
        self.theta_s_grid = self.theta_s_min + self.u_theta * (self.theta_s_max - self.theta_s_min)

        self.lambda_mesh, self.theta_mesh = np.meshgrid(self.lambda_s_grid, self.theta_s_grid)
        self.lambda_s = self.lambda_mesh.flatten()
        self.theta_s = self.theta_mesh.flatten()

        """
        # ... or uniform sampling
        u = (np.arange(self.grid_size) + 0.5) / self.grid_size
        u_max = np.sin(self.theta_s_max) ** 2
        self.theta_s_grid = np.arcsin(np.sqrt(u * u_max))
        """

        # Calculate signal params on grid
        self.n_s = n_eff(self.n_s_1, self.n_s_2, self.lambda_s, self.theta_s, self.t) # signal refractive index
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
        #self.phi_s = np.linspace(self.phi_s_min, self.phi_s_max, self.N_phi, endpoint=False)
        self.phi_s = (
                self.phi_s_min
                + (np.arange(self.N_phi) + 0.5)
                * (self.phi_s_max - self.phi_s_min)
                / self.N_phi
        )
        self.sin_phi = np.sin(self.phi_s)
        self.cos_phi = np.cos(self.phi_s)

        # Early stopping
        self.eps = eps  # numerical tolerance
        self.conv_check = conv_check  # check for convergence at each conv_check iterations
        self.n_conv = n_conv
        self.min_rel_err = min_rel_err  # minimal change in relative dR error that is considered as such
        self.min_abs_error = min_abs_error  # minimal absolute change in dR that is considered as such

        # Parallelization
        self.signal_batch_size = signal_batch_size # number of signal samples that are run on the same process
        self.signal_batches = [
            (start, min(start + self.signal_batch_size, self.N_s))
            for start in range(0, self.N_s, self.signal_batch_size)
        ]
        self.n_processes = n_processes

        self.find_collinear_wavelengths()

    def find_collinear_wavelengths(self, contour_grid_size=1000):
        # Initialize grid
        contour_lambda_grid = np.linspace(self.lambda_p + 100e-9, 4 * self.lambda_p, contour_grid_size)
        contour_theta_grid = np.linspace(self.theta_s_min, self.theta_s_max, contour_grid_size)
        l = np.asarray(contour_lambda_grid).ravel()
        t = np.asarray(contour_theta_grid).ravel()
        lam, theta = np.meshgrid(l, t, indexing="xy")

        # Signal refractive index and wavevector (vectorized in the right shape)
        n_s = n_eff(self.n_s_1, self.n_s_2, lam, theta, self.t)
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

        if len(intersections) == 0:
            self.lambda_s_central = None
            self.lambda_i_central = None
        elif len(intersections) == 1:
            self.lambda_s_central = intersections[0]
            self.lambda_i_central = intersections[0]
        else:
            self.lambda_s_central = intersections[0]
            self.lambda_i_central = intersections[1]

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
                jacobian = (n_i - lambda_i * d_n_i_dlambda_func(lambda_i, theta_i)) / (np.abs(np.cos(theta_i)) * c)
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
        theta_s_batch = self.theta_s[start:end]
        sin_theta = np.sin(theta_s_batch)[:, None]
        signal_prefactor_batch = self.signal_prefactor[start:end]

        # Spherical to cartesian - faster than calling function lots of times
        k_sx = k_s_batch[:, None] * sin_theta * self.cos_phi[None, :]  # (signal_batch_size, N_phi)
        k_sy = k_s_batch[:, None] * sin_theta * self.sin_phi[None, :]  # (signal_batch_size, N_phi)
        k_sz = k_s_batch * np.cos(theta_s_batch)  # (signal_batch_size, ) - we don't store repeated values of k_sz!

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
            _, _, phi_i = cartesian_to_spherical(k_ix, k_iy, k_iz)
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
            jacobian = (n_i - lambda_i * d_n_i_dlambda_func(lambda_i, theta_i)) / (np.abs(np.cos(theta_i)) * c)
            # - calculate other prefactors
            idler_prefactor = jacobian * w_i / n_i ** 2
            gauss_term = np.exp(- self.omega_0 ** 2 * (delta_k_x ** 2 + delta_k_y ** 2) / 2)
            sinc_k_z_term = (sinc_phys(delta_k_z * self.L / 2)) ** 2
            f_val = (self.const_rate_prefactor * signal_prefactor_batch[:, None, None] * idler_prefactor
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

        return f_val, p_val, valid, lambda_i, theta_i, phi_i

    def target_sampler(self, lambda_s, theta_s=0.0, phi_s=0.0, N_i=None, sobol_seed=None, kernel_only=False):
        n_s = n_eff(self.n_s_1, self.n_s_2, lambda_s, theta_s, self.t)

        k_s = 2 * np.pi * n_s / lambda_s
        w_s = 2 * np.pi * c / lambda_s

        k_sx = k_s * np.sin(theta_s) * np.cos(phi_s)
        k_sy = k_s * np.sin(theta_s) * np.sin(phi_s)
        k_sz = k_s * np.cos(theta_s)

        # Sample maximal numer of idlers
        if N_i is None:
            N_i = self.max_N_i
        idler_sampler = qmc.Sobol(
            d=4,
            scramble=True,
            seed=sobol_seed
        )
        idler_points = idler_sampler.random_base2(
            m=int(np.log2(N_i))
        )

        # Calculate transversal momentum mismatch
        delta_k_x = self.gauss_scale * ndtri(idler_points[:, 0])
        delta_k_y = self.gauss_scale * ndtri(idler_points[:, 1])

        # Calculate transversal idler momentum components
        k_ix = -delta_k_x - k_sx
        k_iy = -delta_k_y - k_sy
        k_i_normal = np.hypot(k_ix, k_iy)

        # Retain adaptive stopping without processing one batch at a time (i.e. eliminate idler loop)
        # Generate all samples and calculate the weighted values for all 512 samples in one vectorized operation.
        # Idler refraction index function
        n_i_func = lambda wavelength, theta_i: n_eff(self.n_i_1, self.n_i_2, wavelength, theta_i, self.t)
        d_n_i_dlambda_func = lambda wavelength, theta_i: (
            dn_eff_dlambda(self.n_i_1, self.dn_i_1_dlambda,
                           self.n_i_2, self.dn_i_2_dlambda,
                           wavelength, theta_i, self.t)
        )

        # Any multiplicative factor depending only on the fixed signal state cancels
        # That means that in this version we do not use self.const_rate_prefactor and self.signal_prefactor
        if self.T_I == 0:
            # CW pump => force energy conservation
            w_i = self.w_p - w_s
            lambda_i = 2 * np.pi * c / w_i

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
            _, _, phi_i = cartesian_to_spherical(k_ix, k_iy, k_iz)
            # ...then get longitudinal momentum mismatch
            delta_k_z = np.full_like(k_iz2, np.nan)
            delta_k_z = self.k_p - k_sz - k_iz + self.k_m
            delta_k_z[~valid] = np.nan

            # calculate integrand function value
            # - calculate Jacobian = 1 / (d(omega) / d(k_iz)) analytically
            # k = 2 * pi * n(lambda, theta) / lambda
            # fixed theta: dk/dlambda = 2 * pi * ((1 / lambda) * (dn/dlambda) - n / lambda**2)
            # w = 2 * pi * c / lambda
            # dw/dlambda = - 2 * pi * c / lambda**2
            # => dw/dk = c / (n - lambda * dn/dlambda)
            jacobian = (n_i - lambda_i * d_n_i_dlambda_func(lambda_i, theta_i)) / (np.abs(np.cos(theta_i)) * c)
            # - calculate other prefactors
            idler_prefactor = jacobian * w_i / n_i ** 2
            gauss_term = np.exp(- self.omega_0 ** 2 * (delta_k_x ** 2 + delta_k_y ** 2) / 2)
            sinc_k_z_term = (sinc_phys(delta_k_z * self.L / 2)) ** 2
            f_val = idler_prefactor * sinc_k_z_term * gauss_term

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
            delta_w = self.w_p - w_s - w_i

            # Calculate other prefactors
            idler_prefactor = w_i / n_i ** 2
            gauss_term = np.exp(- self.omega_0 ** 2 * (delta_k_x ** 2 + delta_k_y ** 2) / 2)
            sinc_k_z_term = (sinc_phys(sinc_arg_sampled)) ** 2
            sinc_omega_term = (sinc_phys(delta_w * self.T_I / 2)) ** 2
            f_val = idler_prefactor * sinc_k_z_term * gauss_term * sinc_omega_term

            # Get MC weight
            p_val = (norm.pdf(delta_k_x, loc=0, scale=self.gauss_scale)
                     * norm.pdf(delta_k_y, loc=0, scale=self.gauss_scale)
                     * sinc2_approximation(sinc_arg_sampled) * self.L / 2)

            valid = (
                    np.isfinite(f_val)
                    & np.isfinite(p_val)
                    & (p_val > 1e-300)
            )

        if not kernel_only:
            signal_prefactor = k_s ** 4 * w_s * np.sin(theta_s) / n_s ** 2
            f_val *= (self.const_rate_prefactor * signal_prefactor)

        return f_val, p_val, valid, lambda_i, theta_i, phi_i

    def phase_matched_wavelength_guess(
            self,
            theta_target=0.0,
            contour_grid_size=2000,
    ):
        # Get rough estimate on where dk_z = 0 for given theta
        lam, theta, dkz = self.get_phase_matching_contour(
            contour_grid_size=contour_grid_size,
        )

        theta_values = theta[:, 0]
        # Find theta closest to target
        theta_index = np.argmin(
            np.abs(theta_values - theta_target)
        )

        # extract lambda and delta_k rows for this theta
        lam_row = lam[theta_index, :]
        dkz_row = dkz[theta_index, :]

        branch_mask = (
                np.isfinite(dkz_row)
                & (lam_row < 2 * self.lambda_p)
        )
        if not np.any(branch_mask):
            raise RuntimeError(
                "No valid phase-matching points were found "
                "at the requested theta_target."
            )

        valid_indices = np.flatnonzero(branch_mask)

        # Use index where dk_z is minimal
        minimum_index = valid_indices[
            np.argmin(np.abs(dkz_row[branch_mask]))
        ]

        lambda_guess = lam_row[minimum_index]
        dkz_at_guess = dkz_row[minimum_index]
        theta_used = theta_values[theta_index]

        return lambda_guess, theta_used, dkz_at_guess, lam_row, dkz_row

    def get_phase_matching_peak(
            self,
            theta_target=0.0,
            contour_grid_size=1000,
            coarse_points=61,
            fine_points=101,
            N_i=2 ** 12,
            seed=None,
    ):
        """
        Refine a phase-matching wavelength estimate by maximizing
        the numerical transition-rate estimate.
        """
        lambda_guess, theta, *_ = self.phase_matched_wavelength_guess(
            theta_target=theta_target, contour_grid_size=contour_grid_size
        )
        half_width_theta = self.theta_s_min + 0.5 * (self.theta_s_max - self.theta_s_min) / self.grid_size
        if theta_target == 0.0:
            theta_target += half_width_theta

        half_width_lambda = self.lambda_s_min + 0.5 * (self.lambda_s_max - self.lambda_s_min) / self.grid_size
        lambda_lo = max(
            self.lambda_s_min,
            lambda_guess - half_width_lambda,
        )
        lambda_hi = min(
            self.lambda_s_max,
            lambda_guess + half_width_lambda,
        )

        if lambda_lo >= lambda_hi:
            raise ValueError(
                "Invalid wavelength refinement interval."
            )

        rng = np.random.default_rng(seed)

        def estimate_rate(lambda_s):
            local_seed = int(
                rng.integers(0, 2 ** 32 - 1)
            )

            f_val, p_val, valid, *_ = self.target_sampler(
                lambda_s=lambda_s,
                theta_s=theta_target,
                phi_s=0.0,
                N_i=N_i,
                sobol_seed=local_seed,
                kernel_only=False,
            )

            weighted_val = np.divide(
                f_val,
                p_val,
                out=np.zeros_like(f_val),
                where=valid,
            )

            return np.mean(weighted_val)

        lambda_coarse = np.linspace(
            lambda_lo,
            lambda_hi,
            coarse_points,
        )

        rates_coarse = np.array([
            estimate_rate(lambda_s)
            for lambda_s in lambda_coarse
        ])

        i_coarse = np.argmax(rates_coarse)

        delta_coarse = (
                lambda_coarse[1] - lambda_coarse[0]
        )

        lambda_fine_lo = max(
            self.lambda_s_min,
            lambda_coarse[i_coarse] - 3 * delta_coarse,
        )

        lambda_fine_hi = min(
            self.lambda_s_max,
            lambda_coarse[i_coarse] + 3 * delta_coarse,
        )

        lambda_fine = np.linspace(
            lambda_fine_lo,
            lambda_fine_hi,
            fine_points,
        )

        rates_fine = np.array([
            estimate_rate(lambda_s)
            for lambda_s in lambda_fine
        ])

        i_fine = np.argmax(rates_fine)

        return lambda_fine[i_fine]

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
        f_val, p_val, valid, _, _, _ = self.sampler(args)

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
        #signal_integrals = np.mean(phi_integrals, axis=1)

        dphi = (self.phi_s_max - self.phi_s_min) / self.N_phi
        signal_integrals = dphi * np.sum(phi_integrals, axis=1)

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

    def get_phase_matching_contour(self, contour_grid_size=500):
        """
        Compute transverse phase matching grid.
        :param contour_grid_size: grid size - contour has more point by default, no matter how fine spectrum grid is
        :return:
        """
        # Initialize grid
        contour_lambda_grid = np.linspace(self.lambda_s_min, self.lambda_s_max, contour_grid_size)
        contour_theta_grid = np.linspace(self.theta_s_min, self.theta_s_max, contour_grid_size)
        l = np.asarray(contour_lambda_grid).ravel()
        t = np.asarray(contour_theta_grid).ravel()
        lam, theta = np.meshgrid(l, t, indexing="xy")

        # Signal refractive index and wavevector (vectorized in the right shape)
        n_s = n_eff(self.n_s_1, self.n_s_2, lam, theta, self.t)
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


class Imaging():
    """
    Class for simulating the quantum imaging with undetected light setup.
    """
    def __init__(self, source=SPDC(), imaging_system=Michaelson(source=SPDC()),
                 detector_size=512e-6, detector_pixels=1000,
                 object_func=object_phase_bar, object_plane_size=10e-3, object_plane_pixels=1000,
                 signal_batch_size=24, n_processes=9, seed=None):
        # ------ SPDC source -------
        self.source = source # SPDC class instance
        self.imaging_system = imaging_system # class instance for imaging system
        self.lambda_s_grid = np.linspace(source.lambda_s_min, source.lambda_s_max, source.grid_size)
        #self.dlambda_s = source.lambda_s_max - source.lambda_s_min / (source.grid_size - 1)
        self.theta_s_grid = np.linspace(source.theta_s_min, source.theta_s_max, source.grid_size)
        #self.dtheta_s = source.theta_s_max - source.theta_s_min / (source.grid_size - 1)
        self.dphi_s = (source.phi_s_max - source.phi_s_min) / source.N_phi

        self.lambda_s_central = (source.lambda_s_max + source.lambda_s_min) / 2
        n_s_central = n_eff(source.n_s_1, source.n_s_2, self.lambda_s_central, 0.0, source.t)
        self.k_s_central = 2 * np.pi * n_s_central / self.lambda_s_central

        self.dlambda_s = (source.lambda_s_max - source.lambda_s_min) / source.grid_size
        self.dtheta_s = (source.theta_s_max - source.theta_s_min) / source.grid_size

        self.grid_weight_s = self.dlambda_s * self.dtheta_s * self.dphi_s

        # ------ Detector ------
        # Detector is centered at (x,y) = (0,0)
        self.detector_size = detector_size # size of detector in m
        self.half_detector_size = self.detector_size / 2
        # ...it is a square of size detector_size x detector_size
        self.detector_pixels = detector_pixels # number of detector pixels
        # ...and it contains detector_pixels x detector_pixels pixels
        self.pixel_size = self.detector_size / self.detector_pixels # size of a pixel in m
        self.half_pixel_size = self.pixel_size / 2
        self.pixel_area = self.pixel_size**2
        # Detector grid
        self.x_edges = np.linspace(-self.detector_size / 2, self.detector_size / 2, self.detector_pixels + 1)
        self.y_edges = np.linspace(-self.detector_size / 2, self.detector_size / 2, self.detector_pixels + 1)
        self.x_mesh, self.y_mesh = np.meshgrid(self.x_edges, self.y_edges)
        self.x_grid = self.x_mesh.flatten()
        self.y_grid = self.y_mesh.flatten()
        self.x_centers = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        self.y_centers = 0.5 * (self.y_edges[:-1] + self.y_edges[1:])

        # ------ Object -------
        self.object_func = object_func # function that mathematically represents object properties
        # Object plane representation
        self.object_plane_size = 2 * detector_size / abs(imaging_system.magnification())
        self.object_plane_pixels = object_plane_pixels
        self.object_pixel_size = self.object_plane_size / self.object_plane_pixels
        self.object_x_edges = np.linspace(-self.object_plane_size / 2, self.object_plane_size / 2, self.object_plane_pixels + 1)
        self.object_y_edges = np.linspace(-self.object_plane_size / 2, self.object_plane_size / 2, self.object_plane_pixels + 1)
        self.object_x_centers = 0.5 * (self.object_x_edges[:-1] + self.object_x_edges[1:])
        self.object_y_centers = 0.5 * (self.object_y_edges[:-1] + self.object_y_edges[1:])

        # ------ Parallelization --------
        self.signal_batch_size = signal_batch_size  # number of signal samples that are run on the same process
        self.signal_batches = [
            (start, min(start + self.signal_batch_size, self.source.N_s))
            for start in range(0, self.source.N_s, self.signal_batch_size)
        ]
        self.n_batches = len(self.signal_batches)
        self.n_processes = n_processes # number of processes to which calculation is distributed
        self.seed = seed # seed sample generation
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def simulation_batch(self, args):
        src = self.source
        img_sys = self.imaging_system
        batch_index, start, end, sobol_seed = args

        lambda_s_batch = src.lambda_s[start:end]
        theta_s_batch = src.theta_s[start:end]

        # Loop over signals and perform integration over phi and then also idlers
        f_val, p_val, valid, lambda_i, theta_i, phi_i = src.sampler(args)

        # Apply transition from crystal (medium) to vacuum and propagate idler to the object
        n_i_func = lambda wavelength, theta: n_eff(src.n_i_1, src.n_i_2, wavelength, theta, src.t)
        lambda_i_vac, theta_i_vac = medium_to_air(lambda_i, theta_i, n_i_func)

        # Calculate coordinates after the propagation of idler to the object
        idler_result = img_sys.idler_arm(lambda_i_vac, theta_i_vac, phi_i)
        x_o = idler_result.x
        y_o = idler_result.y

        # Get object transmission coefficient and phase from object function
        t_o, phi_o = self.object_func(x_o, y_o)

        # Add to the MC sum
        weighted_val = np.zeros_like(f_val)
        weighted_val = np.divide(
            f_val,
            p_val,
            out=weighted_val,
            where=valid
        )

        # Constructive interference
        weighted_val_plus = weighted_val * (1 + (t_o * np.cos(phi_o)))
        mean_plus, conv_index_plus = src.adaptive_stopping(weighted_val_plus)
        phi_integrals_plus = np.take_along_axis(
            mean_plus,
            conv_index_plus[..., None],
            axis=-1
        )[..., 0]
        detector_values_plus = self.grid_weight_s * phi_integrals_plus

        # Destructive interference
        weighted_val_minus = weighted_val * (1 - (t_o * np.cos(phi_o)))
        mean_minus, conv_index_minus = src.adaptive_stopping(weighted_val_minus)
        phi_integrals_minus = np.take_along_axis(
            mean_minus,
            conv_index_minus[..., None],
            axis=-1
        )[..., 0]
        detector_values_minus = self.grid_weight_s * phi_integrals_minus

        # Propagate signal to detector
        n_s_func = lambda wavelength, theta: n_eff(src.n_s_1, src.n_s_2, wavelength, theta, src.t)
        lambda_s_vac, theta_s_vac = medium_to_air(
            lambda_s_batch,
            theta_s_batch,
            n_s_func
        )

        n_batch = end - start
        n_phi = src.N_phi

        lambda_s_vac_2d = np.broadcast_to(
            lambda_s_vac[:, None],
            (n_batch, n_phi),
        )
        theta_s_vac_2d = np.broadcast_to(
            theta_s_vac[:, None],
            (n_batch, n_phi),
        )
        phi_s_2d = np.broadcast_to(
            src.phi_s[None, :],
            (n_batch, n_phi),
        )

        # Propagate all signal directions to the detector.
        # Every result has shape: (N_batch, N_phi)
        detector_result = img_sys.detector_arm(lambda_s_vac_2d, theta_s_vac_2d, phi_s_2d)
        x_d = detector_result.x
        y_d = detector_result.y

        # `bin_detector` expects 1D arrays - flatten all arrays
        return (
            batch_index,
            detector_values_plus.ravel(),
            detector_values_minus.ravel(),
            x_o.ravel(),
            y_o.ravel(),
            phi_o.ravel(),
            weighted_val.ravel(),
            x_d.ravel(),
            y_d.ravel()
        )

    def bin_samples(self, image, counts, x, y, vals, size, pixels):
        pixel_size = size / pixels
        half_size = size / 2

        # Shift left edge to 0
        x_shifted = x + half_size
        y_shifted = y + half_size

        # Divide by pixel size to get which pixel this signal propagates to
        # and convert to int pixel index
        px = np.floor(x_shifted / pixel_size).astype(int)
        py = np.floor(y_shifted / pixel_size).astype(int)

        # Valid pixels
        valid = (
                (px >= 0) & (px < pixels) &
                (py >= 0) & (py < pixels)
        )

        # Add to the image
        # np.add.at(A, indices, values) - safe adding
        # - A is the array you modify in place,
        # - indices is a tuple of index arrays, one per axis,
        # - values is broadcastable to the shape of indices and represents values to be added to A.
        np.add.at(
            image,
            (py[valid], px[valid]),
            vals[valid]
        )

        # Track which pixels actually received hits
        np.add.at(
            counts,
            (py[valid], px[valid]),
            1
        )

    def full_simulation(self):
        # Generate seeds ONLY in the parent process.
        if self.seed is not None:
            sobol_seeds = [
                int(self.rng.integers(0, 2 ** 32 - 1))
                for _ in range(self.n_batches)
            ]
        else:
            sobol_seeds = [None] * self.n_batches

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

        plus_image = np.zeros((self.detector_pixels, self.detector_pixels))
        minus_image = np.zeros_like(plus_image)
        object_plane = np.zeros((self.object_plane_pixels, self.object_plane_pixels))

        plus_counts = np.zeros_like(plus_image, dtype=int)
        minus_counts = np.zeros_like(minus_image, dtype=int)
        object_plane_counts = np.zeros_like(object_plane, dtype=int)

        """
        object_x_samples = []
        object_y_samples = []
        object_phi_samples = []
        """
        with (mp.Pool(processes=self.n_processes) as pool):
            for result in tqdm(
                    pool.imap_unordered(self.simulation_batch, tasks),
                    total=self.n_batches,
                    desc="Signal batches",
            ):
                batch_index, detector_values_plus, detector_values_minus, x_o, y_o, phi_o, weights_o, x_d, y_d = result

                """
                object_x_samples.append(x_o)
                object_y_samples.append(y_o)
                object_phi_samples.append(phi_o)
                """

                # Bin this batch's contributions
                self.bin_samples(plus_image, plus_counts, x_d, y_d, detector_values_plus,
                                 self.detector_size, self.detector_pixels)
                self.bin_samples(minus_image, minus_counts, x_d, y_d, detector_values_minus,
                                 self.detector_size, self.detector_pixels)
                self.bin_samples(object_plane, object_plane_counts, x_o, y_o, weights_o,
                                 self.object_plane_size, self.object_plane_pixels)

        """
        x_o = np.concatenate(object_x_samples)
        y_o = np.concatenate(object_y_samples)
        phi_o = np.concatenate(object_phi_samples)
        """

        plus_image[plus_counts == 0] = np.nan
        minus_image[minus_counts == 0] = np.nan

        denominator = plus_image + minus_image

        max_denominator = np.nanmax(denominator)
        min_signal = 1e-30 * max_denominator

        valid_visibility = (
                np.isfinite(plus_image)
                & np.isfinite(minus_image)
                & (denominator > min_signal)
        )

        det_visibility = np.full_like(
            denominator,
            np.nan,
            dtype=float,
        )

        det_visibility[valid_visibility] = ((plus_image[valid_visibility] - minus_image[valid_visibility])
                                            / denominator[valid_visibility])

        return plus_image, minus_image, det_visibility

    def get_kernel(self):
        src = self.source
        img_sys = self.imaging_system

        # Loop over signals and perform integration over phi and then also idlers
        theta_s = 0.0
        lambda_s = src.get_phase_matching_peak(
            theta_target=theta_s, contour_grid_size=1000,
            coarse_points=61, fine_points=101, N_i=2 ** 12, seed=None
        )
        f_val, p_val, valid, lambda_i, theta_i, phi_i = src.target_sampler(
            src.lambda_s_central, theta_s=0.0, phi_s=0.0, N_i=2**15, sobol_seed=None, kernel_only=True
        )

        # Apply transition from crystal (medium) to vacuum and propagate idler to the object
        n_i_func = lambda wavelength, theta: n_eff(src.n_i_1, src.n_i_2, wavelength, theta, src.t)
        lambda_i_vac, theta_i_vac = medium_to_air(lambda_i, theta_i, n_i_func)

        # Calculate coordinates after the propagation of idler to the object
        # For a centered and aligned imaging system, the corresponding ideal object coordinate is approximately (0,0)
        # Thus the idler coordinates themselves are already the relative coordinates
        idler_result = img_sys.idler_arm(lambda_i_vac, theta_i_vac, phi_i)
        x_o = idler_result.x
        y_o = idler_result.y

        n_s_func = lambda wavelength, theta: n_eff(src.n_s_1, src.n_s_2, wavelength, theta, src.t)
        lambda_s_vac, theta_s_vac = medium_to_air(src.lambda_s_central, theta_s, n_s_func)

        # Propagate all signal directions to the detector.
        # Every result has shape: (N_batch, N_phi)
        detector_result = img_sys.detector_arm(lambda_s_vac, theta_s_vac, phi_s=0.0)
        x_d = detector_result.x
        y_d = detector_result.y

        # Use central pixel
        kernel_detector_radius = 2 * self.pixel_size
        central_signal_mask = (
                np.isfinite(x_d)
                & np.isfinite(y_d)
                & (np.abs(x_d) <= kernel_detector_radius)
                & (np.abs(y_d) <= kernel_detector_radius)
        )

        # Create and normalize kernel
        # We do not include object transmission or interference terms when building the kernel
        # The kernel represents the system response before applying an object
        weighted_val = np.divide(
            f_val,
            p_val,
            out=np.zeros_like(f_val),
            where=valid,
        )

        kernel_mask = (
                central_signal_mask
                & valid
                & np.isfinite(x_o)
                & np.isfinite(y_o)
                & np.isfinite(weighted_val)
        )

        kernel_xy, _, _ = np.histogram2d(
            x_o[kernel_mask],
            y_o[kernel_mask],
            bins=[self.object_x_edges, self.object_y_edges],
            weights=weighted_val[kernel_mask],
        )
        kernel = kernel_xy.T

        # Normalize kernel
        kernel = kernel / kernel.sum()

        # Center kernel (fftconvolve will assume this!)
        yy, xx = np.indices(kernel.shape)

        center_of_mass_x = np.sum(xx * kernel)
        center_of_mass_y = np.sum(yy * kernel)

        target_x = kernel.shape[1] // 2
        target_y = kernel.shape[0] // 2

        shift_x = int(np.rint(target_x - center_of_mass_x))
        shift_y = int(np.rint(target_y - center_of_mass_y))

        kernel = np.roll(
            kernel,
            shift=(shift_y, shift_x),
            axis=(0, 1),
        )

        return kernel

    def convolution_method_simulation(self, object_func, kernel):
        # Get object transmission coefficient and phase from object function
        x_obj, y_obj = np.meshgrid(
            self.object_x_centers,
            self.object_y_centers,
            indexing="xy",
        )
        t_o, phi_o = object_func(x_obj, y_obj)
        obj = t_o * np.cos(phi_o)

        # Get object image
        # Use padding because fftconvolve assumes 0 outside array
        pad_y = kernel.shape[0] // 2
        pad_x = kernel.shape[1] // 2

        obj_padded = np.pad(
            obj,
            ((pad_y, pad_y), (pad_x, pad_x)),
            mode="constant",
            constant_values=1.0,
        )

        visibility_padded = fftconvolve(
            obj_padded,
            kernel,
            mode="same",
        )

        t_i = 1.0
        visibility_object = t_i * visibility_padded[
            pad_y:pad_y + obj.shape[0],
            pad_x:pad_x + obj.shape[1],
        ]

        return visibility_object

    def convolve_on_extended_object_plane(
            self,
            object_func,
            kernel,
            extension_factor=2,
            t_i=1.0,
    ):
        """
        Simulates how the object actually extends beyond imaging plane.
        Introduces more natural padding for convolution method.
        :param object_func:
        :param kernel:
        :param extension_factor:
        :param exterior_contrast:
        :param t_i:
        :return:
        """
        if extension_factor < 1:
            raise ValueError(
                "extension_factor must be at least 1."
            )

        kernel = np.asarray(
            kernel,
            dtype=float,
        )

        expected_shape = (
            self.object_plane_pixels,
            self.object_plane_pixels,
        )

        if kernel.shape != expected_shape:
            raise ValueError(
                f"kernel has shape {kernel.shape}, but expected "
                f"{expected_shape}."
            )

        kernel = np.nan_to_num(
            kernel,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if np.any(kernel < 0):
            raise ValueError("kernel must be non-negative.")

        kernel_sum = kernel.sum()

        if kernel_sum <= 0:
            raise ValueError("kernel has zero total weight.")

        kernel = kernel / kernel_sum

        n_original = self.object_plane_pixels
        n_extended = extension_factor * n_original

        extra_pixels = n_extended - n_original

        pad_before = extra_pixels // 2
        pad_after = extra_pixels - pad_before

        crop_start = pad_before
        crop_stop = crop_start + n_original

        # Keep the same physical object-plane pixel size
        pixel_size = self.object_pixel_size

        x_extended_centers = (np.arange(n_extended) - (n_extended - 1) / 2) * pixel_size
        y_extended_centers = (np.arange(n_extended) - (n_extended - 1) / 2) * pixel_size

        x_extended, y_extended = np.meshgrid(
            x_extended_centers,
            y_extended_centers,
            indexing="xy",
        )

        t_o, phi_o = object_func(x_extended, y_extended)
        t_o = np.asarray(t_o, dtype=float)
        phi_o = np.asarray(phi_o, dtype=float)

        # Check shapes
        if t_o.shape != (n_extended, n_extended):
            raise ValueError("object_func returned an unexpected transmission shape.")
        if phi_o.shape != t_o.shape:
            raise ValueError("object_func returned phase with a mismatched shape.")

        object_contrast = (t_o * np.cos(phi_o))
        if not np.all(np.isfinite(object_contrast)):
            raise ValueError("Object contrast contains non-finite values.")

        # The object function itself is evaluated across the large domain.
        # No extra padding is required before convolution because the later
        # crop removes the region influenced by the extended-grid boundary.
        visibility_extended = t_i * fftconvolve(
            object_contrast,
            kernel,
            mode="same",
        )

        visibility_cropped = visibility_extended[
            crop_start:crop_stop,
            crop_start:crop_stop,
        ]

        return visibility_cropped, {
            "visibility_extended": visibility_extended,
            "object_contrast_extended": object_contrast,
            "x_extended_centers": x_extended_centers,
            "y_extended_centers": y_extended_centers,
            "crop_slice": (
                slice(crop_start, crop_stop),
                slice(crop_start, crop_stop),
            ),
        }

    def project_object_visibility_to_detector(
            self,
            visibility_object,
            magnification=None,
            fill_value=np.nan,
    ):
        """
        Map an object-plane convolution image to the detector grid using:

            x_det = M * x_obj
            y_det = M * y_obj.

        Therefore, each detector-grid coordinate is evaluated at:

            x_obj = x_det / M
            y_obj = y_det / M.
        """
        if magnification is None:
            magnification = self.imaging_system.magnification()

        if magnification == 0:
            raise ValueError(
                "Magnification must not be zero."
            )

        visibility_object = np.asarray(
            visibility_object,
            dtype=float,
        )

        expected_shape = (
            self.object_plane_pixels,
            self.object_plane_pixels,
        )

        if visibility_object.shape != expected_shape:
            raise ValueError(
                f"visibility_object has shape "
                f"{visibility_object.shape}, expected "
                f"{expected_shape}."
            )

        interpolator = RegularGridInterpolator(
            (
                self.object_y_centers,
                self.object_x_centers,
            ),
            visibility_object,
            bounds_error=False,
            fill_value=fill_value,
        )

        x_det, y_det = np.meshgrid(
            self.x_centers,
            self.y_centers,
            indexing="xy",
        )

        x_obj = x_det / magnification
        y_obj = y_det / magnification

        sample_points = np.column_stack(
            (
                y_obj.ravel(),
                x_obj.ravel(),
            )
        )

        visibility_detector = interpolator(
            sample_points,
        ).reshape(
            self.detector_pixels,
            self.detector_pixels,
        )

        return visibility_detector