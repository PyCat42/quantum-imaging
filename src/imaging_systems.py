from abc import abstractmethod, ABC

from src.ray_propagation import *

class PropagationResult():
    """Common container for propagation results."""
    def __init__(self, x, y, r, theta):
        self.x = x
        self.y = y
        self.r = r
        self.theta = theta

class ImagingSystem(ABC):
    """
    Common container for imaging system.
    Every new imaging system should inherit this class.
    """
    def __init__(self):
        self.M_idler = None
        self.M_detector = None
        # Each class that inherits this one needs to define its own transfer matrices!

    @abstractmethod
    def build_idler_matrix(self) -> np.ndarray:
        """
        Return the 2x2 ray-transfer matrix from the crystal
        to the object plane.
        """

    @abstractmethod
    def build_detector_matrix(self) -> np.ndarray:
        """
        Return the 2x2 ray-transfer matrix from the crystal
        to the detector plane.
        """

    def propagate_to_plane(self, theta_vac, phi, transfer_matrix):
        """
        Propagates ray with certain starting orientation using
        transfer matrix.
        :param theta_vac: initial polar angle
        :param phi: initial azimuthal angle
        :param transfer_matrix: mathematical representation of mediums and optical elements in path of the ray
        :return: PropagationResult instance
        """
        theta_vac, phi = np.broadcast_arrays(theta_vac, phi)
        original_shape = theta_vac.shape

        theta_flat = theta_vac.ravel()
        phi_flat = phi.ravel()

        r_in_flat = np.zeros_like(theta_flat)

        r_out_flat, theta_out_flat = propagate(
            r_in_flat,
            theta_flat,
            transfer_matrix,
        )

        x_flat, y_flat = get_ray_coordinates(
            r_out_flat,
            phi_flat,
        )

        return PropagationResult(
            x=x_flat.reshape(original_shape),
            y=y_flat.reshape(original_shape),
            r=r_out_flat.reshape(original_shape),
            theta=theta_out_flat.reshape(original_shape),
        )

    def idler_arm(self, lambda_i_vac, theta_i_vac, phi_i):
        """
        Propagates idler ray from crystal to object plane.
        :param lambda_i_vac: initial wavelength
        :param theta_i_vac: initial polar angle
        :param phi_i: initial azimuthal angle
        :return:
        """
        return self.propagate_to_plane(
            theta_vac=theta_i_vac,
            phi=phi_i,
            transfer_matrix=self.M_idler,
        )

    def detector_arm(self, lambda_s_vac, theta_s_vac, phi_s):
        """
        Propagates signal ray from crystal to detector plane.
        :param lambda_s_vac: initial wavelength
        :param theta_s_vac: initial polar angle
        :param phi_s: initial azimuthal angle
        :return:
        """
        return self.propagate_to_plane(
            theta_vac=theta_s_vac,
            phi=phi_s,
            transfer_matrix=self.M_detector,
        )

class Michaelson(ImagingSystem):
    def __init__(self, f_i=0.2, f_d=0.15, ft_1=0.15, ft_2=0.05):
        super().__init__()
        self.f_i = f_i
        self.f_d = f_d
        self.ft_1 = ft_1
        self.ft_2 = ft_2

        # Build specific transfer matrices for this system
        self.M_idler = self.build_idler_matrix()
        self.M_detector = self.build_detector_matrix()

    def build_idler_matrix(self):
        """
        Crystal -> object plane.
        """
        return (
                M_free_space(self.f_i)
                @ M_lens(self.f_i)
                @ M_free_space(self.f_i)
        )

    def build_detector_matrix(self):
        """
        Crystal -> detector plane.
        """
        return (
                M_free_space(self.f_d)
                @ M_lens(self.f_d)
                @ M_free_space(self.f_d + self.ft_1)
                @ M_lens(self.ft_1)
                @ M_free_space(self.ft_1 + self.ft_2)
                @ M_lens(self.ft_2)
                @ M_free_space(self.ft_2)
        )
