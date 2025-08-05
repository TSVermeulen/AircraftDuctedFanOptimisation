"""
Parameterisations
=================

Description
-----------
This module provides methods for calculating the profile coordinates (x,y)
based on a Bezier-Parsec (BP) 3434 parameterisation. It includes methods for
calculating the leading edge and trailing edge thickness and camber
distributions, obtaining reference thickness and camber distributions from a
reference airfoil shape, and extracting key parameters from these distributions.

Additionally, a non-linear least-squares minimization method is included to
obtain the parameterisation for a given input coordinate file.

Classes
-------
AirfoilParameterisation
    A class to calculate airfoil parameterisations using Bezier curves. This
    module provides a class for airfoil parameterisation using Bezier curves.
    Contains full parameterisation, including fitting the parameterisation to
    an existing airfoil coordinate file.

Examples
--------
>>> inputfile = r'Test Airfoils\n0015.dat'
>>> call_class = AirfoilParameterisation()
>>> coefficients = call_class.FindInitialParameterisation(inputfile)

Notes
-----
The generation of a BP3434 parameterisation from a reference airfoil is prone
to errors/invalid parameterisations. It is important to (visually) check the
obtained parameterisation, and to confirm the control points fall within
0 <= x/c <= 1.

When executing the file as a standalone, obtaining the initial parameterisation
will take anywhere between 30 second to a minute, depending on the closeness of
the initial guess.

References
----------
The BP 3434 parameterisation is documented in
https://www.sciencedirect.com/science/article/abs/pii/S0965997810000529.
The derivation of the method, including reasoning for each of the bounds/inputs,
is included in the PhD Thesis of Tim Rogalsky:
"Acceleration of Differential Evolution for Aerodynamic Design",
dated March 2004

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID 4995309
Version: 1.3.1
Date [dd-mm-yyyy]: 24-06-2025

Changelog:
- V1.1: Updated with comments from coderabbitAI.
- V1.0: Adapted version of a parameterisation using only the bezier coefficients
        to give an improved fit to the reference data.
- V1.1: Updated docstring, and working implementation for symmetric profiles
        (i.e. zero camber)
- V1.2: Updated FindInitialparameterisation method to use SLSQP optimization
        rather than least squares to enable correct constraint handling.
- V1.2.1: Previously increased the number of points in the u-vectors for the
          bezier curves to 200. This yields too many in the walls.xxx input file
          for MTSET to handle, causing a crash. Number of points has been
          reduced to 100.
- V1.3: Fixed type hinting. Fixed issue in internal handling/definition of
        b_15 & b_17. Improved accuracy. Updated documentation.
        Refactored findInitialparameterisation.
- V1.3.1: Implemented get_figsize method to ensure plot sizes are consistent for LaTeX thesis document.
"""

# Import standard libraries
from pathlib import Path

# Import 3rd party libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, optimize


def _cot(angle: float) -> float:
    """Cotangent function."""
    return 1 / np.tan(angle)


MAJOR_GRID_ALPHA = 0.6
MINOR_GRID_ALPHA = 0.4

class AirfoilParameterisation:
    """
    This class calculates airfoil parameterisations using Bezier curves.

    It provides methods to calculate the leading edge and trailing edge
    thickness and camber distributions using Bezier curves. It also includes
    methods to obtain reference thickness and camber distributions from a
    reference airfoil shape and to extract key parameters from these
    distributions.

    Methods
    -------
    - BezierCurve3(coeff, u)
        Calculate a 3rd degree Bezier curve.
    - BezierCurve4(coeff, u)
        Calculate a 4th degree Bezier curve.
    - GetCamberAngleDistribution(X, Y)
        Calculate the camber angle distribution over the length of the airfoil.
    - GetReferenceThicknessCamber(reference_file)
        Obtain the thickness and camber distributions from the reference airfoil
        shape.
    - GetReferenceParameters()
        Extract key parameters of the reference profile from the thickness and
        camber distributions.
    - GetThicknessControlPoints(b_8, b_15, r_LE, trailing_wedge_angle)
        Calculate the control points for the thickness distribution Bezier
        curves.
    - GetCamberControlPoints(b_0, b_2, b_17, leading_edge_direction,
                             trailing_camberline_angle)
        Calculate the control points for the camber distribution Bezier curves.
    - ConvertBezier2AirfoilCoordinates(thickness_x, thickness, camber_x, camber)
        Convert Bezier curves to airfoil coordinates.
    - FindInitialParameterisation(reference_file)
        Find the initial parameterisation for the airfoil provided in the
        reference_file.
    """

    # Constants
    N_BEZIER_POINTS = 100
    GA_UPPER_BOUNDS = np.array([0.1, 0.3, 0.7, 0.9, 0.9, 0.5, 0.3, 0.5, 0.2,
                                0.05, 0.005, -0.001, 0.4, 0.3, 0.3])
    GA_LOWER_BOUNDS = np.array([0.01, 0.1, 0, 0, 0, 0.15, 0.01, 0.25, 0, 0,
                                0, -0.2, 0.001, 0.001, 0.001])


    def BezierCurve3(self,
                     coefficients: np.typing.NDArray[np.floating],
                     u: np.typing.NDArray[np.floating],
                     ) -> np.typing.NDArray[np.floating]:
        """
        Calculate a 3rd degree Bezier curve.

        Parameters
         ----------
        - coefficients : np.typing.NDArray[np.floating]
            List of 4 control points for the Bezier curve.
        - u : np.typing.NDArray[np.floating]
            Array ranging from 0 to 1.

        Returns
        -------
        - y : np.typing.NDArray[np.floating]
            An array of the Bezier curve values evaluated at each of the
            points in u.
        """

        # Input checking
        if len(coefficients) != 4:
            raise ValueError(f"Coefficient list must contain exactly 4 elements. Coefficient list contains {len(coefficients)} elements")

        # Calculate the value of y at u using a 3rd degree Bezier curve
        return coefficients[0] * (1 - u) ** 3 + 3 * coefficients[1] * u * (1 - u) ** 2 + 3 * coefficients[2] * u ** 2 * (1 - u) + coefficients[3] * u ** 3


    def BezierCurve4(self,
                     coefficients: np.typing.NDArray[np.floating],
                     u: np.typing.NDArray[np.floating],
                     ) -> np.typing.NDArray[np.floating]:
        """
        Calculate a 4th degree Bezier curve.

        Parameters
        ----------
        - coefficients : np.typing.NDArray[np.floating]
            Array of 5 control points for the Bezier curve.
        - u : np.typing.NDArray[np.floating]
            Array ranging from 0 to 1.

        Returns
        -------
        y : np.typing.NDArray[np.floating]
            An array of the Bezier curve values evaluated at each of the
            points in u.
        """

        # Input checking
        if len(coefficients) != 5:
            raise ValueError(f"Coefficient list must contain exactly 5 elements. Coefficient list contains {len(coefficients)} elements.")

        return coefficients[0] * (1 - u) ** 4 + 4 * coefficients[1] * u * (1 - u) ** 3 + 6 * coefficients[2] * u ** 2 * (1 - u) ** 2 + 4 * coefficients[3] * u ** 3 * (1 - u) + coefficients[4] * u ** 4


    def GetCamberAngleDistribution(self,
                                   x: np.typing.NDArray[np.floating],
                                   y: np.typing.NDArray[np.floating],
                                   ) -> np.typing.NDArray[np.floating]:
        """
        Calculate the camber angle distribution over the length of the airfoil.

        Parameters
        ----------
        - x : np.typing.NDArray[np.floating]
            Array of x-coordinates along the airfoil.
        - y : np.typing.NDArray[np.floating]
            Array of camber values corresponding to the x-coordinates.

        Returns
        -------
        - theta : np.typing.NDArray[np.floating]
            Array of camber gradient angles at each x-coordinate.
        """

        camber_gradient = np.gradient(y, x)

        return np.arctan(camber_gradient)


    def GetReferenceThicknessCamber(self,
                                    reference_file: Path,
                                    ) -> None:
        """
        Obtain the thickness and camber distributions from the reference airfoil
        coordinate file.

        Parameters
        ----------
        - reference_file : Path
            Path to the file containing the reference profile coordinates.
        """

        # Load in the reference profile shape from the reference_file.
        # Assumes coordinates are sorted counter clockwise from TE of the Upper
        # surface.
        # Profile shape must be provided in input file with unit chord length
        # Skips the first row of the profile file as it contains the profile name
        try:
            reference_coordinates = np.genfromtxt(reference_file, dtype=np.floating, skip_header=1)
        except FileNotFoundError as err:
            raise FileNotFoundError(f"The data input file {reference_file} does not exist in the current working directory.") from err

        # Find index of LE coordinate and compute the camber and thickness distributions.
        # LE coordinate index must be the index for x = 0.
        self.idx_LE = (np.abs(reference_coordinates[:, 0] - 0)).argmin()

        if len(reference_coordinates[:self.idx_LE + 1, 0]) > len(reference_coordinates[self.idx_LE:, 0]):
            # If there are more upper coordinates than lower coordinates
            self.lower_coords: np.typing.NDArray[np.floating] = reference_coordinates[self.idx_LE:, 1] - np.flip(reference_coordinates[:self.idx_LE + 1, 1])[0]  # Also shift the lower coordinates to zero if they are not already at 0
            x_camber = reference_coordinates[self.idx_LE:, 0]
            self.upper_coords: np.typing.NDArray[np.floating] = interpolate.make_splrep(x=np.flip(reference_coordinates[:self.idx_LE + 1, 0]),
                                                                                        y=np.flip(reference_coordinates[:self.idx_LE + 1, 1]),
                                                                                        k=3,
                                                                                        s=0)(x_camber) - np.flip(reference_coordinates[:self.idx_LE + 1, 1])[0]
            self.x = np.flip(reference_coordinates[:self.idx_LE + 1, 0])

        elif len(reference_coordinates[:self.idx_LE + 1, 0]) < len(reference_coordinates[self.idx_LE:, 0]):
            # If there are more lower coordinates than upper coordinates
            self.upper_coords: np.typing.NDArray[np.floating] = np.flip(reference_coordinates[:self.idx_LE + 1, 1]) - reference_coordinates[self.idx_LE:, 1][0]  # Also shift the upper coordinates to zero if they are not already at 0
            x_camber = np.flip(reference_coordinates[:self.idx_LE + 1, 0])
            self.lower_coords: np.typing.NDArray[np.floating] = interpolate.make_splrep(x=reference_coordinates[self.idx_LE:, 0],
                                                                                        y=reference_coordinates[self.idx_LE:, 1],
                                                                                        k=3,
                                                                                        s=0)(x_camber) - reference_coordinates[self.idx_LE:, 1][0]
            self.x = reference_coordinates[self.idx_LE:, 0]
        else:
            # If there are an equal number of upper and lower surface coordinates
            self.upper_coords: np.typing.NDArray[np.floating] = np.flip(reference_coordinates[:self.idx_LE + 1, 1]) - np.flip(reference_coordinates[:self.idx_LE + 1, 1])[0]  # Also shift the upper coordinates to zero if they are not already at 0
            self.lower_coords: np.typing.NDArray[np.floating] = reference_coordinates[self.idx_LE:, 1] - np.flip(reference_coordinates[:self.idx_LE + 1, 1])[0]
            x_camber = (np.flip(reference_coordinates[:self.idx_LE + 1, 0]) + reference_coordinates[self.idx_LE:, 0]) / 2
            self.x = x_camber

        # Calculate camber and gradient of camber angle
        y_camber = (self.upper_coords + self.lower_coords) / 2
        theta = self.GetCamberAngleDistribution(x_camber,
                                                y_camber)

        # Calculate thickness distribution
        y_thickness = (self.upper_coords - self.lower_coords) / (2 * np.cos(theta))

        # Calculate x-coordinates of thickness distribution
        x_upper = x_camber - y_thickness * np.sin(theta)

        # Write parameters to self
        self.x_points_camber = x_camber
        self.x_points_thickness = x_upper
        self.camber_distribution = y_camber
        self.camber_gradient_distribution = theta
        self.thickness_gradient_distribution = self.GetCamberAngleDistribution(x_upper,
                                                                               y_thickness)
        self.thickness_distribution = y_thickness
        self.reference_data = reference_coordinates


    def GetReferenceParameters(self) -> dict[str, np.floating]:
        """
        Extract key parameters of the reference profile from the thickness and
        camber distributions.

        Calculates the leading edge radius, leading edge direction,
        trailing edge wedge angle, and trailing edge camber line angle.

        Returns
        -------
        - output_dict: dict[str, any]
            Dictionary containing the following parameters:
                - x_t : float
                    X-coordinate of maximum thickness.
                - y_t : float
                    Maximum thickness.
                - x_c : float
                    X-coordinate of maximum camber.
                - y_c : float
                    Maximum camber.
                - z_TE : float
                    Trailing edge vertical displacement.
                - dz_TE : float
                    Half thickness of the trailing edge.
                - r_LE : float
                    Leading edge radius.
                - leading_edge_direction : float
                    Angle of the leading edge direction.
                - trailing_wedge_angle : float
                    Trailing edge wedge angle.
                - trailing_camberline_angle : float
                    Trailing edge camber line angle.
        """

        # Find the indices for the points of maximum thickness and maximum camber
        self.idx_maxT: int = np.where(self.thickness_distribution == np.max(self.thickness_distribution))[0][0]
        self.idx_maxC: int = np.where(self.camber_distribution == np.max(self.camber_distribution))[0][0]

        x_t: np.floating = self.x_points_thickness[self.idx_maxT]  # X-coordinate of maximum thickness
        y_t: np.floating = self.thickness_distribution[self.idx_maxT]  # Maximum thickness
        x_c: np.floating = self.x_points_camber[self.idx_maxC]  # X-coordinate of maximum camber
        y_c: np.floating = self.camber_distribution[self.idx_maxC]  # Maximum camber
        z_TE: np.floating = self.camber_distribution[-1]  # Trailing edge vertical displacement
        dz_TE: np.floating = self.thickness_distribution[-1]  # Half thickness of the trailing edge

        # To find the radius of curvature of the leading edge, we fit a circle to the points within the first 2 % of the upper surface
        def GetLeadingEdgeRadius(params: tuple[float, float, float],
                                 x: np.typing.NDArray[np.floating],
                                 y: np.typing.NDArray[np.floating]) -> np.typing.NDArray[np.floating]:
            xc, yc, r = params
            return np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r

        LE_points_idx = np.where(self.x_points_thickness < 0.02)[0]
        LE_x = self.x_points_thickness[:LE_points_idx[-1] + 1]
        LE_y = self.thickness_distribution[:LE_points_idx[-1] + 1]
        guess = [0, 0, 0]

        # Perform least squares fitting of circle to data points
        result = optimize.least_squares(GetLeadingEdgeRadius, guess, args=(LE_x, LE_y))
        r_LE = -1 * result.x[-1]  # LE radius multiplied by -1 to comply with sign convention of method

        # Calculate the leading edge direction gamma_LE, trailing wedge angle beta_TE, and trailing camber line angle alpha_TE
        leading_edge_direction = np.atan(self.camber_gradient_distribution[0])
        trailing_wedge_angle = np.atan(-1 * self.thickness_gradient_distribution[-1])  # Multiply gradient by -1 to comply with sign convention
        trailing_camberline_angle = np.atan(-1 * self.camber_gradient_distribution[-1])  # Multiply gradient by -1 to comply with sign convention

        # Return output dictionary
        return {"x_t": x_t,
                "y_t": y_t,
                "x_c": x_c,
                "y_c": y_c,
                "z_TE": z_TE,
                "dz_TE": dz_TE,
                "r_LE": r_LE,
                "leading_edge_direction": leading_edge_direction,
                "trailing_wedge_angle": trailing_wedge_angle,
                "trailing_camberline_angle": trailing_camberline_angle}


    def GetThicknessControlPoints(self,
                                  airfoil_params: dict[str, float],
                                  ) -> tuple[np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], np.typing.NDArray[np.floating]]:
        """
        Calculate the control points for the thickness distribution Bezier curves.

        Parameters
        ----------
        - airfoil_params : dict[str, float]
            Dictionary containing the following parameters:
                - b_8 : float
                    Control point for the thickness curve.
                - b_15 : float
                    Control point for the thickness curve.
                - x_t : float
                    X-coordinate of maximum thickness.
                - y_t : float
                    Maximum thickness.
                - r_LE : float
                    Leading edge radius.
                - dz_TE : float
                    Half thickness of the trailing edge.
                - trailing_wedge_angle : float
                    The trailing edge wedge angle.

        Returns
        -------
        - x_leading_edge_thickness_coeff : np.typing.NDArray[np.floating]
            X-coordinates of the control points for the leading edge thickness Bezier curve.
        - y_leading_edge_thickness_coeff : np.typing.NDArray[np.floating]
            Y-coordinates of the control points for the leading edge thickness Bezier curve.
        -x_trailing_edge_thickness_coeff : np.typing.NDArray[np.floating]
            X-coordinates of the control points for the trailing edge thickness Bezier curve.
        - y_trailing_edge_thickness_coeff : np.typing.NDArray[np.floating]
            Y-coordinates of the control points for the trailing edge thickness Bezier curve.
        """

        # First use the provided b_15 parameter value to construct the relative location of the inflection point
        b_15_coordinate = (airfoil_params["b_15"] - airfoil_params["x_t"]) / (1 - airfoil_params["x_t"])

        # Construct leading edge x coefficients
        x_leading_edge_thickness_coeff = np.array([0,
                                                   0,
                                                   (-3 * airfoil_params["b_8"] ** 2) / (2 * airfoil_params["r_LE"]),
                                                   airfoil_params["x_t"]])

        # Construct leading edge y coefficients
        y_leading_edge_thickness_coeff = np.array([0,
                                                   airfoil_params["b_8"],
                                                   airfoil_params["y_t"],
                                                   airfoil_params["y_t"]])

        x_trailing_edge_thickness_coeff = np.array([airfoil_params["x_t"],
                                                    (7 * airfoil_params["x_t"] + (9 * airfoil_params["b_8"] ** 2) / (2 * airfoil_params["r_LE"])) / 4,
                                                    3 * airfoil_params["x_t"] + (15 * airfoil_params["b_8"] ** 2) / (4 * airfoil_params["r_LE"]),
                                                    b_15_coordinate,
                                                    1])

        # Construct trailing edge y coefficients
        y_trailing_edge_thickness_coeff = np.array([airfoil_params["y_t"],
                                                    airfoil_params["y_t"],
                                                    (airfoil_params["y_t"] + airfoil_params["b_8"]) / 2,
                                                    airfoil_params["dz_TE"] + (1 - b_15_coordinate) * np.tan(airfoil_params["trailing_wedge_angle"]),
                                                    airfoil_params["dz_TE"]])

        return x_leading_edge_thickness_coeff, y_leading_edge_thickness_coeff, x_trailing_edge_thickness_coeff, y_trailing_edge_thickness_coeff


    def GetCamberControlPoints(self,
                               airfoil_params: dict[str, float],
                               ) -> tuple[np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], np.typing.NDArray[np.floating]]:
        """
        Calculate the control points for the camber distribution Bezier curves.

        Parameters
        ----------
        - airfoil_params : dict[str, float]
            Dictionary containing the following parameters:
                - b_0 : float
                    Control point for the camber.
                - b_2 : float
                    Control point for the camber.
                - b_17 : float
                    Control point for the camber.
                - x_c : float
                    X-coordinate of maximum camber.
                - y_c : float
                    Maximum camber.
                - z_TE : float
                    Trailing edge vertical displacement.
                - leading_edge_direction : float
                    Angle of the leading edge direction.
                - trailing_camberline_angle : float
                    Angle of the trailing edge camber line.

        Returns
        -------
        - x_leading_edge_camber_coeff : np.typing.NDArray[np.floating]
            X-coordinates of the control points for the leading edge camber
            Bezier curve.
        - y_leading_edge_camber_coeff : np.typing.NDArray[np.floating]
            Y-coordinates of the control points for the leading edge camber
            Bezier curve.
        - x_trailing_edge_camber_coeff : np.typing.NDArray[np.floating]
            X-coordinates of the control points for the trailing edge camber
            Bezier curve.
        - y_trailing_edge_camber_coeff : np.typing.NDArray[np.floating]
            Y-coordinates of the control points for the trailing edge camber
            Bezier curve.
        """

        # First use the provided parameter values to construct the relative locations of the control points
        b_17_coordinate = (airfoil_params["b_17"] - airfoil_params["x_c"]) / (1 - airfoil_params["x_c"])
        b_0_coordinate = airfoil_params["b_0"] * airfoil_params["x_c"]
        b_2_coordinate = airfoil_params["b_2"] * airfoil_params["x_c"]

        # Construct leading edge x coefficients
        x_leading_edge_camber_coeff = np.array([0,
                                                b_0_coordinate,
                                                b_2_coordinate,  # b_0 and b_2 are inherently bounded to be 0 < b_0 < b_2
                                                airfoil_params["x_c"]])

        # Construct leading edge y coefficients
        y_leading_edge_camber_coeff = np.array([0,
                                                b_0_coordinate * np.tan(airfoil_params["leading_edge_direction"]),
                                                airfoil_params["y_c"],
                                                airfoil_params["y_c"]])

        # Construct trailing edge x coefficients
        x_trailing_edge_camber_coeff = np.array([airfoil_params["x_c"],
                                                 (3 * airfoil_params["x_c"] - airfoil_params["y_c"] * _cot(airfoil_params["leading_edge_direction"])) / 2,
                                                 (-8 * airfoil_params["y_c"] * _cot(airfoil_params["leading_edge_direction"]) + 13 * airfoil_params["x_c"]) / 6,
                                                 b_17_coordinate,
                                                 1])

        # Construct trailing edge y coefficients
        y_trailing_edge_camber_coeff = np.array([airfoil_params["y_c"],
                                                 airfoil_params["y_c"],
                                                 5 * airfoil_params["y_c"] / 6,
                                                 airfoil_params["z_TE"] + (1 - b_17_coordinate) * np.tan(airfoil_params["trailing_camberline_angle"]),
                                                 airfoil_params["z_TE"]])

        return x_leading_edge_camber_coeff, y_leading_edge_camber_coeff, x_trailing_edge_camber_coeff, y_trailing_edge_camber_coeff


    def ConvertBezier2AirfoilCoordinates(self,
                                         thickness_x: np.typing.NDArray[np.floating],
                                         thickness: np.typing.NDArray[np.floating],
                                         camber_x: np.typing.NDArray[np.floating],
                                         camber: np.typing.NDArray[np.floating],
                                         ) -> tuple[np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], np.typing.NDArray[np.floating]]:
        """
        Convert Bezier curves to airfoil coordinates.

        Parameters
        ----------
        - thickness_x : np.typing.NDArray[np.floating]
            Array of x-coordinates for the thickness distribution.
        - thickness : np.typing.NDArray[np.floating]
            Array of thickness values corresponding to the x-coordinates.
        - camber_x : np.typing.NDArray[np.floating]
            Array of x-coordinates for the camber distribution.
        - camber : np.typing.NDArray[np.floating]
            Array of camber values corresponding to the x-coordinates.

        Returns
        -------
        - upper_x : np.typing.NDArray[np.floating]
            Array of x-coordinates for the upper surface of the airfoil.
        - upper_y : np.typing.NDArray[np.floating]
            Array of y-coordinates for the upper surface of the airfoil.
        - lower_x : np.typing.NDArray[np.floating]
            Array of x-coordinates for the lower surface of the airfoil.
        - lower_y : np.typing.NDArray[np.floating]
            Array of y-coordinates for the lower surface of the airfoil.
        """

        thickness_interpolation = interpolate.CubicSpline(thickness_x, thickness)  # Interpolation of bezier thickness distribution
        thickness_distribution = thickness_interpolation(camber_x)  # Use interpolation to get value of thickness at camber points

        # Calculate gradient of camber angle line
        theta = self.GetCamberAngleDistribution(camber_x, camber)

        # Coordinate transformation to create data for upper and lower surface
        upper_x = camber_x - thickness_distribution * np.sin(theta)
        upper_y = camber + thickness_distribution * np.cos(theta)
        lower_x = camber_x + thickness_distribution * np.sin(theta)
        lower_y = camber - thickness_distribution * np.cos(theta)

        return upper_x, upper_y, lower_x, lower_y


    def GenerateBezierUVectors(self,
                               ) -> tuple[np.typing.NDArray[np.floating], np.typing.NDArray[np.floating]]:
        """
        Create u-vectors for Bezier curve generation.

        Returns
        -------
        - u_leading_edge : np.typing.NDArray[np.floating]
            Array of u-values for the leading edge Bezier curve.
        - u_trailing_edge : np.typing.NDArray[np.floating]
            Array of u-values for the trailing edge Bezier curve.
        """

        # Create u-vectors for Bezier curve generation
        pi_factor = np.pi / (2 * (self.N_BEZIER_POINTS - 1))
        i_scaled = np.arange(self.N_BEZIER_POINTS) * pi_factor
        u_leading_edge = 1. - np.cos(i_scaled)  # Space points using a cosine spacing for increased resolution at LE
        u_trailing_edge = np.sin(i_scaled)  # Space points using a sine spacing for increased resolution at TE

        return u_leading_edge, u_trailing_edge


    def ComputeBezierCurves(self,
                            airfoil_params: dict[str, float],
                            ) -> tuple[np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], np.typing.NDArray[np.floating]]:
        """
        Calculate the thickness and camber Bezier distributions.

        Parameters
        ----------
        - airfoil_params : dict[str, float]
            Dictionary containing airfoil parameters.

        Returns
        -------
        - bezier_thickness : np.typing.NDArray[np.floating]
            Array of thickness values along the airfoil.
        - bezier_thickness_x : np.typing.NDArray[np.floating]
            Array of x-coordinates corresponding to the thickness values.
        - bezier_camber : np.typing.NDArray[np.floating]
            Array of camber values along the airfoil.
        - bezier_camber_x : np.typing.NDArray[np.floating]
            Array of x-coordinates corresponding to the camber values.
        """

        # Create Bezier U-vectors
        u_leading_edge, u_trailing_edge = self.GenerateBezierUVectors()

        # Calculate the Bezier curve coefficients for the thickness curves
        x_LE_thickness_coeff, y_LE_thickness_coeff, x_TE_thickness_coeff, y_TE_thickness_coeff = self.GetThicknessControlPoints(airfoil_params)

        # Calculate the leading edge thickness distribution
        y_LE_thickness = self.BezierCurve3(y_LE_thickness_coeff,
                                           u_leading_edge)  # Leading edge thickness represented by 3rd order Bezier curve
        x_LE_thickness = self.BezierCurve3(x_LE_thickness_coeff,
                                           u_leading_edge)  # Leading edge thickness bezier x-coordinates, represented by a 3rd order curve

        # Calculate the trailing edge thickness distribution
        y_TE_thickness = self.BezierCurve4(y_TE_thickness_coeff,
                                           u_trailing_edge[1:])  # Trailing edge thickness represented by 4th order Bezier curve
        x_TE_thickness = self.BezierCurve4(x_TE_thickness_coeff,
                                           u_trailing_edge[1:])  # Trailing edge bezier x-coordinates, represented by 4th order curve

        # Construct full curves by combining LE and TE data
        bezier_thickness = np.concatenate((y_LE_thickness, y_TE_thickness),
                                          axis=0)  # Construct complete thickness curve over length of profile
        bezier_thickness_x = np.concatenate((x_LE_thickness, x_TE_thickness),
                                            axis=0)  # Construct complete array of x-coordinates over length of profile

        # Check the sorting of the thickness curve - if the arrays are not sorted we raise a valueerror as it indicates an infeasible parameterisation
        if not np.all(np.diff(bezier_thickness_x) >= 0):
            raise ValueError("The thickness parameterisation for the profile is infeasible.")

        if airfoil_params["y_c"] > 1e-3:
            # Calculate the Bezier curve coefficients for the camber curves
            x_LE_camber_coeff, y_LE_camber_coeff, x_TE_camber_coeff, y_TE_camber_coeff = self.GetCamberControlPoints(airfoil_params)

            # Calculate the leading edge camber distribution
            y_LE_camber = self.BezierCurve3(y_LE_camber_coeff,
                                            u_leading_edge)  # Leading edge camber represented by 3rd order Bezier curve
            x_LE_camber = self.BezierCurve3(x_LE_camber_coeff,
                                            u_leading_edge)  # Leading edge camber bezier x-coordinates, represented by a 3rd order curve

            # Calculate the trailing edge camber distribution using the parameter b_17
            y_TE_camber = self.BezierCurve4(y_TE_camber_coeff,
                                            u_trailing_edge[1:])  # Trailing edge camber represented by 4th order Bezier curve
            x_TE_camber = self.BezierCurve4(x_TE_camber_coeff,
                                            u_trailing_edge[1:])  # Trailing edge camber bezier x-coordinates, represented by a 4th order curve

            bezier_camber = np.concatenate((y_LE_camber, y_TE_camber),
                                           axis=0)  # Construct complete camber curve over length of profile
            bezier_camber_x = np.concatenate((x_LE_camber, x_TE_camber),
                                             axis=0)  # Construct complete array of x-coordinates over length of profile

            # Check the sorting of the camber curve - if the arrays are not sorted sort them to attempt to fix the profile
            if not np.all(np.diff(bezier_camber_x) >= 0):
                raise ValueError("The camber distribution for the profile is infeasible.")

        else:
            # If camber is zero, handle appropriately
            bezier_camber = np.zeros_like(bezier_thickness)
            bezier_camber_x = bezier_thickness_x

        return bezier_thickness, bezier_thickness_x, bezier_camber, bezier_camber_x


    def ComputeProfileCoordinates(self,
                                  airfoil_params: dict[str, float],
                                  ) -> tuple[np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], np.typing.NDArray[np.floating]]:
        """
        Calculate the airfoil coordinates from the Bezier control points.

        Parameters
        ----------
        - airfoil_params : dict
            Dictionary containing airfoil parameters.

        Returns
        -------
        - upper_x : np.typing.NDArray[np.floating]
            Array of x-coordinates for the upper surface of the airfoil.
        - upper_y : np.typing.NDArray[np.floating]
            Array of y-coordinates for the upper surface of the airfoil.
        - lower_x : np.typing.NDArray[np.floating]
            Array of x-coordinates for the lower surface of the airfoil.
        - lower_y : np.typing.NDArray[np.floating]
            Array of y-coordinates for the lower surface of the airfoil.
        """

        # Obtain the Bezier data
        bezier_thickness, bezier_thickness_x, bezier_camber, bezier_camber_x = self.ComputeBezierCurves(airfoil_params)

        # Calculate the upper and lower surface coordinates from the bezier coordinates
        upper_x, upper_y, lower_x, lower_y = self.ConvertBezier2AirfoilCoordinates(bezier_thickness_x,
                                                                                   bezier_thickness,
                                                                                   bezier_camber_x,
                                                                                   bezier_camber)
        
        # Validate profile shape by checking one-to-one of the x-arrays
        if (not np.all(np.diff(upper_x) >= 0) and not np.all(np.diff(lower_x) >= 0)):
            raise ValueError("The parameterisation for the profile is infeasible.")

        return upper_x, upper_y, lower_x, lower_y


    def CheckOptimizedResult(self,
                             airfoil_params: dict,
                             reference_file: Path = None,
                             ) -> None:
        """
        Check the optimized result by plotting the thickness and camber
        distributions, and the airfoil shape.

        Parameters
        ----------
        - airfoil_params : dict
            A dictionary containing the airfoil parameterisation parameters.
        - reference_file : Path, optional
            The path to the reference file against which the fitting took
            place.
        """

        # Load in the reference profile shape and obtain the relevant parameters
        if reference_file is not None:
            self.GetReferenceThicknessCamber(reference_file)
            self.airfoil_params = self.GetReferenceParameters()

        # Obtain bezier curves
        bezier_thickness, bezier_thickness_x, bezier_camber, bezier_camber_x = self.ComputeBezierCurves(airfoil_params)

        # Compute upper and lower surface coordinates
        upper_x, upper_y, lower_x, lower_y = self.ConvertBezier2AirfoilCoordinates(bezier_thickness_x,
                                                                                   bezier_thickness,
                                                                                   bezier_camber_x,
                                                                                   bezier_camber)

        # Calculate bezier coefficients for the thickness curves for plotting
        x_LE_thickness_coeff, y_LE_thickness_coeff, x_TE_thickness_coeff, y_TE_thickness_coeff = self.GetThicknessControlPoints(airfoil_params)
        x_LE_camber_coeff, y_LE_camber_coeff, x_TE_camber_coeff, y_TE_camber_coeff = self.GetCamberControlPoints(airfoil_params)

        def get_figsize(columnwidth=448.1309, wf=0.5, hf=(5.**0.5-1.0)/2.0, ):
            """
            Parameters
            - wf [float]:  width fraction in columnwidth units
            - hf [float]:  height fraction in columnwidth units.
                            Set by default to golden ratio.
            - columnwidth [float]: width of the column in pt in latex. Get this from LaTeX
                                    using \\showthe\\columnwidth
            Returns:  [fig_width,fig_height]: that should be given to matplotlib
            """

            fig_width_pt = columnwidth*wf
            inches_per_pt = 1.0/72.27               # Convert pt to inch
            fig_width = fig_width_pt*inches_per_pt  # width in inches
            fig_height = fig_width*hf      # height in inches
            return [fig_width, fig_height]

        try:
            plt.rcParams.update({'font.size': 9, "pgf.texsystem": "xelatex", "text.usetex":  True, "pgf.rcfonts": False})

            # Create a single figure with subplots
            fig, axs = plt.subplots(2, 1, figsize=get_figsize(wf=0.95), gridspec_kw={'height_ratios': [1, 1.2]})

            # First row: Thickness and Camber distributions
            axs[0].plot(bezier_thickness_x, bezier_thickness, label="BP thickness", color="tab:blue")
            axs[0].plot(x_LE_thickness_coeff, y_LE_thickness_coeff, '*', color='tab:blue', label="BP thickness control points")
            axs[0].plot(x_TE_thickness_coeff, y_TE_thickness_coeff, '*', color='tab:blue')
            if reference_file is not None:
                axs[0].plot(self.x_points_thickness, self.thickness_distribution, "-.", color="tab:blue", label="Reference thickness")
                axs[0].plot(self.x_points_camber, self.camber_distribution, "-.", color="tab:orange", label="Reference camber")
            axs[0].plot(bezier_camber_x, bezier_camber, "--", label="BP camber", color="tab:orange")
            axs[0].plot(x_LE_camber_coeff, y_LE_camber_coeff, '^', color='tab:orange', label="BP camber control points")
            axs[0].plot(x_TE_camber_coeff, y_TE_camber_coeff, '^', color='tab:orange')
            axs[0].set_title("Thickness and camber distributions")
            axs[0].set_xlabel("Axial coordinate $x/c$ [-]")
            axs[0].set_ylabel("Thickness and camber [-]")
            axs[0].legend(bbox_to_anchor=(1, 1))
            axs[0].grid(which='major', alpha=MAJOR_GRID_ALPHA)
            axs[0].minorticks_on()
            axs[0].grid(which='minor', linewidth=0.25, alpha=MINOR_GRID_ALPHA)

            # Second row: Combined airfoil shape
            axs[1].plot(upper_x, upper_y, label="Modelled profile", color="tab:green")
            axs[1].plot(lower_x, lower_y, color="tab:green")
            if reference_file is not None:
                axs[1].plot(self.reference_data[:, 0], self.reference_data[:, 1], "-.", color="tab:gray", label="Reference profile")
            axs[1].set_title("Combined airfoil shape")
            axs[1].set_xlabel("Axial coordinate $x/c$ [-]")
            axs[1].set_ylabel("Vertical coordinate $y/c$ [-]")
            axs[1].legend(bbox_to_anchor=(1, 1))
            axs[1].grid(which='major', alpha=MAJOR_GRID_ALPHA)
            axs[1].minorticks_on()
            axs[1].grid(which='minor', linewidth=0.25, alpha=MINOR_GRID_ALPHA)

            # Adjust layout
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Warning: Failed to generate plots: {str(e)}")


    def Objective(self,
                  x: np.typing.NDArray[np.floating],
                  reference_file: Path = None) -> float:
        """
        Objective function for minimization.

        Parameters
        ----------
        - x : np.typing.NDArray[np.floating]
            Array of normalised design variables, which are denormalised using
            the guess design vector given by self.guess_design_vector.
        - reference_file : Path, Optional
            The path to the reference file against which the optimisation is to
            take place.

        Returns
        -------
        - squared_fit_error : float
            Sum of squared fit errors of the upper and lower surfaces.
        """

        # Denormalise design vector only for the slsqp optimisation method.
        if reference_file is None:
            # Validate that reference data is loaded
            if not hasattr(self, "x") or not hasattr(self, 'upper_coords') or not hasattr(self, 'lower_coords'):
                raise ValueError("Reference data not loaded. Call GetReferenceThicknessCamber first.")

            x = np.multiply(x, self.guess_design_vector)

        # Reformat the design vector into the required airfoil_params dictionary.
        airfoil_params = {"b_0": x[0],
                          "b_2": x[1],
                          "b_8": x[2] * min(x[6], np.sqrt(-2 * x[11] * x[5] / 3)),
                          "b_15": x[3],
                          "b_17": x[4],
                          "x_t": x[5],
                          "y_t": x[6],
                          "x_c": x[7],
                          "y_c": x[8],
                          "z_TE": x[9],
                          "dz_TE": x[10],
                          "r_LE": x[11],
                          "trailing_wedge_angle": x[12],
                          "trailing_camberline_angle": x[13],
                          "leading_edge_direction": x[14]}

        try:
            upper_x, upper_y, lower_x, lower_y = self.ComputeProfileCoordinates(airfoil_params)

            # Cache interpolation objects to avoid recreating them for each evaluation
            if not hasattr(self, '_upper_interp') or not hasattr(self, '_lower_interp'):
                self._upper_interp = interpolate.make_splrep(self.x,
                                                             self.upper_coords,
                                                             k=3,
                                                             s=0)
                self._lower_interp = interpolate.make_splrep(self.x,
                                                             self.lower_coords,
                                                             k=3,
                                                             s=0)
            reference_upper_y = self._upper_interp(upper_x)
            reference_lower_y = self._lower_interp(lower_x)

            l2_error_upper = np.linalg.norm(upper_y - reference_upper_y)
            l2_error_lower = np.linalg.norm(lower_y - reference_lower_y)
            err = l2_error_lower + l2_error_upper
            return err

        except ValueError:
            # Set a high objective function value in case the parameterisation is invalid.
            return 1e6


    def _slsqp_fitting(self, reference_file: Path) -> tuple[dict[str, float], int]:
        """SLSQP optimization implementation"""

        # Load in the reference profile shape and obtain the relevant parameters
        self.GetReferenceThicknessCamber(reference_file)
        self.airfoil_params = self.GetReferenceParameters()

        # Define a guess of the initial design vector
        self.guess_design_vector = np.array([0.05,
                                             0.2,
                                             0.05 * min(self.airfoil_params["y_t"], np.sqrt(-2 * self.airfoil_params["r_LE"] * self.airfoil_params["x_t"] / 3)),
                                             0.8,
                                             0.8,
                                             self.airfoil_params["x_t"],
                                             self.airfoil_params["y_t"],
                                             self.airfoil_params["x_c"] if self.airfoil_params["x_c"] != 0 else 0.35,
                                             self.airfoil_params["y_c"],
                                             self.airfoil_params["z_TE"],
                                             self.airfoil_params["dz_TE"],
                                             self.airfoil_params["r_LE"],
                                             self.airfoil_params["trailing_wedge_angle"] if self.airfoil_params["trailing_wedge_angle"] != 0 else 0.001,
                                             self.airfoil_params["trailing_camberline_angle"] if self.airfoil_params["trailing_camberline_angle"] != 0 else 0.001,
                                             self.airfoil_params["leading_edge_direction"] if self.airfoil_params["leading_edge_direction"] != 0 else 0.001,
                                             ])

        # Define nonlinear constraint for b8
        def b8_constraint(x):
            x = np.multiply(x, self.guess_design_vector)  # Denormalise design vector
            return np.min([x[6], np.sqrt(-2 * x[11] * x[5] / 3)]) - x[2]

        # Define constraint for bezier control point 1 x-coordinate of TE thickness curve
        def x1_constraint_lower_thickness(x):
            x = np.multiply(x, self.guess_design_vector)  # Denormalise design vector
            return (7 * x[5] + 9 * x[2] / (2 * x[11])) / 4 - x[5]
        
        def x1_constraint_upper_thickness(x):
            x = np.multiply(x, self.guess_design_vector)  # Denormalise design vector
            return 1 - (7 * x[5] + 9 * x[2] / (2 * x[11])) / 4

        # Define constraint for bezier control point 2 x-coordinate of TE thickness curve
        def x2_constraint_lower_thickness(x):
            x = np.multiply(x, self.guess_design_vector)  # Denormalise design vector
            return 2 * x[5] + 15 * x[2] ** 2 / (4 * x[11])
        
        def x2_constraint_upper_thickness(x):
            x = np.multiply(x, self.guess_design_vector)  # Denormalise design vector
            return 1 - 2 * x[5] + 15 * x[2] ** 2 / (4 * x[11])

        # Define constraint for bezier control point 2 x-coordinate of TE camber curve
        def constraint_6_lower(x):
            x = np.multiply(x, self.guess_design_vector)  # Denormalise design vector
            return (3 * x[7] - x[8] / np.tan(x[14])) / 2 - x[7] if x[14] != 0 else x[14]

        def constraint_6_upper(x):
            x = np.multiply(x, self.guess_design_vector)  # Denormalise design vector
            return 1 - (3 * x[7] - x[8] / np.tan(x[14])) / 2 if x[14] != 0 else x[14]

        # Define constraint for control point 3 x-coordinate of TE camber curve
        def constraint_7_lower(x):
            x = np.multiply(x, self.guess_design_vector)  # Denormalise design vector
            return (-8 * x[8] / np.tan(x[14]) + 13 * x[7]) / 6 - x[7] if x[14] != 0 else x[14]

        def constraint_7_upper(x):
            x = np.multiply(x, self.guess_design_vector)  # Denormalise design vector
            return 1 - (-8 * x[8] / np.tan(x[14]) + 13 * x[7]) / 6 if x[14] != 0 else x[14]

        cons = [{'type': 'ineq', 'fun': b8_constraint},
                {'type': 'ineq', 'fun': x1_constraint_lower_thickness},
                {'type': 'ineq', 'fun': x1_constraint_upper_thickness},
                {'type': 'ineq', 'fun': x2_constraint_lower_thickness},
                {'type': 'ineq', 'fun': x2_constraint_upper_thickness},
                {'type': 'ineq', 'fun': constraint_6_lower},
                {'type': 'ineq', 'fun': constraint_6_upper},
                {'type': 'ineq', 'fun': constraint_7_lower},
                {'type': 'ineq', 'fun': constraint_7_upper}]

        optimized_coefficients = optimize.minimize(self.Objective,
                                                   np.ones_like(self.guess_design_vector),
                                                   method="SLSQP",
                                                   bounds=optimize.Bounds(0.95, 1.05),  # Assume the initial guess is reasonably close to the true values, so +/- 5% on the variables should work.
                                                   constraints=cons,
                                                   options={'maxiter': 500,
                                                           'disp': False},
                                                   jac='3-point')

        # Denormalise the found coefficients and write them to the output dictionary
        optimized_coefficients.x = optimized_coefficients.x.astype(float)
        optimized_coefficients.x = np.multiply(optimized_coefficients.x, self.guess_design_vector)

        airfoil_params_optimized = {"b_0": float(optimized_coefficients.x[0]),
                                    "b_2": float(optimized_coefficients.x[1]),
                                    "b_8": float(optimized_coefficients.x[2]),
                                    "b_15": float(optimized_coefficients.x[3]),
                                    "b_17": float(optimized_coefficients.x[4]),
                                    "x_t": float(optimized_coefficients.x[5]),
                                    "y_t": float(optimized_coefficients.x[6]),
                                    "x_c": float(optimized_coefficients.x[7]),
                                    "y_c": float(optimized_coefficients.x[8]),
                                    "z_TE": float(optimized_coefficients.x[9]),
                                    "dz_TE": float(optimized_coefficients.x[10]),
                                    "r_LE": float(optimized_coefficients.x[11]),
                                    "trailing_wedge_angle": float(optimized_coefficients.x[12]),
                                    "trailing_camberline_angle": float(optimized_coefficients.x[13]),
                                    "leading_edge_direction": float(optimized_coefficients.x[14])}

        return airfoil_params_optimized, optimized_coefficients.status


    def _GA_fitting(self, reference_file: Path) -> dict[str, float]:
        """Genetic algorithm optimization implementation"""

        # Lazy-import the pymoo package to avoid unneccesary imports.
        from pymoo.algorithms.soo.nonconvex.ga import GA
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.optimize import minimize
        from pymoo.termination.default import DefaultSingleObjectiveTermination

        # Define the optimisation problem definition class
        class ProblemDefinition(ElementwiseProblem):
            def __init__(self):
                # Reuse a single parameterisation instance
                self.af_param = AirfoilParameterisation()
                self.af_param.GetReferenceThicknessCamber(reference_file)
                self.af_param.GetReferenceParameters()

                super().__init__(n_var=15, n_obj=1, n_eq_constr=0, n_ieq_constr=3,
                                 xl=self.af_param.GA_LOWER_BOUNDS,
                                 xu=self.af_param.GA_UPPER_BOUNDS)

            def _evaluate(self, x, out, *args, **kwargs):
                out["F"] = self.af_param.Objective(x)
                # Compute bound for b_8
                g1 = x[2] - min(x[6], np.sqrt(-2 * x[11] * x[5] / 3))

                # Compute TE camber curve bound to avoid invalid camber shape
                g2 = 8/7 * x[8] / np.tan(x[14]) - (x[7] + 0.05)  # + 0.025 to avoid intersection with x_c

                # Compute TE thickness curve bound to avoid invalid thickness shape
                g4 = -5 * x[2] ** 2 / (8 * x[11]) - (x[5] + 0.05)  # + 0.025 to avoid intersection with x_t
                out["G"] = [g1, g2, g4]


        term_conditions = DefaultSingleObjectiveTermination(xtol=1e-18, cvtol=1e-12, ftol=0.001, period=10, n_max_gen=500, n_max_evals=100000)
        problem = ProblemDefinition()
        algorithm = GA(pop_size=150,
                       eliminate_duplicates=True)
        res = minimize(problem,
                       algorithm,
                       termination=term_conditions,
                       seed=42,
                       verbose=True)

        airfoil_params_optimized = {"b_0": float(res.X[0]),
                                    "b_2": float(res.X[1]),
                                    "b_8": float(res.X[2]),
                                    "b_15": float(res.X[3]),
                                    "b_17": float(res.X[4]),
                                    "x_t": float(res.X[5]),
                                    "y_t": float(res.X[6]),
                                    "x_c": float(res.X[7]),
                                    "y_c": float(res.X[8]),
                                    "z_TE": float(res.X[9]),
                                    "dz_TE": float(res.X[10]),
                                    "r_LE": float(res.X[11]),
                                    "trailing_wedge_angle": float(res.X[12]),
                                    "trailing_camberline_angle": float(res.X[13]),
                                    "leading_edge_direction": float(res.X[14])}

        return airfoil_params_optimized


    def FindInitialParameterisation(self,
                                    reference_file: Path) -> dict[str, float]:
        """
        Find the initial parameterisation for the profile.
        Uses a genetic algorithm to minimize the squared fit error between the
        input reference file and the reconstructed profile shape.

        Parameters
        ----------
        - reference_file : Path
            Path to the file containing the reference profile coordinates.

        Returns
        -------
        - dict[str, float]
            Dictionary containing the optimized airfoil parameters.
        """


        # First we try SLSQP fitting
        airfoil_params_optimized, status = self._slsqp_fitting(reference_file)
        if status == 0:
            return airfoil_params_optimized

        # If SLSQP failed, try a genetic algorithm
        airfoil_params_optimized = self._GA_fitting(reference_file)
        return airfoil_params_optimized


if __name__ == "__main__":
    import time
    call_class = AirfoilParameterisation()

    start_time = time.time()
    inputfile = Path(__file__).resolve().parent / 'Test Airfoils/duct.dat'
    airf_params = call_class.FindInitialParameterisation(inputfile)
    end_time = time.time()
    print(f"Execution of FindInitialParameterisation({inputfile}) took {end_time-start_time} seconds")
    print("-----")
    print(airf_params)

    call_class.CheckOptimizedResult(airf_params,
                                    inputfile)