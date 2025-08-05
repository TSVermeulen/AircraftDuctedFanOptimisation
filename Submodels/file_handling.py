"""
file_handling
=============

Description
-----------
This module provides classes and methods for handling file operations related to 
the conceptual investigation of alternative electric ducted fan architectures. 
It includes functionalities for generating the input files for the MTSET and MTFLO tools, forming 
part of the MTFLOW software suite from MIT.

Classes
-------
fileHandlingMTSET
    Handles the generation of the MTSET input file, walls.xxx.
    This input file contains the axisymmetric bodies present within the domain.

fileHandlingMTFLO
    Handles the generation of the MTFLO input file, tflow.xxx.
    This input file contains the forcing field terms corresponding to the blade rows (rotors and stators).


Examples
--------
>>> n2415_coeff = {"b_0": 0.203, "b_2": 0.319, "b_8": 0.042, "b_15": 0.750, "b_17": 0.679, 
...                "x_t": 0.299, "y_t": 0.060, "x_c": 0.405, "y_c": 0.020, "z_TE": -0.00034, 
...                "dz_TE": 0.0017, "r_LE": -0.024, "trailing_wedge_angle": 0.167, "trailing_camberline_angle": 0.065, 
...                "leading_edge_direction": 0.094, "Chord Length": 1.0, "Leading Edge Coordinates": (0, 2)}
>>> call_class = fileHandlingMTSET(n2415_coeff, n2415_coeff, "test_case", 1.0)
>>> call_class.GenerateMTSETInput()

>>> blading_parameters = [
...     {
...         "root_LE_coordinate": 0.0,
...         "rotational_rate": 10,
...         "blade_count": 18,
...         "radial_stations": np.array([0.1, 1.0]),
...         "chord_length": np.array([0.2, 0.2]),
...         "sweep_angle": np.array([np.pi / 4, np.pi / 4]),
...         "blade_angle": np.array([0, np.pi / 3]),
...         "ref_blade_angle": 0.0,
...         "reference_section_blade_angle": 0.0,
...     },
...     {
...         "root_LE_coordinate": 2.0,
...         "rotational_rate": 0,
...         "blade_count": 10,
...         "radial_stations": np.array([0.1, 1.0]),
...         "chord_length": np.array([0.2, 0.2]),
...         "sweep_angle": np.array([np.pi / 4, np.pi / 4]),
...         "blade_angle": np.array([0, np.pi / 8]),
...         "ref_blade_angle": 0.0,
...         "reference_section_blade_angle": 0.0,
...     },
... ]
>>> design_parameters = [[n2415_coeff, n2415_coeff],
...                      [n2415_coeff, n2415_coeff],
...                      ]
>>> call_class = fileHandlingMTFLO("test_case", 1.0)
>>> call_class.GenerateMTFLOInput(blading_parameters, design_parameters)

Notes
-----
This module is designed to work with the BP3434 profile parameterisation defined in the parameterisations.py file.
Ensure that the input dictionaries are correctly formatted. For details on the specific inputs needed, see the 
different method docstrings.

When executing the file as a standalone, it uses the inputs and calls contained within the if __name__ == "__main__" section. 
This part also imports the time module to measure the time needed to perform each file generation call. This is beneficial in runtime optimization.

References
----------
The coordinate transformation from cartesian space to the developed coordinates m'-theta used to calculate 
s_rel in fileHandlingMTFLO.GenerateMTFLOInput() is documented in the MISES user manual:
https://web.mit.edu/drela/Public/web/mises/mises.pdf

The required input data, limitations, and structures are documented within the MTFLOW user manual:
https://web.mit.edu/drela/Public/web/mtflow/mtflow.pdf

The blade coordinate transformation, and order of calculations, is based on the implementation found in the BladeX module:
https://github.com/mathLab/BladeX 

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 2.1
Date [dd-mm-yyyy]: 03-06-2025

Changelog
---------
- V1.0: Initial working version
- V1.1: Updated test values. Added leading edge coordinate control of centrebody. Added floating point precision of 3 decimals for domain size. Updated input validation logic.
- V1.1.5: Fixed import logic of the Parameterisations module to handle local versus global file execution.
- V1.2.0: Updated class initialization logic and function inputs to enable existing geometry inputs for debugging/validation
- V1.2.1: Fixed duplicate leading edge coordinate in fileHandlingMTSET.GetProfileCoordinates(). Implemented nondimensionalisation of geometric parameters for both MTSET and MTFLO input files. 
- V1.3: Significant reworks to help solve bugs and issues found in validation against the X22A ducted propeller case. Added the grid size as optional input in fileHandlingMTSET. 
        Code now automatically determines degree of bivariate interpolants based on number of radial stations provided in input data. Factorized the GenerateMTFLOInput function. 
        Fixed transformation from planar to cylindrical coordinate system based on the implementation found in the BladeX module. Fixed implementation of circumferential blade thickness and blade slope. 
- V2.0: Removed grouping class to reduce import size in GA optimisation. Updated 1D interpolation to also dynamically change interpolation degree based on input dimension. 
        Updated documentation. Enforced y=0 in rotateProfile since it has no effect on MTFLOW evaluation.
- V2.1: Updated fileHandlingMTFLO to remove interpolation errors by reducing the number of interpolations and only generating inputs at the radial sections where geometry is being defined as input. 
        Switched internal methods to static methods to avoid needing to create dummy initialisations when specific internal methods are needed. 
- V2.2: Removed all interpolations in fileHandlingMTFLO in favor of direct geometry manipulation. 
"""

# Import standard libraries
from pathlib import Path
from typing import Optional

# Import 3rd party lbiraries
import numpy as np
from scipy import interpolate

# Handle local versus global execution of the file with imports
if __name__ == "__main__":
    from Parameterisations import AirfoilParameterisation  # type: ignore
else:
    from .Parameterisations import AirfoilParameterisation  # type: ignore


class fileHandlingMTSET:
    """
    Class for handling the generation of the MTSET input file (walls.xxx).
    
    This class provides methods to generate the input file containing the axisymmetric 
    bodies present within the domain. It handles the calculation of grid sizes and 
    profile coordinates for both the center body and duct.
    """


    def __init__(self, 
                 params_CB: dict,
                 params_duct: dict,
                 analysis_name: str,
                 ref_length: float,
                 external_input : bool = False,
                 domain_boundaries : Optional[list[float]] = None,
                 ) -> None:
        """
        Initialize the fileHandlingMTSET class.
    
        This method sets up the initial state of the class.

        Parameters
        ----------
        - params_CB : dict
            Dictionary containing parameters for the centerbody.
        - params_duct : dict
            Dictionary containing parameters for the duct.
        - analysis_name : str
            Name of the case being handled.
        - ref_length : float
            The reference length used by MTFLOW to non-dimensionalise all the dimensions.
        - external_input : bool, optional
            A control boolean to bypass the input of the "proper" centerbody and duct dictionaries. This is 
            useful when debugging or running cases where pre-existing geometry is to be used, rather than parameterised geometry. 
        - domain_boundaries : list[float], optional
            A list containing the grid boundaries in the format [XFRONT, XREAR, YBOT, YTOP]. Note that these boundaries must already be non-dimensionalised by the reference length!
        """

        # Input validation
        # Required keys for parameter dictionaries
        required_keys = {"Leading Edge Coordinates",
                         "Chord Length",
                         "b_0", "b_2", "b_8", "b_15", "b_17",
                         "x_t", "y_t", "x_c", "y_c",
                         "z_TE", "dz_TE", "r_LE",
                         "trailing_wedge_angle",
                         "trailing_camberline_angle",
                         "leading_edge_direction",
                         }
        
        if domain_boundaries is None:
            domain_boundaries = [None, None, None, None]
        self.domain_boundaries = domain_boundaries

        # Only perform complete input validation if the input dictionaries contain data
        keys_to_check = {"Leading Edge Coordinates", "Chord Length"}
        if not external_input:
            keys_to_check = required_keys
              
        for params, name in [(params_CB, "params_CB"), (params_duct, "params_duct")]:
            missing_keys = keys_to_check - set(params.keys())
            if missing_keys:
                raise ValueError(f"Missing required keys in {name}: {missing_keys}")

        if not isinstance(analysis_name, str):
            raise TypeError("analysis_name must be a string")
            
        if ref_length <= 0:
            raise ValueError("ref_length must be a positive float")
            
        self.centerbody_params = params_CB
        self.duct_params = params_duct
        self.analysis_name = analysis_name
        self.ref_length = ref_length
        self.external_input = external_input

        # Define the Grid size calculation constants
        self.DEFAULT_Y_TOP = 1.0
        self.Y_TOP_MULTIPLIER = 2.5
        self.X_FRONT_OFFSET = 1.0
        self.X_AFT_OFFSET = 1.0

        # Define key paths/directories
        self.parent_dir = Path(__file__).resolve().parent.parent
        self.submodels_path = self.parent_dir / "Submodels"

        # Initialise the airfoil parameterisation class
        self.parameterisation = AirfoilParameterisation()
                        

    def GetGridSize(self) -> list[float, float, float, float]:
        """
        Determine grid size - x & y coordinates of grid boundary. 
        Non-dimensionalises all geometry based on the reference length, in accordance with the MTFLOW documentation. 

        Returns
        -------
        - list[float, float, float, float]
            A list containing the grid boundaries in the format [XFRONT, XREAR, YBOT, YTOP]
        """

        # If all boundaries are provided, return them directly
        if all(entry is not None for entry in self.domain_boundaries):
            return self.domain_boundaries
            
        X_FRONT = self.domain_boundaries[0]
        X_AFT = self.domain_boundaries[1]
        Y_TOP = self.domain_boundaries[3]
        Y_BOT = 0.

        # Only computes the domain boundaries if they are not provided as an input
        if X_FRONT is None:
            # Calculate X-domain front boundary
            X_FRONT = round((min(self.duct_params["Leading Edge Coordinates"][0], 
                            self.centerbody_params["Leading Edge Coordinates"][0]) - self.X_FRONT_OFFSET * self.ref_length) / self.ref_length,
                            3)
        if X_AFT is None:
            # Calculate X-domain aft boundary
            X_AFT = round((max(self.duct_params["Leading Edge Coordinates"][0] + self.duct_params["Chord Length"], 
                          self.centerbody_params["Leading Edge Coordinates"][0] + self.centerbody_params["Chord Length"]) + self.X_AFT_OFFSET * self.ref_length) / self.ref_length,
                          3)
        if Y_TOP is None:
            # Calculate upper Y-domain boundary
            Y_TOP = round(max(self.DEFAULT_Y_TOP, 
                             self.Y_TOP_MULTIPLIER * self.duct_params["Leading Edge Coordinates"][1] / self.ref_length),
                          3)
        
        return [X_FRONT, X_AFT, Y_BOT, Y_TOP]


    def GetProfileCoordinates(self,
                              airfoil_parameters: dict,
                              ) -> np.typing.NDArray[np.floating]:
        """
        Compute the profile coordinates of an airfoil based on given design parameters and leading edge coordinates.
        Note that the outputs are still dimensional, and are yet to be non-dimensionalised!

        Parameters:
        -----------
        - airfoil_parameters : dict
            Dictionary of design parameters for the profile parameterisation.

        Returns:
        --------
        - np.typing.NDArray[np.floating]
            A 2D numpy array of shape (N, 2) containing the x and y coordinates of the airfoil profile.
        """
            
        # Compute the profile x,y coordinates
        upper_x, upper_y, lower_x, lower_y = self.parameterisation.ComputeProfileCoordinates(airfoil_params=airfoil_parameters)
        
        # Multiply with chord length to get correct profile dimensions
        upper_x = upper_x * airfoil_parameters["Chord Length"]
        lower_x = lower_x * airfoil_parameters["Chord Length"]
        upper_y = upper_y * airfoil_parameters["Chord Length"]
        lower_y = lower_y * airfoil_parameters["Chord Length"]

        # Offset airfoil using the (dX, dY) input to shift the profile coordinates to the appropriate 
        # location within the domain 
        upper_x += airfoil_parameters["Leading Edge Coordinates"][0]
        lower_x += airfoil_parameters["Leading Edge Coordinates"][0]
        upper_y += airfoil_parameters["Leading Edge Coordinates"][1]
        lower_y += airfoil_parameters["Leading Edge Coordinates"][1]
        
        # Combine the upper and lower profile coordinates to get the complete coordinate set
        x = np.concatenate((np.flip(upper_x), lower_x[1:]), axis=0) 
        y = np.concatenate((np.flip(upper_y), lower_y[1:]), axis=0) 

        return np.vstack((x, y)).T
        
        
    def GenerateMTSETInput(self,
                           xy_centerbody: Optional[tuple[np.typing.NDArray[np.floating]]] = None,
                           xy_duct: Optional[tuple[np.typing.NDArray[np.floating]]] = None,
                           ) -> None:
        """
        Write the MTSET input file walls.xxx for the given case. 

        Parameters:
        -----------
        - xy_centerbody : tuple[np.typing.NDArray[np.floating]], optional
            Tuple containing the x and y coordinates of the centerbody profile.
        - xy_duct : tuple[np.typing.NDArray[np.floating]], optional
            Tuple containing the x and y coordinates of the duct profile.
        """

        domain_boundaries = self.GetGridSize()

        # Get profiles of centerbody and duct
        # These can be optionally input into the function to bypass the airfoil parameterisation routines. This is 
        # useful if existing geometry is being used, for which no parameterisation has been generated. 
        if xy_centerbody is None and not self.external_input:
            xy_centerbody = self.GetProfileCoordinates(self.centerbody_params)
        if xy_duct is None and not self.external_input:
            xy_duct = self.GetProfileCoordinates(self.duct_params)
            
        # Non-dimensionalise the profile coordinates
        xy_centerbody = np.round(xy_centerbody / self.ref_length, 5)

        # Generate walls.xxx input data structure
        file_path = self.submodels_path / "walls.{}".format(self.analysis_name)
        with open(file_path, "w") as file:
            # Write opening lines of the file
            file.write(self.analysis_name + '\n')
            file.write('    '.join(map(str, domain_boundaries)) + '\n')

            # Write centerbody profile coordinates, using a tab delimiter
            for row in xy_centerbody:
                file.write('    '.join(map(str, row)) + '\n')
        
            # Write duct profile coordinates, using a tab delimiter
            # Only write it if a duct is defined, otherwise skip this step
            if xy_duct is not None:
                # Write separator line, using a tab delimiter
                file.write('    '.join(map(str, [999., 999.])) + '\n')

                # Non-dimensionalise the duct coordinates
                xy_duct = np.round(xy_duct / self.ref_length, 5)

                for row in xy_duct:
                    file.write('    '.join(map(str, row)) + '\n')


class fileHandlingMTFLO:
    """
    Class for handling the generation of the MTFLO input file (tflow.xxx).
        
    This class provides methods to generate the input file containing the blade rows (rotors and stators).
    """


    def __init__(self,
                 analysis_name: str,
                 ref_length: float,
                 centerbody_rotor_thickness: float = 0.18,
                 ) -> None:
        """
        Initialize the fileHandlingMTFLO class.
        This method sets up the initial state of the class.

        Parameters
        ----------
        - analysis_name : str
            Name of the case being handled.
        - ref_length : float
            The reference length used by MTFLOW to non-dimensionalise all the dimensions.
        - centerbody_rotor_thickness : float, optional
            The cutoff radius in meters below which we do not check the circumferential thickness limit to avoid numerical false triggers.
        """

        self.analysis_name= analysis_name
        self.ref_length = ref_length
        self.CENTERBODY_ROTOR_THICKNESS = centerbody_rotor_thickness

        # Define key paths/directories
        self.parent_dir = Path(__file__).resolve().parent.parent
        self.submodels_path = self.parent_dir / "Submodels"

        # Create a common airfoil parameterisation class
        self.parameterisation = AirfoilParameterisation()


    def ValidateBladeThickness(self,
                               local_thickness: float,
                               local_radius: float,
                               blade_count: int) -> None:
        """
        Validate that blade thickness doesn't exceed the complete blockage limit.
        If radius is less than self.CENTERBODY_ROTOR_THICKNESS, the function does nothing.

        Parameters
        ----------
        - local_thickness : float
            The local profile thickness
        - local_radius : float
            The local radius of the blade-to-blade plane
        - blade_count : int
            The total number of blades in the blade-to-blade plane
        """
        
        thickness_limit = 2 * np.pi * local_radius / blade_count
        if local_thickness >= thickness_limit and local_radius > self.CENTERBODY_ROTOR_THICKNESS:
            raise ValueError(f"The cumulative blade thickness exceeds the complete blockage limit of 2PIr at r={local_radius}")
        

    @staticmethod
    def RotateProfile(pitch: float,
                      x_u: np.typing.NDArray[np.floating],
                      x_l: np.typing.NDArray[np.floating],
                      y_u: np.typing.NDArray[np.floating],
                      y_l: np.typing.NDArray[np.floating],
                      ) -> tuple[np.typing.NDArray[np.floating]]:
        """
        Rotate a set of x,y coordinates counter-clockwise over the specified angle.

        Parameters
        ----------
        - pitch : float
            The blade pitch angle in radians.
        - x_u : np.typing.NDArray[np.floating]
            The upper surface x-coordinates.
        - x_l : np.typing.NDArray[np.floating]
            The lower surface x-coordinates.
        - y_u : np.typing.NDArray[np.floating]
            The upper surface y-coordinates.
        - y_l : np.typing.NDArray[np.floating]
            The lower surface y-coordinates

        Returns
        -------
        - rotated_upper_x : np.typing.NDArray[np.floating]
            The x-coordinates of the rotated upper surface.
        - rotated_upper_y : np.typing.NDArray[np.floating]
            The y-coordinates of the rotated upper surface.
        - rotated_lower_x : np.typing.NDArray[np.floating]
            The x-coordinates of the rotated lower surface.
        - rotated_lower_y : np.typing.NDArray[np.floating]
            The y-coordinates of the rotated lower surface.
        """

        # Construct the rotation matrix
        rotation_angle = np.pi / 2 - pitch
        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                    [np.sin(rotation_angle), np.cos(rotation_angle)]])
        
        # Center the coordinates at the mid x and y points
        mid_x = (x_u[-1] + x_u[0]) / 2
        mid_y = ((y_u + y_l) / 2)[len(y_u) // 2]
        shifted_upper_x = x_u - mid_x
        shifted_lower_x = x_l - mid_x
        shifted_upper_y = y_u - mid_y
        shifted_lower_y = y_l - mid_y

        # Perform rotation.
        rotated_upper_points = np.dot(np.column_stack((shifted_upper_x, shifted_upper_y)), rotation_matrix.T)
        rotated_lower_points = np.dot(np.column_stack((shifted_lower_x, shifted_lower_y)), rotation_matrix.T)

        # Shift the rotated coordinates back to the original coordinate system
        rotated_upper_x = rotated_upper_points[:,0] + mid_x
        rotated_upper_y = rotated_upper_points[:,1] + mid_y
        rotated_lower_x = rotated_lower_points[:,0] + mid_x
        rotated_lower_y = rotated_lower_points[:,1] + mid_y

        # Additionally, shift the LE of the profiles back to y = 0 since we do not use rake in the profiles.
        # MTFLOW solves the meridional flowfield, so the LE y-distribution has no effect on the flowfield, meaning we can fix it at 0.
        rotated_upper_y -= rotated_upper_y[0]
        rotated_lower_y -= rotated_lower_y[0]
        
        return rotated_upper_x, rotated_upper_y, rotated_lower_x, rotated_lower_y


    @staticmethod
    def PlanarToCylindrical(y_u: np.typing.NDArray[np.floating],
                            y_l: np.typing.NDArray[np.floating],
                            r: float,
                            ) -> tuple[np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], np.typing.NDArray[np.floating]]:
        """
        Convert the planar airfoil coordinates to cylindrical coordinates.

        Parameters
        ----------
        - y_u : np.typing.NDArray[np.floating]
            The y-coordinates of the upper surface.
        - y_l : np.typing.NDArray[np.floating]
            The y-coordinates of the lower surface.
        - r : float
            The radius of the cylindrical surface.
        
        Returns
        -------
        - y_section_upper : np.typing.NDArray[np.floating]
            The y-coordinates of the upper surface in cylindrical coordinates.
        - y_section_lower : np.typing.NDArray[np.floating]
            The y-coordinates of the lower surface in cylindrical coordinates.
        - y_camber : np.typing.NDArray[np.floating]
            The y-coordinates of the camber line in cylindrical coordinates.
        - z_section_upper : np.typing.NDArray[np.floating]
            The z-coordinates of the upper surface in cylindrical coordinates.
        - z_section_lower : np.typing.NDArray[np.floating]
            The z-coordinates of the lower surface in cylindrical coordinates.
        - z_camber : np.typing.NDArray[np.floating]
            The z-coordinates of the camber line in cylindrical coordinates.
        """

        # Compute camber from the upper and lower surfaces
        camber = (y_u + y_l) / 2
                
        # Compute the theta angles
        if r == 0:
            # Ensure proper handling of calculations at the centerline where the radius is zero.
            theta_up = y_u
            theta_low = y_l
            theta_camber = camber
        else:
            theta_up = y_u / r
            theta_low = y_l / r
            theta_camber = camber / r

        y_section_upper = r * np.sin(theta_up)
        y_section_lower = r * np.sin(theta_low)
        y_camber = r * np.sin(theta_camber)

        z_section_upper = r * np.cos(theta_up)
        z_section_lower = r * np.cos(theta_low)
        z_camber = r * np.cos(theta_camber)

        return y_section_upper, y_section_lower, y_camber, z_section_upper, z_section_lower, z_camber
        

    @staticmethod
    def CircumferentialThickness(y_u: np.typing.NDArray[np.floating],
                                 z_u: np.typing.NDArray[np.floating],
                                 y_l: np.typing.NDArray[np.floating],
                                 z_l: np.typing.NDArray[np.floating],
                                 r: float) -> np.typing.NDArray[np.floating]:
        """
        Compute the circumferential blade thickness distribution

        Parameters
        ----------
        - y_u : np.typing.NDArray[np.floating]
            The y-coordinates of the upper surface.
        - z_u : np.typing.NDArray[np.floating]
            The z-coordinates of the upper surface.
        - y_l : np.typing.NDArray[np.floating]
            The y-coordinates of the lower surface.
        - z_l : np.typing.NDArray[np.floating]
            The z-coordinates of the lower surface.
        - r : float
            The radius of the cylindrical surface.

        Returns
        -------
        - r * angle_range : np.typing.NDArray[np.floating]
            An array representing the circumferential thickness distribution along the blade profile, 
            calculated as the arc length subtended by the angular separation between the upper and lower surfaces.
        """
            
        # Compute subtended angles of the upper and lower surfaces
        theta_upper = np.atan2(y_u, z_u)
        theta_lower = np.atan2(y_l, z_l)

        # Compute the angle range by substracting the upper and lower angle distributions
        angle_range = np.abs(theta_upper - theta_lower)

        # Return the circumferential blade thickness, i.e. the azimuthal blade thickness, using the arc length formula
        return r * angle_range


    @staticmethod
    def GeometricBladeSlope(y_camber: np.typing.NDArray[np.floating],
                            x_camber: np.typing.NDArray[np.floating],
                            z_camber: np.typing.NDArray[np.floating],
                            ) -> tuple[np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], np.typing.NDArray[np.floating]]:
        """
        Compute the geometric blade slope dtheta/dm' in the m'-theta coordinate system. 
        
        Parameters
        ----------
        - y_camber : np.typing.NDArray[np.floating]
            Array of y-coordinates of the (rotated) camber distribution in the cylindrical coordinate system.
        - x_camber : np.typing.NDArray[np.floating]
            Array of x-coordinates of the (rotated) camber distribution in the cylindrical coordinate system.
        - z_camber : np.typing.NDArray[np.floating]
            Array of z-coordinates of the (rotated) camber distribution in the cylindrical coordinate system.

        Returns
        -------
        - blade_slope : np.typing.NDArray[np.floating]
            An array containing the geometric blade slope at every x-coordinate along the blade profile. 
        - m_prime : np.typing.NDArray[np.floating]
            An array containing the m' coordinates at which the blade slope is defined. 
        - theta : np.typing.NDArray[np.floating]
            An array containing the circumferential angles theta of the camber line along the x-coordinates of the blade profile. 
        """               
            
        # Compute the circumferential angle theta
        theta = np.atan2(y_camber, z_camber)
            
        # Compute the radius of the meridional plane
        r = np.sqrt(np.square(y_camber) + np.square(z_camber))

        # Compute the m' coordinate.
        m_prime = np.zeros_like(r) # Initialize m_prime as array of zeros.
        dr = np.diff(r)
        dx = np.diff(x_camber)
        radius_factor = r[:-1] + r[1:] # Equivalent to r[j] + r[j-1]
        distance = np.sqrt(np.square(dr) + np.square(dx))

        if np.all(r == 0):
            # Handle the case where r=0, i.e. the centerline.
            m_prime = x_camber
        else:
            m_prime[1:] = np.cumsum(2 / radius_factor * distance)

        # Construct a cubic spline of the theta-m' curve
        # The blade slope is then found simply by evaluating the gradient of the spline.
        spline = interpolate.make_splrep(m_prime,
                                         theta,
                                         k=3,
                                         s=0)
                                             
        blade_slope = spline(m_prime,
                             nu=1)
            
        return blade_slope, m_prime, theta
    

    def plot_blade_data(self,
                        stage: int,
                        radial_point: float,
                        x_points: np.typing.NDArray[np.floating],
                        rotated_upper_x: np.typing.NDArray[np.floating],
                        rotated_upper_y: np.typing.NDArray[np.floating],
                        rotated_lower_x: np.typing.NDArray[np.floating],
                        rotated_lower_y: np.typing.NDArray[np.floating],
                        circumferential_thickness: np.typing.NDArray[np.floating],
                        blade_slope: np.typing.NDArray[np.floating],
                        m_prime: np.typing.NDArray[np.floating],
                        theta: np.typing.NDArray[np.floating]) -> None:
        """
        Generate plots visualising the blade geometry. 

        Parameters
        ----------
        - stage : int
            The stage number of the blade row.
        - radial_point : float
            The radial point along the blade span.
        - x_points : np.typing.NDArray[np.floating]
            The x-coordinates of the blade profile.
        - rotated_upper_x : np.typing.NDArray[np.floating]
            The x-coordinates of the upper surface of the blade profile.
        - rotated_upper_y : np.typing.NDArray[np.floating] 
            The y-coordinates of the upper surface of the blade profile.
        - rotated_lower_x : np.typing.NDArray[np.floating]
            The x-coordinates of the lower surface of the blade profile.
        - rotated_lower_y : np.typing.NDArray[np.floating]
            The y-coordinates of the lower surface of the blade profile.
        - circumferential_thickness : np.typing.NDArray[np.floating]
            The circumferential thickness distribution along the blade profile.
        - blade_slope : np.typing.NDArray[np.floating]
            The geometric blade slope distribution along the blade profile.
        - m_prime : np.typing.NDArray[np.floating]
            The m' coordinates along the blade profile.
        - theta : np.typing.NDArray[np.floating]
            The circumferential angles theta of the camber line along the blade profile.
        """
        
        import matplotlib.pyplot as plt

        plt.figure(1)
        plt.xlabel("Axial coordinate [m]")
        plt.ylabel("Y coordinate [m]")
        plt.title(f"Rotated profile sections for stage {stage}")
        plt.plot(np.concatenate((rotated_upper_x, np.flip(rotated_lower_x)), axis=0), np.concatenate((rotated_upper_y, np.flip(rotated_lower_y)), axis=0), label=f"R={round(radial_point, 2)} m")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1,1))
        plt.tight_layout()

        plt.figure(2)
        plt.xlabel("Axial coordinate [m]")
        plt.ylabel("Circumferential Blade Thickness $T_\\theta$ [m]")
        plt.title(f"Circumferential thickness distributions for stage {stage}")
        plt.plot(x_points, circumferential_thickness, label=f"R={round(radial_point, 2)} m")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1,1))
        plt.tight_layout()

        plt.figure(3)
        plt.xlabel("Normalised meridional coordinate $||m'||=\\frac{m'}{max(m')}$ [-]")
        plt.ylabel("Blade angle $\\beta$ [deg]")
        plt.title(f"Blade angle distributions for stage {stage}")
        plt.plot(m_prime / m_prime[-1], np.degrees(np.atan(blade_slope)), label=f"R={round(radial_point, 2)} m")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1,1))
        plt.tight_layout()

        plt.figure(4)
        plt.xlabel("Normalised meridional coordinate $||m'||=\\frac{m'}{max(m')}$ [-]")
        plt.ylabel("Circumferential angle $\\theta$ [deg]")
        plt.title(f"Circumferential angle distributions for stage {stage}")
        plt.plot(m_prime / m_prime[-1], np.degrees(theta), label=f"R={round(radial_point, 2)} m")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1,1))
        plt.tight_layout()

        plt.figure(5)
        plt.xlabel("Normalised meridional coordinate $||m'||=\\frac{m'}{max(m')}$ [-]")
        plt.ylabel("Camberline blade slope $\\frac{d \\theta}{dm'}$ [-]")
        plt.title(f"Blade slope distributions for stage {stage}")
        plt.plot(m_prime / m_prime[-1], blade_slope, label=f"R={round(radial_point, 2)} m")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1,1))
        plt.tight_layout()


    @staticmethod
    def _extract_constant_spaced_x_indices(target_array: np.typing.NDArray | list,
                                           sample_size: int) -> list[int]:
        """
        Given a sorted 1D numpy array target_array containing cosine-spaced airfoil x-coordinates,
        this function extracts sample_size values from x_array with an approximately constant spacing.
        It does so by generating the ideal target points and then selecting, for each target,
        the closest value that exists in the original array.
    
        Parameters
        ----------
            - target_array: np.typing.NDArray | list
                1D sorted numpy array or list of x-coordinates.
            - sample_size: int
                The desired number of evenly spaced output samples.
      
        Returns
        -------
        - selected_indices : list[int]
            A list containing the indices corresponding to the constant spaced values.
        """      

        # Create the target values 
        x_targets = np.linspace(target_array[0], target_array[-1], sample_size)

        selected_indices = []
        last_idx = 0  # Ensure selected indices are in order 

        for target in x_targets:
            idx = np.argmin(np.abs(target_array - target))
            if idx < last_idx:
                idx = last_idx  # Avoid traversing the array backwards
            
            selected_indices.append(idx)
            last_idx = idx
        
        return selected_indices
    
        
    def GenerateMTFLOInput(self,
                           blading_params: np.ndarray[dict],
                           design_params: np.ndarray[dict],
                           plot: bool = False,
                           ) -> None:
        """
        Write the MTFLO input file for the given case.

        Parameters:
        -----------
        - blading_params : np.ndarray[dict]
            Array containing the blading parameters for each stage. Each dictionary should include the following keys:
            - "root_LE_coordinate": The leading edge coordinate at the root of the blade.
            - "rotational_rate": The rotational rate of the blade.
            - "blade_count": The number of blades.
            - "radial_stations": Numpy array of the radial stations along the blade span.
            - "ref_blade_angle": The set angle of the blades.
            - "reference_section_blade_angle": The blade angle at the reference section of the blade span. This is used as the value on which the other blade angles are computed.
            - "chord_length": Numpy array of the chord length distribution along the blade span.
            - "sweep_angle": Numpy array of the sweep angle distribution along the blade span.
            - "blade_angle": Numpy array of the twist angle distribution along the blade span.
        - design_params: np.ndarray[dict]
            Nested array containing an equal number of nests as there are stages. Each nested array has dictionary entries equal to the amount of radial stations considered. 
            Each dictionary must contain the following keys:
            - "b_0", "b_2", "b_8", "b_15", "b_17": Coefficients for the airfoil parameterisation.
            - "x_t", "y_t", "x_c", "y_c": Coordinates for the airfoil parameterisation.
            - "z_TE", "dz_TE": Trailing edge parameters.
            - "r_LE": Leading edge radius.
            - "trailing_wedge_angle": Trailing wedge angle.
            - "trailing_camberline_angle": Trailing camberline angle.
            - "leading_edge_direction": Leading edge direction.
            - "Chord Length": The chord length of the blade.
        - plot : bool, optional
            An optional controlling boolean to decide if plots are to be created of the parameters of interest. Default value is False. 
        """

        # Open the tflow.xxx file and start writing the required input data to it
        file_path = self.submodels_path / "tflow.{}".format(self.analysis_name)
        with open(file_path, "w") as file:
            # Write the case name to the file
            file.write('NAME\n')
            file.write(f"{str(self.analysis_name)}\n")
            file.write('END\n \n')

            # Loop over the number of stages and write the data for each stage
            for stage in range(len(blading_params)):
                # First write the "generic" data for the stage
                # This includes the number of blades, the rotational rate, and the data types to be provided
                # Formatting is in-line with the user guide from the MTFLOW documentation

                # Start a stage block
                file.write('STAGE\n \n')

                # Write the number of blades
                file.write('NBLADE\n')
                file.write(str(blading_params[stage]["blade_count"]) + '\n')
                file.write('END\n \n')

                # Write the rotational rate in units of RPS * 2pi * L / V
                file.write('OMEGA\n')
                file.write(str(blading_params[stage]["rotational_rate"]) + '\n')
                file.write('END\n \n')

                # Write the data types to be provided for the stage
                file.write('DATYPE \n')
                file.write('x    r    T    Sr\n')  # Use the x,r coordinates, together with thickness and blade slope
                multipliers = [round(1 / self.ref_length, 6),  round(1 / self.ref_length, 6), round(1 / self.ref_length, 6), 1]  # Add multipliers for each data type
                additions = [0, 0, 0, 0]  # Add additions for each data type
                file.write('*' + '    '.join(map(str, multipliers)) + '\n')
                file.write('+' + '    '.join(map(str, additions)) + '\n')
                file.write('END\n \n')
                    
                # The MTFLO code cannot accept an input file with more than 16x16 points in the streamwise and radial directions for each stage
                # n_points_axial=10 is used to avoid spline interpolation overshoots internally in MTFLO. 
                n_points_axial = 10
                radial_points = blading_params[stage]["radial_stations"]

                # Loop over the radial points and construct the data for each radial point
                # Each radial point is defined as a "section" within the input file
                for idx, r in enumerate(radial_points): 
                    # Create a section in the input file
                    file.write('SECTION\n')

                    # All parameters are normalised using the local chord length, so we need to obtain the local chord in order to obtain the dimensional parameters
                    local_chord = blading_params[stage]["chord_length"][idx]
                   
                    # Create complete airfoil representation from BP3434 parameterisation of the radial section
                    upper_x, upper_y, lower_x, lower_y = self.parameterisation.ComputeProfileCoordinates(design_params[stage][idx])
                    upper_x *= local_chord
                    upper_y *= local_chord
                    lower_x *= local_chord
                    lower_y *= local_chord
                    
                    # Rotate the airfoil profile to the correct angle
                    # The blade pitch is defined with respect to the blade pitch angle at the reference radial station, and thus is corrected accordingly. 
                    blade_pitch = (blading_params[stage]["blade_angle"][idx] + blading_params[stage]["ref_blade_angle"] - blading_params[stage]["reference_section_blade_angle"])
                    rotated_upper_x, rotated_upper_y, rotated_lower_x, rotated_lower_y  = self.RotateProfile(blade_pitch,
                                                                                                             upper_x,
                                                                                                             lower_x,
                                                                                                             upper_y,
                                                                                                             lower_y)

                    # Compute the local leading edge offset at the radial station from the provided interpolant
                    # Use it to offset the x-coordinates of the upper and lower surfaces to the correct position
                    LE_coordinate = blading_params[stage]["root_LE_coordinate"] + r * np.tan(blading_params[stage]["sweep_angle"][idx])
                    rotated_upper_x += LE_coordinate - rotated_upper_x[0]
                    rotated_lower_x += LE_coordinate - rotated_lower_x[0]

                    # Transform the 2D planar airfoils into 3D cylindrical sections
                    y_section_upper, y_section_lower, y_camber, z_section_upper, z_section_lower, z_camber = self.PlanarToCylindrical(rotated_upper_y,
                                                                                                                                      rotated_lower_y,
                                                                                                                                      r)
                    
                    # Compute the circumferential blade thickness
                    if r == 0:
                        # Handle the case at the centerline, where we define the thickness to be zero. 
                        circumferential_thickness = np.zeros_like(upper_x)
                    else:
                        circumferential_thickness = self.CircumferentialThickness(y_section_upper,
                                                                                  z_section_upper,
                                                                                  y_section_lower,
                                                                                  z_section_lower,
                                                                                  r)
                        
                        # Perform check that thickness does not exceed limit of complete blockage (T=2pir/N)
                        # If thickness exceeds limit, raises a ValueError
                        self.ValidateBladeThickness(max(circumferential_thickness), r, blading_params[stage]["blade_count"])

                    # Compute the blade slope in the m'-theta plane. 
                    # Uses the average of the upper and lower x-coordinates to evaluate against. 
                    x_points = (rotated_lower_x + rotated_upper_x) / 2  
                    blade_slope, m_prime, theta = self.GeometricBladeSlope(y_camber,
                                                                           x_points,
                                                                           z_camber)       

                    if plot:
                        self.plot_blade_data(stage,
                                             r,
                                             x_points,
                                             rotated_upper_x,
                                             rotated_upper_y,
                                             rotated_lower_x,
                                             rotated_lower_y,
                                             circumferential_thickness,
                                             blade_slope,
                                             m_prime,
                                             theta)

                    # Compute the sampling indices for the axial points
                    sampling_indices = self._extract_constant_spaced_x_indices(x_points,
                                                                               n_points_axial)
                                                  
                    # Loop over the streamwise points and construct the data for each streamwise point
                    # Each data point consists of the data [x / Lref, r / Lref, T / Lref, Srel]
                    for j in range(n_points_axial):  
                        # Write data to row
                        row = np.array([round((x_points[sampling_indices][j]), 5),
                                        round(r, 5),
                                        round(circumferential_thickness[sampling_indices][j], 5),
                                        round(blade_slope[sampling_indices][j], 5),
                                        ])
                        
                        # Write the row to the file
                        file.write('    '.join(map(str, row)) + '\n')

                    # End the radial section
                    file.write('END\n')
                
                if plot:
                    # Only display plots if the plot boolean is True
                    import matplotlib.pyplot as plt
                    plt.show() 
                 
                # End the stage 
                file.write('\nEND\n \n')
            # End the input file
            file.write('END\n')


if __name__ == "__main__":
    import time

    # Perform test generation of walls.xxx file using dummy inputs
    # Creates a dummy geometry for the centerbody and duct
    centre_body_coeff = {"b_0": 0., "b_2": 0., "b_8": 2.63935800e-02, "b_15": 7.62111322e-01, "b_17": 0, 'x_t': 0.2855061027842137, 'y_t': 0.07513718500645125, 'x_c': 0.5, 'y_c': 0, 'z_TE': -2.3750854491940602e-33, 'dz_TE': 0.0019396795056937765, 'r_LE': -0.01634872585955984, 'trailing_wedge_angle': 0.15684435833921387, 'trailing_camberline_angle': 0.0, 'leading_edge_direction': 0.0, "Chord Length": 2, "Leading Edge Coordinates": (0.3, 0)}
    n2415_coeff = {"b_0": 0.20300919575972556, "b_2": 0.31901972386590877, "b_8": 0.04184620466207193, "b_15": 0.7500824561993612, "b_17": 0.6789808614463232, "x_t": 0.298901583, "y_t": 0.060121131, "x_c": 0.40481558571382253, "y_c": 0.02025376839986754, "z_TE": -0.0003399582707130648, "dz_TE": 0.0017, "r_LE": -0.024240593156029916, "trailing_wedge_angle": 0.16738688797915346, "trailing_camberline_angle": 0.0651960639817597, "leading_edge_direction": 0.09407653642497815, "Chord Length": 2.5, "Leading Edge Coordinates": (0, 1.2)}
    n6409_coeff = {"b_0": 0.07979831, "b_2": 0.20013347, "b_8": 0.02901246, "b_15": 0.74993802, "b_17": 0.78496242, 'x_t': 0.30429947838135246, 'y_t': 0.0452171520304373, 'x_c': 0.4249653844429819, 'y_c': 0.06028051002570214, 'z_TE': -0.0003886462495685791, 'dz_TE': 0.0004425237127035188, 'r_LE': -0.009225474218611841, 'trailing_wedge_angle': 0.10293203348896998, 'trailing_camberline_angle': 0.21034003141636426, 'leading_edge_direction': 0.26559481057525414, "Chord Length": 2.5, "Leading Edge Coordinates": (0, 2)}
    
    starttime = time.time()
    call_class_MTSET = fileHandlingMTSET(centre_body_coeff, n6409_coeff, "test_case", 2)
    call_class_MTSET.GenerateMTSETInput()
    endtime = time.time()
    print("Execution of GenerateMTSETInput() took", endtime - starttime, "seconds")

    # Perform test generation of tflow.xxx file using dummy inputs
    # Creates an input file using 2 stages, a rotor and a stator
    blading_parameters = [{"root_LE_coordinate": 0.5, "rotational_rate": 1., "ref_blade_angle": np.deg2rad(19), "reference_section_blade_angle": np.deg2rad(34), "blade_count": 18, "radial_stations": [0.1, 1.8], "chord_length": [0.2, 0.2], "sweep_angle":[np.pi/16, np.pi/16], "blade_angle": [np.pi / 3, np.pi / 3]},
                          {"root_LE_coordinate": 1., "rotational_rate": 0., "ref_blade_angle": np.deg2rad(19), "reference_section_blade_angle": np.deg2rad(34), "blade_count": 10, "radial_stations": [0.1, 1], "chord_length": [0.2, 0.2], "sweep_angle":[np.pi/8, np.pi/8], "blade_angle": [np.pi / 3, np.pi/8]}]
    design_parameters = [[n2415_coeff, n2415_coeff],
                         [n2415_coeff, n2415_coeff]]
    
    starttime = time.time()
    call_class_MTFLO = fileHandlingMTFLO("test_case", 2)
    call_class_MTFLO.GenerateMTFLOInput(blading_parameters, 
                                        design_parameters)
    endtime = time.time()
    print("Execution of GenerateMTFLOInput() took", endtime - starttime, "seconds")