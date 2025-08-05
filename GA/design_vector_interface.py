"""
design_vector_interface
=======================

Description
-----------
This module provides utilities for efficient access and manipulation of design vectors used in optimization problems.

Classes
-------
DesignVectorInterface
    Class to provide efficient access to design vector elements and decompose design vectors into sub-dictionaries.

Examples
--------
>>> interface = DesignVectorInterface()
>>> interface.DeconstructDesignVector(x_dict)  # Decomposes the design vector into sub-dictionaries

Notes
-----
This module is designed to streamline the handling of design vectors in optimization workflows. It ensures efficient
access to elements and provides error handling for missing keys or indices.

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 2.0

Changelog:
- V1.0: Initial implementation.
- V1.1: Updated documentation and added examples for clarity.
- V1.2: Rework of design vector access. Inclusion of utils.ensure_repo_paths.
- V1.3: Added type hints for better code clarity and maintainability. Added reconstruct_design_vector method.
- V2.0: Renamed MTFLOW to UDFDAC for consistency with written thesis.
"""

# Import standard libraries
import copy

# Import 3rd party libraries
from scipy import interpolate
import numpy as np

# Ensure all paths are correctly setup
from utils import ensure_repo_paths  # type: ignore
ensure_repo_paths()

# Import interfacing modules
import config  # type: ignore
from Submodels.Parameterisations import AirfoilParameterisation  # type: ignore
_PARAMETERISATION = AirfoilParameterisation()

class DesignVectorInterface:
    """
    Simple class to provide efficient access to design vector elements without repeated string formatting,
    and to deconstruct the design vector dictionary x into the expected "sub"-dictionaries required to use the UDFDAC interface """


    def __init__(self) -> None:
        """
        Initialization of the DesignVectorInterface class

        Returns
        -------
        None
        """

        # Import control variables
        self.num_radial = config.NUM_RADIALSECTIONS
        self.num_stages = config.NUM_STAGES
        self.optimize_stages = config.OPTIMIZE_STAGE
        self.rotating = config.ROTATING

        # Initialize the AirfoilParameterisation class for slightly better memory usage
        self.Parameterisation = _PARAMETERISATION

        # Initialize empty ordered keys cache
        self._ordered_keys_cache: list[str] | None = None


    def ComputeDuctRadialLocation(self,
                                  duct_variables: dict,
                                  blade_blading_parameters: list[dict]) -> tuple[dict, list]:
        """
        Compute the y-coordinate of the LE of the duct based on the design variables.

        Parameters
        ----------
        - duct_variables : dict
            A dictionary containing the duct variables.
        - blade_blading_parameters : list[dict]
            The blading parameters for the turbomachinery stage(s).

        Returns
        -------
        - tuple:
            - duct_variables : dict
                The updated duct variables dictionary containing the updated LE y coordinate.
            - blade_blading_parameters : list[dict]
                THe updated blading parameters containing the updated radii of the stator stage(s).
        """

        # Initialize data array for the radial duct coordinates
        radial_duct_coordinates = np.zeros(self.num_stages)

        # Compute the duct x,y coordinates. Note that we are only interested in the lower surface.
        _, _, lower_x, lower_y = self.Parameterisation.ComputeProfileCoordinates(duct_variables)
        lower_x = lower_x * duct_variables["Chord Length"]
        lower_y = lower_y * duct_variables["Chord Length"]

        # Shift the duct x coordinate to the correct location in space
        lower_x += duct_variables["Leading Edge Coordinates"][0]

        # Construct cubic spline interpolant of the duct surface
        order = np.argsort(lower_x)  # Defensively sort the x coordinates to avoid a runtime failure
        dx = np.diff(lower_x[order])
        mask = np.hstack([True, dx > 1e-12]) # keep first point, drop exact duplicates
        order = order[mask]
        duct_interpolant = interpolate.CubicSpline(lower_x[order],
                                                   np.abs(lower_y)[order],  # Take absolute value of y-coordinates since we need the distance, not the actual coordinate
                                                   extrapolate=False)

        x_min, x_max = lower_x[order[0]], lower_x[order[-1]]
        tip_gap = config.tipGap
        for i in range(self.num_stages):
            if not self.rotating[i]:
                continue

            # Extract blading and blade radius
            blading = blade_blading_parameters[i]
            y_tip = blading["radial_stations"][-1]

            # Compute the LE and TE x-coordinates of the tip section
            sweep = np.tan(blading["sweep_angle"][-1])
            x_tip_LE = blading["root_LE_coordinate"] + sweep * y_tip
            projected_chord = blading["chord_length"][-1] * np.cos(np.pi/2 -
                                                                   (blading["blade_angle"][-1] + blading["ref_blade_angle"] - blading["reference_section_blade_angle"]))
            x_tip_TE = x_tip_LE + projected_chord

            # Compute the offsets for the LE and TE of the blade tip
            LE_offset = 0 if not (x_min <= x_tip_LE <= x_max) else float(duct_interpolant(x_tip_LE))  # Set to 0 if duct does not lie above LE
            TE_offset = 0 if not (x_min <= x_tip_TE <= x_max) else float(duct_interpolant(x_tip_TE))  # Set to 0 if duct does not lie above TE

            # Compute the radial location of the duct
            radial_duct_coordinates[i] = y_tip + tip_gap + max(LE_offset, TE_offset)

        # The LE y coordinate of the duct is then the maximum of the computed coordinates to enforce the minimum tip gap everywhere
        if radial_duct_coordinates.any():
            LE_coordinate_duct = float(radial_duct_coordinates.max())
        else:
            LE_coordinate_duct = duct_variables["Leading Edge Coordinates"][1]

        # Update the duct variables
        duct_variables["Leading Edge Coordinates"] = (duct_variables["Leading Edge Coordinates"][0],
                                                      LE_coordinate_duct)

        # Set the radius of all stators equal to this y coordinate to avoid miss-matches in stator sizes.
        for i in range(self.num_stages):
            if not self.rotating[i]:
                r_old = blade_blading_parameters[i]["radial_stations"][-1]
                if r_old:
                    # Simple guard against r=0
                    blade_blading_parameters[i]["radial_stations"] = blade_blading_parameters[i]["radial_stations"] / r_old * LE_coordinate_duct
                else:
                    # If the last entry of radial stations is 0, simply set it to the LE coordinate of the duct. This avoids a divide-by-zero error.
                    blade_blading_parameters[i]["radial_stations"] = np.full_like(blade_blading_parameters[i]["radial_stations"],
                                                                                  LE_coordinate_duct,
                                                                                  dtype=float)

        # Return the updated data
        return duct_variables, blade_blading_parameters


    def SortDesignVector(self,
                         x_dict: dict[str, float | int]) -> list[float | int]:
        """
        Sort the pymoo design vector dictionary based on the keys "x*".

        Parameters
        ----------
        - x_dict : dict[str, float | int]
            The Pymoo design vector dictionary.

        Returns
        - sorted_x: list[float | int]
            A list of the sorted design vector values.
        """

        current_order = sorted(x_dict.keys(), key=lambda k: int(k.lstrip("x")))

        if self._ordered_keys_cache is None:
            self._ordered_keys_cache = current_order

        if self._ordered_keys_cache != current_order:
            self._ordered_keys_cache = current_order
        return [x_dict[k] for k in self._ordered_keys_cache]


    @staticmethod
    def Getb8(b_8_map: float,
              r_le: float,
              x_t: float,
              y_t: float) -> float:
        """
        Helper function to compute the bezier parameter b_8 using the mapping parameter 0 <= b_8_map <= 1
        """

        term = -2 * r_le * x_t / 3
        sqrt_term = 0 if term <= 0 else np.sqrt(term)
        factor = min(y_t, sqrt_term)
        return float(b_8_map * factor)


    def DeconstructDesignVector(self,
                                x_dict: dict[str, float | int],
                                compute_duct: bool = True) -> tuple:
        """
        Decompose the design vector x into dictionaries of all the design variables to match the expected input formats for
        the UDFDAC code interface.

        Parameters
        ----------
        - x_dict : dict[str, float | int]
            The Pymoo design vector dictionary.
        - compute_duct : bool, optional
            If True, compute the duct radial location based on the design variables. Default is True.

        Returns
        -------
        tuple
            A tuple containing the decomposed design variables:
            - centerbody_variables: dict
            - duct_variables: dict
            - blade_design_parameters: list[list[dict]]
            - blade_blading_parameters: list[list[dict]]
            - Lref: float
        """

        # First sort the design vector as we cannot guarantee that it is sorted
        ordered_values = self.SortDesignVector(x_dict)

        # Create an iterator over the design vector values
        it = iter(ordered_values)

        # Define a pointer to count the number of variable parameters
        centerbody_designvar_count = len(config.CENTERBODY_VALUES)
        duct_designvar_count = len(config.DUCT_VALUES)
        section_designvar_count = duct_designvar_count - 2  # -2 since the sections do not use chord length or LE x-coordinate as variable.

        # Deconstruct the centerbody values if it's variable.
        # If the centerbody is constant, read in the centerbody values from config.
        # Note that if the centerbody is variable, we keep the LE coordinate fixed, as the LE coordinate of the duct would already be free to move.
        if config.OPTIMIZE_CENTERBODY:
            try:
                centerbody_vals = [next(it) for _ in range(centerbody_designvar_count)]
            except StopIteration:
                raise ValueError("Design vector is too short for the expected centerbody variables.") from None
            centerbody_variables = {"b_0": 0,
                                    "b_2": 0,
                                    "b_8": self.Getb8(centerbody_vals[0], centerbody_vals[5], centerbody_vals[2], centerbody_vals[3]),
                                    "b_15": centerbody_vals[1],
                                    "b_17": 0,
                                    "x_t": centerbody_vals[2],
                                    "y_t": centerbody_vals[3],
                                    "x_c": 0,
                                    "y_c": 0,
                                    "z_TE": 0,
                                    "dz_TE": centerbody_vals[4],
                                    "r_LE": centerbody_vals[5],
                                    "trailing_wedge_angle": centerbody_vals[6],
                                    "trailing_camberline_angle": 0,
                                    "leading_edge_direction": 0,
                                    "Chord Length": centerbody_vals[7],
                                    "Leading Edge Coordinates": (0, 0)}
        else:
            centerbody_variables = copy.deepcopy(config.CENTERBODY_VALUES)

        # Deconstruct the duct values if it's variable.
        # If the duct is constant, read in the duct values from config.
        if config.OPTIMIZE_DUCT:
            try:
                duct_vals = [next(it) for _ in range(duct_designvar_count)]
            except StopIteration:
                raise ValueError("Design vector is too short for the expected duct variables.") from None
            duct_variables = {"b_0": duct_vals[0],
                              "b_2": duct_vals[1],
                              "b_8": self.Getb8(duct_vals[2], duct_vals[11], duct_vals[5], duct_vals[6]),
                              "b_15": duct_vals[3],
                              "b_17": duct_vals[4],
                              "x_t": duct_vals[5],
                              "y_t": duct_vals[6],
                              "x_c": duct_vals[7],
                              "y_c": duct_vals[8],
                              "z_TE": duct_vals[9],
                              "dz_TE": duct_vals[10],
                              "r_LE": duct_vals[11],
                              "trailing_wedge_angle": duct_vals[12],
                              "trailing_camberline_angle": duct_vals[13],
                              "leading_edge_direction": duct_vals[14],
                              "Chord Length": duct_vals[15],
                              "Leading Edge Coordinates": (duct_vals[16], 0)}
        else:
            duct_variables = copy.deepcopy(config.DUCT_VALUES)

        # Deconstruct the rotorblade parameters if they are variable.
        # If the rotorblade parameters are constant, read in the parameters from config.
        blade_design_parameters = []
        for stage in range(self.num_stages):
            # Initiate empty list for each stage
            stage_design_parameters = []
            if self.optimize_stages[stage]:
                # If the stage is to be optimized, read in the design vector for the blade profiles
                for _ in range(self.num_radial[stage]):
                    # Loop over the number of radial sections and append each section to stage_design_parameters
                    try:
                        section_vals = [next(it) for _ in range(section_designvar_count)]
                    except StopIteration:
                        raise ValueError("Design vector is too short for the expected blade radial section variables.") from None
                    section_parameters = {"b_0": section_vals[0],
                                        "b_2": section_vals[1],
                                        "b_8": self.Getb8(section_vals[2], section_vals[11], section_vals[5], section_vals[6]),
                                        "b_15": section_vals[3],
                                        "b_17": section_vals[4],
                                        "x_t": section_vals[5],
                                        "y_t": section_vals[6],
                                        "x_c": section_vals[7],
                                        "y_c": section_vals[8],
                                        "z_TE": section_vals[9],
                                        "dz_TE": section_vals[10],
                                        "r_LE": section_vals[11],
                                        "trailing_wedge_angle": section_vals[12],
                                        "trailing_camberline_angle": section_vals[13],
                                        "leading_edge_direction": section_vals[14]}
                    stage_design_parameters.append(section_parameters)
            else:
                # If the stage is meant to be constant, read it in from config.
                stage_design_parameters = copy.deepcopy(config.STAGE_DESIGN_VARIABLES[stage])
            # Write the stage nested list to blade_design_parameters
            blade_design_parameters.append(stage_design_parameters)

        blade_blading_parameters = []
        num_operating_conditions = len(config.multi_oper)
        for stage in range(self.num_stages):
            # Initiate empty list for each stage
            stage_blading_parameters = {}
            if self.optimize_stages[stage]:
                # If the stage is to be optimized, read in the design vector for the blading parameters
                stage_blading_parameters["root_LE_coordinate"] = next(it)

                ref_blade_angle_lst = []
                for _ in range(len(config.STAGE_BLADING_PARAMETERS[stage]["ref_blade_angle_lst"])):
                    ref_blade_angle_lst.append(next(it))  # Allow for multi-point variable pitch
                stage_blading_parameters["ref_blade_angle_lst"] = ref_blade_angle_lst
                stage_blading_parameters["ref_blade_angle"] = ref_blade_angle_lst[0]  # Initialise to the first pitch angle

                stage_blading_parameters["reference_section_blade_angle"] = 0  # We take the blade tip as reference section, so the angle is zero.
                stage_blading_parameters["blade_count"] = int(next(it))
                stage_blading_parameters["RPS_lst"] = [next(it) if self.rotating[stage] else 0 for _ in range(num_operating_conditions)]
                stage_blading_parameters["RPS"] = 0  # Initialize the RPS at zero - this will be overwritten later by the appropriate RPS for the operating condition.
                stage_blading_parameters["rotation_rate"] = 0  # Initialize the UDFDAC non-dimensional rotational rate to zero - this will be overwritten later by the appropriate Omega within the problem definition.
                stage_blading_parameters["radial_stations"] = np.linspace(0, 0.5 * next(it), self.num_radial[stage])  # Radial stations are defined as fraction of blade radius * local radius

                # Extract sectional blading parameter lists
                stage_blading_parameters["chord_length"] = [next(it) for _ in range(self.num_radial[stage])]
                sweep_angle = [0]
                sweep_angle.extend([next(it) for _ in range(self.num_radial[stage] - 1)])  # -1 since the root is independent of sweep
                stage_blading_parameters["sweep_angle"] = sweep_angle
                stage_blading_parameters["blade_angle"] = [next(it) for _ in range(self.num_radial[stage] - 1)]  # -1 since we fix the angle at the tip to zero to simplify the design.
                stage_blading_parameters["blade_angle"].append(0)  # Append the tip angle of zero to the list
            else:
                stage_blading_parameters = copy.deepcopy(config.STAGE_BLADING_PARAMETERS[stage])

            # Append the stage blading parameters to the main list
            blade_blading_parameters.append(stage_blading_parameters)

        # Compute the updated duct and blading parameters
        # This must happen after all blade parameters and duct parameters are constructed,
        # since the radial duct location affects the stator blade "diameter"
        if compute_duct:
            duct_variables, blade_blading_parameters = self.ComputeDuctRadialLocation(duct_variables=duct_variables,
                                                                                      blade_blading_parameters=blade_blading_parameters)

        # Write the reference length for UDFDAC
        Lref = blade_blading_parameters[0]["radial_stations"][-1] * 2  # The last entry in radial stations corresponds to the blade tip, so multiply by 2 to get the blade diameter

        return centerbody_variables, duct_variables, blade_design_parameters, blade_blading_parameters, Lref


    def ReconstructDesignVector(self,
                                centerbody_variables: dict[str, float],
                                duct_variables: dict[str, float],
                                blade_design_parameters: list[list[dict[str, float]]],
                                blade_blading_parameters: list[dict[str, float | int]],
                                ) -> dict[str, float | int]:
        """
        Based on the design vector dictionaries, reconstruct the design vector dictionary x.
        This is used to reconstruct the design vector after it has been modified by the repair operator.

        Parameters
        ----------
        - centerbody_variables : dict[str, float]
            A dictionary containing the centerbody variables.
        - duct_variables : dict[str, float]
            A dictionary containing the duct variables.
        - blade_design_parameters : list[list[dict[str, float]]]
            A list of lists of dictionaries containing the blade design parameters for each stage.
        - blade_blading_parameters : list[dict[str, float | int]]
            A list of dictionaries containing the blade blading parameters for each stage.

        Returns
        -------
        - vector : dict[str, float | int]
            A dictionary containing the reconstructed design vector x.
        """

        def extract_b8_map(params: dict[str, float]) -> float:
            """
            Helper function to compute the mapping parameter b_8_map using the bezier parameter b_8
            """

            term = -2 * params["r_LE"] * params["x_t"] / 3
            sqrt_term = 0 if term <= 0 else np.sqrt(term)
            factor = min(params["y_t"], sqrt_term)
            return float(params["b_8"] / factor)


        def profile_section_vars(profile: dict[str, float]) -> list[float]:
            """
            Return the standard 15-var profile section definition.
            """
            return [profile["b_0"],  # b_0
                    profile["b_2"],  # b_2
                    extract_b8_map(profile),  # b_8_map
                    profile["b_15"],  # b_15
                    profile["b_17"],  # b_17
                    profile["x_t"],  # x_t
                    profile["y_t"],  # y_t
                    profile["x_c"],  # x_c
                    profile["y_c"],  # y_c
                    profile["z_TE"],  # z_TE
                    profile["dz_TE"],  # dz_TE
                    profile["r_LE"],  # r_LE
                    profile["trailing_wedge_angle"],  # trailing_wedge_angle
                    profile["trailing_camberline_angle"],  # trailing_camberline_angle
                    profile["leading_edge_direction"]]

        # Initialize variable list
        vector = []

        if config.OPTIMIZE_CENTERBODY:
            # If the centerbody is to be optimised, reconstruct the corresponding variable section.
            vector.append(extract_b8_map(centerbody_variables))  # b_8_map
            vector.append(centerbody_variables["b_15"])  # b_15
            vector.append(centerbody_variables["x_t"])  # x_t
            vector.append(centerbody_variables["y_t"])  # y_t
            vector.append(centerbody_variables["dz_TE"])  # dz_TE
            vector.append(centerbody_variables["r_LE"])  # r_LE
            vector.append(centerbody_variables["trailing_wedge_angle"])  # trailing_wedge_angle
            vector.append(centerbody_variables["Chord Length"])  # Chord Length

        if config.OPTIMIZE_DUCT:
            # If the duct is to be optimised, reconstruct the corresponding variable section.
            vector.extend(profile_section_vars(duct_variables))
            vector.append(duct_variables["Chord Length"])  # Chord Length
            vector.append(duct_variables["Leading Edge Coordinates"][0])  # Leading Edge X-Coordinate

        for i in range(len(config.OPTIMIZE_STAGE)):
            # If (any of) the rotor/stator stage(s) are to be optimised, reconstruct the design variables for the profiles
            if config.OPTIMIZE_STAGE[i]:
                for j in range(config.NUM_RADIALSECTIONS[i]):
                    # Loop over the number of radial sections and append each section to stage_design_parameters
                    vector.extend(profile_section_vars(blade_design_parameters[i][j]))

        for i, opt_stage in enumerate(config.OPTIMIZE_STAGE):
            # Loop over the stages and write the blading parameters to the design vector
            if opt_stage:
                # If the stage is to be optimised, read in the design vector for the blading parameters
                vector.append(blade_blading_parameters[i]["root_LE_coordinate"])  # root_LE_coordinate
                for angle in blade_blading_parameters[i]["ref_blade_angle_lst"]:
                    vector.append(angle)  # Enables variable pitch to be used. 

                # vector.append(blade_blading_parameters[i]["ref_blade_angle"])  # ref_blade_angle
                vector.append(int(blade_blading_parameters[i]["blade_count"]))  # blade_count
                vector.extend([blade_blading_parameters[i]["RPS_lst"][j] for j in range(len(blade_blading_parameters[i]["RPS_lst"]))])  # blade RPS
                vector.append(blade_blading_parameters[i]["radial_stations"][-1] * 2)  # blade diameter

                vector.extend(blade_blading_parameters[i]["chord_length"])  # chord length
                vector.extend(blade_blading_parameters[i]["sweep_angle"][1:])  # Skip the root section since the root is independent of sweep
                vector.extend(blade_blading_parameters[i]["blade_angle"][:-1])  # -1 since we fix the angle at the tip to zero to simplify the design.

        # For a mixed-variable problem, PyMoo expects the vector to be a dictionary, so we convert vector to a dictionary.
        # Note that all variables are given a name xi.
        vector = {f"x{i}": var for i, var in enumerate(vector)}

        return vector