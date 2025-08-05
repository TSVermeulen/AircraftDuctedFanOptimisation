"""
repair
===========

Description
-----------
This module provides repair operators for evolutionary optimization, specifically for airfoil and blade parameterisations.
It ensures that individuals in the population satisfy geometric and physical constraints.

Classes
-------
RepairIndividuals
    A class derived from pymoo's Repair, implementing custom repair logic for airfoil and blade design variables.

Examples
--------
>>> repair = RepairIndividuals()
>>> repaired_population = repair._do(problem, population)

Notes
-----
This module is intended for use with the PyMoo optimization framework.

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Date [dd-mm-yyyy]: [08-06-2025]
Version: 1.5

Changelog:
- V1.0: Initial implementation of repair operators for profile parameterisations.
- V1.1: Added enforcement of positive sweepback for blade leading edge.
- V1.2: Improved one-to-one enforcement for Bezier curves.
- V1.3: Refactored repair logic and updated documentation. Improved robustness of one-to-one enforcing by including additonal equation for gamma_LE.
- V1.4: Made bounds on repair enforce_one2one a reference to the design vector initialisation to ensure single source of truth. Added explicit repair out of bounds operator.
- V1.5: Introduced blade count repair function. Introduced duct LE location repair function. Introduced chord distribution repair function.
"""

# Import standard libraries
import copy
from typing import Any

# Import 3rd party libraries
import numpy as np
from pymoo.core.repair import Repair
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside

# Import local libraries
from utils import ensure_repo_paths #type: ignore
ensure_repo_paths()

import config #type: ignore
from Submodels.Parameterisations import AirfoilParameterisation #type: ignore
from Submodels.file_handling import fileHandlingMTFLO #type: ignore
from design_vector_interface import DesignVectorInterface #type: ignore
from init_designvector import DesignVector #type: ignore


class RepairIndividuals(Repair):
    """
    A repair operator class. This class repairs individuals in the
    population by checking if the profile parameterisations are one-to-one and
    if the sweep angle distribution is physical.
    """


    def __init__(self) -> None:
        """
        Initialization of the RepairIndividuals class.
        """

        # Call the parent class constructor
        super().__init__()

        # Define feasibility offset between consecutive control points
        self.feasibility_offset = config.PROFILE_FEASIBILITY_OFFSET

        # Initialize the airfoil parameterisation class
        self.airfoil_parameterisation = AirfoilParameterisation()

        # Create Bezier U-vectors
        self._u_vectors: tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]] = self.airfoil_parameterisation.GenerateBezierUVectors()

        # Extract BP3434 bounds from the design vector class
        self.BP_bounds = DesignVector().BP_3434_bounds

        # Initialize upper and lower bound lists for the complete design vector array
        self.xu = None
        self.xl = None

        # Initialize the design vector interface
        self.dvi = DesignVectorInterface()


    def _computebezier(self,
                       profile_params: dict[str, float]) -> tuple[tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64], np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]],
                                                                  tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64], np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]]:
        """
        Compute the Bezier curves for the x-coordinates and y-coordinates of the leading and trailing edge thickness and camber distributions
        of the airfoil profile.

        Parameters
        ----------
        - profile_params : dict[str, float]
            Dictionary containing the profile parameters

        Returns
        -------
        - tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
            - Tuple containing the Bezier curves for the leading and trailing edge thickness and camber distributions x-coordinates
                - x_LE_thickness : np.ndarray
                    Bezier curve for the leading edge thickness x-coordinates
                - x_TE_thickness : np.ndarray
                    Bezier curve for the trailing edge thickness x-coordinates
                - x_LE_camber : np.ndarray
                    Bezier curve for the leading edge camber x-coordinates
                - x_TE_camber : np.ndarray
                    Bezier curve for the trailing edge camber x-coordinates
            - Tuple containing the Bezier curves for the leading and trailing edge thickness and camber distributions y-coordinates
                - y_LE_thickness : np.ndarray
                    Bezier curve for the leading edge thickness y-coordinates
                - y_TE_thickness : np.ndarray
                    Bezier curve for the trailing edge thickness y-coordinates
                - y_LE_camber : np.ndarray
                    Bezier curve for the leading edge camber y-coordinates
                - y_TE_camber : np.ndarray
                    Bezier curve for the trailing edge camber y-coordinates
        """

        # Calculate the Bezier curve coefficients for the thickness curves
        (x_LE_thickness_coeff,
         y_LE_thickness_coeff,
         x_TE_thickness_coeff,
         y_TE_thickness_coeff) = self.airfoil_parameterisation.GetThicknessControlPoints(profile_params)

        # Compute the bezier curves for thickness
        u_leading_edge, u_trailing_edge = self._u_vectors
        x_LE_thickness = self.airfoil_parameterisation.BezierCurve3(x_LE_thickness_coeff,
                                                                    u_leading_edge)
        x_TE_thickness = self.airfoil_parameterisation.BezierCurve4(x_TE_thickness_coeff,
                                                                    u_trailing_edge[1:])
        y_LE_thickness = self.airfoil_parameterisation.BezierCurve3(y_LE_thickness_coeff,
                                                                    u_leading_edge)
        y_TE_thickness = self.airfoil_parameterisation.BezierCurve4(y_TE_thickness_coeff,
                                                                    u_trailing_edge[1:])

        if profile_params["y_c"] >= 1e-3:
            # Only calculate the bezier curves for camber if the camber is non-zero
            (x_LE_camber_coeff,
             y_LE_camber_coeff,
             x_TE_camber_coeff,
             y_TE_camber_coeff) = self.airfoil_parameterisation.GetCamberControlPoints(profile_params)

            # Compute the bezier curves for camber
            x_LE_camber = self.airfoil_parameterisation.BezierCurve3(x_LE_camber_coeff,
                                            u_leading_edge)
            x_TE_camber = self.airfoil_parameterisation.BezierCurve4(x_TE_camber_coeff,
                                            u_trailing_edge[1:])
            y_LE_camber = self.airfoil_parameterisation.BezierCurve3(y_LE_camber_coeff,
                                            u_leading_edge)
            y_TE_camber = self.airfoil_parameterisation.BezierCurve4(y_TE_camber_coeff,
                                            u_trailing_edge[1:])

        else:
            x_LE_camber = np.zeros_like(x_LE_thickness)
            x_TE_camber = np.zeros_like(x_TE_thickness)
            y_LE_camber = np.zeros_like(y_LE_thickness)
            y_TE_camber = np.zeros_like(y_TE_thickness)

        return (x_LE_thickness, x_TE_thickness, x_LE_camber, x_TE_camber), \
               (y_LE_thickness, y_TE_thickness, y_LE_camber, y_TE_camber)


    def _enforce_one2one(self,
                         profile_params: dict[str, float]) -> dict[str, float]:
        """
        Enforce that the x-coordinates x bezier curves and y-coordinates y bezier curves for thickness and camber are one to one.
        If the enforcing fails, the function returns the best attempt.

        Parameters
        ----------
        - profile_params : dict[str, float]
            Dictionary containing the profile parameters

        Returns
        -------
        - profile_params : dict[str, float]
            Dictionary containing the profile parameters with adjusted values to ensure one to one mapping
        """

        # Attempt to enforce one to one mapping of the bezier x-curves for 200 attempts.
        modified_profile_params = copy.deepcopy(profile_params)
        
        # Extract the constant values from the profile parameters to avoid repeated dictionary lookups
        r_LE = modified_profile_params["r_LE"]
        x_t = modified_profile_params["x_t"]
        y_t = modified_profile_params["y_t"]
        x_c = modified_profile_params["x_c"]

        for _ in range(200):
            # Compute the bezier curves for the x-coordinates. x_LE_thickness is always one to one, so we can ignore it.
            ((_,
              x_TE_thickness,
              x_LE_camber,
              x_TE_camber),
             (_,
              y_TE_thickness,
              y_LE_camber,
              y_TE_camber)) = self._computebezier(modified_profile_params)

            # Check one to one of all x points
            one_to_one_TE_thickness_x = np.all(np.diff(x_TE_thickness) >= 0)
            one_to_one_LE_camber_x = np.all(np.diff(x_LE_camber) >= 0)
            one_to_one_TE_camber_x = np.all(np.diff(x_TE_camber) >= 0)

            # Check one to one of all y points
            one_to_one_TE_thickness_y = np.all(np.diff(y_TE_thickness) <= 0)  # <=0 since TE thickness should be decreasing
            one_to_one_LE_camber_y = np.all(np.diff(y_LE_camber) >= 0)  # >=0 since LE camber should be increasing
            one_to_one_TE_camber_y = np.all(np.diff(y_TE_camber) <= 0)  # <=0 since TE camber should be decreasing

            # Check if all x points are one to one. If so, we return the updated profile parameters
            if np.all([one_to_one_TE_thickness_x, one_to_one_LE_camber_x, one_to_one_TE_camber_x, one_to_one_TE_thickness_y, one_to_one_LE_camber_y, one_to_one_TE_camber_y]):
                return modified_profile_params

            # Handle TE thickness x points
            if not one_to_one_TE_thickness_x:
                b_15 = modified_profile_params["b_15"]
                b_8 = modified_profile_params["b_8"]
                # Adjust the third x control point to enforce x3 = x_2 + feasibility_offset
                if (b_15 - x_t) / (1 - x_t) < 3 * x_t + 15 * b_8 ** 2 / (4 * r_LE):
                    b_15_adjusted_coor = 3 * x_t + 15 * b_8 ** 2 / (4 * r_LE) + self.feasibility_offset
                    b_15_adjusted = x_t + (1 - x_t) * b_15_adjusted_coor
                    b_15_adjusted = np.clip(b_15_adjusted, self.BP_bounds["b_15"][0], self.BP_bounds["b_15"][1])  # Enfoce b_15 to bounds
                    modified_profile_params["b_15"] = b_15_adjusted

                b_15 = modified_profile_params["b_15"]
                if (3 * x_t + 15 * b_8 ** 2 / (4 * r_LE)) > b_15:
                    # Adjust the second control point to enforce x2 < x3
                    sqrt_term = -10 * x_t * r_LE / 21
                    # Safety check to avoid sqrt of a negative number
                    b_8_adjusted = np.sqrt(sqrt_term) - 1e-2 if sqrt_term > 0 else 0
                    b_8_upper_limit = min(y_t, np.sqrt(-2 * r_LE * x_t / 3))
                    b_8_map = b_8_adjusted / b_8_upper_limit
                    b_8_clipped_map = np.clip(b_8_map, self.BP_bounds["b_8"][0], self.BP_bounds["b_8"][1])
                    b_8_adjusted_clipped = b_8_clipped_map * b_8_upper_limit
                    modified_profile_params["b_8"] = b_8_adjusted_clipped

            # Handle LE camber x points
            if not one_to_one_LE_camber_x:
                b_0 = modified_profile_params["b_0"]
                b_2 = modified_profile_params["b_2"]

                if b_2 < b_0:
                    b_2 = b_0 + self.feasibility_offset
                    modified_profile_params["b_2"] = np.clip(b_2, self.BP_bounds["b_2"][0], self.BP_bounds["b_2"][1])  # Enforce b_2 to bounds

                b_2 = modified_profile_params["b_2"]
                if b_2 > x_c:
                    b_2 = x_c - self.feasibility_offset
                    modified_profile_params["b_2"] = np.clip(b_2, self.BP_bounds["b_2"][0], self.BP_bounds["b_2"][1])  # Enforce b_2 to bounds

            # Handle TE camber x points
            if not one_to_one_TE_camber_x:
                b_17 = modified_profile_params["b_17"]
                y_c = modified_profile_params["y_c"]
                leading_edge_direction = modified_profile_params["leading_edge_direction"]

                if (b_17 - x_c) / (1 - x_c) < (-8 * y_c / np.tan(leading_edge_direction) + 13 * x_c) / 6:
                    b_17_adjusted_coor = (-8 * y_c / np.tan(leading_edge_direction) + 13 * x_c) / 6 + self.feasibility_offset
                    b_17_adjusted = (1 - x_c) * b_17_adjusted_coor + x_c
                    b_17_adjusted = np.clip(b_17_adjusted, self.BP_bounds["b_17"][0], self.BP_bounds["b_17"][1])  # Enforce b_17 to bounds
                    modified_profile_params["b_17"] = b_17_adjusted

                b_17 = modified_profile_params["b_17"]
                if x_c > (3 * x_c - y_c / np.tan(leading_edge_direction))  / 2:
                    gamma_LE_adjusted_x_based = np.atan(y_c / (x_c - 2 * self.feasibility_offset)) + 1e-3
                    gamma_LE_adjusted_b0_based = np.atan(y_c / (modified_profile_params["b_0"] * x_c)) - 1e-3  # based on the LE camber y coordinates

                    # gamma_LE must lie somewhere between the two computed values for it to be feasible, so we simply take the middle value.
                    gamma_LE_adjusted = (gamma_LE_adjusted_x_based + gamma_LE_adjusted_b0_based) / 2

                    modified_profile_params["leading_edge_direction"] = np.clip(gamma_LE_adjusted, self.BP_bounds["leading_edge_direction"][0], self.BP_bounds["leading_edge_direction"][1])  # Enforce gamma_LE to bounds
    
                leading_edge_direction = modified_profile_params["leading_edge_direction"]
                if x_c > (-8 * y_c / np.tan(leading_edge_direction) + 13 * x_c) / 6:
                    y_c_adjusted = 7 / 8 * x_c * np.tan(leading_edge_direction) - 1e-3
                    y_c_adjusted = np.clip(y_c_adjusted, self.BP_bounds["y_c"][0], self.BP_bounds["y_c"][1])  # Enforce y_c to bounds
                    modified_profile_params["y_c"] = y_c_adjusted

            # Handle TE thickness y points
            if not one_to_one_TE_thickness_y:
                # Compute the new trailing edge wedge angle
                beta_TE = np.atan((y_t + modified_profile_params["b_8"] - modified_profile_params["dz_TE"]) / (2 * (1 - modified_profile_params["b_15"]))) - 1e-3
                beta_TE = np.clip(beta_TE, self.BP_bounds["trailing_wedge_angle"][0], self.BP_bounds["trailing_wedge_angle"][1])  # Enforce beta_TE to bounds
                modified_profile_params["trailing_wedge_angle"] = beta_TE

            # Handle LE camber y points
            if not one_to_one_LE_camber_y:
                # Adjust the b_0 control point
                b_0_coor = modified_profile_params["y_c"] / np.tan(modified_profile_params["leading_edge_direction"]) - 1e-3
                b_0 = b_0_coor / x_c
                b_0 = np.clip(b_0, self.BP_bounds["b_0"][0], self.BP_bounds["b_0"][1])  # Enforce b_0 to bounds
                modified_profile_params["b_0"] = b_0

            # # Handle TE camber y points
            if not one_to_one_TE_camber_y:
                # Compute the new trailing camberline angle
                alpha_TE = np.atan((5/6 * modified_profile_params["y_c"] - modified_profile_params["z_TE"]) / (1 - modified_profile_params["b_17"])) + 1e-3
                alpha_TE = np.clip(alpha_TE, self.BP_bounds["trailing_camberline_angle"][0], self.BP_bounds["trailing_camberline_angle"][1])  # Enforce alpha_TE to bounds
                modified_profile_params["trailing_camberline_angle"] = alpha_TE

        return modified_profile_params


    def _enforce_blade_LE_positive_sweepback(self, blading_params: dict[str, Any]) -> dict[str, Any]:
        """
        Enforce that the leading edge x-coordinate of the blade is positively increasing along the span.

        Parameters
        ----------
        - blading_params : dict[str, Any]
            Dictionary containing the blading parameters

        Returns
        -------
        - blading_params : dict[str, Any]
            Dictionary containing the blading parameters with adjusted values to ensure positive sweepback angle
        """

        # Compute the LE x-coordinate distribution for each of the radial sections and enforce it to be positively increasing
        LE_x_coordinate = np.tan(blading_params["sweep_angle"]) * blading_params["radial_stations"]
        LE_x_coordinate_corrected = np.maximum.accumulate(LE_x_coordinate)

        # Extract the corrected sweep angles from the corrected X-coordinate distribution.
        # Skip the first entry since the root is enforced to have sweep=0
        blading_params["sweep_angle"][1:] = np.atan(LE_x_coordinate_corrected[1:] / blading_params["radial_stations"][1:])

        return blading_params


    def _enforce_duct_location(self,
                               blading_params: dict[str, Any],
                               duct_params: dict[str, Any]) -> dict[str, Any]:
        """
        Enforce the duct leading edge is always positioned at/forward of the blade tip leading edge.

        Parameters
        ----------
        - blading_params : dict[str, Any]
            Dictionary containing the blading parameters
        - duct_params : dict[str, Any]
            Dictionary containing the duct parameters

        Returns
        -------
        - duct_params : dict[str, Any]
            Dictionary containing the duct parameters with adjusted LE x coordinate.
        """

        # If the duct is positioned aft of the LE of the blade root, move it forward
        x_old, y_old = duct_params["Leading Edge Coordinates"]
        if x_old > blading_params["root_LE_coordinate"]:
            duct_params["Leading Edge Coordinates"] = (blading_params["root_LE_coordinate"],
                                                       y_old)

        return duct_params


    def _enforce_chord_distribution(self,
                                    blading_params: dict[str, Any]) -> dict[str, Any]:
        """
        Enforce that the blade chord distribution is continuously decreasing from the blade hub to tip.

        Parameters
        ----------
        - blading_params : dict[str, Any]

        Returns
        -------
        - blading_params : dict[str, Any]
        """

        # Extract the current chord distribution and radial stations
        radial_stations = blading_params["radial_stations"]
        chord_distribution = blading_params["chord_length"]

        # Determine midspan value
        midspan_radius = 0.5 * radial_stations[-1]
        midspan_idx = max(i for i, r in enumerate(radial_stations) if r <= midspan_radius)

        # Fix the chord length distribution from the midspan onwards
        chord_distribution[midspan_idx:] = np.minimum.accumulate(chord_distribution[midspan_idx:])
        # Update the blading params
        blading_params["chord_length"] = chord_distribution

        return blading_params


    def _fix_blockage(self,
                      blading_params: dict[str, Any],
                      design_params: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Fix the blade circumferential thickness. If limit of complete blockage is exceeded anywhere along the blade span,
        we simply decrease the blade-count to fix the blockage.

        Parameters
        ----------
        - blading_params : dict[str, Any]
            Dictionary containing the blading parameters
        - design_params : list[dict[str, Any]]
            - List of the Bezier-Parsec design parameters for all defined radial profile sections

        Returns
        -------
        - blading_params : dict[str, Any]
            The repaired blading parameters dictionary
        """

        original_blading_params = copy.deepcopy(blading_params)

        # Use a try-except block to handle cases where the profile shape is infeasible.
        try:
            # Loop over the radial sections
            for i in range(len(design_params)):

                if blading_params["radial_stations"][i] == 0:
                    continue
                # Loop to fix the blockage: decrement blade_count while > 2 to enforce minimum of 2 blades
                evaluating = True
                while evaluating and blading_params["blade_count"] > 2:
                    # First precompute the limit of complete blockage at the radial station
                    complete_blockage = 2 * np.pi * blading_params["radial_stations"][i] / blading_params["blade_count"]

                    upper_x, upper_y, lower_x, lower_y = self.airfoil_parameterisation.ComputeProfileCoordinates(design_params[i])
                    upper_x *= blading_params["chord_length"][i]
                    upper_y *= blading_params["chord_length"][i]
                    lower_x *= blading_params["chord_length"][i]
                    lower_y *= blading_params["chord_length"][i]

                    blade_pitch = (blading_params["blade_angle"][i] + blading_params["ref_blade_angle"] - blading_params["reference_section_blade_angle"])
                    rotated_upper_x, rotated_upper_y, rotated_lower_x, rotated_lower_y  = fileHandlingMTFLO.RotateProfile(blade_pitch,
                                                                                                                        upper_x,
                                                                                                                        lower_x,
                                                                                                                        upper_y,
                                                                                                                        lower_y)

                    LE_coordinate = blading_params["radial_stations"][i] * np.tan(blading_params["sweep_angle"][i])
                    rotated_upper_x += LE_coordinate - rotated_upper_x[0]
                    rotated_lower_x += LE_coordinate - rotated_lower_x[0]

                    y_section_upper, y_section_lower, _, z_section_upper, z_section_lower, _ = fileHandlingMTFLO.PlanarToCylindrical(rotated_upper_y,
                                                                                                                                     rotated_lower_y,
                                                                                                                                     blading_params["radial_stations"][i])

                    # Compute the circumferential blade thickness
                    circumferential_thickness = fileHandlingMTFLO.CircumferentialThickness(y_section_upper,
                                                                            z_section_upper,
                                                                            y_section_lower,
                                                                            z_section_lower,
                                                                            blading_params["radial_stations"][i])

                    # Check if the limit of complete blockage is respected by the design. If not, decrease the blade count by 1
                    max_circumf_thickness = circumferential_thickness.max()
                    if max_circumf_thickness >= complete_blockage:
                        blading_params["blade_count"] -= 1
                        continue

                    # If thickness is okay, break out of the while loop and move to the next radial section
                    evaluating = False

            return blading_params

        except ValueError:
            # If the profile shape is infeasible, return the original blading parameters to avoid crashing the algorithm.
            return original_blading_params


    def _do(self,
            problem: object,
            X: list[dict[str, float | int]], **kwargs) -> list[dict[str, float | int]]:
        """
        Perform a simple repair on all individuals in the population.
        This is not guaranteed to fix the parameterisation, but testing shows it fixes 90% of the infeasible design vectors.

        Parameters
        ----------
        - problem : object
            The optimisation problem.
        - X : list[dict[str, float | int]]
            The list of design vector dictionaries for all individuals in the population.

        Returns
        -------
        - X : list[dict[str, float | int]]
            The repaired list of design vector dictionaries for all individuals in the population.
        """

        # Loop over all individuals in the population and repair them if needed
        for i, individual in enumerate(X):
            # First extract the keys of the integer variables in the design vector dictionary to be able to cast them back to integer after repairing
            int_keys = [key for key, value in individual.items() if isinstance(value, (int, np.integer))]

            # Deconstruct the design vector in to the different design dictionaries
            # We do not need to compute the duct LE y coordinate here, so we can skip this step
            # and speed up the computation by setting compute_duct = False.
            (centerbody_variables,
            duct_variables,
            blade_design_parameters,
            blade_blading_parameters,
            _) = self.dvi.DeconstructDesignVector(individual, compute_duct = False)

            if config.OPTIMIZE_CENTERBODY:
                # Repair the centerbody parameters
                centerbody_variables = self._enforce_one2one(centerbody_variables)

            if config.OPTIMIZE_DUCT:
                # Repair the duct parameters
                duct_variables = self._enforce_one2one(duct_variables)

            for j, optimise_stage in enumerate(config.OPTIMIZE_STAGE):
                if optimise_stage:
                    # Repair the blading parameters
                    blade_blading_parameters[j] = self._enforce_blade_LE_positive_sweepback(blade_blading_parameters[j])
                    blade_blading_parameters[j] = self._enforce_chord_distribution(blade_blading_parameters[j])

                    # Repair the duct LE location
                    duct_variables = self._enforce_duct_location(blade_blading_parameters[j],
                                                                 duct_variables)

                    # Loop over all the radial sections and repair the profile parameters
                    for k in range(config.NUM_RADIALSECTIONS[j]):
                        # Repair the profile parameters
                        blade_design_parameters[j][k] = self._enforce_one2one(blade_design_parameters[j][k])

                    # Repair the blade count
                    blade_blading_parameters[j] = self._fix_blockage(blade_blading_parameters[j], blade_design_parameters[j])

            # Reconstruct the design vector into a singular dictionary
            x = self.dvi.ReconstructDesignVector(centerbody_variables,
                                               duct_variables,
                                               blade_design_parameters,
                                               blade_blading_parameters)

            # Convert design vector into array together with bounds to enforce design variable bounds
            x_array = np.array(list(x.values()))

            # Only extract the bounds of they are not already written in self.
            keys = list(x.keys())
            if self.xu is None or self.xl is None or keys != getattr(self, "_bounds_keys", None):
                if problem is None:
                    raise ValueError("'problem' must expose xl/xu when calling RepairIndividuals._do()")
                self._bounds_keys = keys
                self.xl = np.array([problem.xl[k] for k in keys])
                self.xu = np.array([problem.xu[k] for k in keys])

            # Repair the design vector if it is out of bounds
            x_array = set_to_bounds_if_outside(x_array, self.xl, self.xu)

            # Convert the array back to a dictionary and write the result to X
            # Casts the integer values back to integers to ensure consistent variable types throughout the evaluation.
            x_dict = dict(zip(x.keys(), x_array))
            X[i] = {key: (int(round(val)) if key in int_keys else val) for key, val in x_dict.items()}

        return X


if __name__ == "__main__":
    from init_population import InitPopulation #type: ignore
    from problem_definition import OptimizationProblem #type: ignore

    import time

    pop = InitPopulation("biased").GeneratePopulation()

    pop_dict = [pop.get("X")[i] for i in range(len(pop))]

    # Create an instance of the RepairIndividuals class
    problem = OptimizationProblem()
    start = time.monotonic()
    repair = RepairIndividuals()
    repaired_pop = repair._do(problem, pop_dict)
    print("Repair took:", time.monotonic() - start)

    # Validate that repair worked
    dvi = DesignVectorInterface()
    failures_before = 0
    failures_after = 0

    for individual in pop_dict:
        try:
            dvi.DeconstructDesignVector(individual, compute_duct=False)
        except ValueError:
            failures_before += 1

    for individual in repaired_pop:
        try:
            dvi.DeconstructDesignVector(individual, compute_duct=False)
        except ValueError:
            failures_after += 1

    print(f"Repair fixed {failures_before - failures_after} of {failures_before} invalid individuals")
