"""
init_designvector
==================

Description
-----------
This module defines the DesignVector class, which is used to construct the design vector for the pymoo framework
optimization problem. The design vector supports mixed-variable optimization, including real and integer variables.

Classes
-------
DesignVector
    Class for constructing the design vector based on configuration toggles.

Examples
--------
>>> from init_designvector import DesignVector
>>> from config import cfg
>>> design_vector = DesignVector.construct_vector(cfg)

Notes
-----
This module integrates with the pymoo framework for optimization. Ensure that the configuration module (cfg) is
properly set up with the required toggles and parameters for the design vector construction.

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 1.1

Changelog:
- V1.0: Initial implementation. Extracted from the problem_definition.py file for better modularity and readability.
- V1.1: Implemented optional variable pitch handling.
"""

# Import standard libraries
from types import ModuleType

# Import 3rd party libraries
import numpy as np
from pymoo.core.variable import Real, Integer

class DesignVector:
    """
    This class is used to construct the design vector for the optimisation problem.
    """

    # Define the bounds on the BP3434 parameterisation method.
    BP_3434_bounds = {"b_0": (0.05, 0.1),
                      "b_2": (0.125, 0.3),
                      "b_8": (0.05, 0.7),
                      "b_15": (0.7, 0.95),
                      "b_17": (0.7, 0.95),
                      "x_t": (0.15, 0.4),
                      "y_t": (0.02, 0.3),
                      "x_c": (0.2, 0.5),
                      "y_c": (0, 0.15),
                      "z_TE": (0, 0.05),
                      "dz_TE": (0, 0.005),
                      "r_LE": (-0.2, -0.001),
                      "trailing_wedge_angle": (0.001, 0.4),
                      "trailing_camberline_angle": (0.001, 0.2),
                      "leading_edge_direction": (0.001, 0.2)}

    # Initialize the profile vars as None
    _cached_profile_vars = None

    # Indices for the thickness parameters for the centerbody parameters
    # We force the centerbody to be symmetric, so camber parameters are not needed
    CENTERBODY_VAR_INDICES = [2, 3, 5, 6, 10, 11, 12]
    

    @classmethod
    def _create_profile_vars(cls) -> list:
        """
        Create the profile section variables once and cache them.
        """
        return [Real(bounds=cls.BP_3434_bounds["b_0"]),  # b_0
                Real(bounds=cls.BP_3434_bounds["b_2"]),  # b_2
                Real(bounds=cls.BP_3434_bounds["b_8"]),  # mapping variable for b_8
                Real(bounds=cls.BP_3434_bounds["b_15"]),  # b_15
                Real(bounds=cls.BP_3434_bounds["b_17"]),  # b_17
                Real(bounds=cls.BP_3434_bounds["x_t"]),  # x_t
                Real(bounds=cls.BP_3434_bounds["y_t"]),  # y_t
                Real(bounds=cls.BP_3434_bounds["x_c"]),  # x_c
                Real(bounds=cls.BP_3434_bounds["y_c"]),  # y_c
                Real(bounds=cls.BP_3434_bounds["z_TE"]),  # z_TE
                Real(bounds=cls.BP_3434_bounds["dz_TE"]),  # dz_TE
                Real(bounds=cls.BP_3434_bounds["r_LE"]),  # r_LE
                Real(bounds=cls.BP_3434_bounds["trailing_wedge_angle"]),  # trailing_wedge_angle
                Real(bounds=cls.BP_3434_bounds["trailing_camberline_angle"]),  # trailing_camberline_angle
                Real(bounds=cls.BP_3434_bounds["leading_edge_direction"])]  # leading_edge_direction

    @classmethod
    def profile_section_vars(cls) -> list:
        """
        Return the standard 15-var profile section definition.
        Bounds are based on those presented in:
            Rogalsky T. Acceleration of differential evolution for aerodynamic design.
            Ph.D. Thesis, University of Manitoba; 2004.
        """

        if cls._cached_profile_vars is None:
            cls._cached_profile_vars = cls._create_profile_vars()
        return cls._cached_profile_vars.copy()  # Return a copy to prevent modification.


    @classmethod
    def construct_vector(cls, cfg: ModuleType) -> dict:
        """
        Initialize the pymoo design vector based on the toggles in config.

        Parameters
        ----------
        - cls : class
            The class. used to access the class-attribute BP_3434_bounds.
        - cfg : ModuleType
            The config module containing the design vector configuration.

        Returns
        -------
        - dict
            A dictionary containing the design vector variables and their bounds.
        """

        # Initialize variable list with variable types.
        # This is required to handle the mixed-variable nature of the optimisation, where the blade count is an integer
        vector = []
        if cfg.OPTIMIZE_CENTERBODY:
            # If the centerbody is to be optimised, initialise the variable types
            complete_profile = cls.profile_section_vars()
            section_profile = [complete_profile[i] for i in cls.CENTERBODY_VAR_INDICES]

            vector.extend(section_profile)
            vector.append(Real(bounds=(0.25, 4)))  # Chord Length
        if cfg.OPTIMIZE_DUCT:
            # If the duct is to be optimised, intialise the variable types
            duct_profile = cls.profile_section_vars()
            duct_profile[6] = Real(bounds=(0.05, 0.2))  # set y_t for the duct
            vector.extend(duct_profile)
            vector.append(Real(bounds=(1.0, 1.5)))  # Chord Length
            vector.append(Real(bounds=(-0.5, 0.5)))  # Leading Edge X-Coordinate

        # Pre-calculate the stage parameters to avoid repeated calculations
        stage_configs = []
        num_operating_points = len(cfg.multi_oper)
        for i in range(cfg.NUM_STAGES):
            if cfg.OPTIMIZE_STAGE[i]:
                num_radial_sections = cfg.NUM_RADIALSECTIONS[i]
                stage_configs.append({
                    'index': i,
                    'num_radial_sections': num_radial_sections,
                    'is_rotating': cfg.ROTATING[i],
                    'pitch_angles': len(cfg.STAGE_BLADING_PARAMETERS[i]["ref_blade_angle_lst"])
                })

        # For the stage(s) marked for optimisation, add the profile design variables to the design vector
        for stage_config in stage_configs:
            for _ in range(stage_config['num_radial_sections']):
                vector.extend(cls.profile_section_vars())

        # Add stage-specific variables
        for stage_config in stage_configs:
            i = stage_config['index']
            vector.append(Real(bounds=(0, 0.4)))  # root_LE_coordinate
            for _ in range(stage_config["pitch_angles"]):  # Assign each pitch angle to in case of variable pitch
                vector.append(Real(bounds=(0.1, 5 * np.pi/18)))  # ref_blade_angle from [~5.7deg to 50 deg]
            vector.append(Integer(bounds=(3, 15)))  # blade_count
            if stage_config['is_rotating']:
                for _ in range(num_operating_points):
                    vector.append(Real(bounds=(20, 80)))  # blade RPS
            vector.append(Real(bounds=(1.0, 3.0)))  # blade diameter

            for _ in range(stage_config['num_radial_sections']):
                vector.append(Real(bounds=(0.1, 0.75)))  # chord length
            for _ in range(stage_config['num_radial_sections'] - 1):  # Note the -1 since the root section is independent of sweep.
                vector.append(Real(bounds=(0, np.pi/3)))  # sweep_angle
            for _ in range(stage_config['num_radial_sections'] - 1):  # Note the -1 since the tip has a fixed angle at 0
                vector.append(Real(bounds=(0, np.pi/3)))  # blade_angle

        # For a mixed-variable problem, PyMoo expects the vector to be a dictionary, so we convert vector to a dictionary.
        # Note that all variables are given a name xi.
        var_names = [f"x{i}" for i in range(len(vector))]
        vector = dict(zip(var_names, vector))

        return vector


if __name__ == "__main__":
    import config # type: ignore
    test = DesignVector()
    vector = test.construct_vector(config)
    print(vector)
