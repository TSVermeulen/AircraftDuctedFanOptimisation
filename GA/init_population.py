"""
init_population
===============

Description
-----------
This module provides functionality to initialize populations for optimization problems,
either by generating a biased population based on a reference design or by sampling
randomly within the bounds of the design variables.

Classes
-------
InitPopulation
    A class to initialize populations for optimization problems. It supports generating
    biased populations with perturbations around a reference design or random populations
    sampled uniformly across the bounds.

Examples
--------
>>> import config
>>> import time

>>> start_time = time.time()
>>> init_pop = InitPopulation("biased", config)
>>> population = init_pop.GeneratePopulation()
>>> print(population)
>>> print(f"Population generation took: {time.time() - start_time} seconds")

Notes
-----
This module is designed to work with the pymoo optimization framework and assumes
the design vector configuration is provided in the `config` module.

References
----------
pymoo: https://pymoo.org/

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 1.3

Changelog:
- V1.0: Initial version with tested biased population generation and basic random population generation functionality.
- V1.1: Added support for configurable random seed for reproducibility and improved documentation. Reworked GenerateBiasedPopulation to improve speed using NumPy.
- V1.2: Implemented perturbation for zero-valued real design parameters.
- V1.3: Implemented optional variable pitch handling. 
"""

# Import standard libraries
import numbers

# Import 3rd party libraries
import numpy as np
from pymoo.core.mixed import MixedVariableSampling
from pymoo.core.population import Population
from pymoo.core.individual import Individual

# Import interfacing modules
from init_designvector import DesignVector # type: ignore
import config # type: ignore


class InitPopulation():
    """
    Population generation class to handle generation of the starting point for the genetic algorithm optimisation.
    """


    def __init__(self,
                 population_type: str,
                 **kwargs) -> None:
        """
        Initialisation for the InitPopulation class.

        Parameters
        ----------
        - population_type : str
            A string indicating the type of population to generate. Either 'biased' or 'random'.
            "biased" will generate a population with perturbed individuals based on a reference design, while "random" will generate a random population.
        - **kwargs : dict[str, Any]
            Additional keyword arguments

        Returns
        -------
        None
        """

        self.type = population_type

        # Create a new design vector object
        self.design_vector = DesignVector().construct_vector(config)
        self.design_vector_keys = list(self.design_vector.keys())

        # Set the seed for the random number generator to ensure reproducibility
        seed = kwargs.get("seed", 1)
        self._np_rng = np.random.default_rng(seed)
        np.random.seed(seed)


    def DeconstructDictFromReferenceDesign(self) -> dict:
        """
        Deconstruct the reference design vector dictionaries back into a pymoo design vector.
        This is used to create the initial population for the optimisation problem.

        Parameters
        ----------
        None

        Returns
        -------
        - vector : dict
            A dict of design variables for the optimisation problem.
        """

        # Define a helper function to map the b_8 variable
        def map_b8_value(variable_dict: dict):
            """
            Compute a normalised b_8 mapping variables on [0, 1].
            This normalisation ensures the b_8 variable always matches the constraint:
                             0 <= b_8 <= min(y_t, sqrt(-2*r_LE*x_t/3))
            """

            denominator = min(variable_dict["y_t"],
                              np.sqrt(max(0.0, -2 * variable_dict["x_t"] * variable_dict["r_LE"] / 3)))
            return float(variable_dict["b_8"] / denominator) if denominator > 0 else 0.0

        vector = []
        if config.OPTIMIZE_CENTERBODY:
            # If the centerbody is to be optimised, read in the reference design dictionary and extract the design vector x
            vector.append(map_b8_value(config.CENTERBODY_VALUES))
            vector.append(config.CENTERBODY_VALUES["b_15"])
            vector.append(config.CENTERBODY_VALUES["x_t"])
            vector.append(config.CENTERBODY_VALUES["y_t"])
            vector.append(config.CENTERBODY_VALUES["dz_TE"])
            vector.append(config.CENTERBODY_VALUES["r_LE"])
            vector.append(config.CENTERBODY_VALUES["trailing_wedge_angle"])
            vector.append(config.CENTERBODY_VALUES["Chord Length"])

        if config.OPTIMIZE_DUCT:
            # If the duct is to be optimised, read the reference values into the design vector
            vector.append(config.DUCT_VALUES["b_0"])
            vector.append(config.DUCT_VALUES["b_2"])
            vector.append(map_b8_value(config.DUCT_VALUES))
            vector.append(config.DUCT_VALUES["b_15"])
            vector.append(config.DUCT_VALUES["b_17"])
            vector.append(config.DUCT_VALUES["x_t"])
            vector.append(config.DUCT_VALUES["y_t"])
            vector.append(config.DUCT_VALUES["x_c"])
            vector.append(config.DUCT_VALUES["y_c"])
            vector.append(config.DUCT_VALUES["z_TE"])
            vector.append(config.DUCT_VALUES["dz_TE"])
            vector.append(config.DUCT_VALUES["r_LE"])
            vector.append(config.DUCT_VALUES["trailing_wedge_angle"])
            vector.append(config.DUCT_VALUES["trailing_camberline_angle"])
            vector.append(config.DUCT_VALUES["leading_edge_direction"])
            vector.append(config.DUCT_VALUES["Chord Length"])
            vector.append(config.DUCT_VALUES["Leading Edge Coordinates"][0])

        for i in range(config.NUM_STAGES):
            # If the blade rows are to be optimised, read the reference values into the design vector
            if config.OPTIMIZE_STAGE[i]:
                # Read the reference values into the design vector
                for j in range(config.NUM_RADIALSECTIONS[i]):
                    vector.append(config.STAGE_DESIGN_VARIABLES[i][j]["b_0"])
                    vector.append(config.STAGE_DESIGN_VARIABLES[i][j]["b_2"])
                    vector.append(map_b8_value(config.STAGE_DESIGN_VARIABLES[i][j]))
                    vector.append(config.STAGE_DESIGN_VARIABLES[i][j]["b_15"])
                    vector.append(config.STAGE_DESIGN_VARIABLES[i][j]["b_17"])
                    vector.append(config.STAGE_DESIGN_VARIABLES[i][j]["x_t"])
                    vector.append(config.STAGE_DESIGN_VARIABLES[i][j]["y_t"])
                    vector.append(config.STAGE_DESIGN_VARIABLES[i][j]["x_c"])
                    vector.append(config.STAGE_DESIGN_VARIABLES[i][j]["y_c"])
                    vector.append(config.STAGE_DESIGN_VARIABLES[i][j]["z_TE"])
                    vector.append(config.STAGE_DESIGN_VARIABLES[i][j]["dz_TE"])
                    vector.append(config.STAGE_DESIGN_VARIABLES[i][j]["r_LE"])
                    vector.append(config.STAGE_DESIGN_VARIABLES[i][j]["trailing_wedge_angle"])
                    vector.append(config.STAGE_DESIGN_VARIABLES[i][j]["trailing_camberline_angle"])
                    vector.append(config.STAGE_DESIGN_VARIABLES[i][j]["leading_edge_direction"])

        for i in range(config.NUM_STAGES):
            # If the blade rows are to be optimised, read the reference values into the design vector
            if config.OPTIMIZE_STAGE[i]:
                # Read the reference values into the design vector
                vector.append(config.STAGE_BLADING_PARAMETERS[i]["root_LE_coordinate"])
                for angle in config.STAGE_BLADING_PARAMETERS[i]["ref_blade_angle_lst"]:
                    vector.append(angle)  # Allow for variable pitch by giving each operating point a separate blade angle
                vector.append(int(config.STAGE_BLADING_PARAMETERS[i]["blade_count"]))
                if config.ROTATING[i]:
                    for RPS in config.STAGE_BLADING_PARAMETERS[i]["RPS_lst"]:
                        vector.append(RPS)  # Required for multipoint analysis
                vector.append(config.STAGE_BLADING_PARAMETERS[i]["radial_stations"][-1] * 2)  # The interfaces uses the radial locations, but the design varable is the blade diameter!

                for j in range(config.NUM_RADIALSECTIONS[i]):
                    vector.append(config.STAGE_BLADING_PARAMETERS[i]["chord_length"][j])
                for j in range(config.NUM_RADIALSECTIONS[i] - 1):
                    vector.append(config.STAGE_BLADING_PARAMETERS[i]["sweep_angle"][j + 1])
                for j in range(config.NUM_RADIALSECTIONS[i] - 1):  # -1 since the tip section has a fixed angle of 0
                    vector.append(config.STAGE_BLADING_PARAMETERS[i]["blade_angle"][j])

        # Change vector from a list to a dictionary to match the expected structure of pymoo
        keys = list(self.design_vector.keys())

        # Ensure the vector is the same length as the design vector keys
        vector = {key: value for key, value in zip(keys, vector)}

        return vector


    def GenerateBiasedPopulation(self) -> Population:
        """
        Generate the initial population for the optimisation problem based on an existing solution.

        Parameters
        ----------
        None

        Returns
        -------
        - pop : Population
            The initial population for the optimisation problem as a pymoo Population object.
        """

        # Get the invidivual corresponding to the reference design
        reference_individual = self.DeconstructDictFromReferenceDesign()
        ref = np.array([reference_individual[k] for k in self.design_vector_keys],
                       dtype=float)

        # Extract the lower and upper bounds of the design variables
        lower_bounds = np.array([self.design_vector[k].bounds[0] for k in self.design_vector_keys],
                                dtype=float)
        upper_bounds = np.array([self.design_vector[k].bounds[1] for k in self.design_vector_keys],
                                dtype=float)

        # Verify the reference individual is within the bounds
        ref = np.clip(ref, lower_bounds, upper_bounds)

        # Generate the initial population equal to the POPULATION_SIZE reference_individuals
        pop_dict = [reference_individual.copy() for _ in range(config.INITIAL_POPULATION_SIZE)]

        # Compute masks for the floating point and integer design variables
        real_mask = np.array([np.issubdtype(type(v), np.floating) for v in reference_individual.values()])
        int_mask  = np.array([np.issubdtype(type(v), np.integer)  for v in reference_individual.values()])
        other_mask = ~(real_mask | int_mask)
        if other_mask.any():
            raise TypeError("Non-scalar design variables detected: update initial-population logic.")

        # Compute masks to check which values of the floating point variables are zero
        zero_real_mask = real_mask & (ref == 0)
        nonzero_real_mask = real_mask & (ref != 0)
        span = upper_bounds - lower_bounds

        for i in range(1, len(pop_dict)):
            # Generate some noise for the floating point variables
            # Use uniform noise to ensure an equal sampling across the design space.
            noise = self._np_rng.uniform(-1, 1, size=ref.shape)

            # Apply perturbations
            # For nonzero real values, we use a simple noise perturbation
            perturbed_individual = ref.copy()
            perturbed_individual[nonzero_real_mask] += noise[nonzero_real_mask] * perturbed_individual[nonzero_real_mask] * config.SPREAD_CONTINUOUS
            # For zero real values, we use a perturbation which is equal to some constant value times the design variable span
            perturbed_individual[zero_real_mask] += noise[zero_real_mask] * span[zero_real_mask] * config.ZERO_NOISE * config.SPREAD_CONTINUOUS
            # For integer values, we use an integer noise function
            perturbed_individual[int_mask] += self._np_rng.integers(*config.SPREAD_DISCRETE, size=int_mask.sum())

            # Ensure perturbed individual still falls within the design variable bounds
            perturbed_individual = np.clip(perturbed_individual, lower_bounds, upper_bounds)

            # Convert to dicitonary while casting the discrete variable(s) back to int
            pop_dict[i] = {key: (int(val) if isinstance(reference_individual[key], numbers.Integral) else val)
                           for key, val in zip(self.design_vector_keys, perturbed_individual)}

        # Construct the population object
        individuals = []
        for pop in pop_dict:
            individuals.append(Individual(X=pop))
        pop = Population.create(*individuals)

        return pop


    def GeneratePopulation(self) -> Population|MixedVariableSampling:
        """
        Generate the initial population for the optimisation problem.

        Use either:
        - A biased population where we introduce some perturbations around a known initial design vector
        - A random population where we sample the design vector uniformly across the bounds

        Parameters
        ----------
        None

        Returns
        -------
        - pop : Population|MixedVariableSampling
            The initial population for the optimisation problem as a pymoo Population object or a MixedVariableSampling object.
            The type of population is determined by the `type` parameter.
            - For "biased" type: Returns a fully initialised Population object ready for optimisation.
            - For "random" type: Returns a MixedVariableSampling strategy object that Pymoo will use to generate the population.
        """

        if self.type == "biased":
            # Generate a biased population based on an existing solution
            pop = self.GenerateBiasedPopulation()
        elif self.type == "random":
            # Generate a random population
            pop = MixedVariableSampling()
        else:
            raise ValueError("Invalid population type. Choose 'biased' or 'random'.")

        return pop


if __name__ == "__main__":
    test = InitPopulation("biased")

    biased_pop = test.GeneratePopulation()
