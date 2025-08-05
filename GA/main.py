"""
main
====

Description
-----------
This module defines the main entry point for running a single-threaded optimization problem using the pymoo framework.

Functionality
-------------
- Initializes an optimization problem with mixed-variable support.
- Configures and runs a genetic algorithm (GA) for optimization.
- Saves the results to a dill database for future reference.

Examples
--------
>>> python main.py

Notes
-----
This module integrates with the pymoo framework and requires the problem definition and population initialization modules.
Ensure that all dependencies are installed and properly configured.

References
----------
For more details on pymoo and its capabilities, refer to the official documentation:
https://pymoo.org/

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 1.3

Changelog:
- V1.0: Initial implementation.
- V1.1: Updated documentation to reflect changes in the main module structure and added examples for usage.
- V1.2: Updated to match structure of main-parallelized, but with n_processes = 1.
- V1.3: Removed shared cache as it is no longer implemented. Sorted imports, and formalised termination criteria.
        Switched code from MixedVariableGA to use the UNSGA3 algorithm with mixed variable support.
"""

# Import standard libaries
import datetime
from pathlib import Path

# Import 3rd party libraries
import dill
from pymoo.core.mixed import MixedVariableMating, MixedVariableDuplicateElimination
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.unsga3 import UNSGA3, comp_by_rank_and_ref_line_dist
from pymoo.operators.selection.tournament import TournamentSelection

# Import interface submodels and other dependencies
from utils import ensure_repo_paths, calculate_n_reference_points #type: ignore
ensure_repo_paths()

import config  #type: ignore
from problem_definition import OptimizationProblem #type: ignore
from init_population import InitPopulation #type: ignore
from termination_conditions import GetTerminationConditions #type: ignore
from repair import RepairIndividuals #type: ignore


if __name__ == "__main__":
    """ Initialize the optimization problem and algorithm """
    # Initialize the optimization problem by passing the configuration and the starmap interface of the thread_pool
    if hasattr(config, "PROBLEM_TYPE") and config.PROBLEM_TYPE == "multi_point":
        from multi_point_problem_definition import MultiPointOptimizationProblem # type: ignore
        problem = MultiPointOptimizationProblem(seed=config.GLOBAL_SEED)
    else:
        problem = OptimizationProblem(seed=config.GLOBAL_SEED)

    # Create the reference directions to be used for the optimisation
    ref_dirs = get_reference_directions("energy",
                                        n_dim=len(config.objective_IDs),
                                        n_points=calculate_n_reference_points(config))

    # Initialize the algorithm
    selection_operator = TournamentSelection(func_comp=comp_by_rank_and_ref_line_dist)
    duplicate_operator = MixedVariableDuplicateElimination()
    algorithm = UNSGA3(ref_dirs=ref_dirs,
                       pop_size=config.POPULATION_SIZE,
                       mating=MixedVariableMating(selection=selection_operator,
                                                  eliminate_duplicates=duplicate_operator,
                                                  repair=RepairIndividuals()),
                       sampling=InitPopulation(population_type="biased",
                                               seed=config.GLOBAL_SEED).GeneratePopulation(),
                       eliminate_duplicates=duplicate_operator,
                       selection=selection_operator,
                       repair=RepairIndividuals()
                       )

    # Run the optimization
    res = minimize(problem,
                   algorithm,
                   termination=GetTerminationConditions(),
                   seed=config.GLOBAL_SEED,
                   verbose=True,
                   save_history=True,
                   return_least_infeasible=True)

    # Print some performance metrics
    print(f"Optimization completed in {res.exec_time:.2f} seconds")
    print(f"Best solution found: \nX = {res.X}\nF = {res.F}")

    """ Save the results to a dill file for future reference """
    # This avoids needing to re-run the optimization if the results are needed later.
    # The filename is generated using the process ID and current timestamp to ensure uniqueness.

    # First generate the results folder if it does not exist already
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True, parents=True)

    now = datetime.datetime.now()
    timestamp = f"{now:%y%m%d%H%M%S%f}"
    output_name = results_dir / f"res_pop{config.POPULATION_SIZE}_gen{config.MAX_GENERATIONS}_{timestamp}.dill"
    try:
        with open(output_name, 'wb') as f:
            dill.dump(res, f)
        print(f"Results saved to {output_name}")
    except Exception as e:
        print(f"Error saving results: {e}")