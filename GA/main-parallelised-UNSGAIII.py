"""
main-parallelised-UNSGAIII
====

Description
-----------
This module defines the main entry point for running a multi-threaded optimisation problem using the pymoo framework.
Uses the starmap parallelisation method for flexible parallelisation opportunities.
Uses the Unified-Nondominated Sorting Genetic Algorithm 3 with the mixed-variable implementation.

Functionality
-------------
- Initializes an optimisation problem using UNSGA-III with mixed-variable support.
- Configures and runs a genetic algorithm (GA) for optimisation.
- Saves the results to a dill database for future reference.

Examples
--------
>>> python main-parallelised-UNSGAIII.py

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
Version: 1.0

Changelog:
- V1.0: Initial implementation.
"""

# Import standard libraries
import datetime
import os
import multiprocessing
from pathlib import Path

# Import 3rd party libraries
import dill
from pymoo.core.mixed import MixedVariableMating, MixedVariableDuplicateElimination
from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.unsga3 import UNSGA3, comp_by_rank_and_ref_line_dist
from pymoo.operators.selection.tournament import TournamentSelection

# Ensure parent process has the correct import paths
from utils import ensure_repo_paths, calculate_n_reference_points # type: ignore
ensure_repo_paths()

# Import interface submodels and other dependencies
import config # type: ignore
from problem_definition import OptimizationProblem # type: ignore
from init_population import InitPopulation # type: ignore
from termination_conditions import GetTerminationConditions # type: ignore
from repair import RepairIndividuals # type: ignore


if __name__ == "__main__":
    multiprocessing.freeze_support() # Required for Windows compatibility when using multiprocessing
    if os.name == 'nt':
        multiprocessing.set_start_method('spawn', force=True)

    """ Initialise the thread pool and create the runner """
    total_threads = multiprocessing.cpu_count()
    threads_per_eval = max(1, getattr(config, "THREADS_PER_EVALUATION", 2))
    total_threads_avail = max(0, total_threads - config.RESERVED_THREADS)

    if total_threads_avail < threads_per_eval:
        # No point spawning processes that will immediately contend for the same cores
        n_processes = 0
    else:
        n_processes = total_threads_avail // threads_per_eval

    # Always fall back to at least one serial worker to ensure the script still runs.
    n_processes = max(1, n_processes)

    # Do not spawn more processes than the GA can effectively use
    n_processes = min(n_processes, config.POPULATION_SIZE)

    print(f"Spawning {n_processes} worker processes (total threads: {total_threads}, available: {total_threads_avail}, threads per eval: {threads_per_eval})")
    with multiprocessing.Pool(processes=n_processes,
                              initializer=ensure_repo_paths,
                              maxtasksperchild=100,
                              ) as pool:
        # Create runner
        runner = StarmapParallelization(pool.starmap)

        """ Initialise the optimisation problem and algorithm """
        # Initialise the optimisation problem by passing the configuration and the starmap interface of the thread_pool
        if hasattr(config, "PROBLEM_TYPE") and config.PROBLEM_TYPE == "multi_point":
            from multi_point_problem_definition import MultiPointOptimizationProblem # type: ignore
            problem = MultiPointOptimizationProblem(elementwise_runner=runner,
                                                    seed=config.GLOBAL_SEED)
        else:
            problem = OptimizationProblem(elementwise_runner=runner,
                                          seed=config.GLOBAL_SEED)

        # Create the reference directions to be used for the optimisation
        ref_dirs = get_reference_directions("energy",
                                            n_dim=config.n_objectives,
                                            n_points=calculate_n_reference_points(config))

        # Initialise the algorithm
        duplicate_elimination = MixedVariableDuplicateElimination()
        selection_operator = TournamentSelection(func_comp=comp_by_rank_and_ref_line_dist)
        algorithm = UNSGA3(ref_dirs=ref_dirs,
                           pop_size=config.POPULATION_SIZE,
                           mating=MixedVariableMating(selection=selection_operator,
                                                      eliminate_duplicates=duplicate_elimination,
                                                      repair=RepairIndividuals()),
                           sampling=InitPopulation(population_type="biased",
                                                   seed=config.GLOBAL_SEED).GeneratePopulation(),
                           eliminate_duplicates=duplicate_elimination,
                           selection=selection_operator,
                           repair=RepairIndividuals()
                           )

        # Run the optimisation
        res = minimize(problem,
                       algorithm,
                       termination=GetTerminationConditions(),
                       seed=config.GLOBAL_SEED,
                       verbose=True,
                       save_history=True,
                       return_least_infeasible=True)

        # Print some performance metrics
        print(f"Optimisation completed in {res.exec_time:.2f} seconds")
        print(f"Best solution found: \nX = {res.X}\nF = {res.F}")

        """ Save the results to a dill file for future reference """
        # This avoids needing to re-run the optimisation if the results are needed later.
        # The filename is generated using the process ID and current timestamp to ensure uniqueness.

        # First generate the results folder if it does not exist already
        results_dir = Path(__file__).resolve().parent / "results"
        results_dir.mkdir(exist_ok=True,
                        parents=True)

        now = datetime.datetime.now()
        timestamp = f"{now:%y%m%d%H%M%S%f}"
        output_name = results_dir / f"res_pop{config.POPULATION_SIZE}_eval{config.MAX_EVALUATIONS}_{timestamp}.dill"
        try:
            with open(output_name, 'wb') as f:
                dill.dump(res, f)
            print(f"Results saved to {output_name}")
        except Exception as e:
            print(f"Error saving results: {e}")
