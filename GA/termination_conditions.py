"""
termination
===========

Description
-----------
This module defines a function to set the termination
conditions of the pymoo optimisation.

Examples
--------
>>> term_conditions = GetTerminationConditions()

Notes
-----
Ensure that the `config` module is properly set up with required parameters.

References
----------
For more details on pymoo and its termination options, refer to the official documentation:
https://pymoo.org/

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 1.2

Changelog:
- V1.0: Initial implementation.
- V1.1: Updated termination conditions for much needed improvement in multi-objective optimisation.
- V1.2: Updated consistency in multi-objective termination conditions. 
"""

# Import 3rd party libraries
from pymoo.termination.robust import RobustTermination
from pymoo.termination.ftol import SingleObjectiveSpaceTermination, MultiObjectiveSpaceTermination
from pymoo.termination.cv import ConstraintViolationTermination
from pymoo.termination.xtol import DesignSpaceTermination
from pymoo.termination import get_termination
from pymoo.termination.collection import TerminationCollection

# Import configuration module
import config #type: ignore

def GetTerminationConditions():
    """
    Simple function to set the termination conditions and ensure a single source of truth across different optimisation scripts.
    """
    if len(config.objective_IDs) == 1:
        # Set termination conditions for a single objective optimisation
        term_conditions = TerminationCollection(RobustTermination(SingleObjectiveSpaceTermination(tol=1E-5,
                                                                                                  only_feas=True),
                                                                                                  period=15),  # Change in objective value termination condition
                                                get_termination("n_gen", config.MAX_GENERATIONS),  # Maximum generation count termination condition
                                                RobustTermination(DesignSpaceTermination(tol=1E-3),
                                                                  period=15),  # Maximum change in design vector termination condition
                                                RobustTermination(ConstraintViolationTermination(tol=1E-8, terminate_when_feasible=False),
                                                                  period=15)  # Maximum change in constriant violation termination condition
                                                )
    else:
        # Set termination conditions for a multiobjective optimisation
        term_conditions = TerminationCollection(RobustTermination(MultiObjectiveSpaceTermination(tol=1E-3,
                                                                                          only_feas=True),
                                                                                          period=10),  # Change in objective value termination condition
                                                get_termination("n_gen", config.MAX_GENERATIONS),  # Maximum generation count termination condition
                                                RobustTermination(DesignSpaceTermination(tol=1E-3),
                                                                  period=10),  # Maximum change in design vector termination condition
                                                RobustTermination(ConstraintViolationTermination(tol=1E-8, terminate_when_feasible=False),
                                                                  period=10)  # Maximum change in constriant violation termination condition
                                                )


    return term_conditions