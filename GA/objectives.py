"""
objectives
==========

Description
-----------
This module provides an interface to define and compute objectives for optimization problems.
It includes methods to calculate various sub-objectives such as efficiency, weight, frontal area,
pressure ratio, and multi-point objectives.

Classes
-------
Objectives
    A class to define and compute the objectives for the optimization problem.

Examples
--------
>>> objectives_class = Objectives()
>>> outputs = {"data": {"EtaP": 0.85, "Pressure Ratio": 1.2}}
>>> objective_IDs = [0, 3]
>>> computed_objectives = objectives_class.ComputeObjective(outputs, objective_IDs)
>>> print(computed_objectives)

Notes
-----
This module is designed to work with optimization frameworks such as PyMoo.
The objectives are structured to be compatible with PyMoo's minimization-based approach.

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 3.0

Changelog:
- V1.0: Initial implementation with basic sub-objectives and placeholders for unimplemented methods.
- V1.1: Added ComputeObjective method to handle multiple objectives dynamically.
- V1.2: Improved documentation and added type hints for better clarity.
- V2.0: Refactored code for better modularity and maintainability. Updated examples and notes.
- V2.1: Updated efficiency objective for better performance. Improved efficiency of frontal area objective
- V3.0: Renamed MTFLOW to UDC for consistency with written thesis.
"""

# Import standard libraries
from typing import Any

# Import 3rd party libraries
import numpy as np

# Import config module
import config #type: ignore


# Define type alias for analysis outputs
AnalysisOutputs = dict[str, dict[str, float] | dict[str, dict[str, float]]]

class Objectives:
    """
    A class to define the objectives for the optimization problem.
    """

    def __init__(self,
                 duct_variables : dict[str, Any],
                 oper: list[dict[str, Any]],
                 Lref: float,
                 **kwargs) -> None:
        """
        Initialisation of the Objectives class.

        Parameters
        ----------
        - duct_variables: dict[str, any]
            The duct design variable dictionary.

        Returns
        -------
        None
        """

        # Write the inputs to self
        self.duct_variables = duct_variables
        self.oper = oper
        self.Lref = Lref

        # Define the objective list, and the subset of objectives which are independent of operating condition
        self.objectivelist = [self.Efficiency, self.FrontalArea, self.WettedArea, self.PressureRatio]
        self.constant_objectiveIDs = {1, 2}


    def Efficiency(self,
                   outputs: AnalysisOutputs) -> float:
        """
        Define the efficiency (sub-)objective.
        This sub-objective has identifier 0.

        Parameters
        ----------
        - outputs : AnalysisOutputs
            A dictionary containing the outputs from the forces.xxx file.
            outputs should be structured based on output mode 3 of output_handling.output_processing.GetAllVariables().

        Returns
        -------
        - Propulsive Efficiency: float
            A float of the propulsive efficiency objective, defined as -EtaP.
        """

        return -outputs['data']['EtaP']


    def FrontalArea(self,
                    _outputs: AnalysisOutputs) -> float:
        """
        Define the frontal area (sub-)objective.
        This sub-objective has identifier 1.

        Parameters
        ----------
        - _outputs : AnalysisOutputs
            A dictionary containing the outputs from the forces.xxx file.
            _outputs should be structured based on output mode 3 of output_handling.output_processing.GetAllVariables().

        Returns
        -------
        - frontal_area : float
            The frontal area normalised by the reference frontal area.
        """

        if not hasattr(self, "_airfoil_parameterisation"):
            # Lazy import and cache the AirfoilParameterisation class
            from Submodels.Parameterisations import AirfoilParameterisation  # type: ignore
            self._airfoil_parameterisation = AirfoilParameterisation()

        # To compute the frontal area, we need the maximum radius of the ducted propeller/fan.
        # This can be computed based on the radial LE coordinate of the duct,
        # together with the maximum y-coordinate of the duct profile.

        # For a symmetric profile, this is simply equal to y_t,
        # so we do not need to compute the airfoil upper surface distribution
        if self.duct_variables["y_c"] < 1e-3:
            upper_y = self.duct_variables["y_t"]
        else:
            # For a cambered profile, compute the airfoil coordinates
            # We only care about the upper y coordinates so they are the only ones we store
            _, upper_y, _, _ = self._airfoil_parameterisation.ComputeProfileCoordinates(self.duct_variables)

        # Dimensionalise the y coordinates using the chord length
        chord = self.duct_variables["Chord Length"]
        upper_y = upper_y * chord

        # Compute the maximum radius
        max_radius = self.duct_variables["Leading Edge Coordinates"][1] + np.max(upper_y)

        # Since we deal with axisymmetric geometry, the frontal area is then simply the area of a circle
        frontal_area = np.pi * max_radius ** 2

        # Return the frontal area normalised by the reference frontal area in config
        # This is needed to ensure all objectives are of the same order of magnitude and thus have equal weight to the GA optimiser.
        return frontal_area / config.REF_FRONTAL_AREA


    def WettedArea(self,
                    outputs: AnalysisOutputs) -> float:
        """
        Define the wetted area (sub-)objective.
        This sub-objective has identifier 2.

        Parameters
        ----------
        - outputs : AnalysisOutputs
            A dictionary containing the outputs from the forces.xxx file.
            outputs should be structured based on output mode 3 of output_handling.output_processing.GetAllVariables().

        Returns
        -------
        - wetted_area : float
            The wetted area as taken from the output file forces.analysis_name.
            The wetted area is computed in the normalised coordinate system of UDC,
            so it is multiplied by Lref^2 to obtain the dimensional form.
        """

        return outputs["data"]["Wetted Area"] * self.Lref ** 2


    def PressureRatio(self,
                      outputs: AnalysisOutputs) -> float:
        """
        Define the pressure ratio (sub-)objective.
        This sub-objective has identifier 3.

        Parameters
        ----------
        - outputs : AnalysisOutputs
            A dictionary containing the outputs from the forces.xxx file.
            outputs should be structured based on output mode 3 of output_handling.output_processing.GetAllVariables().

        Returns
        -------
        - Pressure Ratio : float
            A float of the exit pressure ratio objective.
        """

        return - outputs["data"]["Pressure Ratio"]


    def FlightPhaseEnergy(self,
                          outputs: list[AnalysisOutputs] | AnalysisOutputs) -> float:
        """
        Compute the energy used to power the ducted fan during the flight phase.
        Works for both single-point and multi-point optimisation problems.
        This sub-objective has identifier 4.

        Parameters
        ----------
        - outputs : AnalysisOutputs
            A dictionary or list of dictionaries containing the outputs from the forces.xxx file.
            outputs should be structured based on output mode 3 of output_handling.output_processing.GetAllVariables().

        Returns
        -------
        - Energy : float
            A float of the energy used to power the ducted fan during the flight phase.
            This is computed as the product of the power and the time spent in the flight phase.
        """

        if isinstance(outputs, list):
            # If outputs is a list, we are solving for a multi-point optimisation problem, so we sum the total energ for all points.
            energy = 0
            for idx, output in enumerate(outputs):
                power = output["data"]["Total power CP"] * (0.5 * self.oper[idx]["atmos"].density[0] * self.oper[idx]["Vinl"] ** 3 * self.Lref ** 2)
                energy += power * self.oper[idx]["flight_phase_time"] / config.reference_energy
        else:
            # If outputs is a single dictionary, we are solving for a single-point optimisation problem.
            power = outputs["data"]["Total power CP"] * (0.5 * self.oper["atmos"].density[0] * self.oper["Vinl"] ** 3 * self.Lref ** 2)
            energy = power * self.oper["flight_phase_time"] / config.reference_energy

        return energy


    def ComputeObjective(self,
                         analysis_outputs: dict,
                         objective_IDs: list[int],
                         out: dict) -> None:
        """
        Computes the objectives for optimization based on the provided analysis outputs.
        This method evaluates a list of objective functions, specified by their IDs in the
        configuration, and returns their computed values. The objectives are negated to
        convert maximization objectives (e.g., maximize efficiency) into minimization
        objectives, as required by the PyMoo optimization framework.

        Parameters
        ----------
        - analysis_outputs : dict
            A dictionary containing the outputs of the analysis
            required for computing the objectives.
        - out : dict
            A dictionary to store the computed objectives.

        Returns
        -------
        None, the out dictionary is updated in place with the computed objectives.
        """

        objectives = [self.objectivelist[i] for i in objective_IDs]

        computed_objectives = np.fromiter(
            (round(f(analysis_outputs), 5) for f in objectives),
            dtype=float,
            count=len(objectives),
        )

        if 4 in objective_IDs:
            # If the flight phase energy objective is requested, compute it separately
            # and append it to the computed objectives.
            flight_phase_energy = self.FlightPhaseEnergy(analysis_outputs)
            computed_objectives = np.append(computed_objectives, round(flight_phase_energy, 5))

        out["F"] = computed_objectives

        # Check dimension of objectives
        assert out["F"].ndim == 1, "ElementwiseProblem needs a 1-D objective array"


    def ComputeMultiPointObjectives(self,
                                    analysis_outputs: list[dict],
                                    objective_IDs: list[int],
                                    out: dict) -> None:
        """
        Computes the objectives for a multi-point optimization based on the provided analysis outputs.
        This method evaluates a list of objective functions, specified by their IDs in the
        configuration, and returns their computed values. The objectives are negated to
        convert maximization objectives (e.g., maximize efficiency) into minimization
        objectives, as required by the PyMoo optimization framework.

        Parameters
        ----------
        - analysis_outputs : list[dict]
            A list of dictionaries containing the outputs of the analysis
            required for computing the objectives.
        - out : dict
            A dictionary to store the computed objectives.

        Returns
        -------
        None, the out dictionary is updated in place with the computed objectives.
        """

        variable_IDs = [oid for oid in objective_IDs if oid not in self.constant_objectiveIDs and oid !=4]  # Identifiers for the variable objective functions
        constant_IDs = [oid for oid in objective_IDs if oid in self.constant_objectiveIDs and oid !=4]  # Identifiers for the constant variable objective functions

        variable_objectives = [self.objectivelist[i] for i in variable_IDs]  # The variable objectives
        constant_objectives = [self.objectivelist[i] for i in constant_IDs]  # The constant objectives

        # Compute the relevant dimensions and construct the empty objectives output array
        num_outputs = len(analysis_outputs)
        num_varobjectives = len(variable_objectives)
        num_constobjectives = len(constant_objectives)
        computed_objectives = np.empty(num_varobjectives * num_outputs + num_constobjectives, dtype=float)

        # First compute the outputs which are a function of operating condition
        for i, outputs in enumerate(analysis_outputs):
            # Rounds the objective values to 5 decimal figures to match the number of sigfigs given by the UDC outputs to avoid rounding errors.
            computed_objectives[i * num_varobjectives : (i + 1) * num_varobjectives] = np.fromiter(
                (round(obj(outputs), 5) for obj in variable_objectives),
                dtype=float,
                count=num_varobjectives
            )

        # Now compute the constant objectives
        for i, objective in enumerate(constant_objectives):
            computed_objectives[num_varobjectives * num_outputs + i] = round(objective(analysis_outputs[0]), 5)

        # If the flight phase energy objective is requested, compute it separately
        if 4 in objective_IDs:
            flight_phase_energy = self.FlightPhaseEnergy(analysis_outputs)
            computed_objectives = np.append(computed_objectives, round(flight_phase_energy, 5))

        out["F"] = computed_objectives

        # Check dimension of objectives
        assert out["F"].ndim == 1, "ElementwiseProblem needs a 1-D objective array"

if __name__ == "__main__":
    # Run a test of the objectives class

    # Add the parent folder path to the system path
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).resolve().parent.parent
    submodels_path = parent_dir / "Submodels"
    sys.path.extend([str(parent_dir), str(submodels_path)])

    from Submodels.output_handling import output_processing #type: ignore

    objectives_class = Objectives(config.DUCT_VALUES, config.multi_oper, config.BLADE_DIAMETERS[0])
    output = {}
    objectives_class.ComputeObjective(output_processing('x22a_validation').GetAllVariables(0),
                                      config.objective_IDs,
                                      output)
    print(output)