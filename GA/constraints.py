"""
constraints
===========

Description
-----------
This module provides an interface to define and compute constraints for optimization problems.
It includes methods to calculate equality and inequality constraints based on analysis outputs.

Classes
-------
Constraints
    A class to define and compute the constraints for the optimization problem.

Examples
--------
>>> constraints_class = Constraints()
>>> outputs = {"data": {"Total power CP": 0.5, "Total force CT": 0.3}}
>>> Lref = 1.0
>>> out = {}
>>> computed_constraints = constraints_class.ComputeConstraints(outputs, Lref, out)
>>> print(computed_constraints)

Notes
-----
This module is designed to work with optimization frameworks such as PyMoo.
The constraints are structured to be compatible with PyMoo's constraint handling approach.

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 1.7

Changelog:
- V1.0: Initial implementation with basic equality and inequality constraints.
- V1.1: Implemented inequality constraint for efficiency such that eta is always > 0.
- V1.2: Normalised constraints, added 1<T/Tref<1.01 constraint, extracted common power and thrust calculations to separate helper methods.
- V1.3: Implemented multi-point constraint evaluator. Updated documentation. Fixed type hinting.
- V1.4: Implemented constraints on profile parameterisations.
- V1.5: Fixed bug in minimum thrust constraint. Performance improvements by avoiding repeated construction/lookup of data.
- V1.6: Implemented theoretical efficiency limit calculation based on actuator disk theory. Improved efficiency calculations.
- V1.7: Added unified thrust bound constraint to replace the minimum and maximum thrust constraints. Removed deprecated constraints.
"""

# Import standard libraries
import copy
from typing import Any

# Import 3rd party libraries
import numpy as np

# Import analysis configuration and Parameterisation class
import config # type: ignore
from Submodels.Parameterisations import AirfoilParameterisation # type: ignore

# Define type alias for AnalysisOutputs
AnalysisOutputs = dict[str, dict[str, float] | dict[str, dict[str, float]]]

class Constraints:
    """
    Class containing all the constraints for the genetic algorithm optimisation.
    """


    # Large constraint violation value to penalize infeasible designs
    INFEASIBLE_CV = 1e12  # CV = Constraint Violation

    # Define rounding for constraints to match MTFLOW precision
    CONSTRAINT_PRECISION = 5


    def __init__(self,
                 centerbody_values: dict[str, float],
                 duct_values: dict[str, float],
                 blade_blading_values: list[dict[str, float]],
                 design_okay: bool = False,
                 ) -> None:
        """
        Initialisation of the Constraints class.
        """

        # Write the design variable dictionaries to self
        self.centerbody_values = centerbody_values.copy()
        self.duct_values = duct_values.copy()
        self.blade_blading_values = copy.deepcopy(blade_blading_values)  # Use deepcopy due to the nested structure of the design values

        # Define lists of all inequality and equality constraints
        self.ineq_constraints_list = [self.KeepEfficiencyFeasibleUpper, self.ThrustBound, self.FrontalArea]
        self.eq_constraints_list = [self.ConstantPower]

        self.design_okay = design_okay

        # Initialize the airfoil parameterisation class
        self.airfoil_parameterisation = AirfoilParameterisation()

        # Cache filtered constraints to avoid repeated list comprehensions
        self._cached_ineq_constraints = None
        self._cached_eq_constraints = None
        self._cached_constraint_ids = None


    def _calculate_power(self,
                         analysis_outputs: AnalysisOutputs,
                         Lref: float) -> float:
        """
        Helper method to calculate the power in Watts.

        Parameters
        ----------
        - analysis_outputs : AnalysisOutputs
            Outputs from MTFLOW
        - Lref : float
            Reference length

        Returns
        -------
        - Power : float
            A float of the power in Watts
        """
        return analysis_outputs["data"]["Total power CP"] * (0.5 * self.oper["atmos"].density[0] * self.oper["Vinl"] ** 3 * Lref ** 2)


    def _calculate_thrust(self,
                          analysis_outputs: AnalysisOutputs,
                          Lref: float) -> float:
        """
        Helper method to calculate the thrust in Newtons.

        Parameters
        ----------
        - analysis_outputs : AnalysisOutputs
            Outputs from MTFLOW
        - Lref : float
            Reference length

        Returns
        -------
        - Thrust : float
            A float of the thrust in Newtons
        """
        return analysis_outputs["data"]["Total force CT"] * (0.5 * self.oper["atmos"].density[0] * self.oper["Vinl"] ** 2 * Lref ** 2)


    def _calculate_theoretical_efficiency_limit(self,
                                                thrust: float) -> float:
        """
        Calculate the theoretical efficiency limit based on the operating condition and the design parameters.
        This method computes the theoretical efficiency limit based on actuator disk theory.

        Parameters
        ----------
        - thrust : float
            The thrust in Newtons.
        
        Returns
        -------
        - eta_theoretical : float
            The theoretical efficiency limit of the ducted fan.
        """

        # Compute the profile coordinates of the upper surface of the centerbody
        upper_x, upper_y, _, _ = self.airfoil_parameterisation.ComputeProfileCoordinates(self.centerbody_values)

        # Dimensionalise the coordinates using the chord length
        chord = self.centerbody_values["Chord Length"]
        upper_y = upper_y * chord
        upper_x = upper_x * chord

        # We use the LE of the root as x coordinate at which to compute the actuator disk area.
        closest_index = np.abs(upper_x - self.blade_blading_values[0]["root_LE_coordinate"]).argmin()
        centerbody_radius = upper_y[closest_index]

        # Compute the annular area occupied by the fan and use it to find the theoretical efficiency limit.
        A_disk = np.pi * self.blade_blading_values[0]["radial_stations"][-1] ** 2 - np.pi * centerbody_radius ** 2
        term = (0.5 * self.oper["atmos"].density[0] * self.oper["Vinl"] ** 2 * A_disk)

        # If thrust is negative, we cannot compute a theoretical efficiency limit, so we set it to 0.01 to enable the optimiser to still work
        eta_theoretical = 0.01 if thrust <= 0 else 2 / (1 + (thrust / term + 1) ** 0.5)

        return eta_theoretical


    def ConstantPower(self,
                      _analysis_outputs: AnalysisOutputs,
                      _Lref: float,
                      _thrust: float,
                      power: float) -> float:
        """
        Compute the equality constraint for the power coefficient.

        Parameters
        ----------
        - _analysis_outputs : AnalysisOutputs
            A dictionary containing the outputs from the MTFLOW forces output file.
            Must contain all entries corresponding to an execution of
            output_handling.output_processing().GetAllVariables(3)
        - _Lref : float
            The reference length of the analysis. Corresponds to the propeller/fan diameter.
        - _thrust : float
            The thrust in Newtons. Not used here but included to force constant signature.
        - power : float
            The power in Watts.

        Note: this method uses the pre-calculated 'power' value passed directly rather than recalculating it from
        'analysis_outputs' for efficiency.

        Returns
        -------
        - float
            The computed normalised power constraint. This is a scalar value representing the power of the system.
        """

        return (power - self.ref_power) / self.ref_power  # Normalized power constraint
    

    def OnlyImprovedEnergy(self,
                           outputs: list[AnalysisOutputs] | AnalysisOutputs,
                           Lref: float) -> float:
        """
        Define constraint to only allow energies lower than the reference
        Works for both single-point and multi-point optimisation problems.

        Parameters
        ----------
        - outputs : AnalysisOutputs
            A dictionary or list of dictionaries containing the outputs from the forces.xxx file.
            outputs should be structured based on output mode 3 of output_handling.output_processing.GetAllVariables().
        - Lref : float
            A float of the reference length

        Returns
        -------
        - Energy : float
            A float of the energy used to power the ducted fan during the flight phase.
            This is computed as the product of the power and the time spent in the flight phase.
        """

        if isinstance(outputs, list):
            # If outputs is a list, we are solving for a multi-point optimisation problem, so we sum the total energy for all points.
            energy = 0
            for idx, output in enumerate(outputs):
                self.oper = self.multi_oper[idx]  # Set the correct operating condition to compute the power
                power = self._calculate_power(output, Lref)
                energy += power * self.oper["flight_phase_time"]
        else:
            # If outputs is a single dictionary, we are solving for a single-point optimisation problem.
            power = self._calculate_power(outputs, Lref)
            energy = power * self.oper["flight_phase_time"]

        return round(energy / config.reference_energy - 1, self.CONSTRAINT_PRECISION)


    def KeepEfficiencyFeasibleUpper(self,
                               analysis_outputs: AnalysisOutputs,
                               _Lref: float,
                               thrust: float,
                               _power: float) -> float:
        """
        Compute the inequality constraint for the efficiency. Enforces that 
        eta < eta_theoretical, i.e. the theoretical upper efficiency as obtained 
        from actuator disk theory.

        Parameters
        ----------
        - analysis_outputs : AnalysisOutputs
            A dictionary containing the outputs from the MTFLOW forces output file.
            Must contain all entries corresponding to an execution of
            output_handling.output_processing().GetAllVariables(3).
        - _Lref : float
            The reference length of the analysis. Corresponds to the propeller/fan diameter.
            Not used in this method, but required for a uniform constraint function signature.
        - thrust : float
            The thrust in Newtons.
        - _power : float
            The power in Watts. Not used here but included to force constant signature.

        Returns
        -------
        - float
            The computed efficiency constraint. This is a scalar value representing the efficiency of the system.
        """

        # Compute the inequality constraint for the efficiency.
        return analysis_outputs["data"]["EtaP"] - self._calculate_theoretical_efficiency_limit(thrust)


    def ThrustBound(self,
                    _analysis_outputs: AnalysisOutputs,
                    _Lref: float,
                    thrust: float,
                    _power: float) -> float:
        """
        Compute the inequality constraint for the thrust. Enforces that -delta < T/T_ref - 1 <delta.

        Parameters
        ----------
        - _analysis_outputs : AnalysisOutputs
            A dictionary containing the outputs from the MTFLOW forces output file.
            Must contain all entries corresponding to an execution of
            output_handling.output_processing().GetAllVariables(3).
        - _Lref : float
            The reference length of the analysis. Corresponds to the propeller/fan diameter.
            Not used here but included to force constant signature.
        - thrust : float
            The thrust in Newtons.
        - _power : float
            The power in Watts. Not used here but included to force constant signature.
        
        Returns
        -------
        - float
            The computed normalised thrust constraint. This is a scalar value representing the deviation of the thrust of the system from the reference value.
        """

        return abs(thrust / self.ref_thrust - 1) - config.deviation_range


    def FrontalArea(self,
                    _analysis_outputs: AnalysisOutputs,
                    _Lref: float,
                    _thrust: float,
                    _power: float
                    ) -> float:
        """
        Compute the frontal area of the ducted propeller/fan. 
        Enforces that the normalised frontal area is less than config.MAX_FRONTAL_AREA_RATIO.

        Returns
        -------
        - float
            A scalar value representing the frontal area constraint.
        """

        # To compute the frontal area, we need the maximum radius of the ducted propeller/fan.
        # This can be computed based on the radial LE coordinate of the duct,
        # together with the maximum y-coordinate of the duct profile.

        # For a symmetric profile, this is simply equal to y_t, 
        # so we do not need to compute the airfoil upper surface distribution
        if self.duct_values["y_c"] < 1e-3:
            upper_y = self.duct_values["y_t"]
        else:
            # For a cambered profile, compute the airfoil coordinates
            # We only care about the upper y coordinates so they are the only ones we store
            _, upper_y, _, _ = self.airfoil_parameterisation.ComputeProfileCoordinates(self.duct_values)

        # Dimensionalise the y coordinates using the chord length
        chord = self.duct_values["Chord Length"]
        upper_y = upper_y * chord

        # Compute the maximum radius
        max_radius = self.duct_values["Leading Edge Coordinates"][1] + np.max(upper_y)

        # Since we deal with axisymmetric geometry, the frontal area is then simply the area of a circle
        frontal_area = np.pi * max_radius ** 2

        # Return the frontal area normalised by the reference frontal area in config
        # This is needed to ensure all objectives are of the same order of magnitude and thus have equal weight to the GA optimiser.
        return frontal_area / config.REF_FRONTAL_AREA - config.MAX_FRONTAL_AREA_RATIO  # Normalized frontal area constraint. <=0 indicates feasible.


    def _get_filtered_constraints(self):
        """
        Get cached filtered constraints or create them if the cache is invalid.
        """

        current_constraint_ids = (tuple(config.constraint_IDs[0]), tuple(config.constraint_IDs[1]))

        if self._cached_constraint_ids != current_constraint_ids:
            self._cached_ineq_constraints = [self.ineq_constraints_list[i] for i in config.constraint_IDs[0] if i != config.InEqConstraintID.ENERGY_IMPROVEMENT]
            self._cached_eq_constraints = [self.eq_constraints_list[i] for i in config.constraint_IDs[1]]
            self._cached_constraint_ids = current_constraint_ids

        return self._cached_ineq_constraints, self._cached_eq_constraints


    def ComputeConstraints(self,
                           analysis_outputs: AnalysisOutputs,
                           Lref: float,
                           oper: dict[str, Any],
                           out: dict) -> None:
        """
        Compute the inequality and equality constraints based on the provided analysis outputs
        and configuration, and store the results in the output dictionary.

        Parameters
        ----------
        - analysis_outputs: AnalysisOutputs
            A dictionary containing the results of the analysis,
            which are used as inputs to the constraint functions.
        - Lref : float
            A reference length used in the computation of constraints.
        - oper : dict[str, Any]
            The operating conditions dictionary.
        - out : dict
            A dictionary to store the computed constraints. The keys "G" and "H"
            will be populated with the inequality and equality constraints, respectively.

        Returns
        -------
        None, the out dictionary is updated in place with the computed constraints.

        Notes
        -----
            - The function uses `config.constraint_IDs` to determine which constraints to compute.
            - `config.constraint_IDs[0]` specifies the indices of inequality constraints.
            - `config.constraint_IDs[1]` specifies the indices of equality constraints.
            - If no constraints are specified, the corresponding output arrays ("G" or "H")
              will be empty 2D numpy arrays.
        """

        # Copy the operating conditions
        self.oper = oper.copy()

        # Get all inequality and equality constraints based on the constraint IDs
        ineq_constraints, eq_constraints = self._get_filtered_constraints()

        # Compute thrust and power
        thrust = self._calculate_thrust(analysis_outputs, Lref)
        power = self._calculate_power(analysis_outputs, Lref)

        # Set the reference values to self
        # Uses only the first entries in th elist since this is a single-point evaluation.
        self.ref_power = config.P_ref_constr[0]
        self.ref_thrust = config.T_ref_constr[0]

        # Compute the inequality constraints and write them to out["G"]
        # Rounds the constraint values to 5 decimal figures to match the number of sigfigs given by the MTFLOW outputs to avoid rounding errors.
        if ineq_constraints:
            computed_ineq_constraints = [round(constraint(analysis_outputs, Lref, thrust, power), self.CONSTRAINT_PRECISION)
                                         for constraint in ineq_constraints]

            if self.design_okay:
                out["G"] = np.array(computed_ineq_constraints, dtype=float)
            else:
                # If the design is infeasible, set a really high constraint violation to steer the optimizer away.
                out["G"] = np.full(len(computed_ineq_constraints), self.INFEASIBLE_CV, dtype=float)

        # Compute the equality constraints and write them to out["H"]
        # Rounds the constraint values to 5 decimal figures to match the number of sigfigs given by the MTFLOW outputs to avoid rounding errors.
        if eq_constraints:
            computed_eq_constraints = [round(constraint(analysis_outputs, Lref, thrust, power), self.CONSTRAINT_PRECISION)
                                       for constraint in eq_constraints]

            if self.design_okay:
                out["H"] = np.array(computed_eq_constraints, dtype=float)
            else:
                # If the design is infeasible, set a really high constraint violation to steer the optimizer away.
                out["H"] = np.full(len(computed_eq_constraints), self.INFEASIBLE_CV, dtype=float)
        else:
            out["H"] = [[]]


    def _compute_multi_point_constraints(self,
                                         constraint_list: list,
                                         analysis_outputs: list[AnalysisOutputs],
                                         Lref: float,
                                         thrust: np.ndarray,
                                         power: np.ndarray) -> list[float]:
        """Helper method to compute constraints for multi-point analysis."""
        computed_constraints = []

        # Pre-extract reference values to avoid repeated config access
        ref_thrusts = config.T_ref_constr
        ref_powers = config.P_ref_constr

        for i, outputs in enumerate(analysis_outputs):
            self.ref_thrust = ref_thrusts[i]
            self.ref_power = ref_powers[i]
            self.oper = self.multi_oper[i]
            computed_constraints.extend([round(constraint(outputs, Lref, thrust[i], power[i]), self.CONSTRAINT_PRECISION)
                                        for constraint in constraint_list])
        return computed_constraints


    def ComputeMultiPointConstraints(self,
                                     analysis_outputs: list[AnalysisOutputs],
                                     Lref: float,
                                     oper: list[dict[str, Any]],
                                     out: dict) -> None:
        """
        Compute the inequality and equality constraints for a multi-point analysis based on the provided analysis outputs
        and configuration, and store the results in the output dictionary.

        Parameters
        ----------
        - analysis_outputs: list[AnalysisOutputs]
            A list of the output dictionaries containing the results of the analyses,
            which are used as inputs to the constraint functions.
        - Lref : float
            A reference length used in the computation of constraints.
        - oper : list[dict[str, float]]
            A list containing all operating condition dictionaries for each of the operating points in the multi-point analysis.
        - out : dict
            A dictionary to store the computed constraints. The keys "G" and "H"
            will be populated with the inequality and equality constraints, respectively.

        Returns
        -------
        None, the out dictionary is updated in place with the computed constraints.

        Notes
        -----
            - The function uses `config.constraint_IDs` to determine which constraints to compute.
            - `config.constraint_IDs[0]` specifies the indices of inequality constraints.
            - `config.constraint_IDs[1]` specifies the indices of equality constraints.
            - If no constraints are specified, the corresponding output arrays ("G" or "H")
              will be empty 2D numpy arrays.
        """

        # Copy the operating conditions
        self.multi_oper = oper.copy()

        # Get all inequality and equality constraints based on the constraint IDs
        ineq_constraints, eq_constraints = self._get_filtered_constraints()

        num_outputs = len(analysis_outputs)

        # Compute thrust and power
        thrust = np.empty(num_outputs, dtype=float)
        power = np.empty(num_outputs, dtype=float)

        for i in range(num_outputs):
            self.oper = self.multi_oper[i]  # Set the correct operating condition to compute the thrust/power
            thrust[i] = self._calculate_thrust(analysis_outputs[i], Lref)
            power[i] = self._calculate_power(analysis_outputs[i], Lref)


        # Compute the inequality constraints and write them to out["G"]
        # Rounds the constraint values to 5 decimal figures to match the number of sigfigs given by the MTFLOW outputs to avoid rounding errors.
        if ineq_constraints:
            computed_ineq_constraints = self._compute_multi_point_constraints(ineq_constraints, analysis_outputs, Lref, thrust, power)
            if config.InEqConstraintID.ENERGY_IMPROVEMENT in config.constraint_IDs[0]:
                    computed_ineq_constraints.append(self.OnlyImprovedEnergy(analysis_outputs, Lref))

            if self.design_okay:
                out["G"] = np.array(computed_ineq_constraints, dtype=float)
            else:
                # If the design is infeasible, set a really high constraint violation to steer the optimizer away
                out["G"] = np.full(len(computed_ineq_constraints), self.INFEASIBLE_CV, dtype=float)

        # Compute the equality constraints and write them to out["H"]
        # Rounds the constraint values to 5 decimal figures to match the number of sigfigs given by the MTFLOW outputs to avoid rounding errors.
        if eq_constraints:
            computed_eq_constraints = self._compute_multi_point_constraints(eq_constraints, analysis_outputs, Lref, thrust, power)
            if self.design_okay:
                out["H"] = np.array(computed_eq_constraints, dtype=float)
            else:
                # If the design is infeasible, set a really high constraint violation to steer the optimizer away.
                out["H"] = np.full(len(computed_eq_constraints), self.INFEASIBLE_CV, dtype=float)


if __name__ == "__main__":
    # Test execution of constraints using a test-case forces output.

    # Add the parent and submodels paths to the system path
    import sys
    from pathlib import Path
    # Add the parent and submodels paths to the system path if they are not already in the path
    parent_path = str(Path(__file__).resolve().parent.parent)
    submodels_path = str(Path(__file__).resolve().parent.parent / "Submodels")

    if parent_path not in sys.path:
        sys.path.append(parent_path)

    if submodels_path not in sys.path:
        sys.path.append(submodels_path)

    # Import MTFLOW interface submodels and other dependencies
    from Submodels.output_handling import output_processing #type: ignore

    # Extract outputs from the forces output file
    outputs = output_processing(analysis_name='test_case').GetAllVariables(3)

    # Create an instance of the Constraints class
    test = Constraints(config.CENTERBODY_VALUES,
                       config.DUCT_VALUES,
                       config.STAGE_BLADING_PARAMETERS,
                       design_okay=True)

    # Compute the constraints - single point constraint analysis
    output = {}
    test.ComputeConstraints(outputs,
                            Lref=config.BLADE_DIAMETERS[0],
                            oper=config.multi_oper[0],
                            out=output)

    print(output)

    # Compute the constraints - multi-point constraint analysis
    output = {}
    test.ComputeMultiPointConstraints([outputs, outputs],
                                      Lref=config.BLADE_DIAMETERS[0],
                                      oper=config.multi_oper,
                                      out=output)

    print(output)