"""
problem_definition-multistage
==================

Description
-----------
This module defines an optimization problem for a multi-point optimisation at n operating points for the pymoo framework, based on the ElementwiseProblem parent class.
The model is based on the single-point optimisation routine in the problem_definition.py file

Classes
-------
MultiPointOptimizationProblem(ElementwiseProblem)
    Class defining the optimization problem with mixed-variable support.

Examples
--------
>>> problem = MultiPointOptimizationProblem()
>>> out = {}
>>> problem._evaluate(1, out)

Notes
-----
This module integrates with the UDFDAC for aerodynamic analysis. Ensure that the executable and required
input files are present in the appropriate directories. The module is designed to handle mixed-variable optimization
problems, including real and integer variables.

References
----------
For more details on the MTFLOW solver integrated in the UDFDAC and its input/output requirements, refer to the MTFLOW user manual:
https://web.mit.edu/drela/Public/web/mtflow/mtflow.pdf

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 2.0

Changelog:
- V1.0: Initial implementation.
- V2.0: Renamed MTFLOW_caller to UDFDAC for consistency with written thesis. Updated imports to reflect new structure.
"""

# Import standard libraries
import os
import uuid
import datetime
import copy
import contextlib
from pathlib import Path

# Import 3rd party libraries
import numpy as np
from pymoo.core.problem import ElementwiseProblem

# Ensure all paths are correctly setup
from utils import ensure_repo_paths  # type: ignore
ensure_repo_paths()

# Import interface submodels and other dependencies
from Submodels.MTSOL_call import OutputType, ExitFlag  # type: ignore
from objectives import Objectives  # type: ignore
from constraints import Constraints  # type: ignore
from init_designvector import DesignVector  # type: ignore
from design_vector_interface import DesignVectorInterface  # type: ignore
import config  # type: ignore


class MultiPointOptimizationProblem(ElementwiseProblem):
    """
    Class definition of the optimization problem to be solved using the genetic algorithm.
    Inherits from the ElementwiseProblem class from pymoo.core.problem.
    """

    # Define the file names relevant for UDFDAC
    FILE_TEMPLATES = {"walls": "walls.{}",
                      "tflow": "tflow.{}",
                      "forces": "forces.{}",
                      "flowfield": "flowfield.{}",
                      "boundary_layer": "boundary_layer.{}",
                      "tdat": "tdat.{}"}

    # Initialize output dictionary to use in case of an infeasible design.
    # This equals the outputs of the output_handling.output_processing.GetAllVariables(3) method,
    # but is quicker as it does not involve reading a file.
    CRASH_OUTPUTS: dict[str, dict[str, float] | dict[str, dict[str, float]]] = {'data':
                                                                                {'Total power CP': 0.00000,
                                                                                'EtaP': 0.00000,
                                                                                'Total force CT': 0.00000,
                                                                                'Element 2 top CTV': 0.00000,
                                                                                'Element 2 bot CTV': 0.00000,
                                                                                'Axis body CTV': 0.00000,
                                                                                'Viscous CTv': 0.00000,
                                                                                'Inviscid CTi': 0.00000,
                                                                                'Friction CTf': 0.00000,
                                                                                'Pressure CTp': 0.00000,
                                                                                'Pressure Ratio': 0.00000},
                                                        	                    'grouped_data':
                                                                                {'Element 2':
                                                                                {'CTf': 0.00000,
                                                                                'CTp': 0.00000,
                                                                                'top Xtr': 0.00000,
                                                                                'bot Xtr': 0.00000},
                                                                                'Axis Body':
                                                                                {'CTf': 0.00000,
                                                                                'CTp': 0.00000,
                                                                                'Xtr': 0.00000}}}

    _DESIGN_VARS = DesignVector.construct_vector(config)

    _base_oper = copy.deepcopy(config.multi_oper[0])


    def __init__(self,
                 verbose: bool = False,
                 **kwargs) -> None:
        """
        Initialization of the OptimizationProblem class.

        Parameters
        ----------
        - verbose : bool, optional
            Bool to determine if error messages should be printed to the console while running.
        - **kwargs : dict[str, Any]
            Additional keyword arguments.

        Returns
        -------
        None
        """

        self.verbose = verbose

        # Import control variables
        self.num_radial = config.NUM_RADIALSECTIONS
        self.num_stages = config.NUM_STAGES
        self.optimize_stages = config.OPTIMIZE_STAGE

        # Calculate the number of objectives and constraints of the optimization problem
        n_objectives = config.n_objectives
        n_inequality_constraints = len(config.constraint_IDs[0]) * len(config.multi_oper) \
               - sum(1 for con in config.constraint_IDs[0] if con == 3) * (len(config.multi_oper) - 1)
        n_equality_constraints = len(config.constraint_IDs[1]) * len(config.multi_oper)

        # Initialize the parent class
        super().__init__(vars=self._DESIGN_VARS,
                         n_obj=n_objectives,
                         n_ieq_constr=n_inequality_constraints,
                         n_eq_constr=n_equality_constraints,
                         **kwargs)

        # Define key paths/directories
        self.parent_dir = Path(__file__).resolve().parent.parent
        self.submodels_path = self.parent_dir / "Submodels"

        # Validate critical submodels_path exist
        if not self.submodels_path.exists():
            raise SystemError(f"Missing submodels path: {self.submodels_path}")

        # Create folder path to store statefiles
        if config.ARCHIVE_STATEFILES:
            self.dump_folder = self.submodels_path / "Evaluated_tdat_state_files"
            # Check existance of dump folder
            try:
                self.dump_folder.mkdir(exist_ok=True)
            except PermissionError as e:
                raise PermissionError(f"Unable to create dump folder: {self.dump_folder}. Check permissions") from e

        # Define analysisname template
        self.timestamp_format = "%m%d%H%M%S"
        self.analysis_name_template = "{}_{:04d}_{}"

        # Initialize design vector interface
        self.design_vector_interface = DesignVectorInterface()

        # Use lazy-loaded modules (initialized at first use)
        if not hasattr(self, "_lazy_modules_loaded"):
            from UDFDAC import UDFDAC  # type: ignore
            from Submodels.output_handling import output_processing  # type: ignore
            from Submodels.file_handling import fileHandlingMTSET, fileHandlingMTFLO  # type: ignore
            self._UDFDAC = UDFDAC
            self._output_processing = output_processing
            self._fileHandlingMTSET = fileHandlingMTSET
            self._fileHandlingMTFLO = fileHandlingMTFLO
            self._lazy_modules_loaded = True

        # Load base operating conditions
        self.multi_oper = copy.deepcopy(config.multi_oper)


    def SetAnalysisName(self) -> None:
        """
        Generate a unique analysis name and write it to self.
        This is required to enable multi-threading of the optimization problem, and log each state file,
        since each evaluation of UDFDAC requires a unique set of files.

        Returns
        -------
        - None
            The analysis_name is written to self.
        """

        # Generate a timestamp string in the format MMDDHHMMSS
        now = datetime.datetime.now()
        timestamp = now.strftime(self.timestamp_format)

        # Generate a unique identifier using UUID
        unique_id = uuid.uuid4().hex[:12]  # 12 chars max

        # Add a process ID to the analysis name to ensure uniqueness in multi-threaded environments.
        process_id = os.getpid() % 10000  # 4 chars max

        # The analysis name is formatted as: <MMDDHHMMSS>_<process_ID>_<unique_id>.
        # Analysis name has a length of 28 characters, satisfying the maximum length of 32 characters accepted by UDFDAC.
        self.analysis_name = self.analysis_name_template.format(timestamp, process_id, unique_id)

        # Additionally set the tflow file path
        self._tflow_file_path = self.submodels_path / f"tflow.{self.analysis_name}"

        # Invalidate the cached omega-line indices for the new file
        if hasattr(self, "_omega_line_ids"):
            del self._omega_line_ids


    def ComputeReynolds(self) -> None:
        """
        A simple function to compute the inlet Reynolds number,
        and write it to the oper dictionary.

        Returns
        -------
        None
        """

        # Compute the inlet Reynolds number and write it to self.oper
        # Uses Vinl [m/s], Lref [m], and kinematic_viscosity [m^2/s]
        self.oper["Inlet_Reynolds"] = float((self.oper["Vinl"] * self.Lref) / self.oper["atmos"].kinematic_viscosity[0])


    def ComputeOmega(self,
                     idx: int) -> None:
        """
        A simple function to compute the non-dimensional UDFDAC rotational rate Omega,
        and write it to the oper dictionary.

        Parameters
        ----------
        - idx : int
            The index of the operating condition in the multi_oper dictionary.
            This is used to extract the correct RPS from the blading dictionary.

        Returns
        -------
        None
        """

        if idx >= len(self.blade_blading_parameters[0]["RPS_lst"]):
            raise IndexError(f"Expected at least {idx+1} RPS values, but got {len(self.blade_blading_parameters[0]['RPS_lst'])}")

        # Compute the non-dimensional rotational rate Omega for UDFDAC and write it to the blading parameters
        # Multiplied by -1 to comply with sign convention in UDFDAC.

        # First compute the constant non-dimensionalisation factor
        factor = -2 * np.pi * self.Lref / self.oper["Vinl"]

        # Loop over all stages
        for blading_params in self.blade_blading_parameters:
            blading_params["RPS"] = blading_params["RPS_lst"][idx]
            blading_params["rotational_rate"] = float(blading_params["RPS"] * factor)


    def SetOmega(self,
                 oper_idx) -> None:
        """
        A simple function to correctly set the rotational rate Omega in the tflow.analysis_name file.
        This is used in a multi-point analysis to update the tflow file for each analysis rather than regenerating the full tflow file.

        Parameters
        ----------
        - oper_idx : int
            The index of the operating condition in the multi_oper dictionary.
            This is used to extract the correct RPS from the blading dictionary.

        Returns
        -------
        None
        """

        # Compute / update the rotational rates in the blading dictionaries
        self.ComputeOmega(idx=oper_idx)

        with open(self._tflow_file_path, "r+") as file:
            lines = file.readlines()

            # Cache the omega line indices if not already computed.
            if not hasattr(self, "_omega_line_ids"):
                # We assume that the line directly after the one starting with "OMEGA" is the one to update.
                self._omega_line_ids = [i for i, line in enumerate(lines) if line.lstrip().startswith("OMEGA")]

            # Use a local alias to avoid repeated attribute lookups in the loop.
            blade_params = self.blade_blading_parameters

            # Update the omega lines with the correct rotational rates.
            for i, line_idx in enumerate(self._omega_line_ids):
                rate = blade_params[i]["rotational_rate"]
                lines[line_idx + 1] = f"{rate}\n"

            # Write the updated lines back to the file.
            file.seek(0)
            file.writelines(lines)
            file.truncate()  # Ensure no trailing bytes remain from previous contents.


    def CleanUpFiles(self) -> None:
        """
        Move the UDFDAC statefile to a separate folder and clean up temporary files.

        This method:
        1. Moves the tdat statefile to a persistent archive folder.
        2. Removes all temporary UDFDAC input/output files, including the original statefile.

        Note that the output files can always be regenerated from the statefile.

        Returns
        -------
        None
        """

        # Construct filepaths once to reduce string operations
        file_paths = {
            file_type: self.submodels_path / template.format(self.analysis_name)
            for file_type, template in self.FILE_TEMPLATES.items()
        }

        for file_type, file_path in file_paths.items():
            # Move the state file to the dump folder
            if file_type == "tdat" and config.ARCHIVE_STATEFILES:
                if file_path.exists():
                    copied_file = self.dump_folder / file_path.name
                    with contextlib.suppress(FileNotFoundError):
                        # Atomic operation to improve edge case handling
                        file_path.replace(copied_file)
            else:
                # Cleanup all temporary files
                if file_path.exists():
                    file_path.unlink(missing_ok=True)


    def GenerateUDFDACInputs(self,
                             x: dict[str, int | float]) -> bool:
        """
        Generates the input files required for the UDFDAC simulation.
        This method creates the necessary input files for the UDFDAC simulation by utilizing the
        `fileHandlingMTSET/MTFLO` classes from the `Submodels.file_handling` module. It generates two input files:
        - walls.analysis_name: The MTSET input file, which contains the axisymmetric geometries.
        - tflow.analysis_name: The MTFLO blading input file, which contains the blading and design parameters.

        By generating the input files, validation of the design vector is performed, since an infeasible design vector
        will raise a ValueError (somewhere) in the input generation method.

        Parameters
        ----------
        - x : dict[str, int | float]
            The pymoo design vector dictionary.

        Returns
        -------
        - output_generated: bool
            - True if the input files were successfully generated, False if a ValueError occurred
              during the process (indicating potential interpolation issues or infeasible axisymmetric bodies).
        """

        # Generate the MTSET input file containing the axisymmetric geometries and the MTFLO blading input file
        try:
            # Deconstruct the design vector
            (self.centerbody_variables,
            self.duct_variables,
            self.blade_design_parameters,
            self.blade_blading_parameters,
            self.Lref) = self.design_vector_interface.DeconstructDesignVector(x_dict=x)

            # Set the initial non-dimensional omega rates
            self.oper = copy.deepcopy(self._base_oper)

            self._fileHandlingMTSET(params_CB=self.centerbody_variables,
                                    params_duct=self.duct_variables,
                                    analysis_name=self.analysis_name,
                                    ref_length=self.Lref).GenerateMTSETInput()  # Generate the MTSET input file

            self.ComputeMTFLOInputs(oper_idx=0)
            output_generated =  True  # If both input generation routines succeeded, set output_generated to True

        except ValueError as e:
            # Any value error that might occur while generating the MTSET input file will be caused by interpolation issues arising from the input values, so
            # this is an efficient and simple method to check if the axisymmetric bodies are feasible.
            output_generated = False  # If any of the input generation routines raised an error, set output_generated to False
            if self.verbose:
                error_code = "INVALID_DESIGN"
                print(f"[{error_code}] Invalid design vector encountered: {e}")
        except Exception as e:
            # If any unexpected errors occur, log them as well
            output_generated = False
            if self.verbose:
                import traceback
                traceback.print_exc()
                error_code = f"UNEXPECTED_{type(e).__name__}"
                print(f"[{error_code}] Traceback:\n{traceback.format_exc()}")  # Use traceback for more specific error information.

        if not output_generated:
            # Set parameters equal to the config values in case of a crash so that the constraint/objective value calculations do not crash
            self.Lref = config.BLADE_DIAMETERS[0]
            self.duct_variables = copy.copy(config.DUCT_VALUES)
            self.centerbody_variables = copy.copy(config.CENTERBODY_VALUES)
            self.blade_blading_parameters = copy.copy(config.STAGE_BLADING_PARAMETERS)
            self.blade_design_parameters = copy.copy(config.STAGE_DESIGN_VARIABLES)

        return output_generated


    def ComputeMTFLOInputs(self,
                           oper_idx: int) -> bool:
        """
        Compute the correct MTFLO input file based on the operating condition, accounting for the
        possible presence of variable pitch.

        Parameters
        ----------
        - oper_idx : int
            Integer of the operating point index.

        Returns
        -------
        - input_generated : bool
            Boolean to indicate if the MTFLO inputs have been generated successfully.
        """

        # Set the correct pitch angle in the blading params dictionary
        # Loop over all stages
        for blading_params in self.blade_blading_parameters:
            if len(blading_params["ref_blade_angle_lst"]) == len(self.multi_oper):
                # Only update variable pitch if it is used, otherwise leave it as constant.
                blading_params["ref_blade_angle"] = blading_params["ref_blade_angle_lst"][oper_idx]

        # Overwrite the UDFDAC input file to the correct inputs
        # First set the correct nondimensional rotational rate
        self.ComputeOmega(oper_idx)

        try:
            self._fileHandlingMTFLO(analysis_name=self.analysis_name,
                                    ref_length=self.Lref).GenerateMTFLOInput(blading_params=self.blade_blading_parameters,
                                                                             design_params=self.blade_design_parameters,
                                                                             plot=False)  # Generate the MTFLO input file
            input_generated = True

        except ValueError as e:
            # Any value error that might occur while generating the MTFLO input file will indicate an invalid design
            input_generated = False
            if self.verbose:
                error_code = "INVALID_DESIGN"
                print(f"[{error_code}] Invalid design vector encountered: {e}")
        except Exception as e:
            # If any unexpected errors occur, log them as well
            input_generated = False
            if self.verbose:
                import traceback
                traceback.print_exc()
                error_code = f"UNEXPECTED_{type(e).__name__}"
                print(f"[{error_code}] Traceback:\n{traceback.format_exc()}")  # Use traceback for more specific error information.

        if not input_generated:
            # Set parameters equal to the config values in case of a crash so that the constraint/objective value calculations do not crash
            self.Lref = config.BLADE_DIAMETERS[0]
            self.duct_variables = copy.copy(config.DUCT_VALUES)
            self.centerbody_variables = copy.copy(config.CENTERBODY_VALUES)
            self.blade_blading_parameters = copy.copy(config.STAGE_BLADING_PARAMETERS)
            self.blade_design_parameters = copy.copy(config.STAGE_DESIGN_VARIABLES)

        return input_generated


    def _evaluate(self,
                  x:dict,
                  out:dict,
                  *args,
                  **kwargs) -> None:
        """
        Element-wise evaluation function for the multi-point optimisation problem.

        Parameters
        ----------
        - x : dict
            The pymoo design vector dictionary.
        - out : dict
            The pymoo elementwise evaluation output dictionary.
        - *args : tuple
            Additional arguments.
        - **kwargs : dict[str, Any]
            Additional keyword arguments.

        Returns
        -------
        - None
            The output dictionary is modified in-place.
        """

        # Generate a unique analysis name
        self.SetAnalysisName()

        # Generate the UDFDAC input files.
        # If design_okay is false, this indicates an error in the input file generation caused by an infeasible design vector.
        design_okay = self.GenerateUDFDACInputs(x)

        # Only perform the UDFDAC analyses if the input generation has succeeded.
        # Initialise the UDFDAC output list of dictionaries. Use the crash outputs in
        # initialisation to pre-populate them in case of a crash or infeasible design vector
        UDFDAC_outputs = [copy.deepcopy(self.CRASH_OUTPUTS) for _ in range(len(self.multi_oper))]

        if design_okay:
            valid_grid = False
            for idx, operating_point in enumerate(self.multi_oper):
                # Compute the necessary inputs
                self.oper = copy.deepcopy(operating_point)  # Copy the appropriate operating condition dictionary
                self.ComputeReynolds()

                if idx != 0:
                    # Only update tflow file for the second-onward point, since the initial point is written when first generating the input files
                    design_okay = self.ComputeMTFLOInputs(oper_idx=idx)

                UDFDAC_interface = self._UDFDAC(operating_conditions=self.oper,
                                                ref_length=self.Lref,
                                                analysis_name=self.analysis_name,
                                                grid_checked=valid_grid,
                                                run_viscous=True,
                                                **kwargs)
                if design_okay:
                    # Duplicate check required for a multi-point analysis
                    try:
                        # Run UDFDAC
                        exit_flag = UDFDAC_interface.caller(external_inputs=True,
                                                            output_type=OutputType.FORCES_ONLY)

                        # Extract outputs
                        if exit_flag != ExitFlag.CRASH:
                            output_handler = self._output_processing(analysis_name=self.analysis_name)
                            UDFDAC_outputs[idx] = output_handler.GetAllVariables(output_type=0)
                        else:
                            # If the solver outputs are invalid, set the crash outputs and break out of the loop to avoid useless calculation of other operating points.
                            UDFDAC_outputs[idx] = self.CRASH_OUTPUTS
                            break

                    except Exception as e:
                        exit_flag = ExitFlag.CRASH
                        UDFDAC_outputs[idx] = self.CRASH_OUTPUTS
                        if self.verbose:
                            print(f"[UDFDAC_ERROR] OP={idx}, case={self.analysis_name}: {e}")

                    # Set valid_grid to true to skip the grid checking routines for the next operating point if the solver exited with a converged/non-converged solution.
                    if exit_flag in (ExitFlag.SUCCESS, ExitFlag.NON_CONVERGENCE, ExitFlag.CHOKING):
                        valid_grid = True
                else:
                    # If any of the operating points are infeasible, set crash outputs.
                    UDFDAC_outputs[idx] = self.CRASH_OUTPUTS

        # Obtain objective(s)
        # The out dictionary is updated in-place
        Objectives(duct_variables=self.duct_variables,
                   oper=self.multi_oper,
                   Lref=self.Lref).ComputeMultiPointObjectives(analysis_outputs=UDFDAC_outputs,
                                                               objective_IDs=config.objective_IDs,
                                                               out=out)

        # Compute constraints
        # The out dictionary is updated in-place
        Constraints(self.centerbody_variables,
                    self.duct_variables,
                    self.blade_blading_parameters,
                    design_okay).ComputeMultiPointConstraints(analysis_outputs=UDFDAC_outputs,
                                                              Lref=self.Lref,
                                                              oper=self.multi_oper,
                                                              out=out)

        # Cleanup the generated files
        with contextlib.suppress(Exception):
            self.CleanUpFiles()


if __name__ == "__main__":
    """
    Test Block
    """

    test = MultiPointOptimizationProblem()

    # Create a reference vector for testing
    from init_population import InitPopulation  # type: ignore
    from repair import RepairIndividuals  # type: ignore

    ref_pop = InitPopulation(population_type="biased").GeneratePopulation()
    ref_vectors = ref_pop.get("X")
    ref_vectors = RepairIndividuals()._do(test, ref_vectors)
    ref_vector = ref_vectors[0]  # Use the first vector for testing

    # Create an output dictionary to store the results
    output = {}
    test._evaluate(ref_vector, output)
    print(output)
