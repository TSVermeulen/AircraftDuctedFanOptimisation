"""
problem_definition
==================

Description
-----------
This module defines an optimisation problem for the pymoo framework, based on
the ElementwiseProblem parent class.

Classes
-------
OptimisationProblem(ElementwiseProblem)
    Class defining the optimisation problem with mixed-variable support.

Examples
--------
>>> problem = OptimisationProblem()
>>> out = {}
>>> problem._evaluate(1, out)

Notes
-----
This module integrates with the UDC for aerodynamic analysis. Ensure that the
executable and required input files are present in the appropriate directories.
The module is designed to handle mixed-variable optimisation problems, including
real and integer variables.

References
----------
For more details on the MTFLOW solver integrated in the UDC and its input/output
requirements, refer to the MTFLOW user manual:
https://web.mit.edu/drela/Public/web/mtflow/mtflow.pdf

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 2.0

Changelog:
- V1.0:   Initial implementation.
- V1.1:   Improved documentation. Fixed issues with deconstruction of design
          vector. Fixed analysisname generator and switched to using datetime &
          evaluation counter for name generation.
- V1.1.5: Changed analysis name generation to only use datetime to simplify
          naming generation.
- V1.1.6: Updated to remove iter_count from MTFLOW_caller outputs.
- V1.2:   Extracted design vector handling to separate file/class.
- V1.3:   Removed troublesome cache implementation. Cleaned up _evaluate method.
          Created default crash output dictionary to avoid repeated reading of
          crash_outputs forces file. Adjusted GenerateAnalysisName method to
          use 8-char uuid. Updated ComputeOmega method to write omega to the
          blading lists rather than to the oper dictionary.
- V1.4:   Improved robustness of crash handling in MTFLOW. Added conditional
          dump folder generation to avoid unnecessary folder creation.
- V2.0:   Renamed MTFLOW_caller to UDC for consistency with written thesis.
          Updated imports to reflect new structure.
- V2.1:   Updated documentation and formatting. Improved type hints.
"""

# Import standard libraries
import os
import uuid
import copy
import datetime
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

class OptimisationProblem(ElementwiseProblem):
    """
    Class definition of the optimisation problem to be solved using the genetic
    algorithm. Inherits from the ElementwiseProblem class from
    pymoo.core.problem.
    """

    # Define the file names relevant for UDC
    FILE_TEMPLATES = {"walls": "walls.{}",
                      "tflow": "tflow.{}",
                      "forces": "forces.{}",
                      "flowfield": "flowfield.{}",
                      "boundary_layer": "boundary_layer.{}",
                      "tdat": "tdat.{}"}

    # Initialise output dictionary to use in case of an infeasible design.
    # This equals the outputs of the UDC class when no results are produced,
    # but is quicker as it does not involve reading a file.
    CRASH_OUTPUTS = {'data':
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
        Initialisation of the OptimisationProblem class.

        Parameters
        ----------
        - verbose : bool, optional
            Bool to determine if error messages should be printed to the
            console while running.
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

        # Calculate the number of objectives and constraints of the
        # optimisation problem
        n_objectives = config.n_objectives
        n_inequality_constraints = len(config.constraint_IDs[0]) * \
            len(config.multi_oper)
        n_equality_constraints = len(config.constraint_IDs[1]) * \
            len(config.multi_oper)

        # Initialise the parent class
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
            self.dump_folder = self.submodels_path / \
                "Evaluated_tdat_state_files"
            # Check existance of dump folder
            try:
                self.dump_folder.mkdir(exist_ok=True)
            except PermissionError as e:
                raise PermissionError(f"Error creating dump folder \
                                      {self.dump_folder}") from e

        # Define analysisname template
        self.timestamp_format = "%m%d%H%M%S"
        self.analysis_name_template = "{}_{:04d}_{}"

        # Initialise design vector interface
        self.dvec_interface = DesignVectorInterface()

        # Use lazy-loaded modules (initialised at first use)
        # Prevents circular imports and speeds up initial loading time.
        if not hasattr(self, "_lazy_modules_loaded"):
            from UDC import UDC  # type: ignore
            from Submodels.output_handling import output_processing  # type: ignore
            from Submodels.file_handling import fileHandlingMTSET  # type: ignore
            from Submodels.file_handling import fileHandlingMTFLO  # type: ignore
            self._UDC = UDC
            self._output_processing = output_processing
            self._fileHandlingMTSET = fileHandlingMTSET
            self._fileHandlingMTFLO = fileHandlingMTFLO
            self._lazy_modules_loaded = True


    def SetAnalysisName(self) -> None:
        """
        Generate a unique analysis name and write it to self.
        This is required to enable multi-threading of the optimisation problem,
        since each evaluation of UDC requires a unique set of files.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # Generate a timestamp string in the format MMDDHHMMSS
        now = datetime.datetime.now()
        timestamp = now.strftime(self.timestamp_format)

        # Generate a unique identifier using UUID
        unique_id = uuid.uuid4().hex[:12]  # 12 chars max

        # Add a process ID to the analysis name to ensure uniqueness in
        # multi-threaded environments.
        process_id = os.getpid() % 10000  # 4 chars max

        # The analysis name is formatted as:
        # <MMDDHHMMSS>_<process_ID>_<unique_id>.
        # Analysis name has a length of 28 characters, satisfying the maximum
        # length of 32 characters accepted by UDC.
        self.analysis_name = self.analysis_name_template.format(timestamp,
                                                                process_id,
                                                                unique_id)


    def ComputeReynolds(self) -> None:
        """
        A simple function to compute the inlet Reynolds number,
        and write it to the oper dictionary.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # Compute the inlet Reynolds number and write it to self.oper
        # Uses Vinl [m/s], Lref [m], and kinematic_viscosity [m^2/s]
        inlet_re = (self.oper["Vinl"] * self.Lref) / \
            self.oper["atmos"].kinematic_viscosity[0]
        self.oper["Inlet_Reynolds"] = float(inlet_re)


    def ComputeOmega(self) -> None:
        """
        A simple function to compute the non-dimensional UDC rotational rate,
        and write it to the blading parameters.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # Pre-calculate the common factor to avoid repeated computation
        omega_factor = (-2 * np.pi * self.Lref) / self.oper["Vinl"]

        # Process each stage in a single loop
        for blading_params in self.blade_blading_parameters:
            # For a single point analysis, we need to extract/flatten the
            # RPS_list into RPS, which is equivalent to taking the first entry
            # from the list.
            rps = blading_params["RPS_lst"][0]
            blading_params["RPS"] = rps
            blading_params["rotational_rate"] = float(rps * omega_factor)


    def CleanUpFiles(self) -> None:
        """
        Archive the UDC statefile to a separate folder and clean up files.

        This method:
        1. Moves the tdat statefile to a persistent archive folder, if desired.
        2. Removes all temporary UDC input/output files.

        The output files can always be regenerated from the statefile.

        Parameters
        ----------
        None

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
                        # Atomic operation (on same file system only) to improve
                        # edge case handling. Prevents corruption during
                        # concurrent access.
                        file_path.replace(copied_file)
            else:
                # Cleanup all temporary files
                if file_path.exists():
                    file_path.unlink(missing_ok=True)


    def GenerateUDCInputs(self,
                          x: dict[str, float | int]) -> bool:
        """
        Generates the input files required for the UDC simulation.
        It generates two input files:
        - walls.analysis_name: The MTSET input file.
        - tflow.analysis_name: The MTFLO blading input file.

        Includes validation of the design vector is performed, since an
        infeasible design vector will raise a ValueError (somewhere) in the
        input generation method.

        Parameters
        ----------
        - x : dict[str, any]
            The pymoo design vector dictionary.

        Returns
        -------
        - output_generated: bool
            - True if the design vector is feasible, false otherwise.
        """

        # Generate the MTSET input file containing the axisymmetric geometries
        # and the MTFLO blading input file
        try:
            # Deconstruct the design vector
            (self.centerbody_variables,
            self.duct_variables,
            self.blade_design_parameters,
            self.blade_blading_parameters,
            self.Lref) = self.dvec_interface.DeconstructDesignVector(x_dict=x)

            # Set the non-dimensional omega rates
            self.ComputeOmega()

            # Generate the MTSET input file
            self._fileHandlingMTSET(params_CB=self.centerbody_variables,
                                    params_duct=self.duct_variables,
                                    analysis_name=self.analysis_name,
                                    ref_length=self.Lref).GenerateMTSETInput()

            # Generate the MTFLO input file
            self._fileHandlingMTFLO(analysis_name=self.analysis_name,
                                    ref_length=self.Lref).GenerateMTFLOInput(blading_params=self.blade_blading_parameters,
                                                                             design_params=self.blade_design_parameters,
                                                                             plot=False)

            # If both input generation routines succeeded, set output_generated
            output_generated = True

        except ValueError as e:
            # Any value error will be caused by interpolation issues, so
            # this is an efficient and simple method to check feasibility.
            output_generated = False
            if self.verbose:
                error_code = "INVALID_DESIGN"
                print(f"[{error_code}] Invalid design vector encountered: {e}")
        except Exception as e:
            # If any unexpected errors occur, log them as well
            output_generated = False
            if self.verbose:
                # Use traceback for more specific error information.
                import traceback
                error_code = f"UNEXPECTED_{type(e).__name__}"
                print(f"[{error_code}] Traceback:\n{traceback.format_exc()}")

        if not output_generated:
            # Set parameters equal to the config values in case of a crash
            # for downstream handling.
            self.Lref = config.BLADE_DIAMETERS[0]
            self.duct_variables = copy.copy(config.DUCT_VALUES)
            self.centerbody_variables = copy.copy(config.CENTERBODY_VALUES)
            self.blade_blading_parameters = copy.copy(config.STAGE_BLADING_PARAMETERS)
            self.blade_design_parameters = copy.copy(config.STAGE_DESIGN_VARIABLES)

        return output_generated


    def _evaluate(self,
                  x: dict[str, float | int],
                  out: dict[str, np.typing.NDArray[np.floating]],
                  *args,
                  **kwargs) -> None:
        """
        Element-wise evaluation function for a single-point optimisation
        problem.

        Parameters
        ----------
        - x : dict[str, float | int]
            The pymoo design vector dictionary.
        - out : dict[str, np.typing.NDArray[np.floating]]
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

        # Copy the operational conditions
        self.oper = copy.deepcopy(self._base_oper)

        # Generate the UDC input files.
        # If design_okay is false, this indicates an error in the input file
        # generation caused by an infeasible design vector.
        design_okay = self.GenerateUDCInputs(x)

        # Evaluate the design using the UDC if the design is feasible
        if design_okay:
            self.ComputeReynolds()  # Compute the Reynolds number

            UDC_interface = self._UDC(operating_conditions=self.oper,
                                      ref_length=self.Lref,
                                      analysis_name=self.analysis_name,
                                      run_viscous=True,
                                      **kwargs)

            try:
                # Run UDC
                exit_flag, UDC_outputs = UDC_interface.caller(external_inputs=True,
                                                              output_type=OutputType.FORCES_ONLY)

                # Check outputs in case of crashes
                if exit_flag == ExitFlag.CRASH:# or \
                # UDC_outputs.keys() != self.CRASH_OUTPUTS.keys():
                #     if UDC_outputs.keys() != self.CRASH_OUTPUTS.keys():
                #         error_code = "MISSING_OUTPUTS"
                #         print(f"[{error_code}] case={self.analysis_name}: \
                #               Incomplete outputs received from UDC.")
                #         print("Design vector:", x)
                    UDC_outputs = copy.copy(self.CRASH_OUTPUTS)

            except Exception as e:
                if self.verbose:
                    print(f"[UDC_ERROR] case={self.analysis_name}: {e}")
                UDC_outputs = copy.copy(self.CRASH_OUTPUTS)
        else:
            # If the design is infeasible, we load the crash outputs
            # This is a predefined dictionary with all outputs set to 0.
            UDC_outputs = copy.copy(self.CRASH_OUTPUTS)

        # Obtain objective(s)
        # The out dictionary is updated in-place
        Objectives(duct_variables=self.duct_variables,
                   oper=self.oper,
                   Lref=self.Lref).ComputeObjective(analysis_outputs=UDC_outputs,
                                                    objective_IDs=config.objective_IDs,
                                                    out=out)

        # Compute constraints
        # The out dictionary is updated in-place
        Constraints(self.centerbody_variables,
                    self.duct_variables,
                    self.blade_blading_parameters,
                    design_okay).ComputeConstraints(analysis_outputs=UDC_outputs,
                                                    Lref=self.Lref,
                                                    oper=self.oper,
                                                    out=out)

        # Cleanup the generated files
        with contextlib.suppress(Exception):
            self.CleanUpFiles()


if __name__ == "__main__":
    """
    Test Block
    """

    test = OptimisationProblem()

    # Create a reference vector for testing
    from init_population import InitPopulation  # type: ignore
    from repair import RepairIndividuals  # type: ignore

    ref_pop = InitPopulation(population_type="biased").GeneratePopulation()
    ref_vectors = ref_pop.get("X")
    ref_vectors = RepairIndividuals()._do(test, ref_vectors)
    ref_vector = ref_vectors[0]  # Use the first vector for testing

    # Create an output dictionary to store the results
    output = {}

    ref_vector = {'x0': np.float64(0.05), 'x1': np.float64(0.1490905032828976), 'x2': np.float64(0.062219857771399295), 'x3': np.float64(0.9), 'x4': np.float64(0.7046477645216163), 'x5': np.float64(0.18024639981891005), 'x6': np.float64(0.055343872591593624), 'x7': np.float64(0.2047246900784671), 'x8': np.float64(0.011388700499901186), 'x9': np.float64(0.0012034884647541763), 'x10': np.float64(0.00033321467756755374), 'x11': np.float64(-0.08420865994789903), 'x12': np.float64(0.08715196721616161), 'x13': np.float64(0.05), 'x14': np.float64(0.2), 'x15': np.float64(1.0375229659744198), 'x16': np.float64(0.11042233661899682), 'x17': np.float64(0.05762388931051207), 'x18': np.float64(0.2784116432805875), 'x19': np.float64(0.05169553677797759), 'x20': np.float64(0.9), 'x21': np.float64(0.9), 'x22': np.float64(0.24805640272621107), 'x23': np.float64(0.08956183914268077), 'x24': np.float64(0.5), 'x25': np.float64(0.02085636332432073), 'x26': np.float64(0.001023346722805211), 'x27': np.float64(0.0009899778194866766), 'x28': np.float64(-0.07135797605731306), 'x29': np.float64(0.25834267502378727), 'x30': np.float64(0.07560184649348815), 'x31': np.float64(0.12783470691622653), 'x32': np.float64(0.050426062577640006), 'x33': np.float64(0.15112736008158933), 'x34': np.float64(0.05000311953878328), 'x35': np.float64(0.3963365291605321), 'x36': np.float64(0.6325098176114659), 'x37': np.float64(0.2634821609775609), 'x38': np.float64(0.10595146244216813), 'x39': np.float64(0.4093045767194373), 'x40': np.float64(0.016425026135762945), 'x41': np.float64(0.0003101921903649612), 'x42': np.float64(0.0009609128983365676), 'x43': np.float64(-0.052149118509139836), 'x44': np.float64(0.2509506275676825), 'x45': np.float64(0.05), 'x46': np.float64(0.1998184571597516), 'x47': np.float64(0.057618870259595584), 'x48': np.float64(0.13976944816492412), 'x49': np.float64(0.05232064486125312), 'x50': np.float64(0.9), 'x51': np.float64(0.8993644137504563), 'x52': np.float64(0.22315430955119261), 'x53': np.float64(0.050889250908926376), 'x54': np.float64(0.2584605433567202), 'x55': np.float64(0.023017553924105264), 'x56': np.float64(0.0008756138929334285), 'x57': np.float64(0.0006507935166340175), 'x58': np.float64(-0.03621641050217742), 'x59': np.float64(0.21042496825608198), 'x60': np.float64(0.08940587965047296), 'x61': np.float64(0.11155815907022998), 'x62': np.float64(0.06775634039933327), 'x63': np.float64(0.125), 'x64': np.float64(0.06284276384074872), 'x65': np.float64(0.8950283826589028), 'x66': np.float64(0.6061431026179809), 'x67': np.float64(0.3377771173046672), 'x68': np.float64(0.02000012430363885), 'x69': np.float64(0.36681928632681277), 'x70': np.float64(0.010379726400129), 'x71': np.float64(0.0), 'x72': np.float64(0.0003729077270562264), 'x73': np.float64(-0.051940648373555144), 'x74': np.float64(0.04746156250860716), 'x75': np.float64(0.05), 'x76': np.float64(0.074221165707054), 'x77': np.float64(0.14776418438341585), 'x78': np.float64(0.15242898321524867), 'x79': 9, 'x80': np.float64(45.0), 'x81': np.float64(1.9496928723106846), 'x82': np.float64(0.20341527657076233), 'x83': np.float64(0.28537138165826187), 'x84': np.float64(0.2848770275290754), 'x85': np.float64(0.2629067771252505), 'x86': np.float64(0.04936485394061665), 'x87': np.float64(0.10641260540883675), 'x88': np.float64(0.10277339267995997), 'x89': np.float64(0.6771323426340188), 'x90': np.float64(0.7733880786537577), 'x91': np.float64(0.3378696740722167)}
    test._evaluate(ref_vector, output)
    print(output)
