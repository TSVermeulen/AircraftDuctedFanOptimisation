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
Email: T.S.Vermeulen@tudelft.nl
Version: 2.1

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
                (exit_flag, 
                UDC_outputs) = UDC_interface.caller(external_inputs=True,
                                                    output_type=OutputType.FORCES_ONLY)

                # Overwrite outputs in case of crashes
                if exit_flag in (ExitFlag.CRASH, ExitFlag.CHOKING, 
                                 ExitFlag.NOT_PERFORMED):
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

    ref_vector = {'x0': np.float64(0.09253003020815274), 'x1': np.float64(0.16442747283541176), 'x2': np.float64(0.05053214912576813), 'x3': np.float64(0.9), 'x4': np.float64(0.8976189796817896), 'x5': np.float64(0.3304184193966121), 'x6': np.float64(0.05099160712413275), 'x7': np.float64(0.34843725142816895), 'x8': np.float64(0.010585692260310498), 'x9': np.float64(0.0003042612520420299), 'x10': np.float64(0.00072772705427627), 'x11': np.float64(-0.07101650644293878), 'x12': np.float64(0.07285898691890069), 'x13': np.float64(0.050302312367775784), 'x14': np.float64(0.07422587866472143), 'x15': np.float64(1.1726695581849418), 'x16': np.float64(0.07626946646877907), 'x17': np.float64(0.0575280129245654), 'x18': np.float64(0.21365100945565854), 'x19': np.float64(0.07085509891773348), 'x20': np.float64(0.8915178131251633), 'x21': np.float64(0.8665583641597584), 'x22': np.float64(0.34599098197728884), 'x23': np.float64(0.1187688886584023), 'x24': np.float64(0.41599696010042847), 'x25': np.float64(0.013827956628262892), 'x26': np.float64(2.567137137226924e-05), 'x27': np.float64(0.001), 'x28': np.float64(-0.07140035689299523), 'x29': np.float64(0.2097905889231831), 'x30': np.float64(0.051276457458488446), 'x31': np.float64(0.07274443798915403), 'x32': np.float64(0.05), 'x33': np.float64(0.1874081004206576), 'x34': np.float64(0.05000000000000001), 'x35': np.float64(0.8987104710280811), 'x36': np.float64(0.5145540121883229), 'x37': np.float64(0.2620939827499221), 'x38': np.float64(0.13026937848635214), 'x39': np.float64(0.22514613673185646), 'x40': np.float64(0.021265381909524723), 'x41': np.float64(0.00014697935409688651), 'x42': np.float64(0.0005622811322714505), 'x43': np.float64(-0.05772303143229478), 'x44': np.float64(0.3), 'x45': np.float64(0.05032184827280754), 'x46': np.float64(0.2), 'x47': np.float64(0.1), 'x48': np.float64(0.13804807362361537), 'x49': np.float64(0.07336319286912049), 'x50': np.float64(0.8933825027696402), 'x51': np.float64(0.688775898755584), 'x52': np.float64(0.22141793931336626), 'x53': np.float64(0.05110625073317769), 'x54': np.float64(0.29291480807718756), 'x55': np.float64(0.00890398161616589), 'x56': np.float64(0.0011942851482272456), 'x57': np.float64(0.000991359033331071), 'x58': np.float64(-0.017804212945688097), 'x59': np.float64(0.20974475609364207), 'x60': np.float64(0.05), 'x61': np.float64(0.1999941638020577), 'x62': np.float64(0.06296834306446389), 'x63': np.float64(0.23529622172163384), 'x64': np.float64(0.05061347447543771), 'x65': np.float64(0.8346308053531276), 'x66': np.float64(0.8233210851686427), 'x67': np.float64(0.17458042959627734), 'x68': np.float64(0.020006298434876592), 'x69': np.float64(0.3597005762590377), 'x70': np.float64(0.021189370960170553), 'x71': np.float64(0.000895032368927069), 'x72': np.float64(0.0002532019324698469), 'x73': np.float64(-0.04553246719288476), 'x74': np.float64(0.026490080060657322), 'x75': np.float64(0.0648212575787353), 'x76': np.float64(0.10309122953383416), 'x77': np.float64(0.19055233614970088), 'x78': np.float64(0.30254651331428967), 'x79': 4, 'x80': np.float64(52.998462923632324), 'x81': np.float64(1.56807981535195), 'x82': np.float64(0.3621543739928979), 'x83': np.float64(0.23286942648107653), 'x84': np.float64(0.22201606666974683), 'x85': np.float64(0.18312333207171616), 'x86': np.float64(0.06709870258198), 'x87': np.float64(0.14921076253196897), 'x88': np.float64(0.13094063087948288), 'x89': np.float64(0.6741120644605625), 'x90': np.float64(0.5052306524694091), 'x91': np.float64(0.4118643559078663)}
    ref_vector = {'x0': np.float64(0.05), 'x1': np.float64(0.17793882635159), 'x2': np.float64(0.050045297148956565), 'x3': np.float64(0.47765128573425064), 'x4': np.float64(0.8999965785956832), 'x5': np.float64(0.17369043273486623), 'x6': np.float64(0.11241308441790167), 'x7': np.float64(0.2143549426710041), 'x8': np.float64(0.0), 'x9': np.float64(0.00020286132861342888), 'x10': np.float64(0.0007526850910986031), 'x11': np.float64(-0.0895818496714554), 'x12': np.float64(0.0719392487663116), 'x13': np.float64(0.05427664599857429), 'x14': np.float64(0.06752655793537116), 'x15': np.float64(1.20584084954759), 'x16': np.float64(0.10744197305751342), 'x17': np.float64(0.0632508776082393), 'x18': np.float64(0.13590919732654194), 'x19': np.float64(0.05), 'x20': np.float64(0.8842678845393976), 'x21': np.float64(0.8999509724296675), 'x22': np.float64(0.2429883979923818), 'x23': np.float64(0.19827369561779798), 'x24': np.float64(0.36070011467082475), 'x25': np.float64(0.01944625166502599), 'x26': np.float64(0.0012144524488556995), 'x27': np.float64(0.0006144612593388524), 'x28': np.float64(-0.09849512910550139), 'x29': np.float64(0.26658531861804974), 'x30': np.float64(0.05086962879995634), 'x31': np.float64(0.10055726934820007), 'x32': np.float64(0.07111653333609444), 'x33': np.float64(0.22682767987002944), 'x34': np.float64(0.6869916156895552), 'x35': np.float64(0.8996711570721916), 'x36': np.float64(0.8533198895483358), 'x37': np.float64(0.3937600071186033), 'x38': np.float64(0.1883925439592774), 'x39': np.float64(0.34148380665954053), 'x40': np.float64(0.014088077815822709), 'x41': np.float64(0.0012053502477532506), 'x42': np.float64(0.0009997092129680923), 'x43': np.float64(-0.08710389749665505), 'x44': np.float64(0.24273913371037775), 'x45': np.float64(0.05176555411974194), 'x46': np.float64(0.10605161172044174), 'x47': np.float64(0.05), 'x48': np.float64(0.1560666868521758), 'x49': np.float64(0.6867451865385574), 'x50': np.float64(0.9), 'x51': np.float64(0.8986981758315997), 'x52': np.float64(0.2831896658634125), 'x53': np.float64(0.054458120225798656), 'x54': np.float64(0.32822948223737364), 'x55': np.float64(0.023842839685701653), 'x56': np.float64(2.2143361314399997e-06), 'x57': np.float64(0.0006736547235921181), 'x58': np.float64(-0.03854653833925242), 'x59': np.float64(0.07955548341096426), 'x60': np.float64(0.09191083509081967), 'x61': np.float64(0.2), 'x62': np.float64(0.050000248510972255), 'x63': np.float64(0.12808777993512818), 'x64': np.float64(0.07427885132531648), 'x65': np.float64(0.8897577448161127), 'x66': np.float64(0.889852330349503), 'x67': np.float64(0.16426903137516832), 'x68': np.float64(0.020036266604959486), 'x69': np.float64(0.4607704171682947), 'x70': np.float64(0.024914268719784367), 'x71': np.float64(0.000229107861669048), 'x72': np.float64(0.00018509934867890992), 'x73': np.float64(-0.03496434620083292), 'x74': np.float64(0.035719007781643164), 'x75': np.float64(0.09984075291982002), 'x76': np.float64(0.13723804814598142), 'x77': np.float64(0.1200520021145141), 'x78': np.float64(0.38243014359200606), 'x79': 4, 'x80': np.float64(51.85288434219842), 'x81': np.float64(1.6110231373077255), 'x82': np.float64(0.45716013442244463), 'x83': np.float64(0.2203067132725912), 'x84': np.float64(0.21703250767479648), 'x85': np.float64(0.12584342674651183), 'x86': np.float64(0.10981087603474263), 'x87': np.float64(0.09240518785381185), 'x88': np.float64(0.0953536300476101), 'x89': np.float64(0.3914036057672563), 'x90': np.float64(0.5941711838118199), 'x91': np.float64(0.3009740107969327)}
    ref_vector = {'x0': np.float64(0.05435062163162975), 'x1': np.float64(0.16851959093299793), 'x2': np.float64(0.05861116778886588), 'x3': np.float64(0.8994538984715352), 'x4': np.float64(0.8985885496034643), 'x5': np.float64(0.30667184067400854), 'x6': np.float64(0.06233349746902557), 'x7': np.float64(0.21144832357382398), 'x8': np.float64(0.011178944276310533), 'x9': np.float64(0.0003370115340476397), 'x10': np.float64(0.000573408695445748), 'x11': np.float64(-0.05567307635554175), 'x12': np.float64(0.057872521890662554), 'x13': np.float64(0.050274438927041136), 'x14': np.float64(0.07373581397595382), 'x15': np.float64(1.1690861309515648), 'x16': np.float64(0.09001883483525314), 'x17': np.float64(0.0591411733082073), 'x18': np.float64(0.25752315947168486), 'x19': np.float64(0.07812981619773196), 'x20': np.float64(0.8408505653551138), 'x21': np.float64(0.8743306175402952), 'x22': np.float64(0.33566682740092957), 'x23': np.float64(0.22454522476744587), 'x24': np.float64(0.4521964804932839), 'x25': np.float64(0.01804183251669721), 'x26': np.float64(2.61753293463027e-06), 'x27': np.float64(0.0009996123780100065), 'x28': np.float64(-0.07679036978578277), 'x29': np.float64(0.3), 'x30': np.float64(0.05), 'x31': np.float64(0.12172122426421865), 'x32': np.float64(0.05007970090481911), 'x33': np.float64(0.12949934606421595), 'x34': np.float64(0.05029154980969032), 'x35': np.float64(0.8981774299882848), 'x36': np.float64(0.5541121647091336), 'x37': np.float64(0.2568098993669108), 'x38': np.float64(0.13217796308075402), 'x39': np.float64(0.23689920930318928), 'x40': np.float64(0.012690207740524442), 'x41': np.float64(2.758275346555022e-06), 'x42': np.float64(0.0006301938098818066), 'x43': np.float64(-0.05364224436497958), 'x44': np.float64(0.2985900790217241), 'x45': np.float64(0.05), 'x46': np.float64(0.08146895497516152), 'x47': np.float64(0.09778378786877188), 'x48': np.float64(0.13418902783384168), 'x49': np.float64(0.053307637099138716), 'x50': np.float64(0.8974551842702088), 'x51': np.float64(0.6982043430497207), 'x52': np.float64(0.22777451082791572), 'x53': np.float64(0.05122191874365103), 'x54': np.float64(0.41902013780688835), 'x55': np.float64(0.027746593078530564), 'x56': np.float64(0.0010226258521336053), 'x57': np.float64(0.0009996160131230288), 'x58': np.float64(-0.015391045359852784), 'x59': np.float64(0.22432326833623936), 'x60': np.float64(0.08645512721368127), 'x61': np.float64(0.09733057941400085), 'x62': np.float64(0.05046331354924471), 'x63': np.float64(0.24101062474571694), 'x64': np.float64(0.05063896818516331), 'x65': np.float64(0.8844030033883998), 'x66': np.float64(0.6159210411092606), 'x67': np.float64(0.18197440227294592), 'x68': np.float64(0.022766038415430123), 'x69': np.float64(0.3389437187410823), 'x70': np.float64(0.011255026399605374), 'x71': np.float64(0.0009076503350990311), 'x72': np.float64(0.00046564745834986785), 'x73': np.float64(-0.03570253682622168), 'x74': np.float64(0.04105231699523454), 'x75': np.float64(0.05), 'x76': np.float64(0.13480804837073648), 'x77': np.float64(0.16684376058786266), 'x78': np.float64(0.30828943706838424), 'x79': 9, 'x80': np.float64(44.77076518855772), 'x81': np.float64(1.7850934277957224), 'x82': np.float64(0.3521384675293615), 'x83': np.float64(0.3502486928219191), 'x84': np.float64(0.2130949502477144), 'x85': np.float64(0.17825079752905132), 'x86': np.float64(0.0879726787609345), 'x87': np.float64(0.14927304177752712), 'x88': np.float64(0.13220213972416525), 'x89': np.float64(0.6782381076153471), 'x90': np.float64(0.503527911431531), 'x91': np.float64(0.34568462772122255)}
    ref_vector = {'x0': np.float64(0.05), 'x1': np.float64(0.2), 'x2': np.float64(0.05), 'x3': np.float64(0.875), 'x4': np.float64(0.7995682713173954), 'x5': np.float64(0.17606611204288253), 'x6': np.float64(0.0873299008457474), 'x7': np.float64(0.40278895002539195), 'x8': np.float64(0.0), 'x9': np.float64(0.0), 'x10': np.float64(0.0005642751038352118), 'x11': np.float64(-0.08582826353341663), 'x12': np.float64(0.13529388011303586), 'x13': np.float64(0.05), 'x14': np.float64(0.05), 'x15': np.float64(1.0900593077996357), 'x16': np.float64(0.10861901761841408), 'x17': np.float64(0.052224690268491844), 'x18': np.float64(0.21000000000000002), 'x19': np.float64(0.05000000000000001), 'x20': np.float64(0.8400000000000001), 'x21': np.float64(0.8932764709569443), 'x22': np.float64(0.24396395835057735), 'x23': np.float64(0.15825844522056567), 'x24': np.float64(0.421596), 'x25': np.float64(0.019035128589790194), 'x26': np.float64(0.0), 'x27': np.float64(0.0005780056218529611), 'x28': np.float64(-0.09609921338795041), 'x29': np.float64(0.2994791487109442), 'x30': np.float64(0.05026975482093167), 'x31': np.float64(0.09925850950309786), 'x32': np.float64(0.05), 'x33': np.float64(0.22620727802614407), 'x34': np.float64(0.05061459115602705), 'x35': np.float64(0.8863791786320953), 'x36': np.float64(0.8398276130165683), 'x37': np.float64(0.3014575314277771), 'x38': np.float64(0.13132936865580852), 'x39': np.float64(0.4215959999999999), 'x40': np.float64(0.018999999999999996), 'x41': np.float64(0.001210663456833642), 'x42': np.float64(0.0009999913741070487), 'x43': np.float64(-0.07064883116440897), 'x44': np.float64(0.2996459407098272), 'x45': np.float64(0.0685196188061672), 'x46': np.float64(0.08255264181729707), 'x47': np.float64(0.05), 'x48': np.float64(0.15630032271131422), 'x49': np.float64(0.05), 'x50': np.float64(0.7592263903275402), 'x51': np.float64(0.8400000000000001), 'x52': np.float64(0.3111098034986864), 'x53': np.float64(0.0630365269426448), 'x54': np.float64(0.421596), 'x55': np.float64(0.019), 'x56': np.float64(0.0), 'x57': np.float64(0.0009941489874456253), 'x58': np.float64(-0.02982116513703422), 'x59': np.float64(0.14769340640276632), 'x60': np.float64(0.06569154398936515), 'x61': np.float64(0.08255264181729825), 'x62': np.float64(0.05), 'x63': np.float64(0.1900000000000001), 'x64': np.float64(0.07440512177529293), 'x65': np.float64(0.894643056488365), 'x66': np.float64(0.8399999999999994), 'x67': np.float64(0.3110253574212177), 'x68': np.float64(0.02), 'x69': np.float64(0.42159599999999936), 'x70': np.float64(0.00859870178003198), 'x71': np.float64(9.876360659438978e-06), 'x72': np.float64(0.0003287741427297913), 'x73': np.float64(-0.03487572518102231), 'x74': np.float64(0.03973611885769085), 'x75': np.float64(0.05), 'x76': np.float64(0.10971968360129507), 'x77': np.float64(0.1495672948767407), 'x78': np.float64(0.22782445606166768), 'x79': 4, 'x80': np.float64(43.879540715970975), 'x81': np.float64(2.1336), 'x82': np.float64(0.2006300444941277), 'x83': np.float64(0.3152), 'x84': np.float64(0.2367), 'x85': np.float64(0.2205), 'x86': np.float64(0.08479946612535433), 'x87': np.float64(0.10398731099930175), 'x88': np.float64(0.09510758394180742), 'x89': np.float64(0.3963707773542504), 'x90': np.float64(0.5393067388662478), 'x91': np.float64(0.29289370302597156)}
    ref_vector = {'x0': np.float64(0.06862836329507828), 'x1': np.float64(0.16879982968042223), 'x2': np.float64(0.058582379279016626), 'x3': np.float64(0.8933047653227757), 'x4': np.float64(0.770542897433347), 'x5': np.float64(0.19602393504613214), 'x6': np.float64(0.1086457796359572), 'x7': np.float64(0.4131627236911566), 'x8': np.float64(0.0), 'x9': np.float64(0.0003331488656117723), 'x10': np.float64(0.0005244295735420376), 'x11': np.float64(-0.08667905866808216), 'x12': np.float64(0.1265934476976646), 'x13': np.float64(0.05), 'x14': np.float64(0.06034780568304464), 'x15': np.float64(1.172612137632668), 'x16': np.float64(0.07477297475556711), 'x17': np.float64(0.050028317905083906), 'x18': np.float64(0.1544400798666547), 'x19': np.float64(0.051658800473082814), 'x20': np.float64(0.8899496877428007), 'x21': np.float64(0.8860001779974638), 'x22': np.float64(0.23429922812478593), 'x23': np.float64(0.08782961850403123), 'x24': np.float64(0.3874002519773389), 'x25': np.float64(0.015508096442983477), 'x26': np.float64(4.257909026038361e-05), 'x27': np.float64(0.0009903491983153819), 'x28': np.float64(-0.09708574625523717), 'x29': np.float64(0.2997629231257798), 'x30': np.float64(0.05), 'x31': np.float64(0.10485674214976046), 'x32': np.float64(0.05), 'x33': np.float64(0.17404019853534675), 'x34': np.float64(0.05006783466443859), 'x35': np.float64(0.8435128537937141), 'x36': np.float64(0.8692023444577196), 'x37': np.float64(0.2877375753847593), 'x38': np.float64(0.09542092244550118), 'x39': np.float64(0.37220493562404106), 'x40': np.float64(0.01147387543771499), 'x41': np.float64(0.00022699204544033004), 'x42': np.float64(0.0006108779268117336), 'x43': np.float64(-0.0527841396832392), 'x44': np.float64(0.2688632150899689), 'x45': np.float64(0.051228613818918925), 'x46': np.float64(0.05), 'x47': np.float64(0.051319915449876255), 'x48': np.float64(0.1932609218094311), 'x49': np.float64(0.10516641665928939), 'x50': np.float64(0.8997547427574681), 'x51': np.float64(0.61949786447824), 'x52': np.float64(0.3421639108230008), 'x53': np.float64(0.08145163667230101), 'x54': np.float64(0.24028616982224293), 'x55': np.float64(0.01156519789477432), 'x56': np.float64(6.69357812587355e-05), 'x57': np.float64(0.0007739544386753883), 'x58': np.float64(-0.022850944846339396), 'x59': np.float64(0.08767530929929346), 'x60': np.float64(0.05), 'x61': np.float64(0.12252727779147671), 'x62': np.float64(0.06172509040504469), 'x63': np.float64(0.15081872601962315), 'x64': np.float64(0.05000002620938917), 'x65': np.float64(0.4957264085197678), 'x66': np.float64(0.9), 'x67': np.float64(0.17486322102567325), 'x68': np.float64(0.022372426358763683), 'x69': np.float64(0.49686717292940497), 'x70': np.float64(0.011801661689929983), 'x71': np.float64(7.3809628026206e-05), 'x72': np.float64(0.0004869636114035082), 'x73': np.float64(-0.04846087683098008), 'x74': np.float64(0.04850214965399018), 'x75': np.float64(0.06263763880227691), 'x76': np.float64(0.06767719008796821), 'x77': np.float64(0.1473060598192438), 'x78': np.float64(0.2218043020580134), 'x79': 3, 'x80': np.float64(44.4705510075448), 'x81': np.float64(1.96757216116753), 'x82': np.float64(0.4547987052525282), 'x83': np.float64(0.28344372873926904), 'x84': np.float64(0.27948610737056756), 'x85': np.float64(0.21332303597526092), 'x86': np.float64(0.04771237234774799), 'x87': np.float64(0.1517311524676458), 'x88': np.float64(0.1233010324788802), 'x89': np.float64(0.6741684294430199), 'x90': np.float64(0.7167965487418299), 'x91': np.float64(0.3418343491746896)}
    ref_vector = {'x0': np.float64(0.05035561515719036), 'x1': np.float64(0.1644195144056), 'x2': np.float64(0.06224896870945479), 'x3': np.float64(0.8996258463695751), 'x4': np.float64(0.8976158349210307), 'x5': np.float64(0.3213898157364029), 'x6': np.float64(0.05555903719904701), 'x7': np.float64(0.34843725142816895), 'x8': np.float64(0.011276698862858781), 'x9': np.float64(0.0003291709639498397), 'x10': np.float64(0.00072772705427627), 'x11': np.float64(-0.05646907907926193), 'x12': np.float64(0.12677230060204767), 'x13': np.float64(0.05033877631761697), 'x14': np.float64(0.07377465957166386), 'x15': np.float64(1.17336349498811), 'x16': np.float64(0.12096247925622612), 'x17': np.float64(0.05752800530795125), 'x18': np.float64(0.21307824120615546), 'x19': np.float64(0.05782849670685973), 'x20': np.float64(0.8915178131251633), 'x21': np.float64(0.8776286350434881), 'x22': np.float64(0.3513933521955169), 'x23': np.float64(0.11678047137638081), 'x24': np.float64(0.4999994025574745), 'x25': np.float64(0.018224152070037664), 'x26': np.float64(7.888370163527837e-07), 'x27': np.float64(0.0009959914080890574), 'x28': np.float64(-0.09452443370966852), 'x29': np.float64(0.25844489687299266), 'x30': np.float64(0.051243942765154225), 'x31': np.float64(0.07492935282144816), 'x32': np.float64(0.05), 'x33': np.float64(0.17994684882990303), 'x34': np.float64(0.05041337259752426), 'x35': np.float64(0.8447870637362785), 'x36': np.float64(0.8903600424508203), 'x37': np.float64(0.2624160371367633), 'x38': np.float64(0.10749816631069785), 'x39': np.float64(0.22401760009234276), 'x40': np.float64(0.020242372227982447), 'x41': np.float64(0.00017449351171561542), 'x42': np.float64(0.0005806070747494266), 'x43': np.float64(-0.05772303143229478), 'x44': np.float64(0.3), 'x45': np.float64(0.05003754646915319), 'x46': np.float64(0.2), 'x47': np.float64(0.1), 'x48': np.float64(0.13419494064928444), 'x49': np.float64(0.07759220052455401), 'x50': np.float64(0.9), 'x51': np.float64(0.6984991067850737), 'x52': np.float64(0.22247711649902246), 'x53': np.float64(0.05114434847192131), 'x54': np.float64(0.2687409732283227), 'x55': np.float64(0.010796773418389298), 'x56': np.float64(0.0011942839533385494), 'x57': np.float64(0.001), 'x58': np.float64(-0.017970051255863633), 'x59': np.float64(0.20979374312909937), 'x60': np.float64(0.05), 'x61': np.float64(0.1999999588610527), 'x62': np.float64(0.06296775885442824), 'x63': np.float64(0.23547524122150582), 'x64': np.float64(0.05013734056703228), 'x65': np.float64(0.8347723262648988), 'x66': np.float64(0.6077770447049556), 'x67': np.float64(0.1823575919444591), 'x68': np.float64(0.020002017649543718), 'x69': np.float64(0.3597005762590377), 'x70': np.float64(0.021189370960170553), 'x71': np.float64(0.0008783270502365296), 'x72': np.float64(0.0002404749130101065), 'x73': np.float64(-0.045406973932260446), 'x74': np.float64(0.024857729693678676), 'x75': np.float64(0.05), 'x76': np.float64(0.10309122953383416), 'x77': np.float64(0.19055233614970088), 'x78': np.float64(0.3025398913472852), 'x79': 4, 'x80': np.float64(52.98815964875218), 'x81': np.float64(1.56807981535195), 'x82': np.float64(0.3620311670976888), 'x83': np.float64(0.22201608582614632), 'x84': np.float64(0.22201606666974683), 'x85': np.float64(0.15304356171985517), 'x86': np.float64(0.05156827806987072), 'x87': np.float64(0.1490416589736719), 'x88': np.float64(0.13094360943822136), 'x89': np.float64(0.6741120093967852), 'x90': np.float64(0.5055701903633827), 'x91': np.float64(0.41168449068667035)}
    ref_vector = {'x0': np.float64(0.057354539760835446), 'x1': np.float64(0.16940602121113418), 'x2': np.float64(0.05494764641281743), 'x3': np.float64(0.8878415713518093), 'x4': np.float64(0.8976680311924798), 'x5': np.float64(0.3215386950668153), 'x6': np.float64(0.05555903719904701), 'x7': np.float64(0.21125195261977694), 'x8': np.float64(0.011276381405185668), 'x9': np.float64(0.0003291709639498397), 'x10': np.float64(0.0005811633027534018), 'x11': np.float64(-0.10328267416816166), 'x12': np.float64(0.05782898032840142), 'x13': np.float64(0.05031790982032209), 'x14': np.float64(0.07419102501158031), 'x15': np.float64(1.1865948186485755), 'x16': np.float64(0.12133105906081246), 'x17': np.float64(0.05777085369551649), 'x18': np.float64(0.2130215829104652), 'x19': np.float64(0.059022281222251935), 'x20': np.float64(0.8913197773366971), 'x21': np.float64(0.8776254629208743), 'x22': np.float64(0.18782400847420463), 'x23': np.float64(0.09113504996232472), 'x24': np.float64(0.5), 'x25': np.float64(0.015032825459967918), 'x26': np.float64(2.5186183726102793e-06), 'x27': np.float64(0.000999632590534611), 'x28': np.float64(-0.07140035689299523), 'x29': np.float64(0.29388289496231174), 'x30': np.float64(0.05130368501802162), 'x31': np.float64(0.11828708768007995), 'x32': np.float64(0.05019486828540259), 'x33': np.float64(0.1853142095266545), 'x34': np.float64(0.05000000000000001), 'x35': np.float64(0.8986760063292032), 'x36': np.float64(0.5145540121883229), 'x37': np.float64(0.2624035849948511), 'x38': np.float64(0.17412138497792773), 'x39': np.float64(0.23691748515546712), 'x40': np.float64(0.021303430386059603), 'x41': np.float64(0.00014697935409688651), 'x42': np.float64(0.0006033829759857107), 'x43': np.float64(-0.05772303143229478), 'x44': np.float64(0.2968102973298818), 'x45': np.float64(0.050009197607062295), 'x46': np.float64(0.2), 'x47': np.float64(0.09998565673913212), 'x48': np.float64(0.13457319894280298), 'x49': np.float64(0.07759220052455401), 'x50': np.float64(0.8999710291803883), 'x51': np.float64(0.8900737508999793), 'x52': np.float64(0.22488382891243303), 'x53': np.float64(0.05110413026043391), 'x54': np.float64(0.3680622097032429), 'x55': np.float64(0.026504444924876698), 'x56': np.float64(6.469560295593174e-06), 'x57': np.float64(0.000819804880554638), 'x58': np.float64(-0.017588837304853434), 'x59': np.float64(0.22457643335145072), 'x60': np.float64(0.051106732076482066), 'x61': np.float64(0.19446145826154715), 'x62': np.float64(0.0628505738292407), 'x63': np.float64(0.23550167506573094), 'x64': np.float64(0.05013491916554471), 'x65': np.float64(0.8322874582982852), 'x66': np.float64(0.8919919366008909), 'x67': np.float64(0.24977943478013886), 'x68': np.float64(0.020077635630403993), 'x69': np.float64(0.3597005762590377), 'x70': np.float64(0.021189370960170553), 'x71': np.float64(0.0008783818510221151), 'x72': np.float64(0.00046914121750155514), 'x73': np.float64(-0.035500242138512564), 'x74': np.float64(0.026521908304704366), 'x75': np.float64(0.05000495568167973), 'x76': np.float64(0.09382324296097097), 'x77': np.float64(0.18949145523979374), 'x78': np.float64(0.3078684470329587), 'x79': 4, 'x80': np.float64(53.02429007561069), 'x81': np.float64(1.7938744012558505), 'x82': np.float64(0.2108883023537044), 'x83': np.float64(0.3481040128258733), 'x84': np.float64(0.21277833182184774), 'x85': np.float64(0.15288815414807166), 'x86': np.float64(0.06709870258198), 'x87': np.float64(0.14921076253196897), 'x88': np.float64(0.13094360943822136), 'x89': np.float64(0.6784436856284838), 'x90': np.float64(0.5055701903633827), 'x91': np.float64(0.4152317451909224)}
    ref_vector = {'x0': np.float64(0.054817423700434284), 'x1': np.float64(0.16454938871478855), 'x2': np.float64(0.062394107919658603), 'x3': np.float64(0.6380810787805823), 'x4': np.float64(0.8975084631718633), 'x5': np.float64(0.33054034727863835), 'x6': np.float64(0.05109266422386802), 'x7': np.float64(0.35185886216968354), 'x8': np.float64(0.011720856620908309), 'x9': np.float64(0.00032896952213298407), 'x10': np.float64(0.0007231596751261792), 'x11': np.float64(-0.06972114469904085), 'x12': np.float64(0.07282816078109634), 'x13': np.float64(0.05), 'x14': np.float64(0.07568030335214938), 'x15': np.float64(1.1921379180602656), 'x16': np.float64(0.07627969836092395), 'x17': np.float64(0.058218599999584765), 'x18': np.float64(0.14773690908286802), 'x19': np.float64(0.05611908291569801), 'x20': np.float64(0.8979130452385233), 'x21': np.float64(0.6775338723556947), 'x22': np.float64(0.34599098197728884), 'x23': np.float64(0.11652172593803944), 'x24': np.float64(0.49999912586560524), 'x25': np.float64(0.025026291973073638), 'x26': np.float64(2.662310854921433e-06), 'x27': np.float64(0.001), 'x28': np.float64(-0.06843622023223028), 'x29': np.float64(0.20905027586568092), 'x30': np.float64(0.051276457458488446), 'x31': np.float64(0.06539409666874609), 'x32': np.float64(0.0500880129779927), 'x33': np.float64(0.17045359498846294), 'x34': np.float64(0.0712923589410194), 'x35': np.float64(0.8951992358313249), 'x36': np.float64(0.4845963685654864), 'x37': np.float64(0.31366466581214764), 'x38': np.float64(0.10750148431881629), 'x39': np.float64(0.21880029649901783), 'x40': np.float64(0.01839969629696021), 'x41': np.float64(3.1358040344907503e-06), 'x42': np.float64(0.0005643694160710516), 'x43': np.float64(-0.057704949185411816), 'x44': np.float64(0.2544320069258373), 'x45': np.float64(0.05), 'x46': np.float64(0.19774382922856928), 'x47': np.float64(0.1), 'x48': np.float64(0.1897017748583441), 'x49': np.float64(0.07313776938931621), 'x50': np.float64(0.7027882129092855), 'x51': np.float64(0.8854494504806193), 'x52': np.float64(0.22488382891243303), 'x53': np.float64(0.04956894696637333), 'x54': np.float64(0.29206734157949005), 'x55': np.float64(0.014024691476268425), 'x56': np.float64(0.0011942391161052382), 'x57': np.float64(0.0009998132035506507), 'x58': np.float64(-0.01783281578184271), 'x59': np.float64(0.20588526681373595), 'x60': np.float64(0.05), 'x61': np.float64(0.10028984687929322), 'x62': np.float64(0.05031547490217939), 'x63': np.float64(0.235283164937113), 'x64': np.float64(0.05051672096344431), 'x65': np.float64(0.8346308053531276), 'x66': np.float64(0.8981133830768027), 'x67': np.float64(0.1754989579318476), 'x68': np.float64(0.020001247103198613), 'x69': np.float64(0.2959622124002933), 'x70': np.float64(0.024012522187836054), 'x71': np.float64(0.0008947256388042964), 'x72': np.float64(0.0003169812535440497), 'x73': np.float64(-0.045956265144294656), 'x74': np.float64(0.026554599433711936), 'x75': np.float64(0.05), 'x76': np.float64(0.2), 'x77': np.float64(0.18708054347439612), 'x78': np.float64(0.30254651331428967), 'x79': 4, 'x80': np.float64(53.36909851576063), 'x81': np.float64(1.5680749583650315), 'x82': np.float64(0.36082729979397776), 'x83': np.float64(0.24872831447599503), 'x84': np.float64(0.22201606666974683), 'x85': np.float64(0.1571180160346805), 'x86': np.float64(0.06906845434367138), 'x87': np.float64(0.14168944927952953), 'x88': np.float64(0.132596388810178), 'x89': np.float64(0.9812278747420167), 'x90': np.float64(0.5052376518208197), 'x91': np.float64(0.41168449068667035)}
    ref_vector = {'x0': np.float64(0.07074271306829692), 'x1': np.float64(0.16454626963700397), 'x2': np.float64(0.05010329614014347), 'x3': np.float64(0.8947523364866242), 'x4': np.float64(0.8974977811812966), 'x5': np.float64(0.2870353463700017), 'x6': np.float64(0.053243586161978454), 'x7': np.float64(0.3354337291929989), 'x8': np.float64(0.001838080054997404), 'x9': np.float64(0.00030457679312725943), 'x10': np.float64(0.0007246109641191778), 'x11': np.float64(-0.0708696049289823), 'x12': np.float64(0.0744335278433659), 'x13': np.float64(0.05), 'x14': np.float64(0.07419102501158031), 'x15': np.float64(1.1939346986114452), 'x16': np.float64(0.07627689884766449), 'x17': np.float64(0.05801502963051236), 'x18': np.float64(0.21301226965324171), 'x19': np.float64(0.05592043227569649), 'x20': np.float64(0.8915178131251633), 'x21': np.float64(0.877234350025325), 'x22': np.float64(0.3452426135373039), 'x23': np.float64(0.1152590809891906), 'x24': np.float64(0.4991748534109108), 'x25': np.float64(0.013809430837752337), 'x26': np.float64(1.4037346170438995e-06), 'x27': np.float64(0.000999883394899088), 'x28': np.float64(-0.07250412393057155), 'x29': np.float64(0.2582979323047443), 'x30': np.float64(0.05), 'x31': np.float64(0.1004577031422517), 'x32': np.float64(0.050011191960042854), 'x33': np.float64(0.19222015789302663), 'x34': np.float64(0.050450676666809094), 'x35': np.float64(0.8787587094793585), 'x36': np.float64(0.512914656535733), 'x37': np.float64(0.26298729269295185), 'x38': np.float64(0.13256528994822972), 'x39': np.float64(0.22375731886085393), 'x40': np.float64(0.023687723938861167), 'x41': np.float64(0.0001468769181246573), 'x42': np.float64(0.0005617661773095888), 'x43': np.float64(-0.07047993414972384), 'x44': np.float64(0.3), 'x45': np.float64(0.050025877666038776), 'x46': np.float64(0.2), 'x47': np.float64(0.1), 'x48': np.float64(0.13151919317726146), 'x49': np.float64(0.07336677535819125), 'x50': np.float64(0.8929798961428895), 'x51': np.float64(0.6871536463848181), 'x52': np.float64(0.22488382891243303), 'x53': np.float64(0.048049478288513595), 'x54': np.float64(0.2931507229577117), 'x55': np.float64(0.008838781316362404), 'x56': np.float64(0.0011953927432265313), 'x57': np.float64(0.0009998429326825214), 'x58': np.float64(-0.017804408723908097), 'x59': np.float64(0.21090668814894176), 'x60': np.float64(0.05), 'x61': np.float64(0.10523828572389626), 'x62': np.float64(0.052283897712108125), 'x63': np.float64(0.2352940238517895), 'x64': np.float64(0.050277509212110966), 'x65': np.float64(0.8346308053531276), 'x66': np.float64(0.8794608930573073), 'x67': np.float64(0.18395162940587892), 'x68': np.float64(0.020006916275303745), 'x69': np.float64(0.3603184157287996), 'x70': np.float64(0.010747701199472857), 'x71': np.float64(0.0008947256388042964), 'x72': np.float64(0.000464120211100067), 'x73': np.float64(-0.04131710353432434), 'x74': np.float64(0.024859397765657768), 'x75': np.float64(0.05000590090700822), 'x76': np.float64(0.10285409053759106), 'x77': np.float64(0.1463746470824701), 'x78': np.float64(0.3024677021705982), 'x79': 4, 'x80': np.float64(53.4165374984954), 'x81': np.float64(1.56807981535195), 'x82': np.float64(0.3508449501622704), 'x83': np.float64(0.23286040305485828), 'x84': np.float64(0.21060861911015372), 'x85': np.float64(0.18320914067899138), 'x86': np.float64(0.06713281860336148), 'x87': np.float64(0.1492517028221512), 'x88': np.float64(0.13094360943822136), 'x89': np.float64(0.6742674694187962), 'x90': np.float64(0.5040897367147055), 'x91': np.float64(0.41168661963788183)}
    ref_vector = {'x0': np.float64(0.054532249778999786), 'x1': np.float64(0.16403564359081257), 'x2': np.float64(0.050461632950517574), 'x3': np.float64(0.8989534118013895), 'x4': np.float64(0.8976128357486094), 'x5': np.float64(0.3117605472251557), 'x6': np.float64(0.05576624412432181), 'x7': np.float64(0.35232206215066914), 'x8': np.float64(0.011048351160282948), 'x9': np.float64(0.00033574307210337503), 'x10': np.float64(0.0007277484446207025), 'x11': np.float64(-0.08522243955911907), 'x12': np.float64(0.07282816078109634), 'x13': np.float64(0.05504725970146625), 'x14': np.float64(0.07426423105599113), 'x15': np.float64(1.173366450957023), 'x16': np.float64(0.0755590897121223), 'x17': np.float64(0.057187030862985995), 'x18': np.float64(0.21303877600161272), 'x19': np.float64(0.057398167099872055), 'x20': np.float64(0.8861126012710993), 'x21': np.float64(0.8723408631567393), 'x22': np.float64(0.34598735615329795), 'x23': np.float64(0.08100385315315622), 'x24': np.float64(0.5), 'x25': np.float64(0.02049239668308474), 'x26': np.float64(9.896716543639488e-06), 'x27': np.float64(0.0009998981507235998), 'x28': np.float64(-0.08553113208709921), 'x29': np.float64(0.25720593307374395), 'x30': np.float64(0.050047241784825176), 'x31': np.float64(0.10056336994873744), 'x32': np.float64(0.05000038899225358), 'x33': np.float64(0.1734133645682932), 'x34': np.float64(0.06653593162006859), 'x35': np.float64(0.8736454260686816), 'x36': np.float64(0.8606956427977711), 'x37': np.float64(0.31713028970498675), 'x38': np.float64(0.1052325309011066), 'x39': np.float64(0.3463024345724183), 'x40': np.float64(0.01790985253575241), 'x41': np.float64(0.0013247262889993555), 'x42': np.float64(0.0005774354588839431), 'x43': np.float64(-0.09170620560215632), 'x44': np.float64(0.29995723931150525), 'x45': np.float64(0.05), 'x46': np.float64(0.2), 'x47': np.float64(0.1), 'x48': np.float64(0.2085696868532738), 'x49': np.float64(0.6692363388792224), 'x50': np.float64(0.9), 'x51': np.float64(0.6884966669541436), 'x52': np.float64(0.2248830000478399), 'x53': np.float64(0.051150139949940006), 'x54': np.float64(0.3285165807254489), 'x55': np.float64(0.013607393849883232), 'x56': np.float64(1.9233057868024836e-06), 'x57': np.float64(0.0009933661997209573), 'x58': np.float64(-0.0177841459691189), 'x59': np.float64(0.11316390663901306), 'x60': np.float64(0.05), 'x61': np.float64(0.1985000377230955), 'x62': np.float64(0.050101108532276), 'x63': np.float64(0.14607025117705616), 'x64': np.float64(0.05040308794690823), 'x65': np.float64(0.8316032388413133), 'x66': np.float64(0.882816194772045), 'x67': np.float64(0.18236077424167804), 'x68': np.float64(0.02029902427915542), 'x69': np.float64(0.338165739673912), 'x70': np.float64(0.00932115212325643), 'x71': np.float64(0.00012877447999299106), 'x72': np.float64(0.0002360633410762753), 'x73': np.float64(-0.02469992464388624), 'x74': np.float64(0.02878928341580361), 'x75': np.float64(0.05), 'x76': np.float64(0.1386071800036937), 'x77': np.float64(0.19055233614970088), 'x78': np.float64(0.3060676228484556), 'x79': 4, 'x80': np.float64(52.84492526036495), 'x81': np.float64(1.5838722855006213), 'x82': np.float64(0.38531894014281975), 'x83': np.float64(0.22201606666974683), 'x84': np.float64(0.1639960612560437), 'x85': np.float64(0.1639960612560437), 'x86': np.float64(0.06731612087113799), 'x87': np.float64(0.12045692048306522), 'x88': np.float64(0.13094360943822136), 'x89': np.float64(0.39883635882604196), 'x90': np.float64(0.5055701903633827), 'x91': np.float64(0.41168449068667035)}
    ref_vector = {'x0': np.float64(0.09441907973612891), 'x1': np.float64(0.1644824589532731), 'x2': np.float64(0.05041991264573388), 'x3': np.float64(0.8830519846894508), 'x4': np.float64(0.8976154397263706), 'x5': np.float64(0.3299993092735413), 'x6': np.float64(0.053237766619933034), 'x7': np.float64(0.34026565786927365), 'x8': np.float64(0.002028591970330283), 'x9': np.float64(0.00033159844206164717), 'x10': np.float64(0.0007230646010742911), 'x11': np.float64(-0.07060538006690154), 'x12': np.float64(0.07282816078109634), 'x13': np.float64(0.05), 'x14': np.float64(0.06115939157484234), 'x15': np.float64(1.1939273164459931), 'x16': np.float64(0.12086114038153246), 'x17': np.float64(0.058008144592322654), 'x18': np.float64(0.14889135401830755), 'x19': np.float64(0.05784032958017614), 'x20': np.float64(0.8557607238125325), 'x21': np.float64(0.8772325524403954), 'x22': np.float64(0.34609546645690387), 'x23': np.float64(0.11659878314055425), 'x24': np.float64(0.5), 'x25': np.float64(0.013788960181311505), 'x26': np.float64(0.0001248148100887384), 'x27': np.float64(0.0009999968077774007), 'x28': np.float64(-0.06524556526636136), 'x29': np.float64(0.2647008294058789), 'x30': np.float64(0.05003089843359783), 'x31': np.float64(0.06444922948957246), 'x32': np.float64(0.1), 'x33': np.float64(0.1702211268239621), 'x34': np.float64(0.05152138880597886), 'x35': np.float64(0.8947068658104353), 'x36': np.float64(0.5267422457068935), 'x37': np.float64(0.2624035849948511), 'x38': np.float64(0.11026311889785465), 'x39': np.float64(0.2239331652212809), 'x40': np.float64(0.009613513730526036), 'x41': np.float64(3.478463171494034e-06), 'x42': np.float64(0.0005655571313743727), 'x43': np.float64(-0.05768508764658647), 'x44': np.float64(0.3), 'x45': np.float64(0.05), 'x46': np.float64(0.19995354971083165), 'x47': np.float64(0.1), 'x48': np.float64(0.2363054370410226), 'x49': np.float64(0.0734090926073252), 'x50': np.float64(0.764821031210638), 'x51': np.float64(0.6804045523922776), 'x52': np.float64(0.22488382891243303), 'x53': np.float64(0.05112139572408497), 'x54': np.float64(0.2918597568641959), 'x55': np.float64(0.008825176619993776), 'x56': np.float64(0.001195492903824735), 'x57': np.float64(0.001), 'x58': np.float64(-0.017882078812769132), 'x59': np.float64(0.18560746948936782), 'x60': np.float64(0.05), 'x61': np.float64(0.10038404862419571), 'x62': np.float64(0.052340202104699675), 'x63': np.float64(0.23443251556674724), 'x64': np.float64(0.050285122959010956), 'x65': np.float64(0.8346308053531276), 'x66': np.float64(0.8983931479038701), 'x67': np.float64(0.20921086719300058), 'x68': np.float64(0.020006999759965345), 'x69': np.float64(0.29731206501063556), 'x70': np.float64(0.021189370960170553), 'x71': np.float64(0.0008955664080047718), 'x72': np.float64(0.0004346084668307938), 'x73': np.float64(-0.04099733011057777), 'x74': np.float64(0.02463793965966118), 'x75': np.float64(0.05000571914717749), 'x76': np.float64(0.10295442113256419), 'x77': np.float64(0.19068753881119482), 'x78': np.float64(0.30254651331428967), 'x79': 4, 'x80': np.float64(53.02136642615879), 'x81': np.float64(1.56807981535195), 'x82': np.float64(0.45727524317856005), 'x83': np.float64(0.23376021832887126), 'x84': np.float64(0.2223930432002742), 'x85': np.float64(0.18087661537846536), 'x86': np.float64(0.06709870258198), 'x87': np.float64(0.14208871498972095), 'x88': np.float64(0.13094777259984464), 'x89': np.float64(0.6741110012506895), 'x90': np.float64(0.5041046025599504), 'x91': np.float64(0.41168449068667035)}
    test._evaluate(ref_vector, output)
    print(output)
