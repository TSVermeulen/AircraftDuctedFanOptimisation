"""
analyse_performance
===================

Description
-----------
This module provides performance analysis methods for a given ducted fan design. 
It evaluates aerodynamic performance across a tip set angles and rotational 
rates for a given operating condition. It generates performance plots showing 
thrust coefficient versus propulsor efficiency.

The Analyzer class interfaces with the MTFLOW-based UDC (Unified Duct Code) to 
compute performance metrics. 

Notes
-----
Ensure that the MTFLOW executable and required input files are present in the 
appropriate directories. The analyzer validates design vectors for feasibility 
before running simulations and handles crash.

References
----------
For more details on the MTFLOW solver and its input/output requirements, 
refer to the MTFLOW user manual:
https://web.mit.edu/drela/Public/web/mtflow/mtflow.pdf

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@tudelft.nl
Version: 1.0

Changelog:
- V1.0: Initial implementation.
"""

# Import standard libraries
import copy
import contextlib
from pathlib import Path

# Import 3rd party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Enable submodel relative imports
from GA.utils import ensure_repo_paths, get_figsize
ensure_repo_paths()

# Import interface submodels and other dependencies
from MTSOL_call import OutputType, ExitFlag  # type: ignore
from output_handling import output_processing  # type: ignore
from design_vector_interface import DesignVectorInterface  # type: ignore
import config  # type: ignore

# Define plot formatting parameters
plt.rcParams.update({'font.size': 9,
                     "pgf.texsystem": "xelatex",
                     "text.usetex":  True,
                     "pgf.rcfonts": False})

STYLE = ["-.", "--", (0, (3, 5, 1, 5, 1, 5)), ":", (0, (3, 1, 1, 1, 1, 1))]
MARKERS = ["^", "*", "o", "+", "d", "s", "h", "p"]
CLRS = ["tab:blue", "tab:orange", "tab:green",
        "tab:red", "tab:purple", "tab:brown", "tab:pink", 
        "tab:olive", "tab:cyan"]
MS = 4
MAJOR_GRID_ALPHA = 0.6
MINOR_GRID_ALPHA = 0.4
COLUMN_WIDTH = 256.5 # in points, for two-column layout


class Analyzer:
    """
    Class definition for the Analyzer. 
    """

    # Define the relevant file names which must be deleted 
    FILE_TEMPLATES = {"walls": "walls.{}",
                      "tflow": "tflow.{}",
                      "tdat": "tdat.{}"}

    # Set the operating conditions from the config
    oper = copy.deepcopy(config.multi_oper[0])


    def __init__(self,
                 beta_tip: float,
                 rps_range: list[float] | np.typing.NDArray[np.floating],
                 verbose: bool = True,
                 **kwargs) -> None:
        """
        Initialisation of the Analyzer class.

        Parameters
        ----------
        - beta_tip : float
            The blade tip angle in radians.
        - rps_range : list[float] | np.typing.NDArray[np.floating]
            The range of RPS values to analyse.
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
        self.tip_angle_range = beta_tip  # Tip angle in radians
        self.RPS_range = rps_range

        # Define key paths/directories
        self.parent_dir = Path(__file__).resolve().parent
        self.submodels_path = self.parent_dir / "Submodels"

        # Validate critical submodels_path exist
        if not self.submodels_path.exists():
            raise SystemError(f"Missing submodels path: {self.submodels_path}")

        # Define the analysis name
        self.analysis_name = "perf_analysis"

        # Initialise design vector interface
        self.dvec_interface = DesignVectorInterface()

        # Use lazy-loaded modules (initialised at first use)
        # Prevents circular imports and speeds up initial loading time.
        if not hasattr(self, "_lazy_modules_loaded"):
            from UDC import UDC  # type: ignore
            from Submodels.file_handling import fileHandlingMTSET  # type: ignore
            from Submodels.file_handling import fileHandlingMTFLO  # type: ignore
            self._UDC = UDC
            self._fileHandlingMTSET = fileHandlingMTSET
            self._fileHandlingMTFLO = fileHandlingMTFLO
            self._lazy_modules_loaded = True


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
        Clean up the relevant files generated by the UDC after completing the 
        analysis.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        with contextlib.suppress(Exception):
            for template in self.FILE_TEMPLATES.values():
                (self.submodels_path / 
                template.format(self.analysis_name)).unlink(missing_ok=True)


    def GenerateUDCInputs(self,
                          x: dict[str, float | int],
                          RPS: float) -> bool:
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
        - x : dict[str, float | int]
            The design vector dictionary.

        Returns
        -------
        - output_generated: bool
            - True if the design vector is feasible, false otherwise.
        """

        # Generate the MTSET input file containing the axisymmetric geometries
        # and the MTFLO blading input file.
        try:
            # Deconstruct the design vector
            (self.centerbody_variables,
            self.duct_variables,
            self.blade_design_parameters,
            self.blade_blading_parameters,
            self.Lref) = self.dvec_interface.DeconstructDesignVector(x_dict=x)

            # Write the to be analysed angles and rotational rates to the 
            # blading, overwriting the default values from the design vector.
            self.blade_blading_parameters[0]["RPS"] = RPS
            self.blade_blading_parameters[0]["RPS_lst"] = [RPS] * len(config.multi_oper)
            self.blade_blading_parameters[0]["ref_blade_angle"] = self.tip_angle
            self.blade_blading_parameters[0]["ref_blade_angle_lst"] = [self.tip_angle] * len(config.REFERENCE_BLADE_ANGLES) if hasattr(config.REFERENCE_BLADE_ANGLES, 'len') else [self.tip_angle]

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
                  *args,
                  **kwargs) -> pd.DataFrame:
        """
        Evaluation of the UDC to generate a performance curve of TC-eta for a 
        range of tip set angles and RPS values.

        Parameters
        ----------
        - x : dict[str, float | int]
            The pymoo design vector dictionary.
        - *args : tuple
            Additional arguments.
        - **kwargs : dict[str, Any]
            Additional keyword arguments.

        Returns
        -------
        - output : pd.DataFrame
            DataFrame containing the performance results for the sweep.
        """

        results = []
        for tip_angle in self.tip_angle_range:
            self.tip_angle = tip_angle  # Set the current tip angle
            for RPS in self.RPS_range:
                # Generate the UDC input files.
                # If design_okay is false, this indicates an error in the input
                # generation caused by an infeasible design vector.
                design_okay = self.GenerateUDCInputs(x, RPS)

                # Evaluate the design using the UDC if the design is feasible
                if design_okay:
                    self.ComputeReynolds()  # Compute the Reynolds number

                    UDC_interface = self._UDC(operating_conditions=self.oper,
                                            ref_length=self.Lref,
                                            analysis_name=self.analysis_name,
                                            run_viscous=kwargs.get('viscous', 
                                                                   True),
                                            **kwargs)

                    try:
                        # Run UDC
                        (exit_flag,
                        UDC_outputs) = UDC_interface.caller(external_inputs=True,
                                                            output_type=OutputType.FORCES_ONLY)

                        # Overwrite outputs in case of crashes
                        if exit_flag in (ExitFlag.CRASH, ExitFlag.CHOKING,
                                        ExitFlag.NOT_PERFORMED):
                            UDC_outputs = output_processing().GetAllVariables()

                    except Exception as e:
                        UDC_outputs = output_processing().GetAllVariables()
                        if self.verbose:
                            print(f"[UDC_ERROR] case={self.analysis_name}: {e}")

                else:
                    # If the design is infeasible, we load the crash outputs
                    # This is a predefined dictionary with all outputs set to 0.
                    UDC_outputs = output_processing().GetAllVariables()

                # Cleanup the generated files
                self.CleanUpFiles()
                
                # Collect the results and store them in results
                TC = UDC_outputs['data']['Total force CT']
                PC = UDC_outputs['data']['Total power CP']
                EtaP = TC / PC if PC != 0 else 0

                results.append({
                    'beta_tip': round(np.rad2deg(tip_angle), 2),
                    'RPS': RPS,
                    'TC': TC,
                    'PC': PC,
                    'EtaP': EtaP
                })

        # Create output dataframe
        output = pd.DataFrame(results)    
        
        return output
    

    def create_performance_plot(self, 
                                df: pd.DataFrame) -> None:
        """
        Create a performance plot of the thrust coefficient TC as function of 
        the propulsor efficiency. 

        Parameters
        ----------
        - df : pd.DataFrame
            A dataframe containing the TC, PC, and propulsor efficiency data 
            for each analysed tip set angle and rotational rate. 

        Returns
        -------
        - None
        """

        # Group by beta_tip
        groups = df.groupby('beta_tip')

        # Create the plot
        plt.figure(figsize=get_figsize(columnwidth=COLUMN_WIDTH, wf=1),
                constrained_layout=True)

        i = 0
        for beta, group in groups:
            plt.plot(group['TC'], group['EtaP'], 
                     marker=MARKERS[i % len(MARKERS)], 
                     color=CLRS[i % len(CLRS)], ms=MS, 
                     linewidth=0.5, label=f"$\\beta_{{tip}}$={beta}$^\\circ$")
            i += 1

        # Plot formatting
        plt.xlabel('Thrust Coefficient $T_C$ [-]')
        plt.ylabel('Propulsor Efficiency $\\eta_P$ [-]')
        plt.grid(which='major', alpha=MAJOR_GRID_ALPHA)
        plt.grid(which='minor', alpha=MINOR_GRID_ALPHA, linewidth=0.25)
        plt.legend()
        plt.minorticks_on()

        plt.show()
    

    def analyse_performance(self, 
                            x: dict[str, float | int] | None = None,
                            store_to_csv: bool = False) -> None:
        """
        Analyze and visualize performance data.
        This method evaluates performance based on provided design variables
        and creates a performance plot. 
        It can either compute performance from scratch or load previously 
        computed results from a CSV file.

        Parameters
        ----------
        - x : dict[str, float | int] | None, optional
            Dictionary containing the design variables 
            If None, existing performance data is loaded from 
            'performance_data.csv'. Default is None.
        - store_to_csv : bool, optional
            If True and x is provided, saves the computed performance data to
            'performance_data.csv'. Default is False.
        
        Returns
        -------
        None
        """

        # Collect the output dataframe
        if x is not None:
            output = self._evaluate(x)

            # Store the dataframe to a csv file if desired
            if store_to_csv:
                output.to_csv('performance_data.csv', index=False)
        else:
            # If no vector is passed read the existing performance_data.csv
            # file instead. 
            output = pd.read_csv('performance_data.csv')

        # Create the performance plot
        self.create_performance_plot(output)


if __name__ == "__main__":
    # Create a reference vector for testing
    from problem_definition import OptimisationProblem  # type: ignore
    from init_population import InitPopulation  # type: ignore
    from repair import RepairIndividuals  # type: ignore
    import time

    start_time = time.monotonic()
    ref_pop = InitPopulation().GeneratePopulation()
    ref_vectors = ref_pop.get("X")
    ref_vectors = RepairIndividuals()._do(OptimisationProblem(), ref_vectors)
    ref_vector = ref_vectors[0]  # Use the first vector for testing

    rps_values = np.arange(20, 44, 2) 

    tip_angles = np.linspace(np.deg2rad(5), np.deg2rad(25), 4)
    tip_angles = [np.deg2rad(14.5)]

    Analyzer(verbose=True, 
             beta_tip=tip_angles, 
             rps_range=rps_values,
             kwargs={'viscous':True}).analyse_performance(ref_vector, 
                                                          store_to_csv=True)
