"""
UDFDAC.py
=================

Description
-----------
This module provides a complete interface with the MTFLOW programs, from input 
file generation through to output generation, including convergence handling, 
choking handling, crash detection, and grid refinement in the form of the Unified Ducted Fan Design and Analysis Code (UDFDAC). 
It builds on the MTSET_call, MTFLO_call, and MTSOL_call classes to construct the complete interface.
The module is able to maintain a logging file to allow for debugging through the 
use of an optional control boolean. 

Classes
-------
UDFDAC : class
    A class to execute the complete MTFLOW interface. Effectively a detailed "wrapper" class of the MTSET_call, MTFLO_call, 
    and MTSOL_call classes. 

Examples
--------
>>> import time

>>> oper = {"Inlet_Mach": 0.6,
>>>         "Inlet_Reynolds": 5e6,
>>>         "N_crit": 9,
>>>         }

>>> analysisName = "test_case"
    
>>> # Roughly basing the blade design on the CFM Leap engine (approximate chord lengths)
>>> blading_parameters = [{"root_LE_coordinate": 0.5, "rotational_rate": 0.75, "blade_count": 15, "radial_stations": [0.1, 1.15], "chord_length": [0.3, 0.2], "sweep_angle":[np.pi/16, np.pi/16], "blade_angle": [0, np.pi/8]},
>>>                       {"root_LE_coordinate": 1.1, "rotational_rate": 0., "blade_count": 15, "radial_stations": [0.1, 1.3], "chord_length": [0.15, 0.1], "sweep_angle":[np.pi/16, np.pi/16], "blade_angle": [0, np.pi/8]}]
    
>>> # Model the fan and stator blades using a uniform naca2415 profile along the blade span
>>> n2415_coeff = {"b_0": 0.20300919575972556, "b_2": 0.31901972386590877, "b_8": 0.04184620466207193, "b_15": 0.7500824561993612, "b_17": 0.6789808614463232, "x_t": 0.298901583, "y_t": 0.060121131, "x_c": 0.40481558571382253, "y_c": 0.02025376839986754, "z_TE": -0.0003399582707130648, "dz_TE": 0.0017, "r_LE": -0.024240593156029916, "trailing_wedge_angle": 0.16738688797915346, "trailing_camberline_angle": 0.0651960639817597, "leading_edge_direction": 0.09407653642497815}
>>> design_parameters = [[n2415_coeff, n2415_coeff],
>>>                      [n2415_coeff, n2415_coeff]]
    
>>> # Model the duct using a naca 6409 profile
>>> duct_parameters = {"b_0": 0.07979831, "b_2": 0.20013347, "b_8": 0.02901246, "b_15": 0.74993802, "b_17": 0.78496242, 'x_t': 0.30429947838135246, 'y_t': 0.0452171520304373, 'x_c': 0.4249653844429819, 'y_c': 0.06028051002570214, 'z_TE': -0.0003886462495685791, 'dz_TE': 0.0004425237127035188, 'r_LE': -0.009225474218611841, 'trailing_wedge_angle': 0.10293203348896998, 'trailing_camberline_angle': 0.21034003141636426, 'leading_edge_direction': 0.26559481057525414, "Chord Length": 3.0, "Leading Edge Coordinates": (0, 1.2)}
    
>>> # Model the centrebody using a naca 0025 profile
>>> centrebody_parameters = {"b_0": 0., "b_2": 0., "b_8": 7.52387039e-02, "b_15": 7.46448823e-01, "b_17": 0, 'x_t': 0.29842005729819904, 'y_t': 0.12533559300869632, 'x_c': 0, 'y_c': 0, 'z_TE': 0, 'dz_TE': 0.00277173368735548, 'r_LE': -0.06946118699675888, 'trailing_wedge_angle': np.float64(0.27689037361278407), 'trailing_camberline_angle': 0.0, 'leading_edge_direction': 0.0, "Chord Length": 4, "Leading Edge Coordinates": (0.3, 0)}

>>> start_time = time.time()
>>> class_call = UDFDAC(operating_conditions=oper,
>>>                     centrebody_params=centrebody_parameters,
>>>                     duct_params=duct_parameters,
>>>                     blading_parameters=blading_parameters,
>>>                     design_parameters=design_parameters,
>>>                     analysis_name=analysisName,
>>>                     ).caller()
>>> end_time = time.time()
    
>>> print(f"Execution of UDFDAC.caller() took {end_time - start_time} second")

Notes
-----
N/A

References
----------
The required input data, limitations, and structures are documented within the MTFLOW user manual:
https://web.mit.edu/drela/Public/web/mtflow/mtflow.pdf

[1] https://www.easa.europa.eu/sites/default/files/dfu/EASA_E110_TCDS_Issue_13_LEAP-1A-1C.pdf?form=MG0AV3

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID 4995309
Version: 1.5

Changelog:
- V1.0: Initial version. Lacks proper crash handling and choking handling in the HandleExitFlag() method, but otherwise complete. 
- V1.1: Cleaned up following successful implementation of validation case. Crash handling is now done within MTSOL_call. 
- V1.2: Cleaned up import statements, resolved infite loop issue in gridtest, and added MTSOL loop exit for non-convergence. 
- V1.3: Removed HandleExitFlag() method as it is not needed. Extracted choking handling to a separate method.
- V1.4: Cleaned up imports. Implemented chdir context manager. Switched to pathlib for path operations. Cleaned up/streamlined exit flags. Implemented OutputType enum class. 
- V1.5: Updated to remove iter_count output from MTSOl_call. Updated documentation. Reduced no of streamwise points to 150 to reduce computational cost. Adopted named constants for grid coefficients and streamwise points. 
- V2.0: Renamed MTFLOW_caller to UDFDAC for consistency with written thesis. Updated to reflect new structure.
"""

# Import standard libraries
import os
from contextlib import contextmanager
from pathlib import Path

# Import 3rd party libraries
import numpy as np

# Import interfacing modules
from Submodels.MTSET_call import MTSET_call # type: ignore 
from Submodels.MTFLO_call import MTFLO_call # type: ignore 
from Submodels.MTSOL_call import MTSOL_call, ExitFlag, OutputType # type: ignore 
from Submodels.file_handling import fileHandlingMTSET, fileHandlingMTFLO # type: ignore 


@contextmanager
def change_working_directory(dir: Path):
    """
    Context manager to temporarily change the working directory.

    Parameters
    ----------
    - dir : Path
        Path to which the working directory needs to be changed.
    """

    current_dir = Path.cwd()
    try:
        os.chdir(dir)
        yield
    finally:
        os.chdir(current_dir)


class UDFDAC:
    """
    Wrapper class to execute the complete MTFLOW evaluation cycle.
    This class handles the setup, execution, and error handling for the MTFLOW evaluation process, which includes generating input files, constructing grids, and running solvers.
    """

    INITIAL_GRID_E_COEFF = 0.8
    INITIAL_GRID_X_COEFF = 0.8
    INITIAL_STREAMWISE_POINTS = 150


    def __init__(self,
                 operating_conditions: dict,
                 ref_length: float,
                 analysis_name: str,
                 run_viscous: bool = True,
                 **kwargs
                 ) -> None:
        """
        Initialize the UDFDAC class.

        This method sets up the initial state of the class.

        Parameters
        ----------
        - operating_conditions : dict
            A dictionary containing at least the following entries: Inlet_Mach, Inlet_Reynolds, N_crit, i.e. the inlet Mach number, Reynolds number, and critical amplification factor N       
        - ref_length : float
            Reference length used to non-dimensionalise all geometric parameters. By convention, this is equal to the fan diameter. 
        - analysis_name : str
            String of the casename
        - run_viscous : bool, optional
            Optional control boolean to determine if a viscous or inviscid analysis should be performed. Default is True.
        - **kwargs : dict, optional
            Additional keyword arguments.
            - seed
            - centrebody_params : dict
                Dictionary containing parameters for the centerbody.
            - duct_params : dict
                Dictionary containing parameters for the duct.
            - blading_parameters : list[dict]
                List containing the blading parameters for each stage. Each dictionary should include the following keys:
                    - "root_LE_coordinate": The leading edge coordinate at the root of the blade.
                    - "rotational_rate": The rotational rate of the blade.
                    - "blade_count": The number of blades.
                    - "radial_stations": List of the radial stations along the blade span.
                    - "chord_length": List of the chord length distribution along the blade span.
                    - "sweep_angle": List of the sweep angle distribution along the blade span.
                    - "blade_angle": List of the blade angle distribution along the blade span.
                    - "ref_blade_angle": The measured set angle at the reference section (in radians).
                    - "reference_section_blade_angle": The actual reference blade angle at the reference section (in radians).
            - design_parameters : list[list[dict]]
                List containing an equal number of nested lists as there are stages. Each nested list contains an equal number of dictionaries as there are radial stations. 
                Each dictionary must contain the following keys:
                    - "b_0", "b_2", "b_8", "b_15", "b_17": Coefficients for the airfoil parameterisation.
                    - "x_t", "y_t", "x_c", "y_c": Coordinates for the airfoil parameterisation.
                    - "z_TE", "dz_TE": Trailing edge parameters.
                    - "r_LE": Leading edge radius.
                    - "trailing_wedge_angle": Trailing wedge angle.
                    - "trailing_camberline_angle": Trailing camberline angle.
                    - "leading_edge_direction": Leading edge direction.
                    - "Chord Length": The chord length of the blade.
        """

        # Unpack class inputs
        self.operating_conditions = operating_conditions
        self.analysis_name = analysis_name
        self.ref_length = ref_length

        # Define key submodels directory
        self.submodels_path = Path(__file__).resolve().parent / "Submodels"

        # Define control boolean for the viscous analysis
        self.run_visc = run_viscous


    def caller(self,
               external_inputs: bool = False,
               output_type: OutputType = OutputType.FORCES_ONLY,
               grid_checked: bool = False,
               **kwargs
               ) -> ExitFlag:
        """ 
        Executes a complete MTSET-MTFLO-MTSOL evaluation, while handling grid issues and choking issues. 

        Parameters
        ----------
        - external_inputs : bool, optional
            A boolean controlling the generation of the MTFLO and MTSET input files. If true, assumes walls.analysis_name and tflow.analysis_name have been generated outside of UDFDAC. 
            This is useful for debugging or validation against existing, external data. 
        - output_type : OutputType, optional
            An enum to determine which output files to generate. OutputType.FORCES_ONLY generates only the forces file, while OutputType.ALL_FILES generates all files.
        - grid_checked : bool, optional
            A boolean to indicate if the grid for the case being run has been checked already. This speeds up batch analyses as it skips the grid checking routine. 

        Returns
        -------
        - ExitFlag
            The total exit flag of the MTFLOW analysis
        """
            
        # --------------------
        # Change working directory to the submodels folder using the context manager.
        # Execute all code within the context manager
        # --------------------

        with change_working_directory(self.submodels_path):
            # --------------------
            # First step is generating the MTSET input file - walls.analysis_name if needed
            # --------------------

            # Check if all required inputs are present in case the inputs need to be constructed            
            if not external_inputs:
                required_kwargs = ("centrebody_params","duct_params","blading_parameters","design_parameters")
                missing = [k for k in required_kwargs if kwargs.get(k) is None]
                if missing:
                    raise ValueError(f"Missing mandatory kwargs for MTSET/MTFLO input: {', '.join(missing)}")
                
                self.centrebody_params = kwargs.get('centrebody_params')
                self.duct_params = kwargs.get('duct_params')
                self.blading_parameters = kwargs.get('blading_parameters')
                self.design_parameters = kwargs.get('design_parameters')
                fileHandlingMTSET(params_CB=self.centrebody_params,
                                  params_duct=self.duct_params,
                                  case_name=self.analysis_name,
                                  ref_length=self.ref_length).GenerateMTSETInput()
                
            # --------------------
            # Check the grid by running a simple, fan-less, inviscid low-Mach case. If there is an issue with the grid MTSOL will crash
            # Hence we can check the grid by checking the exit flag
            # --------------------
                
            # Initialize count of grid checks and exit flag
            check_count = 0
            exit_flag = ExitFlag.NOT_PERFORMED
                
            while exit_flag != ExitFlag.SUCCESS:                    
                # If the grid is incorrect, change grid parameters and rerun MTSET to update the grid. 
                # The updated e and x coefficients reduce the number of streamwise points on the airfoil elements (by 0.1 * Npoints), 
                # while yielding a more "rounded/elliptic" grid due to the reduced x-coefficient.
                if check_count == 0:
                    # For the initial attempt, use the default values, but with an increased streamwise resolution
                    streamwise_points = self.INITIAL_STREAMWISE_POINTS
                    grid_e_coeff = self.INITIAL_GRID_E_COEFF
                    grid_x_coeff = self.INITIAL_GRID_X_COEFF
                elif check_count == 1:
                    # Revert back to the default number of streamwise points - this can help reduce likeliness of self-intersecting grid   
                    streamwise_points = self.INITIAL_STREAMWISE_POINTS  
                    grid_e_coeff = 0.7
                    grid_x_coeff = 0.9
                else: 
                    exit_flag = ExitFlag.CRASH  # If the grid is still incorrect after the repair attempt, we assume that the grid is not fixable and exit the loop
                    break
                                    
                # Generate the grid
                MTSET_call(analysis_name=self.analysis_name,
                           grid_e_coeff=grid_e_coeff,
                           grid_x_coeff=grid_x_coeff,
                           streamwise_points=streamwise_points).caller()
                
                if grid_checked:
                    exit_flag = ExitFlag.SUCCESS
                    break
                
                exit_flag  = MTSOL_call(operating_conditions={"Inlet_Mach": 0.15, "Inlet_Reynolds": 0., "N_crit": self.operating_conditions["N_crit"]},
                                                 analysis_name=self.analysis_name).caller(run_viscous=False,
                                                                                          generate_output=False)
                                
                check_count += 1

            # --------------------
            # Generate the MTFLO input file tflow.analysis_name if needed
            # --------------------
            if not external_inputs and exit_flag != ExitFlag.CRASH:
                fileHandlingMTFLO(case_name=self.analysis_name,
                                  ref_length=self.ref_length).GenerateMTFLOInput(blading_params=self.blading_parameters,
                                                                                 design_params=self.design_parameters)
            
            # --------------------
            # Execute MTSOl solver
            # Passes the exit flag to determine if any issues have occurred. 
            # --------------------
            if exit_flag != ExitFlag.CRASH:   
                try:    
                    MTFLO_call(self.analysis_name).caller() #Load in the blade row(s) from MTFLO
                except Exception as e:
                    # If MTFLO fails, set the exit flag to crash so that we don't execute the MTSOL evaluation.
                    exit_flag = ExitFlag.CRASH
                    print(f"MTFLO failed: {e}")

                if exit_flag != ExitFlag.CRASH:
                    # Execute MTSOL    
                    exit_flag = MTSOL_call(operating_conditions=self.operating_conditions,
                                        analysis_name=self.analysis_name).caller(run_viscous=self.run_visc,
                                                                                generate_output=True,
                                                                                output_type=output_type)
        return exit_flag


if __name__ == "__main__":
    import time
    from ambiance import Atmosphere

    fan_diameter = 2.3  # meters

    # Collect sea level atmosphere properties and calculate the mach number and reynolds number at the inlet
    atmosphere = Atmosphere(0)
    speed = 200  # m/s

    Inlet_Reynolds = (speed * fan_diameter / (atmosphere.kinematic_viscosity))[0]
    Inlet_Mach = (speed / atmosphere.speed_of_sound)[0]

    # Construct the operating conditions dictionary
    oper = {"Inlet_Mach": Inlet_Mach,
            "Inlet_Reynolds": Inlet_Reynolds,
            "N_crit": 9,
            }

    analysisName = "test_case"

    # Construct the non-dimensional RPM for the rotor
    RPM = 3894  # max rpm of the low speed spool of the CFM LEAP-1A engine according to [1]
    RPS = RPM / 60  # Hz
    Omega = -RPS * np.pi * 2 * fan_diameter / speed  # Non-dimensional RPM
    
    # Roughly basing the blade design on the CFM Leap engine (approximate chord lengths)
    blading_parameters = [{"root_LE_coordinate": 0.5, "rotational_rate": Omega, "blade_count": 15, "ref_blade_angle": np.deg2rad(29), "reference_section_blade_angle": np.deg2rad(19), "radial_stations": [0.1, 1.15], "chord_length": [0.3, 0.2], "sweep_angle":[np.pi/16, np.pi/16], "blade_angle": [0, np.pi/8]},
                          {"root_LE_coordinate": 1.1, "rotational_rate": 0., "blade_count": 15, "ref_blade_angle": np.deg2rad(-29), "reference_section_blade_angle": np.deg2rad(-19),"radial_stations": [0.1, 1.3], "chord_length": [0.15, 0.1], "sweep_angle":[np.pi/16, np.pi/16], "blade_angle": [0, np.pi/8]}]
    
    # Model the fan and stator blades using a uniform naca2415 profile along the blade span
    n2415_coeff = {"b_0": 0.20300919575972556, "b_2": 0.31901972386590877, "b_8": 0.04184620466207193, "b_15": 0.7500824561993612, "b_17": 0.6789808614463232, "x_t": 0.298901583, "y_t": 0.060121131, "x_c": 0.40481558571382253, "y_c": 0.02025376839986754, "z_TE": -0.0003399582707130648, "dz_TE": 0.0017, "r_LE": -0.024240593156029916, "trailing_wedge_angle": 0.16738688797915346, "trailing_camberline_angle": 0.0651960639817597, "leading_edge_direction": 0.09407653642497815}
    design_parameters = [[n2415_coeff, n2415_coeff],
                         [n2415_coeff, n2415_coeff]]
    
    # Model the duct using a naca 6409 profile
    duct_parameters = {"b_0": 0.07979831, "b_2": 0.20013347, "b_8": 0.02901246, "b_15": 0.74993802, "b_17": 0.78496242, 'x_t': 0.30429947838135246, 'y_t': 0.0452171520304373, 'x_c': 0.4249653844429819, 'y_c': 0.06028051002570214, 'z_TE': -0.0003886462495685791, 'dz_TE': 0.0004425237127035188, 'r_LE': -0.009225474218611841, 'trailing_wedge_angle': 0.10293203348896998, 'trailing_camberline_angle': 0.21034003141636426, 'leading_edge_direction': 0.26559481057525414, "Chord Length": 3.0, "Leading Edge Coordinates": (0, 1.2)}
    
    # Model the centrebody using a naca 0025 profile
    centrebody_parameters = {"b_0": 0., "b_2": 0., "b_8": 7.52387039e-02, "b_15": 7.46448823e-01, "b_17": 0, 'x_t': 0.29842005729819904, 'y_t': 0.12533559300869632, 'x_c': 0, 'y_c': 0, 'z_TE': 0, 'dz_TE': 0.00277173368735548, 'r_LE': -0.06946118699675888, 'trailing_wedge_angle': np.float64(0.27689037361278407), 'trailing_camberline_angle': 0.0, 'leading_edge_direction': 0.0, "Chord Length": 4, "Leading Edge Coordinates": (0.3, 0)}

    start_time = time.time()
    solver = UDFDAC(operating_conditions=oper,
                    ref_length=fan_diameter,
                    analysis_name=analysisName)

    exit_flag = solver.caller(external_inputs=False,
                              centrebody_params=centrebody_parameters,
                              duct_params=duct_parameters,
                              blading_parameters=blading_parameters,
                              design_parameters=design_parameters)
    end_time = time.time()

    print(f"Execution of UDFDAC.caller() took {end_time - start_time} second")
