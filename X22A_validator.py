"""
X22A_validation
===============

Description
-----------
This file is an implementation of all classes and functions to validate the MTFLOW codes and input generation routines against
experimental wind tunnel data for the X22A ducted propeller powertrain unit, contained in NASA TN-D-4142.

Notes
-----
N/A

References
----------
[1] - https://ntrs.nasa.gov/api/citations/19670025554/downloads/19670025554.pdf
[2] - https://apps.dtic.mil/sti/tr/pdf/AD0447814.pdf?form=MG0AV3
[3] - https://arc.aiaa.org/doi/10.2514/1.C037541

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID 4995309
Version: 1.0

Changelog:
- V1.0: Initial complete validated version.
- V1.1: Added documentation and type hinting.
"""

# Import standard libraries
import os
from pathlib import Path
from contextlib import contextmanager

# Import 3rd party libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from ambiance import Atmosphere

# Enable submodel relative imports
from GA.utils import ensure_repo_paths, get_figsize
ensure_repo_paths()

# Import interfacing modules
from Submodels.Parameterisations import AirfoilParameterisation # type: ignore
from Submodels.file_handling import fileHandlingMTFLO, fileHandlingMTSET # type: ignore
from Submodels.output_handling import output_processing # type: ignore
from Submodels.MTSOL_call import MTSOL_call # type: ignore
from Submodels.MTSET_call import MTSET_call # type: ignore
from Submodels.MTFLO_call import MTFLO_call # type: ignore

# Define key paths/directories
submodels_path = Path(__file__).resolve().parent / "Submodels"

# Define plot formatting
plt.rcParams.update({'font.size': 9, "pgf.texsystem": "xelatex", "text.usetex":  True, "pgf.rcfonts": False})
MAJOR_GRID_ALPHA = 0.6
MINOR_GRID_ALPHA = 0.4

# First we define some constants and the operating conditions which will be analysed
REFERENCE_BLADE_ANGLE = np.array([np.deg2rad(19), np.deg2rad(29)])  # radians, converted from degrees
ANALYSIS_NAME = "X22A_validation"  # Analysis name for MTFLOW
ALTITUDE = 0  # m
FAN_DIAMETER = 7 * 0.3048  # m, taken from [3] and converted to meters from feet
L_REF = FAN_DIAMETER  # m, reference length for use by MTFLOW


def GenerateMTFLOBlading(Omega: float,
                         ref_blade_angle: float,
                         plot : bool,
                         ) -> tuple[list, list]:
    """
    Generate MTFLO blading.
    The blading parameters are based on Figure 3 in [1].

    Parameters
    ----------
    - Omega : float
        The non-dimensional rotational speed of the rotor, as defined in the MTFLOW documentation in units of Vinl/Lref
    - ref_blade_angle : float
        The blade set angle, in radians.
    - plot : bool
        A control boolean to determine whether the propeller blade geometry is plotted.

    Returns
    -------
    - (list, list):
        A tuple of two lists:
        - blading_parameters : list
            A list containing dictionaries with the blading parameters.
        - design_parameters : list
            A list containing dictionaries with the design parameters for each radial station.
    """

    # Start defining the MTFLO blading inputs
    propeller_parameters = {"root_LE_coordinate": 0.1495672948767407, "rotational_rate": Omega, "ref_blade_angle": ref_blade_angle, "reference_section_blade_angle": np.deg2rad(20), "blade_count": 3, "radial_stations": np.array([0,
                                                                                                                                                                                                                      0.2,
                                                                                                                                                                                                                      0.3,
                                                                                                                                                                                                                      0.4,
                                                                                                                                                                                                                      0.5,
                                                                                                                                                                                                                      0.6,
                                                                                                                                                                                                                      0.7,
                                                                                                                                                                                                                      0.8,
                                                                                                                                                                                                                      0.9,
                                                                                                                                                                                                                      1]).astype(float) * FAN_DIAMETER / 2,
                                                                                                                                                                                    "chord_length": np.array([0.3510,
                                                                                                                                                                                                              0.3510,
                                                                                                                                                                                                              0.3152,
                                                                                                                                                                                                              0.2794,
                                                                                                                                                                                                              0.2528,
                                                                                                                                                                                                              0.2413,
                                                                                                                                                                                                              0.2367,
                                                                                                                                                                                                              0.2309,
                                                                                                                                                                                                              0.2251,
                                                                                                                                                                                                              0.2205]).astype(float),
                                                                                                                                                                                    "blade_angle": np.array([np.deg2rad(53.6),
                                                                                                                                                                                                             np.deg2rad(53.6),
                                                                                                                                                                                                             np.deg2rad(46.8),
                                                                                                                                                                                                             np.deg2rad(39.5),
                                                                                                                                                                                                             np.deg2rad(32.3),
                                                                                                                                                                                                             np.deg2rad(26.4),
                                                                                                                                                                                                             np.deg2rad(22.3),
                                                                                                                                                                                                             np.deg2rad(19.1),
                                                                                                                                                                                                             np.deg2rad(16.8),
                                                                                                                                                                                                             np.deg2rad(15.5)]).astype(float)}

    horizontal_strut_parameters = {"root_LE_coordinate": 0.57785, "rotational_rate": 0, "ref_blade_angle": 0, "reference_section_blade_angle": 0, "blade_count": 4, "radial_stations": np.array([0.05,
                                                                                                                                                                                    1]) * 1.15,
                                                                                                                                                                                    "chord_length": np.array([0.57658,
                                                                                                                                                                                                              0.14224]),
                                                                                                                                                                                    "blade_angle": np.array([np.deg2rad(90),
                                                                                                                                                                                                             np.deg2rad(90)]),
                                                                                                                                                                                    "sweep_angle": np.array([0,
                                                                                                                                                                                                             0])}

    diagonal_strut_parameters = {"root_LE_coordinate": 0.577723, "rotational_rate": 0, "ref_blade_angle": 0, "reference_section_blade_angle": 0, "blade_count": 2, "radial_stations": np.array([0.05,
                                                                                                                                                                                    1]) * 1.15,
                                                                                                                                                                                    "chord_length": np.array([0.10287,
                                                                                                                                                                                                              0.10287]),
                                                                                                                                                                                    "blade_angle": np.array([np.deg2rad(90),
                                                                                                                                                                                                             np.deg2rad(90)]),
                                                                                                                                                                                    "sweep_angle": np.array([0,
                                                                                                                                                                                                             0])}

    # Construct blading list
    blading_parameters = [propeller_parameters,
                          horizontal_strut_parameters,
                          diagonal_strut_parameters]

    # Define the sweep angles
    # Note that this is approximate, since the rotation of the chord line is not completely accurate when rotating a complete profile
    sweep_angle = np.zeros_like(blading_parameters[0]["chord_length"])
    root_blade_angle = (np.deg2rad(53.6) + blading_parameters[0]["ref_blade_angle"] - blading_parameters[0]["reference_section_blade_angle"])
    root_rotation_angle = np.pi / 2 - root_blade_angle

    root_LE = blading_parameters[0]["root_LE_coordinate"] # The location of the root LE is arbitrary for computing the sweep angles.
    root_mid_chord = root_LE + (0.3510 / 2) * np.cos(root_rotation_angle)
    for i in range(len(blading_parameters[0]["chord_length"])):
        blade_angle = (blading_parameters[0]["blade_angle"][i] + blading_parameters[0]["ref_blade_angle"] - blading_parameters[0]["reference_section_blade_angle"])
        rotation_angle = np.pi / 2 - blade_angle

        # Compute sweep such that the midchord line is constant.
        local_LE = root_mid_chord - (blading_parameters[0]["chord_length"][i] / 2) * np.cos(rotation_angle)
        if blading_parameters[0]["radial_stations"][i] != 0:
            sweep_angle[i] = np.atan((local_LE - root_LE) / (blading_parameters[0]["radial_stations"][i]))
        else:
            sweep_angle[i] = 0
    blading_parameters[0]["sweep_angle"] = sweep_angle

    # Create plot of the propeller blade
    # Chord lengths are approximate due to the incomplete rotation implementation.
    if plot:
        x_LE_arr = np.zeros_like(blading_parameters[0]["chord_length"])
        x_TE_arr = np.zeros_like(x_LE_arr)
        x_mid_arr = np.zeros_like(x_LE_arr)

        for section in range(len(blading_parameters[0]["radial_stations"])):
            rotation_angle = np.pi/2 - (blading_parameters[0]["blade_angle"][section] + blading_parameters[0]["ref_blade_angle"] - blading_parameters[0]["reference_section_blade_angle"])
            x_LE = root_LE + blading_parameters[0]["radial_stations"][section] * np.tan(blading_parameters[0]["sweep_angle"][section])

            x_TE = x_LE + blading_parameters[0]["chord_length"][section] * np.cos(rotation_angle)
            x_mid_arr[section] = (x_LE + x_TE) / 2
            x_LE_arr[section] = x_LE
            x_TE_arr[section] = x_TE
            plt.plot([x_LE, x_TE], [blading_parameters[0]["radial_stations"][section], blading_parameters[0]["radial_stations"][section]], label=f"r={round(blading_parameters[0]["radial_stations"][section], 2)} m")

        plt.xlabel("Axial Location [m]")
        plt.ylabel("Radial location [m]")
        plt.title("Propeller Blade Input Geometry")
        plt.grid(which='major', alpha=MAJOR_GRID_ALPHA)
        plt.plot(x_LE_arr, blading_parameters[0]["radial_stations"], "-.k")
        plt.plot(x_TE_arr, blading_parameters[0]["radial_stations"], "-.k")
        plt.plot(x_mid_arr, blading_parameters[0]["radial_stations"], "-.k")
        plt.legend(loc='upper left', bbox_to_anchor=(1,1))
        plt.tight_layout()
        plt.show()



    # Obtain the parameterisations for the profile sections.
    local_dir_path = Path(__file__).resolve().parent / 'Validation/Profiles'
    R02_fpath = local_dir_path / 'X22_02R.dat'
    R03_fpath = local_dir_path / 'X22_03R.dat'
    R04_fpath = local_dir_path / 'X22_04R.dat'
    R05_fpath = local_dir_path / 'X22_05R.dat'
    R06_fpath = local_dir_path / 'X22_06R.dat'
    R07_fpath = local_dir_path / 'X22_07R.dat'
    R08_fpath = local_dir_path / 'X22_08R.dat'
    R09_fpath = local_dir_path / 'X22_09R.dat'
    R10_fpath = local_dir_path / 'X22_10R.dat'
    Hstrut_fpath = local_dir_path / 'Hstrut.dat'
    Dstrut_fpath = local_dir_path / 'Dstrut.dat'

    filenames = [R02_fpath, R03_fpath, R04_fpath, R05_fpath, R06_fpath, R07_fpath, R08_fpath, R09_fpath, R10_fpath, Hstrut_fpath, Dstrut_fpath]

    # First check if all files are present
    missing_files = [f for f in filenames if not f.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing files: {', '.join(map(str, missing_files))}")

    # Compute parameterisation for the airfoil section at r=0.2R
    # Note that we keep this section constant for r=0.1R and r=0.15R and equal to that of r=0.2R
    R01_section = AirfoilParameterisation().FindInitialParameterisation(reference_file=R02_fpath)
    # Compute parameterisation for the airfoil section at r=0.3R
    R03_section = AirfoilParameterisation().FindInitialParameterisation(reference_file=R03_fpath)
    # Compute parameterisation for the airfoil section at r=0.4R
    R04_section = AirfoilParameterisation().FindInitialParameterisation(reference_file=R04_fpath)
    # Compute parameterisation for the mid airfoil section
    R05_section = AirfoilParameterisation().FindInitialParameterisation(reference_file=R05_fpath)
    # Compute parameterisation for the airfoil section at r=0.6R
    R06_section = AirfoilParameterisation().FindInitialParameterisation(reference_file=R06_fpath)
    # Compute parameterisation for the airfoil section at r=0.7R
    R07_section = AirfoilParameterisation().FindInitialParameterisation(reference_file=R07_fpath)
    # Compute parameterisation for the airfoil section at r=0.8R
    R08_section = AirfoilParameterisation().FindInitialParameterisation(reference_file=R08_fpath)
    # Compute parameterisation for the airfoil section at r=0.9R
    R09_section = AirfoilParameterisation().FindInitialParameterisation(reference_file=R09_fpath)
    # Compute parameterisation for the tip airfoil section
    R10_section = AirfoilParameterisation().FindInitialParameterisation(reference_file=R10_fpath)
    # Compute parameterisation for the horizontal struts
    Hstrut_section = AirfoilParameterisation().FindInitialParameterisation(reference_file=Hstrut_fpath)
    # Compute parameterisation for the diagonal struts
    Dstrut_section = AirfoilParameterisation().FindInitialParameterisation(reference_file=Dstrut_fpath)

    # Construct blading list
    design_parameters = [[R01_section, R01_section, R03_section, R04_section, R05_section, R06_section, R07_section, R08_section, R09_section, R10_section],
                         [Hstrut_section, Hstrut_section],
                         [Dstrut_section, Dstrut_section]]

    return blading_parameters, design_parameters


def GenerateMTFLOInput(blading_parameters: list,
                       design_parameters: list,
                       display_plot: bool) -> None:
    """
    Generate the MTFLO input file tflow.X22A_validation

    Parameters
    ----------
   - blading_parameters : list
        A list containing dictionaries with the blading parameters.
    - design_parameters : list
        A list containing dictionaries with the design parameters for each radial station.
    - display_plot : bool
        A control boolean to determine whether the blade data input for MTFLO is plotted.

    Returns
    -------
    None
    """

    fileHandlingMTFLO(analysis_name=ANALYSIS_NAME,
                      ref_length=L_REF).GenerateMTFLOInput(blading_params=blading_parameters,
                                                           design_params=design_parameters,
                                                           plot=display_plot)


def GenerateMTSETGeometry() -> None:
    """
    Generate the duct and center body geometry. Uses a combination of analytical representation, and smoothing interpolations to obtain the axisymmetric geometries.
    """

    # --------------------
    # Generate duct geometry
    # --------------------

    # First define the upper surface based on the given coordinates in [1]
    # Note that we must flip the arrays to comply with the formatting expected by MTFLOW (TE-LE-TE)
    # Dimensions are originally provided in inches, so they are converted to meters
    upper_x = np.flip(np.array([0, 0.613, 1.225, 2.45, 3.675, 4.9, 7.35, 9.8, 10.25, 12.25, 14.7, 17.75, 19.6, 23.7, 24.5, 29.4, 34.3, 39.2, 44.1, 46.55, 49])) * 2.54 / 100  # x coordinates of the duct upper surface
    upper_y = np.flip(np.array([47.625, 48.695, 49.096, 49.609, 49.953, 50.205, 50.535, 50.710, 50.745, 50.779, 50.763, 50.6575, 50.552, 50.25, 50.164, 49.649, 49.038, 48.344, 47.576, 47.160, 46.722])) * 2.54 / 100  # y coordinates of the duct upper surface

    # The lower surface must be reconstructed from parts defined by the shapes and coordinates given in [1]
    lower_x = np.array([])
    lower_y = np.array([])

    # The nose is represented by a ellipse for which the points are given as:
    P1 = (0, 47.625)  # Vertex 1
    P2 = (10.25, 42.365)  # Vertex 2
    C = (10.25, 47.625)  # Center

    # We can then calculate the semi-major and semi-minor axes of the ellipse as:
    a = P2[0]
    b = C[1] - P2[1]
    h, k = C

    # Construct an array of x values along which to reconstruct the leading edge nose shape,
    # and use it to calculate the corresponding y values
    # Both x_nose and y_nose are still in inches
    x_nose = np.linspace(P1[0], C[0], 30)
    y_nose = (k - b * np.sqrt(1 - ((x_nose - h) ** 2 / a ** 2)))

    lower_x = np.append(lower_x, x_nose)
    lower_y = np.append(lower_y, y_nose)

    # Raw approximated data for aft section of the lower surface
    # Data taken from a graph digitized by Bram Meijerink.
    # Note that the y values are not matching the graph from [1], so the data is scaled.
    # After scaling, a smoothing interpolation using a univariatespline is used to create a more physical curve.
    aft_x_raw = np.array([17.75, 21.4, 24.2, 28, 31.1, 33.4, 35.9, 38, 39.6, 41.7, 43.9, 45.5, 47.2, 48.4, 49])
    aft_y_raw = np.array([42.375, 42.4562, 42.6226, 42.8506, 43.1218, 43.3298, 43.5878, 43.7874, 43.9954, 44.2618, 44.4946, 44.661, 44.869, 44.9938, 45.05])
    scale_factor = (46.65 - lower_y[-1]) / (aft_y_raw[-1] - lower_y[-1])
    aft_y_raw_scaled = lower_y[-1] + scale_factor * (aft_y_raw - lower_y[-1])

    # Smooth the aft y data using a univariatespline
    x = np.linspace(aft_x_raw[0], aft_x_raw[-1], 30)
    aft_y_smoothed = interpolate.UnivariateSpline(aft_x_raw, aft_y_raw_scaled, s=0.01)(x)

    # Construct full lower surface array and convert from inches to meters
    lower_x = np.append(lower_x, x) * 2.54 / 100
    lower_y = np.append(lower_y, aft_y_smoothed) * 2.54 / 100

    # Transform the xy arrays for the duct to the correct format
    x_duct = np.concatenate((upper_x, lower_x[1:]), axis=0)
    y_duct = np.concatenate((upper_y, lower_y[1:]), axis=0)
    xy_duct = np.vstack((x_duct, y_duct)).T

    # --------------------
    # Generate centre body geometry
    # --------------------

    # Data taken from [1]
    centerbody_x = np.flip(np.array([66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 43, 40, 34, 32, 30, 25, 22, 19, 15, 11, 6.3, 2.3, 0.7, 0.5, 0]) * 2.54 / 100) - 3.67 * 2.54 / 100
    centerbody_y = np.flip(np.array([1.72, 1.72, 1.73, 1.74, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 4.2, 4.4, 4.6, 4.8, 5, 5.2, 5.6, 6.2, 7.1, 7.8, 8.5, 8.7, 8.2, 7.5, 6.5, 5.6, 4.4, 3.3, 1.7, 1.0, 0]) * 2.54 / 100)

    # Perform smoothing interpolation on the centerbody geometry
    interpolated_centerbody_x = centerbody_x[0] + ((1 - np.cos(np.linspace(0, np.pi, 30))) / 2) * (centerbody_x[-1] - centerbody_x[0])  #  cosine spacing for increased resolution at LE and TE


    interpolated_centerbody_y = interpolate.UnivariateSpline(centerbody_x,
                                                             centerbody_y,
                                                             k=3,
                                                             s=5
                                                             )(interpolated_centerbody_x)


    CENTERBODY_VALUES = {"b_0": 0.05, "b_2": 0.125, "b_8": 7.52387039e-02, "b_15": 7.46448823e-01, "b_17": 0.8, 'x_t': 0.29842005729819904, 'y_t': 0.12533559300869632, 'x_c': 0.3, 'y_c': 0., 'z_TE': 0., 'dz_TE': 0.00277173368735548, 'r_LE': -0.06946118699675888, 'trailing_wedge_angle': 0.27689037361278407, 'trailing_camberline_angle': 0., 'leading_edge_direction': 0., "Chord Length": 1.5, "Leading Edge Coordinates": (0., 0.)}
    
    plt.figure("cb geometry", figsize=get_figsize(wf=0.75))
    naca_x, naca_y, _, _, = AirfoilParameterisation().ComputeProfileCoordinates(CENTERBODY_VALUES)
    naca_x *= CENTERBODY_VALUES["Chord Length"]
    naca_y *= CENTERBODY_VALUES["Chord Length"]

    plt.plot((centerbody_x[:14] - centerbody_x[0]), centerbody_y[:14] , "x", label="True profile coordinates")
    plt.plot((interpolated_centerbody_x - interpolated_centerbody_x[0]), interpolated_centerbody_y , "-.", label="Smoothed representation")
    plt.plot(naca_x, naca_y, label="NACA 0025 representation")
    plt.xlabel("Axial coordinate $x$ [m]")
    plt.ylabel("Radial coordinate $z$ [m]")
    plt.grid(which='major', alpha=MAJOR_GRID_ALPHA)
    plt.grid(which='minor', linewidth=0.25, alpha=MINOR_GRID_ALPHA)
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.ylim(bottom=0)
    plt.show()



    # Transform the data to the correct format
    # Ensures leading edge data point only occurs once to make sure a smooth spline is constructed, in accordance with the MTFLOW documentation.
    centerbody_x_complete = np.concatenate((np.flip(interpolated_centerbody_x), interpolated_centerbody_x[1:]), axis=0)
    centerbody_y_complete = np.concatenate((np.flip(interpolated_centerbody_y), -interpolated_centerbody_y[1:]), axis=0)
    xy_centerbody = np.vstack((centerbody_x_complete, centerbody_y_complete)).T

    # Generate MTSET input file walls.X22A_validation
    # To run the GenerateMTSETInput() function, we need to define the params_CB and params_duct dictionaries, so fill them with the minimum required inputs
    params_CB = {"Leading Edge Coordinates": (centerbody_x.min(),0), "Chord Length": (lower_x.max() - lower_x.min())}
    params_duct = {"Leading Edge Coordinates": (x_duct.min(), upper_y[-1]), "Chord Length": (lower_x.max() - lower_x.min())}

    # Generate the walls.x22a_validation
    fileHandlingMTSET(params_CB=params_CB,
                      params_duct=params_duct,
                      analysis_name=ANALYSIS_NAME,
                      ref_length=L_REF,
                      external_input=True).GenerateMTSETInput(xy_centerbody=xy_centerbody,
                                                              xy_duct=xy_duct)


def ChangeOMEGA(omega: float) -> None:
    """
    Rather than generating the tflow file for each advance ratio from scratch,
    simply change omega in the tflow file.

    Parameters
    ----------
    - omega : float
        The non-dimensional rotational speed to be entered into the tflow input file.
    """

    # Open the tflow.analysis_name file
    tflow_path = Path(f"tflow.{ANALYSIS_NAME}")
    with tflow_path.open("r") as file:
        lines = file.readlines()

    omega_line = 11  # Line at which the rotational rate is defined
    updated_omega = f"{omega} \n"
    lines[omega_line] = updated_omega

    # Write the updated tflow data back to the file
    with tflow_path.open("w") as file:
        file.writelines(lines)


@contextmanager
def pushd(path: Path):
    """
    Context manager to handle the required change in working directory.

    Parameters
    ----------
    - path : Path
        The path to which the working directory must be changed.
    """

    original_path = Path.cwd()
    if not path.is_dir():
        raise FileNotFoundError(f"Cannot change directory - {path} does not exist")
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_path)


def ExecuteParameterSweep(omega: np.typing.NDArray[np.floating],
                          inlet_mach: np.typing.NDArray[np.floating],
                          reynolds_inlet: np.typing.NDArray[np.floating],
                          reference_angle: float,
                          generate_plots: bool = False,
                          streamwise_points: int = 400,
                          ) -> tuple[np.typing.NDArray[np.floating], np.typing.NDArray[np.floating], np.typing.NDArray[np.floating]]:
    """
    Perform a parameter sweep over a range of OMEGA, inlet Mach numbers, and Reynolds numbers.

    Parameters
    ----------
    - omega : np.typing.NDArray[np.floating]
        Array of non-dimensional rotational speeds of the rotor.
    - inlet_mach : np.typing.NDArray[np.floating]
        Array of inlet Mach numbers.
    - reynolds_inlet : np.typing.NDArray[np.floating]
        Array of inlet Reynolds numbers.
    - reference_angle : float
        The set angle of the propeller blade
    - generate_plots : bool, optional
        If true, generates plots of the input geometry and rotor input data. Default is False.
    - streamwise_points : int, optional
        Number of streamwise points for the grid generation. Default is 400.

    Returns
    -------
    - tuple
        - CT_outputs : np.typing.NDArray[np.floating]
            Array of thrust coefficients.
        - CP_outputs : np.typing.NDArray[np.floating]
            Array of power coefficients.
        - EtaP_outputs : np.typing.NDArray[np.floating]
            Array of propulsive efficiencies.
    """

    # Create the MTSET geometry and write the input file walls.ANALYSIS_NAME
    GenerateMTSETGeometry()

    # Construct the MTFLO blading using omega=0 and reference blade angle.
    # Perform parameterisation can be optionally set to true in case different profiles are used compared to the default inputs
    blading_parameters, design_parameters = GenerateMTFLOBlading(Omega=0,
                                                                 ref_blade_angle=reference_angle,
                                                                 plot=generate_plots)

    # Change working directory to the submodels folder
    with pushd(submodels_path):
        # Generate the MTFLO input file
        GenerateMTFLOInput(blading_parameters,
                        design_parameters,
                        display_plot=generate_plots)

        # Perform analysis for all omega, Mach, and Re combinations defined at the top of the file
        CT_outputs = np.zeros_like(omega)
        CP_outputs = np.zeros_like(omega)
        EtaP_outputs = np.zeros_like(omega)

        for i in range(len(omega)):
            # Create the grid
            MTSET_call(analysis_name=ANALYSIS_NAME,
                    streamwise_points=streamwise_points,
                    ).caller()

            # Update the blade parameters to the correct omega
            ChangeOMEGA(omega[i])

            #Load in the blade row(s) from MTFLO
            MTFLO_call(ANALYSIS_NAME).caller()

            # Define operating conditions
            oper = {"Inlet_Mach": inlet_mach[i],
                    "Inlet_Reynolds": reynolds_inlet[i],
                    "N_crit": 9,
                    }

            # Execute MTSOL
            MTSOL_call(operating_conditions=oper,
                    analysis_name=ANALYSIS_NAME,
                    ).caller(run_viscous=True,
                                generate_output=True)

            # Collect outputs from the forces.xxx file
            CT, CP, etaP = output_processing(ANALYSIS_NAME).GetCTCPEtaP()
            print(f"Omega: {omega[i]}, CT: {CT}, CP: {CP}, etaP: {etaP}")

            CT_outputs[i] = CT
            CP_outputs[i] = CP
            EtaP_outputs[i] = etaP

    return CT_outputs, CP_outputs, EtaP_outputs


if __name__ == "__main__":
    # Advance ratio range to be used for the validation, together with freestream velocity.
    J = np.array([0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3])  # -
    FREESTREAM_VELOCITY = np.ones_like(J) * 26  # m/s, tweaked to get acceptable values of RPS/OMEGA for the advance ratio range considered.

    # Compute the rotational speed of the rotor in rotations per second
    RPS = FREESTREAM_VELOCITY / (J * FAN_DIAMETER)  # Hz
    print(f"RPM (Should be between 1200-2590 RPM) [-]: {RPS * 60}")

    # Use the calculated rotational speed to obtain the non-dimensional Omega used as input into MTFLOW
    OMEGA = -(RPS * 2 * np.pi) * L_REF / FREESTREAM_VELOCITY
    print(f"Omega [-]: {OMEGA}")

    # Construct atmosphere object to obtain the atmospheric properties at the cruise altitude
    # These properties can then be used to compute the inlet mach number and reynolds number
    atmosphere = Atmosphere(ALTITUDE)
    inlet_mach = (FREESTREAM_VELOCITY / atmosphere.speed_of_sound)
    print(f"Mach [-]: {inlet_mach}")
    reynolds_inlet = (FREESTREAM_VELOCITY * L_REF / (atmosphere.kinematic_viscosity))
    print(f"Reynolds [-]: {reynolds_inlet}")

    # Check the dynamic pressure range
    dyn_press = atmosphere.density * FREESTREAM_VELOCITY ** 2 / 2  # Pa
    dyn_press = dyn_press * 0.0208854  # Convert to psf
    print(f"Dynamic pressure (Should be < 106 psf) [psf]: {dyn_press}")

    # Initialize output dictionaries and perform parameter sweep.
    nrows = len(J)
    ncols = len(REFERENCE_BLADE_ANGLE)
    CT = np.zeros((ncols, nrows))
    CP = np.zeros((ncols, nrows))
    etaP = np.zeros((ncols, nrows))

    for i in range(len(REFERENCE_BLADE_ANGLE)):
        print(f"Analysing beta_{75}={round(np.rad2deg(REFERENCE_BLADE_ANGLE[i]), 2)} deg")
        CT_out, CP_out, eta_out = ExecuteParameterSweep(omega=OMEGA,
                                                        inlet_mach=inlet_mach,
                                                        reynolds_inlet=reynolds_inlet,
                                                        reference_angle=REFERENCE_BLADE_ANGLE[i],
                                                        generate_plots=False,
                                                        streamwise_points=200)
