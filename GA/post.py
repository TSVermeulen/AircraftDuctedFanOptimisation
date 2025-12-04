"""
post
====

Description
----------
This module provides post-processing functionality for analyzing and
visualising the results of a Pymoo optimisation. It includes methods for
loading optimization results, extracting population data, and generating
comparative plots for axisymmetric geometries, blading data,
and blade design profiles.

Classes
--------
PostProcessing
    A class to handle post-processing of optimization results,
    including data extraction and visualization.

Examples
--------
>>> output = Path('res_pop20_gen20_250506220150593712.dill')
>>> processing_class = PostProcessing(fname=output)
>>> res = processing_class.load_res()
>>> processing_class.ExtractPopulationData(res)
>>> processing_class.main()

Notes
-----
This module assumes the presence of a Pymoo optimization results file in `.dill`
format. It integrates with various utility modules for handling design vectors,
parameterisations, and file operations. Ensure that the required dependencies
and configuration files are correctly set up before using this module.

References
----------
For more details on the Pymoo framework, refer to the official documentation:
https://pymoo.org/

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@tudelft.nl
Version 2.2

Changelog:
- V1.0: Initial implementation of plotting capabilities of outputs.
- V1.1: Added convergence property plotter. Added 3D blade geometry plotting
        capability.
- V2.0: Significantly expanded functionality. Implemented multi-analysis
        comparison. Updated formatting of graphs. Implemented uniform
        linestyle/color/marker definitions.
- V2.1: Expanded objective space plotting capabilities to enable cumilative
        objective space plotting from multiple analyses. Updated formatting.
- V2.2: Updated formatting to work for double-column layouts.
"""

# Import standard libraries
import copy
from pathlib import Path
from typing import Any

# Import 3rd party libraries
import dill
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter, FuncFormatter, MaxNLocator
from pymoo.visualization.scatter import Scatter
from pymoo.visualization.pcp import PCP
from pymoo.util.running_metric import RunningMetric
from pymoo.core.result import Result
from scipy.spatial.distance import pdist

# Ensure all paths are correctly setup
from utils import ensure_repo_paths
ensure_repo_paths()

# Import interfacing modules
import config # type: ignore
from Submodels.Parameterisations import AirfoilParameterisation # type: ignore
from design_vector_interface import DesignVectorInterface # type: ignore
from Submodels.file_handling import fileHandlingMTFLO #type: ignore
from utils import get_figsize #type: ignore
from init_population import InitPopulation # type: ignore
from objectives import Objectives # type: ignore

# Adjust open figure warning
plt.rcParams['figure.max_open_warning'] = 50
plt.rcParams.update({'font.size': 9,
                     "pgf.texsystem": "xelatex",
                     "text.usetex":  True,
                     "pgf.rcfonts": False})

# Define linestyles, markers, and colors
# Linestyles: dash-dot, dashed, dash dot dot
STYLE = ["-.", "--", (0, (3, 5, 1, 5, 1, 5)), ":", (0, (3, 1, 1, 1, 1, 1))]
MARKERS = ["^", "*", "o", "+", "d", "s", "h", "p"]
CLRS = ["tab:blue", "tab:orange", "tab:green",
        "tab:red", "tab:purple", "tab:olive", "tab:cyan"]
MS = 5
MAJOR_GRID_ALPHA = 0.6
MINOR_GRID_ALPHA = 0.4
COLUMN_WIDTH = 256.5 # in points, for two-column layout

class PostProcessing:
    """
    A class to handle post-processing of optimization results.

    This class provides comprehensive post-processing functionality for analyzing
    and visualizing results from Pymoo optimizations. It supports single-analysis
    and multi-analysis comparisons, including geometry visualization, convergence
    analysis, and objective space plotting.

    Attributes
    ----------
    - fname : Path
        The filename or path of the .dill file to be loaded.
    - base_dir : Path
        The base directory for resolving relative file paths.

    Examples
    --------
    >>> output = Path('res_pop20_gen20_250506220150593712.dill')
    >>> processing = PostProcessing(fname=output)
    >>> res = processing.load_res()
    >>> processing.extract_population_data(res)
    >>> processing.main()
    """

    _airfoil_param = AirfoilParameterisation()  # shared, read-only
    _objectives_class = Objectives(duct_variables=config.DUCT_VALUES,
                                   oper=config.multi_oper,
                                   Lref=1)  # Initialized with dummy values


    def __init__(self,
                 fname: Path,
                 base_dir: Path | None = None) -> None:
        """
        Initialization of the PostProcessing class.

        Parameters
        ----------
        - fname : Path
            The filename or path of the .dill file to be loaded.
            If not an absolute path it will be relative to base_dir.
        - base_dir : Path, optional
            The base directory to use if fname is not an absolute path.
            Defaults to the directory containing this script.

        Returns
        -------
        None
        """

        # If base_dir is not provided, use the script's directory
        if base_dir is None:
            self.base_dir = Path(__file__).resolve().parent
        else:
            self.base_dir = base_dir

        # Coerce fname to Path and resolve it if it's not already absolute
        fname = Path(fname)
        self.results_path = fname if fname.is_absolute() else \
            (self.base_dir / fname).resolve()

        # Validate file extension
        if self.results_path.suffix.lower() != '.dill':
            raise ValueError(f"Incorrect file extension: {self.results_path.suffix}")


    def load_res(self) -> Result:
        """
        Load and return the optimization results from the specified .dill file.

        Parameters
        ----------
        None

        Returns
        -------
        - res : Result
            The reconstructed pymoo optimisation results object
        """

        try:
            # Open and load the results file
            with self.results_path.open('rb') as f:
                # ignore=False ensures we get an error if the object
                # cannot be reconstructed.
                res = dill.load(f, ignore=False)

            return res

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error: Results file not found.") from e

        except dill.UnpicklingError as e:
            raise RuntimeError(f"Error loading results: {e}") from e


    def _extract_data(self,
                      population: list[dict[str, float | int]]) -> tuple:
        """
        Helper method to extract and deconstruct vectors from a population.

        Parameters
        ----------
        - population : list[dict[str, float | int]]
            A list of individuals from the pymoo population.

        Returns
        -------
        - tuple : (CB_data, duct_data, blading_data, design_data)
            - CB_data : list[dict[str, float | int]]
                List of control body data dictionaries for each individual.
            - duct_data : list[dict[str, float | int]]
                List of duct data dictionaries for each individual.
            - blading_data : list[dict[str, float | int]]
                List of blading data dictionaries for each individual.
            - design_data : list[list[dict[str, float | int]]]
                List of blade design data dictionaries for each individual.
        """
        vec_interface = DesignVectorInterface()

        decomposed_data = [vec_interface.DeconstructDesignVector(individual.X)
                           for individual in population]
        CB_data = [data[0] for data in decomposed_data]
        duct_data = [data[1] for data in decomposed_data]
        design_data = [data[2] for data in decomposed_data]
        blading_data = [data[3] for data in decomposed_data]

        return CB_data, duct_data, blading_data, design_data


    def extract_population_data(self,
                              res: Result) -> None:
        """
        Extract all population data from the results,
        deconstruct the design vectors, and write all data to lists in self.

        Parameters
        ----------
        - res : Result
            The reconstructed pymoo results object.

        Returns
        -------
        None
        """

        # Write data from the full population to self
        (self.CB_data,
         self.duct_data,
         self.blading_data,
         self.design_data) = self._extract_data(res.pop)

        # Write data from the optimum individuals to self
        (self.CB_data_opt,
         self.duct_data_opt,
         self.blading_data_opt,
         self.design_data_opt) = self._extract_data(res.opt)


    def compare_axisymmetric_geometry(self,
                                    reference: dict[str, Any],
                                    optimised: list[dict[str, Any]],
                                    individual: bool = False) -> None:
        """
        Generate plots of the original and optimised
        (normalised) axisymmetric profiles.

        Parameters
        ----------
        - reference : dict[str, Any]
            The reference geometry.
        - optimised : list[dict[str, Any]]
            A list of the optimised geometry.
        - individual : bool, optional
            An optional bool to determine if individual comparison plots
            between each optimised individual and the reference design should
            be generated. Default value is False.

        Returns
        -------
        None
        """

        # Compute the original geometry (x,y) coordinates
        (original_upper_x,
        original_upper_y,
        original_lower_x,
        original_lower_y) = self._airfoil_param.ComputeProfileCoordinates(reference)

        # Precompute the concatenated original geometry coordinates to avoid
        # repeated operatings while plotting
        original_x = np.concatenate((original_upper_x,
                                     np.flip(original_lower_x)), axis=0)
        original_y = np.concatenate((original_upper_y,
                                     np.flip(original_lower_y)), axis=0)

        # Create grouped figure to compare the geometry between the reference
        # and the optimised designs
        grouped_fig, ax1 = plt.subplots()

        # First plot the original geometry
        ax1.plot(original_x,
                 original_y,
                 "k-.",
                 label="Original Geometry",
                 )

        # Loop over all individuals in the final population and plot
        # their geometries
        for i, geom in enumerate(optimised):
            # Compute the optimised geometry
            (opt_upper_x,
            opt_upper_y,
            opt_lower_x,
            opt_lower_y) = self._airfoil_param.ComputeProfileCoordinates(geom)

            # Compute the concatenated optimised x and y coordinates
            opt_x = np.concatenate((opt_upper_x, np.flip(opt_lower_x)), axis=0)
            opt_y = np.concatenate((opt_upper_y, np.flip(opt_lower_y)), axis=0)

            # Plot the optimised geometry
            ax1.plot(opt_x,
                     opt_y,
                     label=f"Individual {i}",
                     )

            if individual:
                # Create figure for the individual comparison plot
                plt.figure(f"Comparison for individual {i}")
                # plot the original geometry
                plt.plot(original_x,
                         original_y,
                         "k-.",
                         label="Original Geometry",
                         )
                plt.plot(opt_x,
                         opt_y,
                         label=f"Individual {i}",
                         )
                plt.legend(bbox_to_anchor=(1,1))
                plt.grid(which='major', alpha=MAJOR_GRID_ALPHA)
                plt.grid(which='minor', linewidth=0.25, alpha=MINOR_GRID_ALPHA)
                plt.minorticks_on()
                plt.tight_layout()
                plt.xlabel("Normalised axial coordinate $x/c$ [-]")
                plt.ylabel("Normalised perpendicular coordinate $y/c$ [-]")
                # plt.axis('equal')
                plt.show()
                plt.close()

        ax1.grid(which='both')
        ax1.minorticks_on()
        ax1.set_xlabel("Normalised axial coordinate $x/c$ [-]")
        ax1.set_ylabel("Normalised perpendicular coordinate $y/c$ [-]")

        if len(optimised) <= 40:
            # Avoid generating the legend if the population size is too big
            # to ensure plots remain somewhat clear
            ax1.legend(loc='upper left', bbox_to_anchor=(1,1))
        grouped_fig.tight_layout()


    def construct_blade_profile(self,
                              design:list[dict[str, Any]],
                              section_idx: int) -> tuple:
        """
        Function to compute the rotated upper and lower profile coordinates
        for a blade section.

        Parameters
        ----------
        - design : list[dict[str, Any]]
            The list of the design dictionaries for each blade profile in the
            stage.
        - section_idx : int
            The index of the radial section being constructed.

        Returns
        -------
        - tuple [np.ndarray]
            - upper_x
            - upper_y
            - lower_x
            - lower_y
        """

        # Create complete airfoil representation
        (upper_x,
         upper_y,
         lower_x,
         lower_y) = self._airfoil_param.ComputeProfileCoordinates(design[section_idx])

        return upper_x, upper_y, lower_x, lower_y


    def _plot_scalar_blading_parameter(self,
                                       x: np.typing.NDArray,
                                       k: int,
                                       key: str,
                                       reference_value: float | int,
                                       optimised_blading: list[dict[str, Any]],
                                       base_bar_width: float,
                                       labels: list[str] | None = None) -> None:
        """
        Helper method to plot scalar blading parameters.

        Parameters
        ----------
        - x : np.typing.NDArray
            The x positions for the bars.
        - k : int
            The index of the current parameter being plotted.
        - key : str
            The key of the blading parameter to plot.
        - reference_value : float | int
            The reference value of the blading parameter.
        - optimised_blading : list[dict[str, Any]]
            The list of optimised blading parameter dictionaries.
        - base_bar_width : float
            The base width of the bars.
        - labels : list[str], optional
            The labels for each optimised individual. Defaults to None.

        Returns
        -------
        None
        """

        ref_val = reference_value
        if key == "ref_blade_angle":
            ref_val = np.rad2deg(ref_val)

        # Plot the reference data
        plt.bar(x[k], ref_val, width=base_bar_width, label="Reference",
                color='black', hatch='//', edgecolor='white')

        # Plot the optimised blading parameters
        for j, opt_vals in enumerate(optimised_blading):
            label = f"Individual {j}" if labels is None else labels[j]

            opt_val = opt_vals[key]
            if key == "ref_blade_angle":
                opt_val = np.rad2deg(opt_val)
            plt.bar(x[k] + (j + 1) * base_bar_width, opt_val,
                    width=base_bar_width, label=label, color=CLRS[j])


    def _plot_rps_blading_parameter(self,
                                    x: np.typing.NDArray,
                                    k: int,
                                    reference_rps: list[float],
                                    optimised_blading: list[dict[str, Any]],
                                    base_bar_width: float,
                                    labels: list[str] | None = None) -> None:
        """
        Helper method to plot RPS blading parameters.

        Parameters
        ----------
        - x : np.typing.NDArray
            The x positions for the bars.
        - k : int
            The index of the current parameter being plotted.
        - reference_rps : list[float]
            The reference RPS values.
        - optimised_blading : list[dict[str, Any]]
            The list of optimised blading parameter dictionaries.
        - base_bar_width : float
            The base width of the bars.
        - labels : list[str], optional
            The labels for each optimised individual. Defaults to None.

        Returns
        -------
        None
        """

        num_rps = len(reference_rps)
        sub_bar_width = base_bar_width / num_rps

        # Plot reference RPS values
        for r, r_val in enumerate(reference_rps):
            offset = (r - (num_rps - 1) / 2) * sub_bar_width
            plt.bar(x[k] + offset, r_val, width=sub_bar_width,
                    label="Reference" if r == 0 else "",
                    color='black', hatch='//', edgecolor='white')

        # Plot optimised RPS values
        for j, opt_vals in enumerate(optimised_blading):
            label = (f"Individual {j}" if r == 0 else "") if labels is None \
                else labels[j]

            opt_rps = opt_vals["RPS_lst"]
            for r, opt_r_val in enumerate(opt_rps):
                offset = (r - (num_rps - 1) / 2) * sub_bar_width
                plt.bar(x[k] + offset + (j + 1) * base_bar_width, opt_r_val,
                        width=sub_bar_width,
                        label=label,
                        color=CLRS[j])


    def _plot_radial_stations_parameter(self,
                                        x: np.typing.NDArray,
                                        k: int,
                                        reference_value: list[float],
                                        optimised_blading: list[dict[str, Any]],
                                        base_bar_width: float,
                                        labels: list[str] | None = None) -> None:
        """
        Helper method to plot radial stations parameters (blade diameter).

        Parameters
        ----------
        - x : np.typing.NDArray
            The x positions for the bars.
        - k : int
            The index of the current parameter being plotted.
        - reference_value : list[float]
            The reference radial station values.
        - optimised_blading : list[dict[str, Any]]
            The list of optimised blading parameter dictionaries.
        - base_bar_width : float
            The base width of the bars.
        - labels : list[str], optional
            The labels for each optimised individual. Defaults to None.

        Returns
        -------
        None
        """

        plt.bar(x[k], max(reference_value) * 2, width=base_bar_width,
                color='black', hatch="//", edgecolor="white")

        for j, opt_vals in enumerate(optimised_blading):
            label = f"Individual {j}" if labels is None else labels[j]

            opt_val = opt_vals["radial_stations"]
            opt_val = max(opt_val) * 2
            # Time 2 as the radial stations are defined over the blade radius.
            plt.bar(x[k] + (j + 1) * base_bar_width, opt_val,
                    width=base_bar_width, label=label,
                    color=CLRS[j])


    def _plot_blading_bar_chart(self,
                                stage_idx: int,
                                reference_blading: list[dict[str, Any]],
                                optimised_blading: list[dict[str, Any]],
                                labels: list[str] | None = None) -> None:
        """
        Helper method to create bar chart comparing blading parameters.

        Parameters
        ----------
        - stage_idx : int
            The index of the blade stage being plotted.
        - reference_blading : list[dict[str, Any]]
            The reference blading data. Each dictionary in the list corresponds
            to a stage.
        - optimised_blading : list[dict[str, Any]]
            The optimised blading data. Each nested list corresponds to an
            individual in the final optimised population.
        - labels : list[str], optional
            The labels for each optimised individual. Defaults to None.

        Returns
        -------
        None
        """

        variables = [
            "$x_{root}$ [m]",
            "$\\beta_{tip}$ [deg]",
            "$B$ [-]",
            "$\\Omega$ [s$^{-1}$]",
            "$R$ [m]"
        ]

        keys = [
            "root_LE_coordinate",
            "ref_blade_angle",
            "blade_count",
            "RPS_lst",
            "radial_stations"
        ]

        num_indiv = len(optimised_blading)
        x = np.arange(len(keys))
        base_bar_width = 0.8 / (num_indiv + 1)

        plt.figure("Bar Chart with blading parameters",
                   figsize=get_figsize(wf=1, columnwidth=COLUMN_WIDTH))
        for k, key in enumerate(keys):
            if key == "radial_stations":
                self._plot_radial_stations_parameter(x,
                                                     k,
                                                     reference_blading[stage_idx][key],
                                                     optimised_blading,
                                                     base_bar_width,
                                                     labels)
            elif key == "RPS_lst":
                self._plot_rps_blading_parameter(x,
                                                 k,
                                                 reference_blading[stage_idx][key],
                                                 optimised_blading,
                                                 base_bar_width,
                                                 labels)
            else:
                self._plot_scalar_blading_parameter(x,
                                                    k,
                                                    key,
                                                    reference_blading[stage_idx][key],
                                                    optimised_blading,
                                                    base_bar_width,
                                                    labels)

        # Format the plot
        plt.xticks(x + (base_bar_width * num_indiv) / 2, variables,
                   rotation=90)
        plt.title("Comparison of Reference vs Optimized Design Variables")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        if len(optimised_blading) <= 40:
            # Avoid generating the legend if the population size is too big
            # to ensure plots remain somewhat clear
            plt.legend(by_label.values(), by_label.keys(),
                       loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(axis='y', which='major',
                 alpha=MAJOR_GRID_ALPHA)
        plt.grid(axis='y', which='minor', linewidth=0.25,
                 alpha=MINOR_GRID_ALPHA)
        plt.yscale('log')
        plt.ylim(bottom=1e-2, top=1e2)
        plt.minorticks_on()
        plt.tight_layout()


    def _plot_sectional_blading_data(self,
                                     reference_blading: list[dict[str, Any]],
                                     optimised_blading: list[dict[str, Any]],
                                     labels: list[str] | None = None) -> None:
        """
        Helper method to plot sectional blading data
        (chord length, sweep angle, blade angle).

        Parameters
        ----------
        - reference_blading : list[dict[str, Any]]
            The reference blading data. Each dictionary in the list corresponds
            to a stage.
        - optimised_blading : list[dict[str, Any]]
            The optimised blading data. Each nested list corresponds to an
            individual in the final optimised population.
        - labels : list[str], optional
            The labels for each optimised individual. Defaults to None.

        Returns
        -------
        None
        """

        fig, ax = plt.subplots(nrows=2, ncols=2,
                               constrained_layout=True,
                               figsize=get_figsize(wf=1,
                                                   hf=0.7,         
                                                   columnwidth=COLUMN_WIDTH))

        # Set cyclers and grids
        for row in range(2):
            for col in range(2):
                if row == 1 and col == 1:
                    # Skip the bottom-right subplot (used for legend)
                    continue
                ax[row, col].minorticks_on()
                ax[row, col].grid(which='major',
                                  alpha=MAJOR_GRID_ALPHA)
                ax[row, col].grid(which='minor',
                                  linewidth=0.25,
                                  alpha=MINOR_GRID_ALPHA)
                ax[row, col].set_xlabel("Normalised radius [-]")
                ax[row, col].set_xlim([0,1])

        # Plot reference data
        # Use the radial location as fraction rather than absolute dimension
        # to remove the effect of different diameters
        ax[0,0].plot(reference_blading[0]["radial_stations"] / reference_blading[0]["radial_stations"][-1],
                    reference_blading[0]["chord_length"],
                    label="Reference design", color='black', marker="x", ms=MS)
        ax[0,0].set_title("Chord length [m]")

        ax[0,1].plot(reference_blading[0]["radial_stations"] / reference_blading[0]["radial_stations"][-1],
                    np.rad2deg(reference_blading[0]["sweep_angle"]),
                    label="Reference design", color='black', marker="x", ms=MS)
        ax[0,1].set_title("Sweep angle [deg]")

        ax[1,0].plot(reference_blading[0]["radial_stations"] / reference_blading[0]["radial_stations"][-1],
                    np.rad2deg(reference_blading[0]["blade_angle"]),
                    label="Reference design", color='black', marker="x", ms=MS)
        ax[1,0].set_title("Blade angle [deg]")

        # Plot optimised data
        # Use the radial location as fraction rather than absolute dimension
        # to remove the effect of different diameters
        for j, opt_vals in enumerate(optimised_blading):
            label = f"Individual {j}" if labels is None else labels[j]

            ax[0,0].plot(opt_vals["radial_stations"] / opt_vals["radial_stations"][-1],
                        opt_vals["chord_length"],
                        label=label, marker=MARKERS[j],
                        linestyle=STYLE[j], color=CLRS[j], ms=MS)

            ax[0,1].plot(opt_vals["radial_stations"] / opt_vals["radial_stations"][-1],
                        np.rad2deg(opt_vals["sweep_angle"]),
                        label=label, marker=MARKERS[j],
                        linestyle=STYLE[j], color=CLRS[j], ms=MS)

            ax[1,0].plot(opt_vals["radial_stations"] / opt_vals["radial_stations"][-1],
                        np.rad2deg(opt_vals["blade_angle"]),
                        label=label, marker=MARKERS[j],
                        linestyle=STYLE[j], color=CLRS[j], ms=MS)

        # Use bottom-right subplot for legend
        ax[1,1].axis('off')
        handles, labels = ax[0,0].get_legend_handles_labels()
        if len(optimised_blading) <= 40:
            # Avoid generating the legend if the population size is too big
            # to ensure plots remain somewhat clear
            ax[1,1].legend(handles, labels, loc='center', ncol=1)


    def compare_blading_data(self,
                             reference_blading: list[dict[str, Any]],
                             optimised_blading: list[list[dict[str, Any]]]) -> None:
        """
        Generate plots of the blading data for the final population members
        and the initial reference design.

        Parameters
        ----------
        - reference_blading : list[dict[str, Any]]
            The reference blading data. Each dictionary in the list corresponds
            to a stage.
        - optimised_blading : list[dict[str, Any]]
            The optimised blading data. Each nested list corresponds to an
            individual in the final optimised population.

        Returns
        -------
        None
        """

        # Generate bar charts and sectional plots for each optimized stage
        for stage_idx, opt_stage in enumerate(config.OPTIMIZE_STAGE):
            if opt_stage:
                # Create bar chart comparison
                self._plot_blading_bar_chart(stage_idx,
                                             reference_blading,
                                             optimised_blading)

                # Create sectional data plots
                self._plot_sectional_blading_data(reference_blading,
                                                  optimised_blading)


    def _plot_single_blade_profile(self,
                                   design: list[dict[str, Any]],
                                   section_idx: int,
                                   label: str,
                                   color: str,
                                   linestyle: str = '-') -> None:
        """
        Helper method to plot a single blade profile.

        Parameters
        ----------
        - design : list[dict[str, Any]]
            The design data for the blade stage
        - section_idx : int
            The radial section index
        - label : str
            Label for the plot legend
        - color : str
            Color for the plot
        - linestyle : str, optional
            Line style for the plot. Defaults to '-'

        Returns
        -------
        None
        """

        (upper_x,
         upper_y,
         lower_x,
         lower_y) = self.construct_blade_profile(design, section_idx)

        # Plot the blade profile
        plt.plot(np.concatenate((upper_x, np.flip(lower_x))),
                np.concatenate((upper_y, np.flip(lower_y))),
                label=label, color=color, linestyle=linestyle)


    def _plot_reference_blade_profile(self,
                                      reference_design: list[list[dict[str, Any]]],
                                      stage_idx: int,
                                      section_idx: int) -> None:
        """
        Helper method to plot the reference blade profile.

        Parameters
        ----------
        - reference_design : list[list[dict[str, Any]]]
            The reference design data
        - stage_idx : int
            The stage index
        - section_idx : int
            The radial section index

        Returns
        -------
        None
        """

        self._plot_single_blade_profile(reference_design[stage_idx],
                                        section_idx,
                                        "Reference",
                                        "k",
                                        "-")


    def _plot_multiple_optimum_designs(self,
                                       multi_optimum_designs: list,
                                       reference_design: list[list[dict[str, Any]]],
                                       stage_idx: int) -> None:
        """
        Helper method to plot multiple optimum designs for a given stage.

        Parameters
        ----------
        - multi_optimum_designs : list
            List of optimum design data
        - reference_design : list[list[dict[str, Any]]]
            The reference design data
        - stage_idx : int
            The stage index to plot

        Returns
        -------
        None
        """

        radial_coordinates = np.linspace(0,
                                         1,
                                         config.NUM_RADIALSECTIONS[stage_idx])
        colors = plt.cm.tab10(np.linspace(0,
                                          1,
                                          len(multi_optimum_designs)))

        for j, radial_coordinate in enumerate(radial_coordinates):
            plt.figure(f"BladeProfileComparison_R{round(radial_coordinate, 3)}_Stage{stage_idx}")

            # Plot each optimum design
            for opt_idx, current_opt_design in enumerate(multi_optimum_designs):
                self._plot_single_blade_profile(current_opt_design[stage_idx],
                                                j,
                                                f"Optimised (Ind {opt_idx})",
                                                colors[opt_idx])

            # Plot reference design
            self._plot_reference_blade_profile(reference_design,
                                               stage_idx,
                                               j)

            # Format the plot
            plt.legend()
            plt.title(f"Blade profile at r={round(radial_coordinate, 3)}R for multiple optima")
            plt.minorticks_on()
            plt.grid(which='major',
                     alpha=MAJOR_GRID_ALPHA)
            plt.grid(which='minor',
                     linewidth=0.25,
                     alpha=MINOR_GRID_ALPHA)
            plt.xlabel('Normalised chordwise coordinate $x/c$ [-]')
            plt.ylabel('Normalised perpendicular coordinate $y/c$ [-]')
            plt.tight_layout()


    def _plot_single_optimum_design(self,
                                    optimised_design: list[list[dict[str, Any]]],
                                    reference_design: list[list[dict[str, Any]]],
                                    stage_idx: int,
                                    individual: int | str) -> None:
        """
        Helper method to plot a single optimum design for a given stage.

        Parameters
        ----------
        - optimised_design : list[list[dict[str, Any]]]
            The optimised design data
        - reference_design : list[list[dict[str, Any]]]
            The reference design data
        - stage_idx : int
            The stage index to plot
        - individual : int | str
            The individual identifier for labeling

        Returns
        -------
        None
        """

        radial_coordinates = np.linspace(0,
                                         1,
                                         config.NUM_RADIALSECTIONS[stage_idx])

        for j, radial_coordinate in enumerate(radial_coordinates):
            plt.figure(f"BladeProfileComparison_R{round(radial_coordinate, 3)}_Stage{stage_idx}")

            # Plot optimised design
            self._plot_single_blade_profile(optimised_design[stage_idx],
                                            j,
                                            "Optimised",
                                            'tab:blue')

            # Plot reference design
            self._plot_reference_blade_profile(reference_design,
                                               stage_idx,
                                               j)

            # Format the plot
            plt.legend()
            plt.title(f"Blade profile at r={round(radial_coordinate, 3)}R for individual: {individual}")
            plt.minorticks_on()
            plt.grid(which='major',
                     alpha=MAJOR_GRID_ALPHA)
            plt.grid(which='minor',
                     linewidth=0.25,
                     alpha=MINOR_GRID_ALPHA)
            plt.xlabel('Normalised chordwise coordinate $x/c$ [-]')
            plt.ylabel('Normalised perpendicular coordinate $y/c$ [-]')
            plt.tight_layout()


    def compare_blade_design_data(self,
                               reference_design: list[list[dict[str, Any]]],
                               res: Result,
                               individual: int | str = "opt",
                               optimised_design: list[list[dict[str, Any]]] | None = None) -> None:
        """
        Compares the blade design data of a reference design with an
        optimised design and generates plots for visual comparison at various
        radial sections.

        Parameters
        ----------
        - reference_design : list[list[dict[str, Any]]]
            The reference blade design data, structured as a list of stages,
            where each stage contains a list of dictionaries representing
            radial sections.
        - res : Result
            The optimisation result containing the design vector of the
            optimised design.
        - individual : int | str, optional
            Specifies which individual design to compare against. If "opt",
            the optimum design from the optimization result is used. If an
            integer, the corresponding individual from the `optimised_design`
            list is used. Defaults to "opt".
        - optimised_design list[list[dict[str, Any]]], optional
            The optimized blade design data, structured similarly to
            `reference_design`. Required if `individual` is an integer.
            Defaults to None.

        Returns
        -------
        None
        """

        if individual == "opt":
            # Handle optimum design case
            if isinstance(res.X, dict):
                (_,
                 _,
                 optimised_design,
                 _,
                 _) = DesignVectorInterface().DeconstructDesignVector(res.X)

                # Plot single optimum for each optimized stage
                for i in range(len(config.OPTIMIZE_STAGE)):
                    if config.OPTIMIZE_STAGE[i]:
                        self._plot_single_optimum_design(optimised_design,
                                                         reference_design,
                                                         i,
                                                         individual)
            else:
                # Multiple optimum designs
                multi_optimum_designs = []
                for design_dict in res.X:
                    (_,
                     _,
                     design_opt,
                     _,
                     _) = DesignVectorInterface().DeconstructDesignVector(design_dict)
                    multi_optimum_designs.append(design_opt)

                # Plot multiple optima for each optimized stage
                for i in range(len(config.OPTIMIZE_STAGE)):
                    if config.OPTIMIZE_STAGE[i]:
                        self._plot_multiple_optimum_designs(multi_optimum_designs,
                                                            reference_design,
                                                            i)
        else:
            # Handle individual index case
            if optimised_design is None:
                raise ValueError("'optimised_design' must be supplied when 'individual' is an int.")

            optimised_design = copy.deepcopy(optimised_design[individual])

            # Plot individual design for each optimized stage
            for i in range(len(config.OPTIMIZE_STAGE)):
                if config.OPTIMIZE_STAGE[i]:
                    self._plot_single_optimum_design(optimised_design,
                                                     reference_design,
                                                     i,
                                                     individual)


    def generate_convergence_statistics(self,
                                        res: Result) -> None:
        """
        Generate some graphs to analyse the convergence behaviour of the
        optimisation. Analyses:
            - The convergence of the best and average objective values.
            - Diversity of the design vectors
            - Constraint violation

        Parameters
        ----------
        - res : Result
            The optimization result containing the history of the optimisation.

        Returns
        -------
        None
        """

        def three_sig_figs_ticks(x, _):
            return f"{x:.3g}"

        # First visualise the convergence of the objective values
        n_gen = np.linspace(1, len(res.history), len(res.history))
        generational_optimum = np.array([e.opt[0].F for e in res.history])
        avg_objectives = np.array([np.mean(e.pop.get("F"), axis=0) for
                                   e in res.history])
        nadir_objectives = np.array([np.max(e.pop.get("F"), axis=0) for
                                     e in res.history])

        if avg_objectives.shape[1] == 1:
            plt.figure("Objective value", figsize=get_figsize(wf=0.95,
                                                              hf=0.75,
                                                              columnwidth=COLUMN_WIDTH))
            # For a single-objective problem, plot the best, avg, and worst generational values.
            avg_objectives = avg_objectives.squeeze()
            nadir_objectives = nadir_objectives.squeeze()
            if config.objective_IDs[0] == config.ObjectiveID.ENERGY:
                plt.plot(n_gen, generational_optimum,
                         "-.x", label='Generational optimum')
                plt.plot(n_gen, avg_objectives,
                         "--*", label="Generational average")
                plt.plot(n_gen, nadir_objectives,
                         ":o", label="Generational worst")
            elif config.objective_IDs[0] == config.ObjectiveID.EFFICIENCY:
                plt.plot(n_gen, -generational_optimum,
                         "-.x", label='Generational optimum')
                plt.plot(n_gen, -avg_objectives,
                         "--*", label="Generational average")
                plt.plot(n_gen, -nadir_objectives,
                         ":o", label="Generational worst")
            plt.minorticks_on()
            plt.grid(which='major',
                     alpha=MAJOR_GRID_ALPHA)
            plt.grid(which='minor',
                     linewidth=0.25,
                     alpha=MINOR_GRID_ALPHA)
            plt.xlabel("Generation [-]")
            plt.ylabel("Propulsive efficiency $\\eta_p$ [-]")
            plt.ylim(bottom=0.4, top=1)
            plt.legend()
            plt.tight_layout()

        else:
            # For a multi-objective poblem, we plot the running metric
            ngen = res.algorithm.n_gen
            running_metric = RunningMetric(period=ngen)

            plt.figure("Running metric", figsize=get_figsize(wf=0.95,
                                                             hf=0.75,
                                                             columnwidth=COLUMN_WIDTH))
            for e in res.history:
                if e != res.history[-1]:
                    running_metric.update(e)
                    tau = e.n_gen
                    f = running_metric.delta_f
                    x = np.arange(len(f)) + 1
                    v = [max(ideal, nadir) >1E-5 for ideal, nadir in
                         zip(running_metric.delta_ideal,
                             running_metric.delta_nadir)]

                    plt.plot(x, f, "-.", alpha=0.6)

                else:
                    running_metric.update(e)
                    tau = e.n_gen
                    f = running_metric.delta_f
                    x = np.arange(len(f)) + 1
                    v = [max(ideal, nadir) >1E-5 for ideal, nadir in
                         zip(running_metric.delta_ideal,
                             running_metric.delta_nadir)]

                    plt.plot(x, f, label=f"Final generation: {tau}", alpha=0.9)

                    for k in range(len(v)):
                        if v[k]:
                            plt.plot(np.array([k + 1, k + 1]), [0, f[k]],
                                     color="black", linewidth=0.5, alpha=0.5)
                            plt.plot(np.array([k + 1]), [f[k]], "o",
                                     color="black", alpha=0.5, markersize=2)

            plt.xlabel("Generation [-]")
            plt.ylabel("Change in objective space $\\Delta f$ [-]")
            plt.gca().yaxis.set_major_formatter(FuncFormatter(three_sig_figs_ticks))
            plt.yscale("symlog")
            plt.legend()
            plt.grid(which='major',
                     alpha=MAJOR_GRID_ALPHA)
            plt.grid(which='minor',
                     linewidth=0.25,
                     alpha=MINOR_GRID_ALPHA)
            plt.minorticks_on()
            plt.tight_layout()

            try:
                # Find the first generation where all individuals are feasible
                CV = np.array([np.max(e.pop.get("CV"), axis=0) for
                               e in res.history])
                feasible_at_gen = np.where(CV <= 0.0)[0].min()
                plt.axvline(n_gen[feasible_at_gen],
                            color="grey",
                            linestyle="--",
                            label="All individuals feasible")
            except ValueError:
                # ValuError indicates there were still individuals in the final
                # population with a non-zero CV, so simply pass if
                # this is the case
                pass

        # Visualise diversity of the design vectors,
        # measured through the averaged standard deviation of all variables
        # of the generation
        diversity = []

        # Extract the key ordering of the first optimum individual and use
        # it to ensure all individuals are ordered the same
        x_keys = list(res.opt.get("X")[0].keys())

        # Compute the mean standard deviation of each population
        for e in res.history:
            X = np.array([[d[k] for k in x_keys] for d in e.pop.get("X")])
            X_min = X.min(axis=0)
            X_max = X.max(axis=0)
            range_ = X_max - X_min
            range_[range_ < 1e-9] = 1 # Avoid divide-by-zero for constant variables
            X_norm = (X - X_min) / (range_)
            # Use pdist to compute pairwise distances and then take the mean
            diversity.append(np.mean(pdist(X_norm, metric="euclidean")))

        plt.figure()
        plt.title("Normalised population diversity")
        plt.plot(n_gen, diversity, "-x")
        plt.minorticks_on()
        plt.grid(which='major',
                 alpha=MAJOR_GRID_ALPHA)
        plt.grid(which='minor',
                 linewidth=0.25,
                 alpha=MINOR_GRID_ALPHA)
        plt.xlabel("Generation [-]")
        plt.ylabel("Mean Euclidean distance [-]")
        plt.tight_layout()

        # Visualise the maximum constraint violation of each population
        max_constraint_violation = []
        for e in res.history:
            CV = e.pop.get("CV")
            max_violation = np.max(np.abs(CV), axis=1)
            max_constraint_violation.append(np.max(max_violation))

        plt.figure()
        plt.title("Average constraint violation of the population")
        plt.plot(n_gen, max_constraint_violation, "-x")
        plt.minorticks_on()
        plt.grid(which='major',
                 alpha=MAJOR_GRID_ALPHA)
        plt.grid(which='minor',
                 linewidth=0.25,
                 alpha=MINOR_GRID_ALPHA)
        plt.xlabel("Generation [-]")
        plt.ylabel("Constraint violation $CV_{avg}$ [-]")
        plt.ylim(top=1.5, bottom=-0.1)
        plt.tight_layout()


        # Create a 2x2 grid of subplots
        fig = plt.figure(figsize=get_figsize(wf=0.75,
                                             hf=0.9,
                                             columnwidth=COLUMN_WIDTH))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.25,1])

        ax_big = fig.add_subplot(gs[0, :])  # span top row
        ax_big.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1 = fig.add_subplot(gs[1,0], sharex=ax_big)
        ax2 = fig.add_subplot(gs[1,1], sharex=ax_big)

        if avg_objectives.ndim == 1:
            ax_big.set_title("Objective function")
            if config.objective_IDs[0] == config.ObjectiveID.ENERGY:
                ax_big.plot(n_gen, generational_optimum,
                            "-",
                            label='Generational optimum')
                ax_big.plot(n_gen, avg_objectives,
                            "--",
                            label="Generational average")
                ax_big.plot(n_gen, nadir_objectives,
                            "-.", label="Generational worst")
                ax_big.set_ylabel("Normalised flight energy \n $E_{flight} / E_{flight_{ref}}$ [-]")
                ax_big.set_ylim(0.8, 1.2)
            elif config.objective_IDs[0] == config.ObjectiveID.EFFICIENCY:
                ax_big.plot(n_gen, -generational_optimum,
                            "-", label='Generational optimum')
                ax_big.plot(n_gen, -avg_objectives,
                            "--", label="Generational average")
                ax_big.plot(n_gen, -nadir_objectives,
                            "-.", label="Generational worst")
                ax_big.set_ylabel("Propulsive efficiency $\\eta_p$ [-]")
                ax_big.set_ylim(0.3, 0.9)

            ax_big.grid(which='major', alpha=MAJOR_GRID_ALPHA)
            ax_big.set_xlabel("Generation [-]")
            ax_big.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        else:
            # For a multi-objective poblem, we plot the running metric
            ax_big.set_title("Running metric")
            ngen = res.algorithm.n_gen
            running_metric = RunningMetric(period=ngen)

            ax_big.set_title("Running metric")
            for e in res.history:
                if e != res.history[-1]:

                    running_metric.update(e)
                    tau = e.n_gen
                    f = running_metric.delta_f
                    x = np.arange(len(f)) + 1
                    v = [max(ideal, nadir) >1E-5 for ideal, nadir in zip(running_metric.delta_ideal, running_metric.delta_nadir)]

                else:
                    running_metric.update(e)
                    tau = e.n_gen
                    f = running_metric.delta_f
                    x = np.arange(len(f)) + 1
                    v = [max(ideal, nadir) >1E-5 for ideal, nadir in zip(running_metric.delta_ideal, running_metric.delta_nadir)]

                    ax_big.plot(x, f, label="Final population", alpha=0.9)

                    for k in range(len(v)):
                        if v[k]:
                            ax_big.plot([k + 1, k + 1], [0, f[k]], color="black", linewidth=0.5, alpha=0.5)
                            ax_big.plot([k + 1], [f[k]], "o", color="black", alpha=0.5, markersize=2)

            ax_big.set_xlabel("Generation [-]")
            ax_big.set_ylabel("Change in objective space $\\Delta f$ [-]")
            ax_big.set_yscale("symlog")
            ax_big.yaxis.set_major_formatter(FuncFormatter(three_sig_figs_ticks))
            ax_big.grid(which='major', alpha=MAJOR_GRID_ALPHA)
            ymin, _ = ax_big.get_ylim()
            ax_big.set_ylim(bottom=ymin - 0.01)
            ax_big.minorticks_on()

        # Bottom-left subplot: Diversity
        ax1.set_title("Normalised diversity")
        ax1.plot(n_gen, diversity, "-")
        ax1.set_xlabel("Generation [-]")
        ax1.set_ylabel("Mean Euclidean \n distance [-]")
        ax1.grid(which='major', alpha=MAJOR_GRID_ALPHA)
        ax1.tick_params(axis='x', which='minor', bottom=False, top=False)

        # Bottom-right subplot: Constraint Violation
        ax2.set_title("Average constraint violation")
        ax2.plot(n_gen, max_constraint_violation, "-")
        ax2.set_xlabel("Generation [-]")
        ax2.set_ylabel("Constraint violation \n $CV_{avg}$ [-]")
        ax2.set_ylim(-0.1, 1.5)
        ax2.grid(which='major', alpha=MAJOR_GRID_ALPHA)
        ax2.tick_params(axis='x', which='minor', bottom=False, top=False)

        # Add vertical line for all individuals feasible
        legend_loc = 'lower right' if len(config.objective_IDs) == 1 else 'upper right'
        try:
            CV = np.array([np.max(e.pop.get("CV"), axis=0) for e in res.history])
            feasible_at_gen = np.where(CV <= 0.0)[0].min()
            ax_big.axvline(n_gen[feasible_at_gen], color="grey", linestyle="--", label="All individuals feasible")
            ax1.axvline(n_gen[feasible_at_gen], color="grey", linestyle="--", label="All individuals feasible")
            ax2.axvline(n_gen[feasible_at_gen], color="grey", linestyle="--", label="All individuals feasible")
            ax_big.legend(loc=legend_loc)
        except ValueError:
            ax_big.legend(loc=legend_loc)
            pass

        plt.tight_layout()


    def plot_objective_space(self,
                           res: Result) -> None:
        """
        Visualise the objective space for all feasible solutions.

        Parameters
        ----------
        - res : Result
            The optimization result containing the design vector of
            the optimised design.

        Returns
        -------
        None
        """

        # Collect the objective values of the complete evaluated solution set
        F_all = np.vstack([gen.pop.get("F") for gen in res.history])

        # Collect the constraint violations for all evaluated solutions
        CV_all = np.vstack([gen.pop.get("CV") for gen in res.history])

        # Select only the feasible designs
        feasible_mask = np.all(CV_all <= 0, axis=1)
        F_feasible = F_all[feasible_mask]

        # Extract the initial population for plotting to show the improvement
        F_initial = res.history[0].pop.get("F")
        CV_initial = res.history[0].pop.get("CV")
        feasible_mask = np.all(CV_initial <= 0, axis=1)
        F_initial_feasible = F_initial[feasible_mask]

        if len(F_initial[0]) > 2:
            # Use pymoo built-in plotting in case more than 2 objectives are used
            plot = Scatter(figsize=get_figsize(wf=0.75, hf=0.75, columnwidth=COLUMN_WIDTH),
                           title="Objective space for the feasible evaluated solution set")
            plot.add(F_initial_feasible[0],
                     marker="x",
                     facecolor="tab:blue",
                     s=75,
                     label="Reference Design")
            if len(F_initial_feasible) > 1:
                plot.add(F_initial_feasible[1:],
                         marker="^",
                         facecolor="tab:green",
                         s=35,
                         label="Initial MOO population")
            plot.add(config.ref_objectives,
                     marker="*",
                     facecolor="tab:purple",
                     s=35,
                     label="SOO solution")
            plot.add(config.frontcons_objectives,
                     marker="+",
                     facecolor="tab:cyan",
                     s=35,
                     label="SOO $g_3$ solution")
            plot.add(F_feasible,
                     facecolor='none',
                     edgecolor='black',
                     s=10,
                     label="Evaluated MOO solutions")
            plot.add(res.F,
                     facecolor='tab:red',
                     s=35,
                     label="Optimum MOO solutions")
            plot.legend = True
            plot.tight_layout = True
            plot.show()
        elif len(F_initial[0]) == 2:
            # For a 2-objective plot
            fig, ax = plt.subplots(figsize=get_figsize(wf=0.75,
                                                       hf=0.6,
                                                       columnwidth=COLUMN_WIDTH),
                                   constrained_layout=True)
            ax.set_title("Objective space for the feasible evaluated solution set")
            ax.set_xlabel("Propulsive efficiency $\\eta_p$ [-]")
            ax.set_ylabel("Normalised frontal area $A_{front} / A_{front_{ref}}$ [-]")
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))


            if len(F_initial_feasible) > 1:
                ax.scatter(F_initial_feasible[1:, 0] * -1,
                           F_initial_feasible[1:, 1],
                           s=20,
                           marker="^",
                           color="tab:green",
                           label="Initial MOO population")
            ax.scatter(F_feasible[:,0] * -1,
                       F_feasible[:,1],
                       marker="o",
                       facecolor="none",
                       s=10,
                       alpha=0.5,
                       edgecolor="black",
                       label="Evaluated MOO designs")
            ax.scatter(config.ref_objectives[0] * -1,
                       config.ref_objectives[1],
                       marker="*",
                       s=35,
                       facecolor="tab:purple",
                       label="SOO solution")
            ax.scatter(config.frontcons_objectives[0] * -1,
                       config.frontcons_objectives[1],
                       marker="+",
                       facecolor="tab:cyan",
                       s=35,
                       label="SOO $g_3$ solution")
            ax.scatter(res.F[:,0] * -1,
                       res.F[:,1],
                       marker="o",
                       color="tab:red",
                       s=35,
                       label="Optimum MOO solutions")
            ax.scatter(F_initial_feasible[0,0] * -1,
                       F_initial_feasible[0,1],
                       s=75,
                       marker="h",
                       color="tab:pink",
                       label="Reference design")
            ax.legend(bbox_to_anchor=(1,1))
            ax.minorticks_on()
        else:
            # Don't plot the single objective scatter
            return


    def analyse_design_space(self,
                             res: Result,
                             idx_list: list[int]
                             ) -> None:
        """
        Visualise the feasible design space for the design variables whose
        indices are given in idx_list.

        Parameters
        ----------
        - res : Result
            The optimisation result containing the design vector of the
            optimised design.
        - idx_list : list[int]
            A list of integers which correspond to the indices of the design
            variables which need to be plotted. To determine which integer value
            correspond to which design variable, inspect the
            init_designvector class.

        Returns
        -------
        None
        """

        # Collect all evaluated design vectors
        X_all_dicts = [design for gen in res.history for
                       design in gen.pop.get("X")]

        # Collect the constraint violations for all evaluated solutions
        CV_all = np.array([design for gen in res.history for
                           design in gen.pop.get("CV")])

        # Convert the feasible solution set to arrays for plotting
        # Selects only the design variables whose indices are given in idx_list.
        keys = [f"x{i}" for i in idx_list]
        X_all_arr = np.array([[d[k] for k in keys] for d in X_all_dicts])

        # Select only the feasible design vectors
        feasible_mask = np.all(CV_all <=0, axis=1)
        X_feasible_arr = X_all_arr[feasible_mask]

        # Create parallel coordinate plot for the design variables
        pcp = PCP(labels=keys)
        pcp.add(X_feasible_arr)
        pcp.tight_layout = True
        pcp.show()

        # Create scatter plot for the design variables
        scatter = Scatter(labels=keys)
        scatter.add(X_feasible_arr).show()


    def create_blade_geometry_plots(self,
                                    blading: list[list[dict[str, Any]]],
                                    design: list[list[list[dict[str, Any]]]],
                                    reference_blading: list[list[dict[str,Any]]],
                                    reference_design: list[list[list[dict[str, Any]]]]) -> None:
        """
        Generate 2D and 3D comparison plots of blade geometries for each
        optimised stage compared to the reference in the ducted fan design.
        This method visualizes the blade sections by constructing their
        geometry using provided blading and design data.

        For each stage marked for optimization, it creates:
          - A 2D plot showing the airfoil profiles at multiple radial stations.
          - A 3D plot displaying the full blade surface by stacking the airfoil
          sections along the span.

        The method utilizes geometry construction and transformation utilities
        from the fileHandlingMTFLO class and airfoil coordinate conversion
        from the _airfoil_param attribute.

        Parameters
        ----------
        - blading : list[list[dict[str, Any]]]
            A list containing blading data for each stage. Each stage is
            represented as a list of dictionaries with geometric and
            aerodynamic properties at various radial stations.
        - design : list[list[list[dict[str, Any]]]]
            A list containing design data for each stage. Each stage is
            represented as a list of lists of dictionaries with design
            parameters for the blade sections.
        - reference_blading : list[list[dict[str, Any]]]
            A list containing the reference blading data for each stage.
            Each stage is represented as a list of dictionaries with geometric
            and aerodynamic properties at various radial stations.
        - reference_design : list[list[list[dict[str, Any]]]]
            A list containing the reference design data for each stage.
            Each stage is represented as a list of lists of dictionaries
            with design parameters for the blade sections.

        Returns
        -------
        None

        Notes
        -----
        - Only stages specified in the global config.OPTIMIZE_STAGE list are
          plotted.
        """

        # Create plot for each stage:
        for i in range(len(blading)):
            # Only compute the plots if the stage is to be optimized
            if config.OPTIMIZE_STAGE[i]:
                # Set up figures for 2D and 3D plotting
                _, ax2d = plt.subplots(figsize=(12, 8))
                fig3d = plt.figure(figsize=(12, 8))
                ax3d = fig3d.add_subplot(111, projection='3d')

                # Compute the data for the optimised design
                x_data = []
                y_data = []
                r_data = []
                radial_points = blading[i]["radial_stations"]
                colors = plt.cm.tab10(np.linspace(0, 1, len(radial_points)))
                for idx, r in enumerate(radial_points):
                    # All parameters are normalised using the local chord
                    # length, so we need to obtain the local chord in order
                    # to obtain the dimensional parameters
                    local_chord = blading[i]["chord_length"][idx]

                    # Create complete airfoil representation from BP3434
                    # parameterisation of the radial section
                    (upper_x,
                     upper_y,
                     lower_x,
                     lower_y) = self._airfoil_param.ComputeProfileCoordinates(design[i][idx])
                    upper_x *= local_chord
                    upper_y *= local_chord
                    lower_x *= local_chord
                    lower_y *= local_chord

                    # Rotate the airfoil profile to the correct angle
                    # The blade pitch is defined with respect to the blade
                    # pitch angle at the reference radial station, and thus is
                    # corrected accordingly.
                    blade_pitch = blading[i]["blade_angle"][idx] + blading[i]["ref_blade_angle"] - blading[i]["reference_section_blade_angle"]
                    (rotated_upper_x,
                     rotated_upper_y,
                     rotated_lower_x,
                     rotated_lower_y)  = fileHandlingMTFLO.RotateProfile(blade_pitch,
                                                                         upper_x,
                                                                         lower_x,
                                                                         upper_y,
                                                                         lower_y)

                    # Compute the local leading edge offset at the radial
                    # station from the provided interpolant. Use it to offset
                    # the x-coordinates of the upper and lower surfaces to
                    # the correct position
                    LE_coordinate = blading[i]["root_LE_coordinate"] + \
                        r * np.tan(blading[i]["sweep_angle"][idx])
                    rotated_upper_x += LE_coordinate - rotated_upper_x[0]
                    rotated_lower_x += LE_coordinate - rotated_lower_x[0]

                    # Concatenate the upper and lower data sets
                    rotated_x = np.concatenate((rotated_upper_x,
                                                np.flip(rotated_lower_x)),
                                               axis=0)
                    rotated_y = np.concatenate((rotated_upper_y,
                                                np.flip(rotated_lower_y)),
                                               axis=0)

                    # Plot the 2D profile on 2D axes
                    ax2d.plot(rotated_x, rotated_y,
                              color=colors[idx], label="Optimised")

                    # Append the section to the list for 3d plotting
                    x_data.append(rotated_x)
                    y_data.append(rotated_y)
                    r_data.append(np.full_like(rotated_x, r))

                    # Plot the blade section in the 3D plot
                    # Each section is defined at constant r
                    ax3d.plot(rotated_x,
                              rotated_y,
                              np.full_like(rotated_x, r),
                              color='black')

                # Compute the data for the reference design
                reference_x_data = []
                reference_y_data = []
                reference_r_data = []
                reference_radial_points = reference_blading[i]["radial_stations"]
                for idx, r in enumerate(reference_radial_points):
                    # All parameters are normalised using the local chord
                    # length, so we need to obtain the local chord in order to
                    # obtain the dimensional parameters
                    local_chord = reference_blading[i]["chord_length"][idx]

                    # Create complete airfoil representation from BP3434
                    # parameterisation of the radial section
                    (upper_x,
                     upper_y,
                     lower_x,
                     lower_y) = self._airfoil_param.ComputeProfileCoordinates(reference_design[i][idx])
                    upper_x *= local_chord
                    upper_y *= local_chord
                    lower_x *= local_chord
                    lower_y *= local_chord

                    # Rotate the airfoil profile to the correct angle
                    # The blade pitch is defined with respect to the blade pitch
                    # angle at the reference radial station, and thus is
                    # corrected accordingly.
                    blade_pitch = (blading[i]["blade_angle"][idx] + blading[i]["ref_blade_angle"] - blading[i]["reference_section_blade_angle"])
                    (rotated_upper_x,
                     rotated_upper_y,
                     rotated_lower_x,
                     rotated_lower_y)  = fileHandlingMTFLO.RotateProfile(blade_pitch,
                                                                         upper_x,
                                                                         lower_x,
                                                                         upper_y,
                                                                         lower_y)

                    # Compute the local leading edge offset at the radial
                    # station from the provided interpolant. Use it to offset
                    # the x-coordinates of the upper and lower surfaces to
                    # the correct position
                    LE_coordinate = blading[i]["root_LE_coordinate"] + \
                        r * np.tan(blading[i]["sweep_angle"][idx])
                    rotated_upper_x += LE_coordinate - rotated_upper_x[0]
                    rotated_lower_x += LE_coordinate - rotated_lower_x[0]

                    # Concatenate the upper and lower data sets
                    rotated_x = np.concatenate((rotated_upper_x,
                                                np.flip(rotated_lower_x)),
                                               axis=0)
                    rotated_y = np.concatenate((rotated_upper_y,
                                                np.flip(rotated_lower_y)),
                                               axis=0)

                    # Plot the 2D profile on 2D axes
                    ax2d.plot(rotated_x, rotated_y,
                              "-.", color=colors[idx], label="Reference")

                    # Append the section to the list for 3d plotting
                    reference_x_data.append(rotated_x)
                    reference_y_data.append(rotated_y)
                    reference_r_data.append(np.full_like(rotated_x, r))

                # Convert all data to arrays - this is needed to use the
                # plot_surface method.
                x_data = np.array(x_data)
                y_data = np.array(y_data)
                r_data = np.array(r_data)
                reference_x_data = np.array(reference_x_data)
                reference_y_data = np.array(reference_y_data)
                reference_r_data = np.array(reference_r_data)

                # Plot the optimised and reference blade surfaces in 3D
                ax3d.plot_surface(x_data,
                                  y_data,
                                  r_data,
                                  color='tab:blue',
                                  label="Optimised")
                ax3d.plot_surface(reference_x_data,
                                  reference_y_data,
                                  reference_r_data,
                                  alpha=0.5,
                                  color='tab:orange',
                                  label="Reference")


                # Format 2D plot
                ax2d.set_title("2D Projection of Blade Geometry at Each Radial Section")
                ax2d.set_xlabel("Axial Coordinate [m]")
                ax2d.set_ylabel("Thickness/Height Coordinate [m]")
                ax2d.minorticks_on()
                ax2d.grid(which='major',
                          alpha=MAJOR_GRID_ALPHA)
                ax2d.grid(which='minor',
                          linewidth=0.25,
                          alpha=MINOR_GRID_ALPHA)
                handles, labels = ax2d.get_legend_handles_labels()
                unique = dict(zip(labels, handles))
                legend2d = ax2d.legend(unique.values(), unique.keys())
                for line in legend2d.get_lines():
                    line.set_color("black")

                # Format 3D plot
                ax3d.set_title("3D Blade Geometry")
                ax3d.set_xlabel("Axial Coordinate [m]")
                ax3d.set_ylabel("Thickness/Height Coordinate [m]")
                ax3d.set_zlabel("Radial Coordinate [m]")
                ax3d.minorticks_on()
                ax3d.grid(which='major',
                          alpha=MAJOR_GRID_ALPHA)
                ax3d.grid(which='minor',
                          linewidth=0.25,
                          alpha=MINOR_GRID_ALPHA)
                ax3d.legend()

                # Force the aspect ratio of the axes to be the same
                x_limits = ax3d.get_xlim3d()
                y_limits = ax3d.get_ylim3d()
                z_limits = ax3d.get_zlim3d()

                x_range = abs(x_limits[1] - x_limits[0])
                y_range = abs(y_limits[1] - y_limits[0])
                z_range = abs(z_limits[1] - z_limits[0])
                max_range = max(x_range, y_range, z_range)

                x_mid = np.mean(x_limits)
                y_mid = np.mean(y_limits)
                z_mid = np.mean(z_limits)

                ax3d.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
                ax3d.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
                ax3d.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])

            plt.show()


    def _create_individual_meridional_plot(self,
                                           color: str,
                                           linestyle: str,
                                           label: str,
                                           fname: Path = None,
                                           reference: bool = False,
                                           individual_idx: int = None) -> None:
        """
        Create a meridional view of the optimised ducted fan.

        Parameters
        ----------
        - color : str
            Color to use for the plot.
        - linestyle : str
            Linestyle to use for the plot.
        - label : str
            Label for the plot.
        - fname : Path, optional
            Optional path for the analysis from which to load the results.
            Required if reference is false.
        - reference : bool, optional
            Optional bool to determine if the reference design should be plotted.
            Default is false. The reference design is always plotted in
            grey with a dash-dot line.
        - individual_idx : int, optional
            Optional index of the individual to plot. If not provided, all
            individuals are plotted.

        Returns
        -------
        None
        """

        # Load in the design data for the current analysis
        if not reference:
            self.results_path = fname if fname.is_absolute() else \
                (self.base_dir / fname).resolve()
            res = self.load_res()
            (CB_data,
             duct_data,
             blading_data,
             design_data) = self._extract_data(res.opt)
        else:
            reference_vector = InitPopulation().DeconstructDictFromReferenceDesign()

            vec_interface = DesignVectorInterface()

            data = vec_interface.DeconstructDesignVector(reference_vector)
            CB_data = [data[0]]
            duct_data = [data[1]]
            design_data = [data[2]]
            blading_data = [data[3]]

        # Loop over all optimized individuals for the analysis.
        # All outputs of _extract_data are lists of the same length as
        # the number of optimum solutions.
        indices = [individual_idx] if individual_idx is not None \
            else range(len(CB_data))
        for j in indices:
            # Compute meridional (upper) coordinates for the centerbody
            (CB_upper_x,
             CB_upper_y,
             _,
             _) = self._airfoil_param.ComputeProfileCoordinates(CB_data[j])
            CB_upper_x *= CB_data[j]["Chord Length"]
            CB_upper_y *= CB_data[j]["Chord Length"]

            if reference:
                plt.plot(CB_upper_x, CB_upper_y, color="k")

            # Compute meridional coordinates for the duct
            (DUCT_upper_x,
             DUCT_upper_y,
             DUCT_lower_x,
             DUCT_lower_y) = self._airfoil_param.ComputeProfileCoordinates(duct_data[j])
            DUCT_upper_x *= duct_data[j]["Chord Length"]
            DUCT_upper_y *= duct_data[j]["Chord Length"]
            DUCT_lower_x *= duct_data[j]["Chord Length"]
            DUCT_lower_y *= duct_data[j]["Chord Length"]
            DUCT_upper_x += duct_data[j]["Leading Edge Coordinates"][0]
            DUCT_lower_x += duct_data[j]["Leading Edge Coordinates"][0]
            DUCT_upper_y += duct_data[j]["Leading Edge Coordinates"][1]
            DUCT_lower_y += duct_data[j]["Leading Edge Coordinates"][1]

            duct_x = np.concatenate((DUCT_upper_x,
                                     np.flip(DUCT_lower_x)),
                                    axis=0)
            duct_y = np.concatenate((DUCT_upper_y,
                                     np.flip(DUCT_lower_y)),
                                    axis=0)

            # Create plot for each stage:
            for k in range(len(blading_data[j])):
                radial_points = blading_data[j][k]["radial_stations"]
                outline_x_LE = []
                outline_x_TE = []

                # Loop over the radial sections
                for idx, r in enumerate(radial_points):
                    # All parameters are normalised using the local chord
                    # length, so we need to obtain the local chord in order
                    # to obtain the dimensional parameters
                    local_chord = blading_data[j][k]["chord_length"][idx]

                    # Create complete airfoil representation from BP3434
                    # parameterisation of the radial section
                    (upper_x,
                     upper_y,
                     lower_x,
                     lower_y) = self._airfoil_param.ComputeProfileCoordinates(design_data[j][k][idx])
                    upper_x *= local_chord
                    upper_y *= local_chord
                    lower_x *= local_chord
                    lower_y *= local_chord

                    # Rotate the airfoil profile to the correct angle
                    # The blade pitch is defined with respect to the blade
                    # pitch angle at the reference radial station, and thus
                    # is corrected accordingly.
                    blade_pitch = (blading_data[j][k]["blade_angle"][idx] + blading_data[j][k]["ref_blade_angle"] - blading_data[j][k]["reference_section_blade_angle"])
                    (rotated_upper_x,
                     _,
                     rotated_lower_x,
                     _) = fileHandlingMTFLO.RotateProfile(blade_pitch,
                                                          upper_x,
                                                          lower_x,
                                                          upper_y,
                                                          lower_y)

                    # Compute the local leading edge offset at the radial
                    # station from the provided interpolant. Use it to offset
                    # the x-coordinates of the upper and lower surfaces to the
                    # correct position
                    LE_coordinate = blading_data[j][k]["root_LE_coordinate"] + \
                        r * np.tan(blading_data[j][k]["sweep_angle"][idx])
                    rotated_upper_x += LE_coordinate - rotated_upper_x[0]
                    rotated_lower_x += LE_coordinate - rotated_lower_x[0]

                    outline_x_LE.append(min(rotated_upper_x.min(),
                                            rotated_lower_x.min()))
                    outline_x_TE.append(max(rotated_upper_x.max(),
                                            rotated_lower_x.max()))
                outline_r = np.concatenate((radial_points,
                                            np.flip(radial_points)),
                                           axis=0)
                outline_x = np.concatenate((outline_x_LE,
                                            np.flip(outline_x_TE)),
                                           axis=0)

                if config.OPTIMIZE_STAGE[k]:
                    plt.plot(outline_x, outline_r,
                             color=color, linestyle=linestyle)
                else:
                    plt.plot(outline_x, outline_r,
                             color=color, linestyle=linestyle, linewidth=0.75)
            plt.plot(duct_x, duct_y,
                     color=color, linestyle=linestyle, label=label)


    def create_multi_analysis_meridional_plots(self,
                               fnames: list[Path],
                               labels: list[str]) -> None:
        """
        Create a meridional plots of the blade geometry of the optimum
        individuals for the different analyses.

        Parameters
        ----------
        - fnames : list[Path]
            A list of paths for the different results files to be loaded.
        - labels : list[str]
            A list of strings of the corresponding labels for the plot.

        Returns
        -------
        None
        """

        plt.figure("Meridional plot of the ducted fan designs",
                   figsize=get_figsize(wf=1, hf=0.8, columnwidth=COLUMN_WIDTH),
                   constrained_layout=True)
        self._create_individual_meridional_plot(color="k",
                                                linestyle="-",
                                                label="Reference design",
                                                reference=True)

        # --- Collect all individuals and labels ---
        all_fnames = []
        all_labels = []
        all_indices = []
        for fname, label in zip(fnames, labels):
            self.results_path = fname if fname.is_absolute() else \
                (self.base_dir / fname).resolve()
            res = self.load_res()
            CB_data, _, _, _ = self._extract_data(res.opt)
            for idx in range(len(CB_data)):
                all_fnames.append(fname)
                all_indices.append(idx)
                if len(CB_data) > 1:
                    all_labels.append(f"{label} (ind {idx})")
                else:
                    all_labels.append(label)

        for i, (fname, idx) in enumerate(zip(all_fnames, all_indices)):
            # Load in the design data for the current analysis
            self._create_individual_meridional_plot(color=CLRS[i],
                                                    linestyle=STYLE[i],
                                                    label=all_labels[i],
                                                    fname=fname,
                                                    reference=False,
                                                    individual_idx=idx)

        plt.xlabel("Axial coordinate $x$ [m]")
        plt.ylabel("Radial coordinate $z$ [m]")
        plt.minorticks_on()
        plt.grid(which="major", alpha=MAJOR_GRID_ALPHA)
        plt.axis('equal')
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc="upper center", ncols=2)


    def create_multi_analysis_duct_plots(self,
                                     fnames: list[Path],
                                     labels: list[str]) -> None:
        """
        Create a plot of the duct profiles for the optimised individuals of
        multiple analyses.

        Parameters
        ----------
        - fnames : list[Path]
            A list of paths for the different results files to be loaded.
        - labels : list[str]
            A list of strings of the corresponding labels for the plot.

        Returns
        -------
        None
        """

        # --- Collect all individuals and labels ---
        all_duct_data = []
        all_labels = []
        for fname, label in zip(fnames, labels):
            self.results_path = fname if fname.is_absolute() else \
                (self.base_dir / fname).resolve()
            res = self.load_res()
            _, duct_data, _, _ = self._extract_data(res.opt)
            for idx, indiv in enumerate(duct_data):
                all_duct_data.append(indiv)
                if len(duct_data) > 1:
                    all_labels.append(f"{label} (ind {idx})")
                else:
                    all_labels.append(label)

        def get_ref_profile():
            (ux,
             uy,
             lx,
             ly) = self._airfoil_param.ComputeProfileCoordinates(config.DUCT_VALUES)
            return np.concatenate((ux,
                                   np.flip(lx))), np.concatenate((uy,
                                                                  np.flip(ly)))

        def plot_profiles(ax):
            ref_x, ref_y = get_ref_profile()
            ax.plot(ref_x, ref_y,
                    color="k",
                    linestyle="-",
                    label="Reference design")

            for i, data in enumerate(all_duct_data):
                ux, uy, lx, ly = self._airfoil_param.ComputeProfileCoordinates(data)
                opt_x = np.concatenate((ux, np.flip(lx)))
                opt_y = np.concatenate((uy, np.flip(ly)))
                ax.plot(opt_x, opt_y,
                        label=all_labels[i],
                        color=CLRS[i],
                        linestyle=STYLE[i])

        def create_profile_only_figure():
            fig, ax = plt.subplots(figsize=get_figsize(wf=0.75, columnwidth=COLUMN_WIDTH))
            plot_profiles(ax)
            ax.set_xlabel("Axial coordinate $x/c$ [-]")
            ax.set_ylabel("Vertical coordinate $y/c$ [-]")
            ax.minorticks_on()
            ax.grid(which='major',
                    alpha=MAJOR_GRID_ALPHA)
            ax.grid(which='minor',
                    linewidth=0.25,
                    alpha=MINOR_GRID_ALPHA)
            ax.legend()
            fig.tight_layout()

        def create_comparison_figure():
            fig, axs = plt.subplots(3, 1,
                                    figsize=get_figsize(wf=0.75, hf=1, columnwidth=COLUMN_WIDTH),
                                    gridspec_kw={'height_ratios': [1.75,
                                                                   1.125,
                                                                   1.125]})
            fig.align_labels()

            ref_x, ref_y = get_ref_profile()
            axs[0].plot(ref_x,
                        ref_y,
                        linestyle="-",
                        color="k",
                        label="Reference design")

            for i, data in enumerate(all_duct_data):
                ux, uy, lx, ly = self._airfoil_param.ComputeProfileCoordinates(data)
                opt_x = np.concatenate((ux, np.flip(lx)))
                opt_y = np.concatenate((uy, np.flip(ly)))

                axs[0].plot(opt_x,
                            opt_y,
                            label=all_labels[i],
                            color=CLRS[i],
                            linestyle=STYLE[i])


                (bez_thick,
                 bx_thick,
                 bez_camb,
                 bx_camb) = self._airfoil_param.ComputeBezierCurves(data)
                (bez_thick_ref,
                 bx_thick_ref,
                 bez_camb_ref,
                 bx_camb_ref) = self._airfoil_param.ComputeBezierCurves(config.DUCT_VALUES)
                axs[1].plot(bx_thick_ref,
                            bez_thick_ref,
                            linestyle="-",
                            color="k",
                            label="Reference design")
                axs[2].plot(bx_camb_ref,
                            bez_camb_ref,
                            linestyle="-",
                            color="k",
                            label="Reference design")

                axs[1].plot(bx_thick,
                            bez_thick,
                            label=all_labels[i],
                            color=CLRS[i],
                            linestyle=STYLE[i])
                axs[2].plot(bx_camb,
                            bez_camb,
                            label=all_labels[i],
                            color=CLRS[i],
                            linestyle=STYLE[i])

            axs[0].set_ylabel("Vertical coordinate $y/c$ [-]")
            axs[0].minorticks_on()
            axs[0].grid(which='major',
                        alpha=MAJOR_GRID_ALPHA)
            axs[0].grid(which='minor',
                        linewidth=0.25,
                        alpha=MINOR_GRID_ALPHA)
            axs[0].set_title("Duct profile")
            axs[0].set_xlim(-0.01, 1.01)

            axs[1].set_ylabel("Thickness [-]")
            axs[1].minorticks_on()
            axs[1].grid(which='major',
                        alpha=MAJOR_GRID_ALPHA)
            axs[1].grid(which='minor',
                        linewidth=0.25,
                        alpha=MINOR_GRID_ALPHA)
            axs[1].set_title("Thickness distribution")
            axs[1].set_xlim(-0.01, 1.01)

            axs[2].set_ylabel("Camber [-]")
            axs[2].set_xlabel("Axial coordinate $x/c$ [-]")
            axs[2].minorticks_on()
            axs[2].grid(which='major',
                        alpha=MAJOR_GRID_ALPHA)
            axs[2].grid(which='minor',
                        linewidth=0.25,
                        alpha=MINOR_GRID_ALPHA)
            axs[2].set_title("Camber distribution")
            axs[2].set_xlim(-0.01, 1.01)

            axs[0].set_xticklabels([])
            axs[1].set_xticklabels([])
            axs[0].set_xlabel("")
            axs[1].set_xlabel("")

            # Legend below figure
            handles, legend_labels = axs[0].get_legend_handles_labels()
            fig.legend(handles,
                       legend_labels,
                       loc='lower center',
                       ncol=max(2, len(legend_labels) // 2),
                       frameon=True)
            plt.tight_layout(rect=[0, 0.08, 1, 1])

        # --- Execute both plots ---
        create_profile_only_figure()
        create_comparison_figure()


    def create_multi_analysis_blade_profile_plots(self,
                                              fnames: list[Path],
                                              labels: list[str],
                                              stage_idx: int) -> None:
        """
        Create blade profile subplots for each radial section, overlaying
        profiles from multiple analyses.

        Parameters
        ----------
        - fnames : list[Path]
            A list of paths for the different results files to be loaded.
        - labels : list[str]
            A list of strings of the corresponding labels for the plot.
        - stage_idx : int
            The stage index to plot.

        Returns
        -------
        None
        """

        all_designs = []
        all_labels = []

        # Load and process each design
        for fname, label in zip(fnames, labels):
            self.results_path = fname if fname.is_absolute() else \
                (self.base_dir / fname).resolve()
            res = self.load_res()
            _, _, _, design_data = self._extract_data(res.opt)

            for idx, indv in enumerate(design_data):
                all_designs.append(indv)

                if len(design_data) > 1:
                    all_labels.append(f"{label} (ind {idx})")
                else:
                    all_labels.append(label)


        # Load reference design
        reference_design = config.STAGE_DESIGN_VARIABLES

        num_radial = config.NUM_RADIALSECTIONS[stage_idx]
        radial_coords = np.linspace(0, 1, num_radial)

        # Layout: 2 plots per row max
        num_cols = 2
        num_rows = int(np.ceil(num_radial / num_cols))

        fig, axes = plt.subplots(num_rows,
                                 num_cols,
                                 figsize=get_figsize(wf=1,
                                                     columnwidth=COLUMN_WIDTH),
                                 sharex=True)
        axes = axes.flatten()

        for j, (ax, radial_coord) in enumerate(zip(axes, radial_coords)):
            plt.sca(ax)

            for i, design in enumerate(all_designs):
                self._plot_single_blade_profile(design[stage_idx],
                                                j,
                                                label=all_labels[i],
                                                color=CLRS[i],
                                                linestyle=STYLE[i])

            self._plot_reference_blade_profile(reference_design,
                                               stage_idx,
                                               j)

            ax.set_title(f"r={round(radial_coord, 2)}R")
            ax.set_xlim(-0.01, 1.01)
            ax.grid(which='major',
                    alpha=MAJOR_GRID_ALPHA)
            ax.grid(which='minor',
                    linewidth=0.25,
                    alpha=MINOR_GRID_ALPHA)
            ax.minorticks_on()

        # Create legend
        handles, legend_labels = axes[num_radial - 1].get_legend_handles_labels()
        fig.legend(handles,
                   legend_labels,
                   loc='lower center',
                   bbox_to_anchor=(0.5, -0.01),
                   ncol=len(legend_labels),
                   frameon=True)

        # Hide unused subplots
        for ax in axes[num_radial:]:
            ax.axis('off')

        for idx, ax in enumerate(axes[:num_radial]):
            if idx % num_cols != 0:
                ax.set_ylabel('')
            if idx < (num_rows - 1) * num_cols:
                ax.set_xlabel('')

        # Shared axis labels, subplot title, and formatting
        fig.text(0.5, 0.15, 'Axial coordinate $x/c$ [-]',
                 ha='center')
        fig.text(0.02, 0.5, 'Vertical coordinate $y/c$ [-]',
                 va='center',
                 rotation='vertical')
        fig.tight_layout(rect=[0.025, 0.15, 0.98, 1])


    def create_multi_analysis_blading_bar_chart(self,
                                            fnames: list[Path],
                                            labels: list[str]) -> None:
        """
        Create bar chart comparison of blading parameters across
        multiple analyses.

        Parameters
        ----------
        - fnames : list[Path]
            A list of file paths to the analysis result files.
        - labels : list[str]
            A list of corresponding labels for each analysis.

        Returns
        -------
        None
        """

        all_blading_data = []
        all_labels = []

        # Load and unpack each design
        for fname, label in zip(fnames, labels):
            self.results_path = fname if fname.is_absolute() else \
                (self.base_dir / fname).resolve()
            res = self.load_res()
            _, _, blading_data, _ = self._extract_data(res.opt)

            for idx, indv in enumerate(blading_data):
                all_blading_data.append(indv)

                if len(blading_data) > 1:
                    all_labels.append(f"{label} (ind {idx})")
                else:
                    all_labels.append(label)

        reference_vector = InitPopulation().DeconstructDictFromReferenceDesign()
        vec_interface = DesignVectorInterface()

        data = vec_interface.DeconstructDesignVector(reference_vector)
        reference_blading = data[3]

        for stage_idx, is_optimised in enumerate(config.OPTIMIZE_STAGE):
            if is_optimised:
                # Prepare stage-specific data to satisfy your existing functions
                optimised_stage_blading = [opt[stage_idx] for
                                           opt in all_blading_data]
                self._plot_blading_bar_chart(stage_idx,
                                             reference_blading,
                                             optimised_stage_blading,
                                             all_labels)


    def create_multi_analysis_sectional_blading_plot(self,
                                                 fnames: list[Path],
                                                 labels: list[str]) -> None:
        """
        Create sectional comparison plots of blading data across multiple
        analyses.

        Parameters
        ----------
        - fnames : list[Path]
            A list of file paths to the analysis result files.
        - labels : list[str]
            A list of corresponding labels for each analysis.

        Returns
        -------
        None
        """

        all_blading_data = []
        all_labels = []

        # Load and unpack each design
        for fname, label in zip(fnames, labels):
            self.results_path = fname if fname.is_absolute() else \
                (self.base_dir / fname).resolve()
            res = self.load_res()
            _, _, blading_data, _ = self._extract_data(res.opt)

            for idx, indv in enumerate(blading_data):
                all_blading_data.append(indv)

                if len(blading_data) > 1:
                    all_labels.append(f"{label} (ind {idx})")
                else:
                    all_labels.append(label)

        reference_vector = InitPopulation().DeconstructDictFromReferenceDesign()
        vec_interface = DesignVectorInterface()

        data = vec_interface.DeconstructDesignVector(reference_vector)
        reference_blading = data[3]

        for stage_idx, is_optimised in enumerate(config.OPTIMIZE_STAGE):
            if is_optimised:
                optimised_stage_blading = [opt[stage_idx] for
                                           opt in all_blading_data]
                self._plot_sectional_blading_data(reference_blading,
                                                  optimised_stage_blading,
                                                  all_labels)


    def create_multi_analysis_objective_space_plot(self,
                                                   fnames: list[Path],
                                                   labels: list[str]) -> None:
        """
        Create a single objective space plot containing all analysed feasible
        individuals across multiple analyses.

        Parameters
        ----------
        - fnames : list[Path]
            A list of file paths to the analysis result files.
        - labels : list[str]
            A list of corresponding labels for each analysis.

        Returns
        -------
        None
        """

        def _computeFrontalAreaObjective(x: dict[str, float | int]) -> float:
            """
            Simple helper function to compute the frontal area from the
            design vectors of a single-objective optimisation to enable
            plotting on the same design space as the MOO.

            Parameters
            ----------
            - x : dict[str, float | int]
                The design vector for which the frontal area must
                be computed.

            Returns
            -------
            - frontal_area : float
                The frontal area objective.
            """

            # Deconstruct the design vector
            vec_interface = DesignVectorInterface()
            decomposed_data = vec_interface.DeconstructDesignVector(x)

            duct_data = decomposed_data[1]

            # Compute the area objective corresponding to the design vector.
            # Use the initialized objectives class to avoid repeated object
            # creation and destruction
            self._objectives_class.duct_variables = duct_data
            area_objective = self._objectives_class.FrontalArea({})

            return area_objective

        # Format the objective space plot
        fig, ax = plt.subplots(figsize=get_figsize(wf=1, hf=0.6,
                                                   columnwidth=COLUMN_WIDTH),
                               constrained_layout=True)
        # ax.set_title("Objective space for the feasible solution sets")
        ax.set_xlabel("Propulsor efficiency $\\eta_p$ [-]")
        ax.set_ylabel("Normalised frontal area \n $A_{front} / A_{front_{ref}}$ [-]")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        for i, fname in enumerate(fnames):
            # Load in the design data for the current analysis
            self.results_path = fname if fname.is_absolute() \
                else (self.base_dir / fname).resolve()
            res = self.load_res()

            # Collect the constraint violations for all evaluated solutions,
            # and use them to define a mask for the feasible designs
            CV_all = np.vstack([gen.pop.get("CV") for gen in res.history])
            feasible_mask = np.all(CV_all <= 0, axis=1)

            if len(res.opt.get('F')[0]) != 2:
                # First extract all design vectors
                X_all = np.array([design for gen in res.history for
                                  design in gen.pop.get("X")])

                X_all_feasible = X_all[feasible_mask]

                # Get the efficiency objectives from the SOO
                F_all_SOO = np.vstack([gen.pop.get("F") for gen in res.history])
                F_all_SOO_feasible = F_all_SOO[feasible_mask]

                # Now compute the frontal area objective
                area_objective = np.array([_computeFrontalAreaObjective(x) for
                                           x in X_all_feasible])

                # Construct the complete "equivalent" MOO solution array
                F_feasible = np.column_stack((F_all_SOO_feasible.flatten(),
                                                  area_objective))

                # Extract the initial population for plotting to show
                # the improvement
                F_initial_SOO = res.history[0].pop.get("F")
                X_initial = res.history[0].pop.get("X")
                CV_initial = res.history[0].pop.get("CV")
                feasible_mask = np.all(CV_initial <= 0, axis=1)

                F_initial_feasible_SOO = F_initial_SOO[feasible_mask]
                X_initial_feasible = X_initial[feasible_mask]

                area_initial = np.array([_computeFrontalAreaObjective(x) for x
                                         in X_initial_feasible])
                F_initial_feasible = np.column_stack((F_initial_feasible_SOO.flatten(),
                                                      area_initial))

                # Extract the optimum solution for plotting
                F_opt_SOO = res.opt.get("F")
                X_opt_SOO = res.opt.get("X")[0]
                area_opt_SOO = _computeFrontalAreaObjective(X_opt_SOO)
                F_opt = np.column_stack((F_opt_SOO, area_opt_SOO))
            else:
                # For an MOO simple collect the objective values of the
                # complete evaluated solution set
                F_all = np.vstack([gen.pop.get("F") for gen in res.history])
                F_feasible = F_all[feasible_mask]

                # Extract the initial population for plotting to show
                # the improvement
                F_initial = res.history[0].pop.get("F")
                CV_initial = res.history[0].pop.get("CV")
                feasible_mask = np.all(CV_initial <= 0, axis=1)
                F_initial_feasible = F_initial[feasible_mask]

                # Extract the non-dominated solution for plotting
                F_opt = res.opt.get("F")

            ax.scatter(F_feasible[:,0] * -1, F_feasible[:,1],
                       marker='o', facecolor="none", s=20, alpha=0.3,
                       edgecolors=CLRS[i], label=f"Evaluated designs")
            ax.scatter(F_opt[:,0] * -1, F_opt[:,1],
                       marker="X", color="tab:orange", s=100,
                       label="Optimized design")
            ax.scatter(F_initial_feasible[0,0] * -1, F_initial_feasible[0,1],
                       s=50, marker="H", color="tab:red",
                       label="Reference design")

        # Legend below figure
        handles, legend_labels = ax.get_legend_handles_labels()
        by_label = dict(zip(legend_labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, -0.3), loc="upper center", ncols=2)

        ax.minorticks_on()


    def compare_multiple_runs(self,
                              fnames: list[Path],
                              labels: list[str],
                              stage_idx: int = 0) -> None:
        """
        Simple function which groups the multi-run comparison postprocessing
        routines for ease-of-use.

        Parameters
        ----------
        - fnames : list[Path]
            A list of the paths of the results for the runs which
            are compared.
        - labels : list[str]
            A list of strings for the labels of each run in the plots.
        - stage_idx : int, optional
            An optional integer for the stage for which the profiles are to
            be compared.

        Returns
        -------
        None
        """

        # Visualise the complete evaluated objective space of the analyses
        self.create_multi_analysis_objective_space_plot(fnames,
                                                        labels)
        plt.show()
        plt.close('all')

        # Create meridional plots of the designs
        self.create_multi_analysis_meridional_plots(fnames,
                                                    labels)
        plt.show()
        plt.close('all')

        # Create a comparison plot of the optimised duct designs
        self.create_multi_analysis_duct_plots(fnames,
                                              labels)
        plt.show()
        plt.close('all')

        # Create blade profile plots for the optimised designs
        self.create_multi_analysis_blade_profile_plots(fnames,
                                                       labels,
                                                       stage_idx)
        plt.show()
        plt.close('all')

        # Create bar chart comparisons of key blading parameters
        self.create_multi_analysis_blading_bar_chart(fnames,
                                                     labels)
        plt.show()
        plt.close('all')

        # Create sectional blading parameter comparison plots
        self.create_multi_analysis_sectional_blading_plot(fnames,
                                                          labels)
        plt.show()
        plt.close('all')


    def main(self) -> None:
        """
        Main post-processing method. Only for single-analysis post-processing.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # Load in the results object and extract the population data to self
        res = self.load_res()
        self.extract_population_data(res)

        print("Original solution:")
        print(res.history[0].pop[0].get("F"))
        print("Optimum individual(s):")
        print(res.opt.get("X"))
        print("F:", res.opt.get("F"))
        print("G:", res.opt.get("G"))
        print("CV_max:", res.pop.get("CV").max())
        print("CV_avg:", res.pop.get("CV").mean())
        print("Evaluation time:", res.exec_time)

        # Visualise the convergence behavior of the solution
        self.generate_convergence_statistics(res)
        plt.show()
        plt.close('all')

        # Visualise the objective space
        self.plot_objective_space(res)
        plt.show()
        plt.close('all')

        # Plot the centerbody designs
        if config.OPTIMIZE_CENTERBODY:
            # First plot the complete final population
            self.compare_axisymmetric_geometry(config.CENTERBODY_VALUES,
                                               self.CB_data)
            plt.show()
            plt.close('all')

            # Plot the optimum solution set
            self.compare_axisymmetric_geometry(config.CENTERBODY_VALUES,
                                               self.CB_data_opt)
            plt.show()
            plt.close('all')

        # Plot the duct designs
        if config.OPTIMIZE_DUCT:
            # First plot the complete final population
            self.compare_axisymmetric_geometry(config.DUCT_VALUES,
                                               self.duct_data)
            plt.show()
            plt.close('all')

            # Plot the optimum solution set
            self.compare_axisymmetric_geometry(config.DUCT_VALUES,
                                               self.duct_data_opt)
            plt.show()
            plt.close('all')

        # Plot the optimised stage designs
        for i in range(len(config.OPTIMIZE_STAGE)):

            if config.OPTIMIZE_STAGE[i]:
                # First plot the complete final population
                self.compare_blading_data(config.STAGE_BLADING_PARAMETERS,
                                          self.blading_data)
                plt.show()
                plt.close('all')

                # Plot the optimum solution set
                self.compare_blading_data(config.STAGE_BLADING_PARAMETERS,
                                          self.blading_data_opt)
                plt.show()
                plt.close('all')

                # Plot the optimum solution set
                self.compare_blade_design_data(reference_design=config.STAGE_DESIGN_VARIABLES,
                                               res=res,
                                               individual="opt")
                plt.show()
                plt.close('all')

                # Plot the optimum solution set
                for j in range(len(self.blading_data_opt)):
                    self.create_blade_geometry_plots(self.blading_data_opt[j],
                                                     self.design_data_opt[j],
                                                     config.STAGE_BLADING_PARAMETERS,
                                                     config.STAGE_DESIGN_VARIABLES)
                    plt.show()
                    plt.close('all')


if __name__ == "__main__":
    # Initialize the post-processing class with an initial results file
    # This file is overwritten when using the compare_multiple_runs method
    output = Path('Results/res_pop100_eval11000_250628082743263171.dill')
    processing_class = PostProcessing(fname=output)

    # Multi-point single-objective results for the endurance mission
    # processing_class.compare_multiple_runs(fnames=[Path("Results/res_pop200_eval500_251124182952404967.dill")],
    #                                        labels=["Optimized"])

    # Single-point single-objective results for the endurance cruise condition. 
    processing_class.compare_multiple_runs(fnames=[Path("Results/res_pop200_eval500_251119042735540563.dill")],
                                           labels=["Optimized"])

