"""
output_handling
=============

Description
-----------
This module provides classes and methods to process and visualise the output
of MTFLOW in terms of the flowfield and boundary layer data.

Classes
-------
output_visualisation()
    A class to plot the streamline parameters and boundary layer data for the
    converged MTSOL case.
output_processing()
    A class responsible for the post-processing of the MTFLOW output data.

Examples
--------
>>> test = output_visualisation(analysis_name='test_case')
>>> create_individual_plots = False
>>> test.PlotOutputs(plot_individual=create_individual_plots)

Notes
-----
The CreateBoundaryLayerPLots() method is only executed if the
boundary_layer.analysis_name file exists in the local working directory.

References
----------
None

Versioning
------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 1.4

Changelog:
- V1.0: Initial working version, containing only the plotting capabilities
        based on the flowfield.analysis_name and boundary_layer.analysis_name
        files. The output_processing() class is still a placeholder.
- V1.1: Added the output_processing() class to read the forces.analysis_name
        file and extract the thrust and power coefficients.
- V1.2: Updated GetAllVariables() method to remove empty strings to increase
        robustness and avoid runtime errors in case MTSOL.GetAvgValues() adds
        additional whitelines.
- V1.3: Fixed issue with file handling where regex patterns expected mandatory
        spaces, which would not be the case for negative values.
- V1.4: Revamped output_processing method to enable console-based output
        handling. Updated documentation and type hinting.
"""

# Import standard libraries
import re
import time
from pathlib import Path

# Import 3rd party libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class output_visualisation:
    """
    This class handles the visualization of flowfield and boundary layer data
    from MTFLOW analysis.
    """

    # Define the columns from the flowfield file
    FLOWFIELD_COLUMNS = ['x', 'y', 'rho/rhoinf', 'p/pinf', 'u/Uinf', 'v/Uinf',
                         'Vtheta/Uinf', 'q/Uinf', 'm/rhoinf Uinf', 'M', 'Cp',
                         'Cp0', '(q/Uinf)^2']

    # Define the columns from the boundary layer file
    BOUNDARY_LAYER_COLUMNS = ['x', 'r', 's', 'b0', 'Cp', 'Ue/Uinf',
                              'rhoe/rhoinf', 'Me', 'Hk', 'R_theta', 'delta*',
                              'theta', 'theta*', 'delta**', 'Cf/2', 'CD',
                              'ctau', 'm', 'P', 'K', 'Delta*', 'Theta',
                              'Theta*', 'Delta**', 'Gl', 'Gt']


    def __init__(self,
                 analysis_name: str) -> None:
        """
                 Initialize an output_visualisation instance for a given analysis and prepare input file paths and existence flags.
                 
                 Sets path attributes for the Submodels directory and expected files (flowfield, walls, tflow, boundary_layer), and records boolean flags indicating whether optional inputs (walls, tflow, boundary layer) are present. Raises FileNotFoundError if the required flowfield file is missing.
                 
                 Parameters:
                     analysis_name (str): Name/identifier of the analysis used as the filename extension (e.g., for file flowfield.<analysis_name>).
                 """

        self.analysis_name = analysis_name

        # Define key paths/directories
        self.parent_dir = Path(__file__).resolve().parent.parent
        self.submodels_path = self.parent_dir / "Submodels"

        # Validate if the required files exist
        self.flowfield_path = self.submodels_path / \
            f"flowfield.{self.analysis_name}"
        self.walls_path = self.submodels_path / \
            f"walls.{self.analysis_name}"
        self.tflow_path = self.submodels_path / \
            f"tflow.{self.analysis_name}"
        self.boundary_layer_path = self.submodels_path / \
            f"boundary_layer.{self.analysis_name}"

        if not self.flowfield_path.exists():
            raise FileNotFoundError(f"The file flowfield.{self.analysis_name}"
                                    " was not found.")

        if self.walls_path.exists():
            self.walls = True
        else:
            self.walls = False
        if self.tflow_path.exists():
            self.tflow = True
        else:
            self.tflow = False

        # Check if the boundary layer file exists, and if so,
        # set viscous_exists to True
        if self.boundary_layer_path.exists():
            self.viscous_exists = True
        else:
            self.viscous_exists = False

        # Set the maximum number of figures that can be opened before
        # raising a warning
        plt.rcParams['figure.max_open_warning'] = 100


    def GetFlowfield(self) -> tuple[list[pd.DataFrame], pd.DataFrame]:
        """
        Parse the flowfield file into per-streamline blocks and a single concatenated DataFrame.
        
        Reads the file at self.flowfield_path, splits content into blocks separated by blank lines, ignores lines that start with '#', and converts numeric rows into DataFrames using self.FLOWFIELD_COLUMNS.
        
        Returns:
            tuple[list[pd.DataFrame], pd.DataFrame]:
                - block_dfs: List of DataFrames, one per streamline block containing flowfield variables.
                - df: A single DataFrame containing all flowfield rows concatenated.
        
        Raises:
            IOError: If the flowfield file cannot be read.
        """

        try:
            with open(self.flowfield_path, 'r') as file:
                data = file.read()
        except IOError as e:
            raise IOError(f"Failed to read the flowfield data: {e}") from e

        # Split the data into blocks for each streamline
        blocks = data.strip().split('\n\n')
        all_data = []
        block_dfs = []

        # Load in the numbers but not text or comments
        for block in blocks:
            block_data = []
            lines = block.strip().split('\n')
            for line in lines:
                if not line.startswith('#'):
                    all_data.append([float(x) for x in line.split()])
                    block_data.append([float(x) for x in line.split()])

            # Convert block data to DataFrame and add it to the list of
            # block DataFrames
            block_df = pd.DataFrame(block_data, columns=self.FLOWFIELD_COLUMNS)
            block_dfs.append(block_df)

        #Construct the dataframe
        df = pd.DataFrame(all_data, columns=self.FLOWFIELD_COLUMNS)

        return block_dfs, df


    def GetBoundaryLayer(self) -> list[pd.DataFrame]:
        """
        Parse the boundary_layer.{analysis_name} file into a list of DataFrames, one per boundary-layer element.
        
        The file is split into blocks separated by blank lines; comment lines starting with `#` are ignored. Each returned DataFrame uses the column names defined by `self.BOUNDARY_LAYER_COLUMNS`.
        
        Returns:
            list[pd.DataFrame]: List of DataFrames, one per boundary-layer element, each containing the viscous variables.
        """

        try:
            with open(self.boundary_layer_path, 'r') as file:
                data = file.read()
        except IOError as e:
            raise IOError(f"Failed to read the boundary layer data: {e}") from e

        # Split the data into blocks for each streamline
        blocks = data.strip().split('\n\n')
        element_dfs = []

        # Load in the numbers but not text or comments
        for block in blocks:
            element_data = []
            lines = block.strip().split('\n')
            for line in lines:
                if not line.startswith('#'):
                    element_data.append([float(x) for x in line.split()])

            # Convert block data to DataFrame and add it to the list of
            # block DataFrames
            element_df = pd.DataFrame(element_data,
                                      columns=self.BOUNDARY_LAYER_COLUMNS)
            element_dfs.append(element_df)

        return element_dfs


    def ReadGeometry(self) -> list[np.typing.NDArray[np.floating]]:
        """
        Read axisymmetric body outlines from the walls.{analysis_name} file.
        
        Each returned element is a NumPy array of shape points (rows of floating-point coordinates) for one body. The file is read starting at the third line; sections are separated by a line containing "999.0    999.0". Empty trailing sections are ignored.
        
        Returns:
            list[np.typing.NDArray[np.floating]]: A list where each entry is a 2D NumPy array of float coordinates for a single body.
        
        Raises:
            IOError: If the walls file cannot be opened or read.
        """

        try:
            with open(self.walls_path, 'r') as file:
                lines = file.readlines()
        except IOError as e:
            raise IOError(f"Failed to read the geometry data: {e}") from e

        # Initialize an empty shapes and current_shape list
        shapes = []
        current_shape = []

        # Start reading in data from the 3rd line onwards, as the first 2
        # lines do not contain geometry points
        for line in lines[2:]:
            if "999.0    999.0" in line:
                shapes.append(np.array(current_shape))
                current_shape = []
            else:
                current_shape.append([float(x) for x in line.split()])
        if current_shape:
            shapes.append(np.array(current_shape))

        return shapes


    def ReadBlades(self) -> list[np.typing.NDArray[np.floating]]:
        """
        Parse blade outlines from the tflow.{analysis_name} file and return per-stage blade point arrays.
        
        Each stage yields a NumPy array of 2D points (x, y) describing the blade outline for that stage; within each array the first point corresponds to the leading point and the last point corresponds to the trailing point. Sections in the file are delimited by the keywords "STAGE", "SECTION", and "END".
        
        Returns:
            list[np.typing.NDArray[np.floating]]: A list where each element is an (N, 2) NumPy array of points for a stage's blade outline.
        
        Raises:
            IOError: If the tflow file cannot be read.
        """

        try:
            with open(self.tflow_path, 'r') as file:
                lines = file.readlines()
        except IOError as e:
            raise IOError(f"Failed to read the tflow file: {e}") from e

        # Create blade outlines for each blade row (i.e. each stage)
        stages_outlines = []

        current_outline = []
        in_section = False

        for line in lines:
            if line.strip() == "STAGE":
                if current_outline:
                    stages_outlines.append(np.array(current_outline))
                current_outline = []
                in_section = False
            elif line.strip() == "SECTION":
                in_section = True
            elif line.strip() == "END":
                in_section = False
                if current_outline:
                    stages_outlines.append(np.array(current_outline))
                    current_outline = []
            elif in_section:
                points = [float(x) for x in line.split()]
                # Extract only the leading (first) and trailing (last) points
                if not current_outline:
                    current_outline.append(points[:2])  # Leading point
                current_outline.append(points[:2])  # Trailing point

        # Append the last outline if any
        if current_outline:
            stages_outlines.append(np.array(current_outline, dtype=float))

        return stages_outlines


    def CreateContours(self,
                       df: pd.DataFrame,
                       shapes: list[np.typing.NDArray[np.floating]],
                       blades: list[np.typing.NDArray[np.floating]],
                       figsize: tuple[float, float] = (6.4, 4.8),
                       cmap: str = 'viridis',
                       ) -> None:
        """
                       Generate filled contour plots for each flowfield variable and overlay axisymmetric bodies and blade outlines.
                       
                       Parameters:
                           df (pd.DataFrame): Complete flowfield data containing columns 'x', 'y' and variables named in self.FLOWFIELD_COLUMNS.
                           shapes (list[np.typing.NDArray[np.floating]]): List of (N,2) arrays for axisymmetric body coordinates to fill on the plots.
                           blades (list[np.typing.NDArray[np.floating]]): List of (M,2) arrays for blade outline coordinates to plot over the contours.
                           figsize (tuple[float, float], optional): Figure size passed to matplotlib; defaults to (6.4, 4.8).
                           cmap (str, optional): Matplotlib colormap name used for filled contours; defaults to 'viridis'.
                       """

        # Close any existing figures to free memory
        plt.close('all')

        # Create a contour plot for every variable
        for var in self.FLOWFIELD_COLUMNS[2:]:
            plt.figure(figsize=figsize)
            plt.tricontourf(df['x'],
                            df['y'],
                            df[var],
                            levels=100,
                            cmap=cmap,
                            )
            plt.colorbar(label=var + ' [-]')

            for shape in shapes:
                plt.fill(shape[:,0], shape[:,1], 'dimgrey')

            for blade in blades:
                plt.plot(blade[:,0], blade[:,1], 'k-.')

            plt.xlabel('Axial coordinate $x/L_{ref}$ [-]')
            plt.ylabel('Radial coordinate $r/L_{ref}$ [-]')
            plt.ylim(bottom=0)
            plt.minorticks_on()
            plt.grid()
            plt.title(f'Contour Plot of {var}')

        plt.show()


    def CreateStreamlinePlots(self,
                              blocks: list[pd.DataFrame],
                              plot_individual_streamlines: bool = False,
                              ) -> None:
        """
                              Plot streamline distributions for each flowfield parameter grouped as total, interior, and exterior streams, and optionally create per-streamline plots.
                              
                              Parameters:
                                  blocks (list[pd.DataFrame]): List of DataFrames, each containing flowfield quantities for a single streamline (columns include 'x', 'Vtheta/Uinf' and other parameters from FLOWFIELD_COLUMNS).
                                  plot_individual_streamlines (bool): If True, generate separate plots for every individual streamline and parameter (can produce many figures). Default is False.
                              """

        # Close any existing figures to free memory
        plt.close('all')

        # Create streamline plots for all streamlines and all variables in
        # self.FLOWFIELD_COLUMNS
        for param in self.FLOWFIELD_COLUMNS[2:]:  # Skipping x and y
            # Create plot window, define plot title and axis labels
            plt.figure()
            plt.title(f"{param} streamline distribution")
            plt.xlabel('Axial coordinate $x/L_{ref}$ [-]')
            plt.ylabel(f'{param} [-]')

            # Plot all streamlines
            for i, df in enumerate(blocks):
                plt.plot(df['x'], df[param], label=f'Streamline {i + 1}')

            # Set grid and minor ticks
            plt.minorticks_on()
            plt.grid(which='both')

            # Create plot window for interior streamlines,
            # define plot title and axis labels
            plt.figure()
            plt.title(f"{param} interior streamline distribution")
            plt.xlabel('Axial coordinate $x/L_{ref}$ [-]')
            plt.ylabel(f'{param} [-]')

            # Plot interior streamlines
            for i, df in enumerate(blocks):
                if (df["Vtheta/Uinf"].abs() > 0).any():
                    plt.plot(df['x'], df[param], label=f'Streamline {i + 1}')

            # Set grid and minor ticks
            plt.minorticks_on()
            plt.grid(which='both')

            # Create plot window for exterior streamlines,
            # define plot title and axis labels
            plt.figure()
            plt.title(f"{param} exterior streamline distribution")
            plt.xlabel('Axial coordinate $x/L_{ref}$ [-]')
            plt.ylabel(f'{param} [-]')

            # Plot exterior streamlines
            for i, df in enumerate(blocks):
                if not (df["Vtheta/Uinf"].abs() > 0).any():
                    plt.plot(df['x'], df[param], label=f'Streamline {i + 1}')

            # Set grid and minor ticks
            plt.minorticks_on()
            plt.grid(which='both')

            #Show all streamline plots
            plt.show()

        if plot_individual_streamlines:
            # Create individual streamline plots for all variables in
            # self.FLOWFIELD_COLUMNS
            for i,df in enumerate(blocks):
                if i != 0:
                    for p in self.FLOWFIELD_COLUMNS[2:]:  # Skipping x and y
                        # Create plot window, define plot tile and axis labels
                        plt.figure()
                        plt.title(f"{p} distribution for streamline {i + 1}")
                        plt.xlabel('Axial coordinate $x/L_{ref}$ [-]')
                        plt.ylabel(f'{p} [-]')

                        # Plot the streamline distribution
                        plt.plot(df['x'], df[p], ms=3, marker="x")

                        # Set grid and minor ticks
                        plt.minorticks_on()
                        plt.grid(which='both')

                    #Show all plots for the streamline
                    plt.show()


    def CreateBoundaryLayerPlots(self,
                                 blocks : list[pd.DataFrame]) -> None:
        """
                                 Plot boundary-layer profiles for each axisymmetric surface.
                                 
                                 Each DataFrame in `blocks` is expected to contain an 'x' column and the boundary-layer parameters named in `self.BOUNDARY_LAYER_COLUMNS`. The function creates one figure per boundary-layer parameter (excluding 'x' and 'r') and plots that parameter versus axial coordinate for every surface.
                                 
                                 Parameters:
                                     blocks (list[pd.DataFrame]): List of DataFrames, one per surface, containing boundary-layer data.
                                 
                                 Returns:
                                     None
                                 """

        # Close any existing figures to free memory
        plt.close('all')

        # Create a plot for each boundary layer quantity,
        # except the x and r coordinates.
        for param in self.BOUNDARY_LAYER_COLUMNS[2:]:  # skip x and r
            plt.figure()
            plt.title(f"{param} boundary layer distributions")
            plt.xlabel('Axial coordinate $x/L_{ref}$ [-]')
            plt.ylabel(f'{param} [-]')

            # Plot all streamlines
            for i, df in enumerate(blocks):
                plt.plot(df['x'], df[param],
                         label=f'Surface {i + 1}', ms=3, marker="x")

            # Set grid and minor ticks
            plt.legend()
            plt.minorticks_on()
            plt.grid(which='both')

        plt.show()


    def PlotOutputs(self,
                    plot_individual: bool = False) -> None:
        """
                    Generate and display all analysis plots for the current analysis.
                    
                    Loads flowfield data and, if available, geometry and blade outlines to create contour plots;
                    generates streamline plots (optionally per-streamline) and boundary layer plots when viscous data exists.
                    
                    Parameters:
                        plot_individual (bool): If True, produce individual per-streamline plots in addition to aggregated streamline plots. Default False.
                    """

        # Load in the flowfield into blocks for each streamline and an
        # overall dataframe
        blocks, df = self.GetFlowfield()

        # Create contour plots from the flowfield
        if self.walls and self.tflow:
            # Read in the axi-symmetric geometry
            bodies = self.ReadGeometry()
            # Read in the blade outlines
            blades = self.ReadBlades()

            self.CreateContours(df, bodies, blades)

        # Create the streamline plots
        self.CreateStreamlinePlots(blocks,
                                   plot_individual_streamlines=plot_individual)

        # Load in the boundary layer data and create the boundary layer plots
        # if a boundary layer data file exists
        if self.viscous_exists:
            boundary_layer_blocks = self.GetBoundaryLayer()
            self.CreateBoundaryLayerPlots(boundary_layer_blocks)


class output_processing:
    """
    A class responsible for post-processing MTFLOW output data.
    """

    def __init__(self,
                 analysis_name: str | None = None):
        """
                 Initialize the output_processing helper, set up repository paths, and locate the forces file for a given analysis.
                 
                 When an analysis name is provided, constructs the Submodels forces file path (forces.<analysis_name>) and verifies it exists.
                 
                 Parameters:
                     analysis_name (str | None): Analysis identifier used as the forces file extension; if None, no forces path is set.
                 
                 Raises:
                     FileNotFoundError: If an analysis_name is provided but the corresponding forces.<analysis_name> file does not exist under Submodels.
                 """

        self.analysis_name = analysis_name

        # Define key paths/directories
        self.parent_dir = Path(__file__).resolve().parent.parent
        self.submodels_path = self.parent_dir / "Submodels"

        # Validate if the required forces file exist
        self.forces_path = None
        if self.analysis_name is not None:
            self.forces_path = self.submodels_path / \
                f"forces.{self.analysis_name}"

            if not self.forces_path.exists():
                raise FileNotFoundError(f"The file "
                                        f"forces.{self.analysis_name} "
                                        "was not found.")


    def GetAllVariables(self,
                        output_type: int = 0,
                        forces_data: list[str] | None = None,
                        ) -> dict[str, float | dict[str, float]]:
        """
                        Parse forces output lines and return selected summary and element-level force metrics.
                        
                        Parameters:
                            output_type (int): Which results to return:
                                0 — both general data and grouped element breakdowns (default);
                                1 — general output data only;
                                2 — grouped element/axis breakdowns only.
                            forces_data (list[str] | None): Optional list of lines containing forces output to parse.
                                If None, the method reads from the instance's forces file path and may raise ValueError
                                if the path is not defined.
                        
                        Returns:
                            dict[str, float | dict[str, float]]: When output_type is 0, a dictionary with keys
                            "data" and "grouped_data":
                                - "data": mapping of named general outputs (e.g., "Total force CT", "Total power CP",
                                          "EtaP", "Pressure Ratio", "Wetted Area", etc.) to floats.
                                - "grouped_data": mapping of element/region names (e.g., "Element 2", "Axis Body") to
                                  dictionaries of their breakdown values (e.g., "CTf", "CTp", "Xtr", etc.) as floats.
                            When output_type is 1, returns the "data" mapping only. When output_type is 2, returns the
                            "grouped_data" mapping only.
                        
                        Raises:
                            ValueError: if forces_data is None and the instance has no forces path, or if output_type is not 0, 1, or 2.
                            OSError: if reading the forces file fails.
                        """

        # Only read the forces data from a file if it is not
        # provided as an argument
        if forces_data is None:
            if self.forces_path is None:
                raise ValueError("Forces path is not defined. Please provide "
                                 "forces_data or initialize the class with "
                                 "an analysis_name.")
            
            # Short sleep to ensure file has finished reading/writing to
            time.sleep(0.25)

            try:
                with open(self.forces_path, 'r') as file:
                    # Read the file contents
                    forces_file_contents = file.readlines()
            except OSError as e:
                raise OSError(f"Failed to open forces file") from e
        else:
            forces_file_contents = forces_data

        # Replace the newline characters with empty strings.
        # Also remove any empty lines from the list
        forces_file_contents = [s.replace('\n', '').strip()
                                for s in forces_file_contents if s.strip()]

        # Define a unified number pattern
        number_pattern = r"(?:[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?Infinity)"

        # Define regex patterns.
        total_CP_etaP_pattern = fr"CP\s*=\s*({number_pattern})\s+EtaP\s*=\s*({number_pattern})"
        total_CT_pattern = fr"Total force\s+CT\s*=\s*({number_pattern})"
        top_CTV_pattern = fr"top CTV\s*=\s*({number_pattern})"
        bot_CTV_pattern = fr"bot CTv\s*=\s*({number_pattern})"
        axis_body_CTV_pattern = fr"Axis body\s+CTv\s*=\s*({number_pattern})"
        viscous_inviscid_pattern = fr"CTv\s*=\s*({number_pattern})\s+CTi\s*=\s*({number_pattern})"
        friction_pressure_pattern = fr"CTf\s*=\s*({number_pattern})\s+CTp\s*=\s*({number_pattern})"
        element_breakdown_pattern = (
            fr"CTf\s*=\s*({number_pattern})\s+CTp\s*=\s*({number_pattern})"
            fr"\s+top Xtr\s*=\s*({number_pattern})\s+bot Xtr\s*=\s*({number_pattern})"
        )
        axis_body_breakdown_pattern = fr"CTf\s*=\s*({number_pattern})\s+CTp\s*=\s*({number_pattern})\s+Xtr\s*=\s*({number_pattern})"
        P_ratio_pattern = fr"Pexit/Po\s*=\s*({number_pattern})"
        wetted_area_pattern = fr"Total\s*:\s*({number_pattern})"

        # Initialise output dictionaries.
        data = {}
        grouped_data = {}

        # Use regex to extract values from the line.
        # Only search for the data if desired based on the output_type
        # integer provided.
        for idx, line in enumerate(forces_file_contents):

            if idx == 0:
                continue

            elif idx == 3 and output_type in (0, 1, 3):
                match = re.search(total_CP_etaP_pattern, line)
                if match is not None:
                    data["Total power CP"] = match.group(1)
                    data["EtaP"] = match.group(2)
                else:
                    data["Total power CP"] = 0
                    data["EtaP"] = 0

            elif idx == 4 and output_type in (0, 1, 3):
                match = re.search(total_CT_pattern, line)
                if match is not None:
                    data["Total force CT"] = match.group(1)
                else:
                    data["Total force CT"] = 0

            elif idx == 5 and output_type in (0, 1, 3):
                match = re.search(top_CTV_pattern, line)
                if match is not None:
                    data["Element 2 top CTV"] = match.group(1)
                else:
                    data["Element 2 top CTV"] = 0

            elif idx == 6 and output_type in (0, 1, 3):
                match = re.search(bot_CTV_pattern, line)
                if match is not None:
                    data["Element 2 bot CTV"] = match.group(1)
                else:
                    data["Element 2 bot CTV"] = 0

            elif idx == 7 and output_type in (0, 1, 3):
                match = re.search(axis_body_CTV_pattern, line)
                if match is not None:
                    data["Axis body CTV"] = match.group(1)
                else:
                    data["Axis body CTV"] = 0

            elif idx == 9 and output_type in (0, 1, 3):
                viscous_inviscid_match = re.search(viscous_inviscid_pattern,
                                                   line)
                if viscous_inviscid_match is not None:
                    data["Viscous CTv"] = viscous_inviscid_match.group(1)
                    data["Inviscid CTi"] = viscous_inviscid_match.group(2)
                else:
                    data["Viscous CTv"] = 0
                    data["Inviscid CTi"] = 0

            elif idx == 10 and output_type in (0, 1, 3):
                friction_pressure_match = re.search(friction_pressure_pattern,
                                                    line)
                if friction_pressure_match is not None:
                    data["Friction CTf"] = friction_pressure_match.group(1)
                    data["Pressure CTp"] = friction_pressure_match.group(2)
                else:
                    data["Friction CTf"] = 0
                    data["Pressure CTp"] = 0

            elif idx == 11 and output_type in (0, 2, 3):
                match = re.search(element_breakdown_pattern, line)
                if match is not None:
                    CTf = match.group(1)
                    CTp = match.group(2)
                    top_Xtr = match.group(3)
                    bot_Xtr = match.group(4)
                    grouped_data["Element 2"] = {"CTf": CTf,
                                                "CTp": CTp,
                                                "top Xtr": top_Xtr,
                                                "bot Xtr": bot_Xtr}
                else:
                    grouped_data["Element 2"] = {"CTf": 0,
                                                "CTp": 0,
                                                "top Xtr": 0,
                                                "bot Xtr": 0}

            elif idx == 12 and output_type in (0, 2, 3):
                match = re.search(axis_body_breakdown_pattern, line)
                if match is not None:
                    CTf = match.group(1)
                    CTp = match.group(2)
                    Xtr = match.group(3)
                    grouped_data["Axis Body"] = {"CTf": CTf,
                                                "CTp": CTp,
                                                "Xtr": Xtr}
                else:
                    grouped_data["Axis Body"] = {"CTf": 0,
                                                "CTp": 0,
                                                "Xtr": 0}

            elif idx == 14 and output_type in (0, 1, 3):
                match = re.search(P_ratio_pattern, line)
                if match is not None:
                    data["Pressure Ratio"] = match.group(1)
                else:
                    data["Pressure Ratio"] = 0

            elif idx == 21 and output_type in (0, 1, 3):
                match = re.search(wetted_area_pattern, line)
                if match is not None:
                    data["Wetted Area"] = match.group(1)
                else:
                    data["Wetted Area"] = 0

        # Convert contents of all dictionaries to floats
        data = {key: float(value) for key, value in data.items()}
        grouped_data = {key: {k: float(v) for k, v in value.items()} for key,
                        value in grouped_data.items()}

        # Construct output dictionary
        output = {}
        if output_type == 0:
            output["data"] = data
            output["grouped_data"] = grouped_data
        elif output_type == 1:
            output = data
        elif output_type == 2:
            output = grouped_data
        else:
            raise ValueError(f"Invalid output type: {output_type}."
                             " Output type should be 0-2.")

        return output


    def GetCTCPEtaP(self) -> tuple[float, float, float]:
        """
        Extract total thrust coefficient, total power coefficient, and propulsive efficiency from the parsed forces output.
        
        Returns:
            tuple (CT, CP, EtaP): CT is the total thrust coefficient, CP is the total power coefficient, and EtaP is the propulsive efficiency for the analysed case.
        """

        data = self.GetAllVariables(1)

        total_CP = data["Total power CP"]
        EtaP = data["EtaP"]
        total_CT = data["Total force CT"]

        return total_CT, total_CP, EtaP


if __name__ == "__main__":
    # Example usage for the output_visualisation class
    # test = output_visualisation(analysis_name='x22a_validation')

    # create_individual_plots = True
    # test.PlotOutputs(plot_individual=create_individual_plots)

    # Example usage for the output_processing class
    start = time.monotonic()
    test = output_processing(analysis_name='f')
    test.GetAllVariables(0)
    print(time.monotonic() - start)
    test.GetCTCPEtaP()