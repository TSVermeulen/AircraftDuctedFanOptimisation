"""
output_handling
=============

Description
-----------
This module provides classes and methods to process and visualise the output of MTFLOW in terms of the flowfield and boundary layer data.

Classes
-------
output_visualisation()
    A class to plot the streamline parameters and boundary layer data for the converged MTSOL case.
output_processing()
    A class responsible for the post-processing of the MTFLOW output data.

Examples
--------
>>> test = output_visualisation(analysis_name='test_case')
>>> create_individual_plots = False
>>> test.PlotOutputs(plot_individual=create_individual_plots)

Notes
-----
The CreateBoundaryLayerPLots() method is only executed if the boundary_layer.analysis_name file exists in the local working directory.

References
----------
None

Versioning
------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 1.3

Changelog:
- V1.0: Initial working version, containing only the plotting capabilities based on the flowfield.analysis_name and boundary_layer.analysis_name files. The output_processing() class is still a placeholder.
- V1.1: Added the output_processing() class to read the forces.analysis_name file and extract the thrust and power coefficients.
- V1.2: Updated GetAllVariables() method to remove empty strings to increase robustness and avoid runtime errors in case MTSOL.GetAvgValues() adds additional whitelines.
- V1.3: Fixed issue with file handling where regex patterns expected mandatory spaces, which would not be the case for negative values.
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
    This class handles the visualization of flowfield and boundary layer data from MTFLOW analysis.

    Methods
    -------
    - __init__(self, analysis_name: str = None) -> None
        Initializes the output_visualisation class with the given analysis name.

    - GetFlowfield(self) -> tuple[list[pd.DataFrame], pd.DataFrame]
        Loads the flowfield data from the flowfield.analysis_name file and returns it as a list of DataFrames for each streamline and a combined DataFrame.

    - GetBoundaryLayer(self) -> list[pd.DataFrame]
        Loads the boundary layer data from the boundary_layer.analysis_name file and returns it as a list of DataFrames for each surface.

    - ReadGeometry(self) -> list[np.ndarray]
        Reads the geometry data from the walls.analysis_name file and returns it as a list of numpy arrays for each axisymmetric body.

    - ReadBlades(self) -> list[np.ndarray[float]]
        Reads the blade geometries from the tflow.analysis_name file and returns them as a list of numpy arrays for each blade row.

    - CreateContours(self, df: pd.DataFrame, shapes: list[np.ndarray[float]], blades: list[np.ndarray[float]], figsize: tuple[float, float] = (6.4, 4.8), cmap: str = 'viridis') -> None
        Creates contour plots for each parameter in the flowfield data and overlays the axisymmetric body shapes and blade outlines.

    - CreateStreamlinePlots(self, blocks: list[pd.DataFrame], plot_individual_streamlines: bool = False) -> None
        Creates streamline plots for each parameter in the flowfield data, with options to plot individual streamlines.

    - CreateBoundaryLayerPlots(self, blocks: list[pd.DataFrame]) -> None
        Creates plots for each boundary layer quantity for the axisymmetric surfaces.

    - PlotOutputs(self, plot_individual: bool = False) -> None
        Generates all output plots for the analysis, including flowfield contours, streamline plots, and boundary layer plots (if applicable).
    """

    # Define the columns from the flowfield file
    FLOWFIELD_COLUMNS = ['x', 'y', 'rho/rhoinf', 'p/pinf', 'u/Uinf', 'v/Uinf', 'Vtheta/Uinf',
                         'q/Uinf', 'm/rhoinf Uinf', 'M', 'Cp', 'Cp0', '(q/Uinf)^2']

    # Define the columns from the boundary layer file
    BOUNDARY_LAYER_COLUMNS = ['x', 'r', 's', 'b0', 'Cp', 'Ue/Uinf', 'rhoe/rhoinf', 'Me', 'Hk', 'R_theta',
                              'delta*', 'theta', 'theta*', 'delta**', 'Cf/2', 'CD', 'ctau', 'm', 'P', 'K',
                              'Delta*', 'Theta', 'Theta*', 'Delta**', 'Gl', 'Gt']


    def __init__(self,
                 analysis_name: str) -> None:
        """
        Initialize the output_visualisation class.

        This method sets up the initial state of the class.

        Parameters
        ----------
        - analysis_name : str
            A string of the analysis name. Must equal the filename extension used for walls.xxx, tflow.xxx, tdat.xxx, boundary_layer.xxx, and flowfield.xxx.

        Returns
        -------
        None
        """

        self.analysis_name = analysis_name

        # Define key paths/directories
        self.parent_dir = Path(__file__).resolve().parent.parent
        self.submodels_path = self.parent_dir / "Submodels"

        # Validate if the required files exist
        self.flowfield_path = self.submodels_path / f"flowfield.{self.analysis_name}"
        self.walls_path = self.submodels_path / f"walls.{self.analysis_name}"
        self.tflow_path = self.submodels_path / f"tflow.{self.analysis_name}"
        self.boundary_layer_path = self.submodels_path / f"boundary_layer.{self.analysis_name}"

        if not self.flowfield_path.exists():
            raise FileNotFoundError(f"The required file flowfield.{self.analysis_name} was not found.")

        if self.walls_path.exists():
            self.walls = True
        else:
            self.walls = False
        if self.tflow_path.exists():
            self.tflow = True
        else:
            self.tflow = False

        # Check if the boundary layer file exists, and if so, set viscous_exists to True
        if self.boundary_layer_path.exists():
            self.viscous_exists = True
        else:
            self.viscous_exists = False

        # Set the maximum number of figures that can be opened before raising a warning
        plt.rcParams['figure.max_open_warning'] = 100


    def GetFlowfield(self) -> tuple[list[pd.DataFrame], pd.DataFrame]:
        """
        Load in the flowfield.analysis_name file and write it to a Pandas dataframe.

        Returns
        -------
        - tuple[list, pd.DataFrame] :
            - block_dfs : list[pd.DataFrame]
                List of nested DataFrames of the flow variables for each streamline.
            - df : pd.DataFrame
                A Pandas DataFrame containing the flowfield values across all streamlines.
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

            # Convert block data to DataFrame and add it to the list of block DataFrames
            block_df = pd.DataFrame(block_data, columns=self.FLOWFIELD_COLUMNS)
            block_dfs.append(block_df)

        #Construct the dataframe
        df = pd.DataFrame(all_data, columns=self.FLOWFIELD_COLUMNS)

        return block_dfs, df


    def GetBoundaryLayer(self) -> list[pd.DataFrame]:
        """
        Load in the boundary_layer.analysis_name file and write the data for each element to a Pandas dataframe.

        Returns
        - list[pd.DataFrame] :
            A list of nested DataFrames with the viscous variables for each boundary layer.
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

            # Convert block data to DataFrame and add it to the list of block DataFrames
            element_df = pd.DataFrame(element_data, columns=self.BOUNDARY_LAYER_COLUMNS)
            element_dfs.append(element_df)

        return element_dfs


    def ReadGeometry(self,
                     ) -> list[np.typing.NDArray[np.floating]]:
        """
        Read in the centrebody and duct geometry from the walls.analysis_name file

        Returns
        -------
        - shapes : list[np.typing.NDArray[np.floating]]
            A list of nested arrays, where each array contains the geometry of one of the axisymmetric bodies.
        """

        try:
            with open(self.walls_path, 'r') as file:
                lines = file.readlines()
        except IOError as e:
            raise IOError(f"Failed to read the geometry data: {e}") from e

        # Initialize an empty shapes and current_shape list
        shapes = []
        current_shape = []

        # Start reading in data from the 3rd line onwards, as the first 2 lines do not contain geometry points
        for line in lines[2:]:
            if "999.0    999.0" in line:
                shapes.append(np.array(current_shape))
                current_shape = []
            else:
                current_shape.append([float(x) for x in line.split()])
        if current_shape:
            shapes.append(np.array(current_shape))

        return shapes


    def ReadBlades(self,
                   ) -> list[np.typing.NDArray[np.floating]]:
        """
        Read the blade geometries from the tflow.analysis_name file.

        Returns
        -------
        - list[np.ndarray[float]]
            A collection of blade outlines where each outline is stored as a NumPy array
            containing the leading and trailing points.
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
        Create contour plots for every parameter in the flowfield.analysis_name file.
        Plots the axisymmetric bodies in dimgrey to generate the complete flowfield.

        Parameters
        ----------
        - df : pd.DataFrame
            The dataframe of the complete flowfield.
        - shapes : list[np.ndarray[float]]
            A nested list with the coordinates of all the axisymmetric bodies.
        - blades : list
            A nested list with the coordinates of the outlines of the rotor/stator blades in the domain.
        - figsize : tuple[float, float], optional
            A tuple with the figure size. Default value corresponds to the internal default of matplotlib.pyplot.
        - cmap : str, optional
            A string with the colourmap to be used for the contourplots. Default value is the viridis colourmap.

        Returns
        -------
        None
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
        Plot the total, interior, exterior, and optional individual streamlines for all logged parameters.

        Parameters
        ----------
        - plot_individual_streamlines : bool, optional
            A control boolean to determine if plots for each individual streamline should be generated. This is useful for debugging, but generates a very large amount of plots (11 plots times the number of streamlines, 45).
            Default is False.

        Returns
        -------
        None
        """

        # Close any existing figures to free memory
        plt.close('all')

        # Create streamline plots for all streamlines and all variables in self.FLOWFIELD_COLUMNS
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

            # Create plot window for interior streamlines, define plot title and axis labels
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

            # Create plot window for exterior streamlines, define plot title and axis labels
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
            # Create individual streamline plots for all variables in self.FLOWFIELD_COLUMNS
            for i,df in enumerate(blocks):
                if i != 0:
                    for param in self.FLOWFIELD_COLUMNS[2:]:  # Skipping x and y
                        # Create plot window, define plot tile and axis labels
                        plt.figure()
                        plt.title(f"{param} distribution for streamline {i + 1}")
                        plt.xlabel('Axial coordinate $x/L_{ref}$ [-]')
                        plt.ylabel(f'{param} [-]')

                        # Plot the streamline distribution
                        plt.plot(df['x'], df[param], ms=3, marker="x")

                        # Set grid and minor ticks
                        plt.minorticks_on()
                        plt.grid(which='both')

                    #Show all plots for the streamline
                    plt.show()


    def CreateBoundaryLayerPlots(self,
                                 blocks : list[pd.DataFrame]) -> None:
        """
        Plot the boundary layer quantities for each of the axi-symmetric surfaces

        Parameters
        ----------
        - blocks : list[pd.DataFrame]
            A nested list of dataframes containing the boundary layer quantities for each surface.

        Returns
        -------
        None
        """

        # Close any existing figures to free memory
        plt.close('all')

        # Create a plot for each boundary layer quantity, except the x and r coordinates.
        for param in self.BOUNDARY_LAYER_COLUMNS[2:]:  # skip x and r
            plt.figure()
            plt.title(f"{param} boundary layer distributions")
            plt.xlabel('Axial coordinate $x/L_{ref}$ [-]')
            plt.ylabel(f'{param} [-]')

            # Plot all streamlines
            for i, df in enumerate(blocks):
                plt.plot(df['x'], df[param], label=f'Surface {i + 1}', ms=3, marker="x")

            # Set grid and minor ticks
            plt.legend()
            plt.minorticks_on()
            plt.grid(which='both')

        plt.show()


    def PlotOutputs(self,
                    plot_individual: bool = False,
                    ) -> None:
        """
        Generate all output plots for the analysis.

        Parameters
        ----------
        - plot_individual : bool, optional
            A controlling boolean to determine if plots for each individual streamline should be generated.
            Default value is False.
        """

        # Load in the flowfield into blocks for each streamline and an overall dataframe
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

        # Load in the boundary layer data and create the boundary layer plots if a boundary layer data file exists
        if self.viscous_exists:
            boundary_layer_blocks = self.GetBoundaryLayer()
            self.CreateBoundaryLayerPlots(boundary_layer_blocks)


class output_processing:
    """
    A class responsible for post-processing MTFLOW output data.

    Methods
    -------
    - __init__(self, analysis_name: str = None)
        Initializes the output_processing class with the given analysis name and validates the existence of required files.

    - GetCTCPEtaP(self) -> tuple[float, float, float]
        Reads the forces.analysis_name file and extracts the thrust coefficient (CT), power coefficient (CP), and propulsive efficiency (EtaP).
    """

    def __init__(self,
                 analysis_name: str):
        """
        Class Initialisation.

        Parameters
        ----------
        - analysis_name : str
            A string of the analysis name. Must equal the filename extension used for walls.xxx, tflow.xxx, tdat.xxx, boundary_layer.xxx, and flowfield.xxx.
        """

        self.analysis_name = analysis_name

        # Define key paths/directories
        self.parent_dir = Path(__file__).resolve().parent.parent
        self.submodels_path = self.parent_dir / "Submodels"

        # Validate if the required forces file exist
        self.forces_path = self.submodels_path / f"forces.{self.analysis_name}"

        if not self.forces_path.exists():
            raise FileNotFoundError(f"The required file forces.{self.analysis_name} was not found.")


    def GetAllVariables(self,
                        output_type : int = 0,
                        ) -> dict[str, float | dict[str, float]]:
        """
        Read the forces.analysis_name file and return the variables and their values.

        Parameters
        ----------
        - output_type : int
            An integer indicating the type of output desired from the method:
            - '0' : All outputs
            - '1' : General Output data only
            - '2' : Element output data only

        Returns
        -------
        - output : dict[str, dict[str, float]]
            A nested dictionary containing:
            - oper : A dictionary containing the operating conditions
            - data : A dictionary containing the general output data
            - grouped_data : A dictionary containing the element breakdowns for the duct and centerbody
        """

        # Short sleep to ensure file has finished reading/writing to
        time.sleep(0.25)

        try:
            with open(self.forces_path, 'r') as file:
                # Read the file contents, and replace the newline characters with empty strings.
                # Also remove any empty lines from the list
                forces_file_contents = file.readlines()
                forces_file_contents = [s for s in forces_file_contents if s.strip()]
                forces_file_contents = [s.replace('\n', '') for s in forces_file_contents]
        except OSError as e:
            raise OSError(f"An error occurred opening the forces.{self.analysis_name} file: {e}") from e

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
        # Only search for the data if desired based on the output_type integer provided.
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
                viscous_inviscid_match = re.search(viscous_inviscid_pattern, line)
                if viscous_inviscid_match is not None:
                    data["Viscous CTv"] = viscous_inviscid_match.group(1)
                    data["Inviscid CTi"] = viscous_inviscid_match.group(2)
                else:
                    data["Viscous CTv"] = 0
                    data["Inviscid CTi"] = 0

            elif idx == 10 and output_type in (0, 1, 3):
                friction_pressure_match = re.search(friction_pressure_pattern, line)
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
        grouped_data = {key: {k: float(v) for k, v in value.items()} for key, value in grouped_data.items()}

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
            raise ValueError(f"Invalid output type passed to GetAllVariables: {output_type}. output type should be 0-2.")

        return output


    def GetCTCPEtaP(self) -> tuple[float, float, float]:
        """
        Read the forces.analysis_name file and return the thrust and power coefficients with the propulsive efficiency.

        Parameters
        ----------
        None

        Returns
        -------
        - tuple[float, float, float]
            A tuple of the form (CT, CP, EtaP) containing the thrust and power coefficients, together with the propulsive efficiency for the analysed case
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