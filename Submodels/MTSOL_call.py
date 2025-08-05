"""
MTSOL_call
=============

Description
-----------
This module provides an interface to interact with the MTSOL executable from Python.
It creates a subprocess for the MTSOL executable, executes an inviscid, and viscid if desired, solve,
and writes the output data to the state file and the forces.xxx output file.

Classes
-------
ExitFlag
    An Enum class with exit flags for the MTSOL interface
MTSOL_call
    A class to handle the interface between Python and the MTSOL executable.

Examples
--------
>>> analysisName = "test_case"
>>> oper = {"Inlet_Mach": 0.25,
>>>         "Inlet_Reynolds": 5.000E6,
>>>         "N_crit": 9,
>>>         }
>>> test = MTSOL_call(oper, analysisName).caller(run_viscous=True)

Notes
-----
This module is designed to work with the MTSOL executable. Ensure that the executable and the input state file, tdat.xxx,
are present in the same directory as this Python file. When executing the file as a standalone, it uses the inputs
and calls contained within the if __name__ == "__main__" section. This part also imports the time module to measure
the time needed to perform each file generation call. This is beneficial in runtime optimization.

References
----------
The required input data, limitations, and structures are documented within the MTFLOW user manual:
https://web.mit.edu/drela/Public/web/mtflow/mtflow.pdf

Versioning
------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 1.5

Changelog:
- V0.0: File created with empty class as placeholder.
- V0.9: Minimum Working Example. Lacks crash handling and pressure ratio definition.
- V0.9.5: Cleaned up inputs, removing file_path and changing it to a constant.
- V1.0: With implemented choking handling, pressure ratio definition is no longer needed. Added choking exit flag. Cleaned up/updated HandleExitFlag() method. Added critical amplification factor as input.
- V1.1: Added file processing check to ensure that the forces file is copied before it is deleted. Added a check to ensure that the MTSOL executable is present in the same directory as this Python file.
- V1.2: Added watchdog to check if the forces file is created before copying it.
- V1.3: Added crash handling for the inviscid solve. Added a function to set all values to zero in case of a crash during the inviscid solve. Added a function to handle non-convergence by averaging over the last self.SAMPLE_SIZE iterations. Added a function to handle the exit flag of the solver execution. Added a function to handle the convergence of individual surfaces in case of a crash during the viscous solve.
- V1.4: Full rework of FileCreatedHandling(). Revamped file processing. Cleaned up imports. Removed shell=True from process initialisation. Switched to pathlib for path handling. Revamped individual surface convergence
- V1.4.1: Removed iter_count from outputs.
- V1.5: Implemented efficiency check to verify inviscid solution is a physical result. This enables skipping of viscous solve for such cases, which can significantly speed up computations in batch analyses.
"""

# Import standard libraries
import subprocess, os, re, time, shutil, queue, threading
from pathlib import Path
from collections import OrderedDict
from contextlib import ExitStack
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from enum import Enum

# Conditionally load the correct locking module based on the operating system
if os.name == 'nt':
    import msvcrt
else:
    import fcntl

# Import 3rd party libraries
import numpy as np

# Import custom libraries
from output_handling import output_processing


class FileCreatedHandling(FileSystemEventHandler):
    """
    Simple class to handle checking if forces.analysis_name file has been modified.
    """

    # File waiting constants
    BACKOFF = 0.01
    MAX_BACKOFF = 0.1

    def __init__(self,
                 file_path: Path,
                 destination: Path) -> None:
        self.file_path = file_path
        self.destination = destination
        self.file_processed = False


    def is_file_free(self, file_path: Path) -> bool:
        """
        Checks if the file is free by attempting to lock a small portion of it.
        On Windows, it uses msvcrt.locking; on Unix-like systems, it uses fcntl.flock.

        Returns True if the lock can be acquired (indicating the file is likely free),
        otherwise False.
        """
        try:
            with open(file_path, 'rb') as f:
                if os.name == 'nt':  # Windows
                    # Try to lock the first byte in a non-blocking way.
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    # Immediately unlock.
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)

                else:
                    # Attempt a non-blocking exclusive lock.
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # Release the lock.
                    fcntl.flock(f, fcntl.LOCK_UN)
            return True
        except (IOError, OSError, PermissionError):
            return False


    def wait_until_file_free(self,
                             file_path: Path,
                             timeout: float = 5) -> bool:
        """ Helper function to wait until the file_path has finished being written to, and is available. """
        start_time = time.monotonic()
        backoff = self.BACKOFF
        while (time.monotonic() - start_time) < timeout:
            if self.is_file_free(file_path):
                return True
            time.sleep(backoff)
            # Adjust the backoff based on remaining time to avoid overshooting the timeout
            backoff = min(self.MAX_BACKOFF, backoff * 2)
        return False


    def on_modified(self, event):
        if Path(event.src_path).name == self.file_path.name:
            if self.wait_until_file_free(self.file_path):
                shutil.copy(self.file_path, self.destination)
                self.file_processed = True
            else:
                print(f"Warning: File {self.file_path} was still busy after timeout")
                self.file_processed = True  # Still mark the file as processed to avoid hanging


    def is_file_processed(self) -> bool:
        """
        Return whether the file has been processed.

        Returns
        -------
        - bool
            True if the file has been processed, False otherwise.
        """
        return self.file_processed


class ExitFlag(Enum):
    """
    Enum class to define the exit flags for the MTSOL solver.

    The exit flags are used to determine the status of the solver execution.

    Attributes
    ----------
    SUCCESS : int
        Successful completion of the solver execution.
    CRASH : int
        MTSOL crash - likely related to the grid resolution.
    NON_CONVERGENCE : int
        Non-convergence, to be handled by the HandleNonConvergence function.
    NOT_PERFORMED : int
        Not performed, with no iterations executed or outputs generated.
    COMPLETED : int
        Finished the iteration/operation, but not converged.
    CHOKING : int
        Choking occurs somewhere in the solution, indicated by the 'QSHIFT' message in the MTSOL console output
    """

    SUCCESS = -1
    CRASH = 0
    NON_CONVERGENCE = 1
    NOT_PERFORMED = 2
    COMPLETED = 3
    CHOKING = 4


class OutputType(Enum):
    """
    Enum class to define the output types to be generated by MTSOL
    """

    FORCES_ONLY = 0
    ALL_FILES = 1


class CompletionType(Enum):
    """
    Enum class to define the completion type to be checked for in MTSOL.
    """

    ITERATION = 1
    OUTPUT = 2
    PARAM_CHANGE = 3
    VISCOUS_TOGGLE = 4


class MTSOL_call:
    """
    Class to handle the interface between MTSOL and Python
    """

    # Define file templates as constants
    FILE_TEMPLATES = {'forces': "forces.{}",
                      'flowfield': "flowfield.{}",
                      'boundary_layer': "boundary_layer.{}"}

    def __init__(self,
                 operating_conditions: dict,
                 analysis_name: str,
                 ) -> None:
        """
        Initialize the MTSOL_call class.

        This method sets up the initial state of the class.

        Parameters
        ----------
        - operating_conditions : dict
            A dictionary containing the operating conditions for the MTSOL analysis. The dictionary needs to contain:
                - Inlet_Mach: the inlet Mach number
                - Inlet_Reynolds: the inlet Reynolds number, calculated using L=1m
                - N_crit: the critical amplification factor
        - analysis_name : str
            A string of the analysis name.

        Returns
        -------
        None
        """

        self.operating_conditions = operating_conditions
        self.analysis_name = analysis_name

        # Define constants for the class
        self.ITER_STEP_SIZE = 2  # Step size in which iterations are performed in MTSOL
        self.SAMPLE_SIZE = 10  # Number of iterations to use to average over in case of non-convergence.
        self.ITER_LIMIT_INVISC = 50  # Maximum number of iterations to perform before non-convergence is assumed in an inviscid solution.
        self.ITER_LIMIT_VISC = 50  # Maximum number of iterations to perform before non-convergence is assumed in a viscous solution

        # Define key paths/directories
        self.submodels_path = Path(__file__).resolve().parent

        # Define filepath of MTSOL as being in the same folder as this Python file
        self.fpath = self.submodels_path / 'mtsol.exe'
        if not self.fpath.exists():
            raise FileNotFoundError("MTSOL executable not found at {}".format(self.fpath))

        # Define filepaths
        self.filepaths = {'forces': self.submodels_path / self.FILE_TEMPLATES['forces'].format(self.analysis_name),
                          'flowfield': self.submodels_path / self.FILE_TEMPLATES['flowfield'].format(self.analysis_name),
                          'boundary_layer': self.submodels_path / self.FILE_TEMPLATES['boundary_layer'].format(self.analysis_name)}

        # Define the dump folder to write non-converged forces output files to
        self.dump_folder = self.submodels_path / "MTSOL_output_files"
        self.dump_folder.mkdir(exist_ok=True,
                               parents=True)

        # Define forces non-convergence file template
        self.forces_template_nonconv = 'forces.{}.{}'

        # Define regular expression patterns for the GetAverageValues method
        self.var_value_pattern = re.compile(r'([\w\s]+?)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')  # Regular expression for key=value pairs.We use (?:...) for the exponent part so that findall returns just two groups.
        self.value_pattern = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')  # Regular expression for generic numeric value (scientific-notated numbers, etc.)


    def StdinWrite(self,
                   command: str) -> None:
        """
        Simple function to write commands to the subprocess stdin in order to pass commands to MTSOL

        Parameters
        ----------
        - command : str
            The text-based command to pass to MTSOL

        Returns
        -------
        None
        """

        self.process.stdin.write(f"{command} \n")
        self.process.stdin.flush()


    def GenerateProcess(self,
                        ) -> None:
        """
        Create MTSOL subprocess

        Requires that the executable, mtsol.exe, and the input file, tdat.xxx are present in the same directory as this
        Python file.
        """

        # Add a shutdown event to signal thread termination or reset the existing one
        if not hasattr(self, "shutdown_event"):
            self.shutdown_event = threading.Event()
        else:
            self.shutdown_event.clear()

        # Stop any orphaned reader threads if they exist before starting the new subprocess
        if getattr(self, "reader", None) and self.reader.is_alive():
            if getattr(self, "process", None) and self.process.stdout:
                self.process.stdout.close()
            # Signal the thread to stop
            self.shutdown_event.set()
            self.reader.join(timeout=5)
            self.reader = None

        # Generate the subprocess and write it to self
        # First check if the process already exists. If it does, close it before starting the new subprocess
        if getattr(self, "process", None) and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)  # Wait for the process to terminate after killing

        self.process = subprocess.Popen([self.fpath, self.analysis_name],
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL,
                                        text=True,
                                        bufsize=1,
                                        cwd=self.submodels_path  # keep all relative I/O in one place
                                        )

        # Initialize output reader thread.
        # Uses an unbounded queue to prevent deadlocks if output accumulates faster than processed
        self.output_queue = queue.Queue()

        def output_reader(out, q):
            """ Helper function to read the output on a separate thread """
            try:
                while not getattr(self, "shutdown_event", threading.Event()).is_set():
                    if self.process.poll() is not None:
                        # Check if the process is still running before reading stdout.
                        break
                    line = out.readline()
                    if not line:
                        break
                    q.put(line)
            except ValueError:
                # Catch value errors separately to avoid spamming console out
                # with "I/O operation on closed file" errors during MTSOl
                # crashes/restarts
                pass
            except Exception as e:
                print(e)

        self.reader = threading.Thread(target=output_reader, args=(self.process.stdout, self.output_queue))
        self.reader.daemon = True

        # Start the reader thread
        self.reader.start()

        # Check if subprocess is started successfully
        if self.process.poll() is not None:
            raise ImportError(f"MTSOL or tdat.{self.analysis_name} not found in {self.fpath}") from None


    def SetOperConditions(self,
                          ) -> None:
        """
        Set the inlet Mach number and critical amplification factor, and set the Reynolds number equal to zero to ensure an inviscid case is obtained.

        Returns
        -------
        None
        """

        # Enter the Modify solution parameters menu
        self.StdinWrite("m")

        # Write inlet Mach number
        self.StdinWrite(f"M {self.operating_conditions['Inlet_Mach']}")

        # Wait for completion of processing operating condition change
        self.WaitForCompletion(completion_type=CompletionType.PARAM_CHANGE)

        # Set critical amplification factor to N=9 rather than the default N=7
        self.StdinWrite(f"N {self.operating_conditions['N_crit']}")

        # Wait for completion of processing operating condition change
        self.WaitForCompletion(completion_type=CompletionType.PARAM_CHANGE)

        # Set the Reynolds number to 0 to ensure an inviscid solve is performed initially
        self.StdinWrite("R 0")

        # Wait for completion of processing operating condition change
        self.WaitForCompletion(completion_type=CompletionType.PARAM_CHANGE)

        # Set momentum/entropy conservation Smom flag
        self.StdinWrite("S 4")

        # Wait for completion of processing operating condition change
        self.WaitForCompletion(completion_type=CompletionType.PARAM_CHANGE)

        # Exit the modify solution parameters menu
        self.StdinWrite("")


    def ToggleViscous(self) -> None:
        """
        Toggle the viscous setting for all surfaces by setting the inlet Reynolds number.
        Note that the Reynolds number must be defined using the MTFLOW reference length LREF.

        Returns
        -------
        None
        """

        # Enter the Modify solution parameters menu
        self.StdinWrite("m")

        # Set the viscous Reynolds number, calculated using the length self.LREF
        self.StdinWrite(f"R {self.operating_conditions['Inlet_Reynolds']}")

        # Wait for completion of processing operating condition change
        self.WaitForCompletion(completion_type=CompletionType.PARAM_CHANGE)

        # Exit the Modify solution parameters menu
        self.StdinWrite("")


    def SetViscous(self,
                   surface_ID: list[int] | int,
                   mode: str = "enable"
                   ) -> None:
        """
        Set the viscous toggle for the given surface identifier(s).

        Parameters
        ----------
        - surface_ID : list[int] | int
            IDs of the surface which are to be toggled. For a ducted fan, the ID should be either 1, 3, or 4.
        - mode : str, optional
            Optional string to determine which toggling mode to use: enable or disable.

        Returns
        -------
        None
        """

        if mode == "enable":
            lookfor_flag = ExitFlag.COMPLETED
        elif mode == "disable":
            lookfor_flag = ExitFlag.SUCCESS
        else:
            raise ValueError(f"Unknown mode in SetViscous: {mode}")

        # Input validation to format surface_ID into a list if a pure integer is provided
        if isinstance(surface_ID, int):
            surface_ID = [surface_ID]

        # Enter the Modify solution parameters menu
        self.StdinWrite("m")
        self.WaitForCompletion(completion_type=CompletionType.PARAM_CHANGE)

        # Toggle the given surfaces to enable their viscous setting only if the surface was disabled
        for ID in surface_ID:
            # First check the current state of the surface
            flag = self.WaitForCompletion(completion_type=CompletionType.VISCOUS_TOGGLE, surface_ID=ID)
            while flag == lookfor_flag:
                self.StdinWrite(f"V{ID}")
                flag = self.WaitForCompletion(completion_type=CompletionType.VISCOUS_TOGGLE, surface_ID=ID)
                if flag != lookfor_flag:
                    break
        # Exit the Modify solution parameters menu
        self.StdinWrite("")


    def WaitForCompletion(self,
                          completion_type: CompletionType = CompletionType.ITERATION,
                          output_file: str = None,
                          surface_ID: int = None,
                          ) -> ExitFlag:
        """
        Monitor the console output to verify the completion of a command.

        Parameters
        ----------
        - completion_type : CompletionType, optional
            A CompletionType enum to determine what to check for. Default value if CompletionType.ITERATION
        - output_file : str, optional
            A string of the output file for which the completion is to be monitored. Either 'forces', 'flowfield', or 'boundary_layer'.
            Note that the file extension (i.e. casename), should not be included!
        - surface_ID : int, optional
            An integer of the surface which is to be inspected.

        Returns
        -------
        - exit_flag : ExitFlag
            Exit flag indicating the status of the solver execution.
        """

        # Define an exponential time_delay function to limit CPU usage while the solver is working
        def sleep_time(delta_t: float) -> float:
            return min(0.1, 0.01 * (2 ** min(10, int(delta_t))))

        # Check the console output to ensure that commands are completed
        timer_start = time.monotonic()
        time_out = 45  # seconds
        while (time.monotonic() - timer_start) <= time_out:
            # First check if the subprocess has terminated to ensure fail fast if this is the case
            if self.process.poll() is not None:
                return ExitFlag.CRASH

            # Read the output from the output thread
            try:
                adaptive_timeout = min(0.025 * (1 + int((time.monotonic() - timer_start) // 5)), 0.25)
                line = self.output_queue.get(timeout=adaptive_timeout)
            except queue.Empty:
                if self.process.poll() is not None:
                    return ExitFlag.CRASH
                time.sleep(0.01)
                continue

            # Once iteration is complete, return the completed exit flag
            if line.startswith(' =') and completion_type == CompletionType.ITERATION:
                return ExitFlag.COMPLETED

            # Once the iteration is converged, return the converged exit flag
            elif 'Converged' in line and completion_type == CompletionType.ITERATION:
                return ExitFlag.SUCCESS

            # If choking occurs, return the exit flag to choking
            elif ' *** QSHIFT: Mass flow or Pexit must be a DOF!' in line and completion_type == CompletionType.ITERATION:
                return ExitFlag.CHOKING

            # Once the solution is written to the state file, or the forces/flowfield file is written, return the completed exit flag
            # The succesful forces/flowfield writing can be detected from the prompt to overwrite the file or to enter a filename
            elif ('Solution written to state file' in line
                  or line.startswith((' File exists.  Overwrite?  Y', 'Enter filename'))) and completion_type == CompletionType.OUTPUT:
                if output_file is not None:
                    max_wait_time = 10  # Maximum wait time in seconds
                    # Wait for the file creation to be finished
                    target_path = self.submodels_path / self.FILE_TEMPLATES[output_file].format(self.analysis_name)
                    start_time = time.monotonic()
                    while not target_path.exists() and (time.monotonic() - start_time) <= max_wait_time:
                        time.sleep(sleep_time(time.monotonic() - start_time))

                return ExitFlag.COMPLETED

            # When changing the operating conditions, check for the end of the modify parameters menu
            elif line.startswith(' ----') and completion_type == CompletionType.PARAM_CHANGE:
                return ExitFlag.COMPLETED

            # When toggling viscous surfaces, check for the star at the corresponding line:
            elif line.startswith(' T{} *'.format(surface_ID)) and completion_type == CompletionType.VISCOUS_TOGGLE:
                return ExitFlag.SUCCESS
            elif line.startswith(' T{}    '.format(surface_ID)) and completion_type == CompletionType.VISCOUS_TOGGLE:
                return ExitFlag.COMPLETED

            # If the solver crashes, return the crash exit flag
            # A crash can be detectede either by the MTSOL subprocess exiting, or neg. temp. lines in the console output
            elif self.process.poll() is not None or line.startswith(' *** Neg. temp.'):
                return ExitFlag.CRASH

        # If timer ran out while waiting for completion, assume the solver has crashed/hung
        return ExitFlag.NON_CONVERGENCE


    def WriteStateFile(self,
                       ) -> None:
        """
        Writes the current solution to the state file tdat.analysis_name.

        Returns
        -------
        None
        """

        # Update the solution state file
        self.StdinWrite("W")

        # Check if the solution state file is written successfully
        self.WaitForCompletion(completion_type=CompletionType.OUTPUT)


    def GenerateSolverOutput(self,
                             output_type: OutputType = OutputType.FORCES_ONLY) -> None:
        """
        Generate all output files for the current analysis.
        If a viscous analysis was performed, the boundary layer data is also dumped to the corresponding file.
        Requires that MTSOL is in the main menu when starting this function.

        Parameters
        ----------
        - output_type : OutputType
            An enum to determine which output files need to be generated:
                - OutputType.FORCES_ONLY: Generate only the forces file (default)
                - OutputType.ALL_FILES: Generate all files (forces, flowfield, boundary_layer)

        Returns
        -------
        None
        """

        # First check if MTSOL is still running. If MTSOl has crashed, restart MTSOL based on the last saved statefile
        if getattr(self, "process", None) and self.process.poll() is not None:
            self.GenerateProcess()

        # Dump the forces data
        self.StdinWrite("F")
        self.StdinWrite(self.FILE_TEMPLATES['forces'].format(self.analysis_name))
        if self.filepaths["forces"].exists():
            self.StdinWrite("Y")  # Overwrite existing file

        # Check if the forces file is written successfully
        self.WaitForCompletion(completion_type=CompletionType.OUTPUT,
                               output_file='forces')

        if output_type == OutputType.FORCES_ONLY:
            return

        # Dump the flowfield data
        self.StdinWrite("D")
        self.StdinWrite(self.FILE_TEMPLATES['flowfield'].format(self.analysis_name))
        if self.filepaths["flowfield"].exists():
            self.StdinWrite("Y")  # Overwrite existing file

        # Check if the flowfield file is written successfully
        self.WaitForCompletion(completion_type=CompletionType.OUTPUT,
                               output_file='flowfield')

        # Dump the boundary layer data
        self.StdinWrite("B")
        self.StdinWrite(self.FILE_TEMPLATES['boundary_layer'].format(self.analysis_name))
        if self.filepaths["boundary_layer"].exists():
            self.StdinWrite("Y")  # Overwrite existing file

        # Check if the boundary layer file is written successfully
        self.WaitForCompletion(completion_type=CompletionType.OUTPUT,
                               output_file='boundary_layer')


    def ExecuteSolver(self,
                      ) -> ExitFlag:
        """
        Execute the MTSOL solver for the current analysis.

        Returns
        -------
        - exit_flag : ExitFlag
            Exit flag indicating the status of the solver execution.
        """

        # Ensure an iteration limit is defined - fall back to the inviscid iteration count as default
        if not hasattr(self, "ITER_LIMIT"):
            raise AttributeError("ITER_LIMIT not set before ExecuteSolver()")

        # Initialize iteration count and exit flag
        self.iter_counter = 0
        exit_flag = ExitFlag.NOT_PERFORMED

        # Keep converging until the iteration count exceeds the limit
        while (self.iter_counter < self.ITER_LIMIT) and (exit_flag not in (ExitFlag.SUCCESS, ExitFlag.CHOKING, ExitFlag.CRASH)):
            #Execute next iteration(s)
            self.StdinWrite(f"x {self.ITER_STEP_SIZE}")

            # Check the exit flag to see if the solution has converged
            # If the solution has converged, break out of the iteration loop
            exit_flag = self.WaitForCompletion(completion_type=CompletionType.ITERATION)

            if exit_flag != ExitFlag.CRASH:
                # Increase the iteration counter for all non-crash states
                self.iter_counter += self.ITER_STEP_SIZE

            # Break on terminal states
            if exit_flag in (ExitFlag.SUCCESS, ExitFlag.CHOKING, ExitFlag.CRASH):
                break

        # If the solver has not converged within self.ITER_LIMIT iterations, set the exit flag to non-convergence
        if exit_flag not in (ExitFlag.SUCCESS, ExitFlag.CHOKING, ExitFlag.CRASH):
            exit_flag = ExitFlag.NON_CONVERGENCE

        # Return the exit flag
        return exit_flag


    def GetAverageValues(self,
                         ) -> None:
        """
        Read the output force files from the MTSOL_output_files directory and average the values to obtain the assumed true values in case of non-convergence.

        Returns
        -------
        None
        """

        # Extract output files from the dump folder
        output_files = list(self.dump_folder.glob(f"forces.{self.analysis_name}.*"))
        if not output_files:
            return

        # Indices of header lines that should not be averaged
        skip_lines = frozenset({0, 1, 2, 3, 4, 5, 13, 25, 26, 31, 36, 41})

        with ExitStack() as stack, open(self.filepaths['forces'], 'w') as averaged_file:
            file_handlers = [stack.enter_context(open(output_file, 'r', newline='')) for output_file in output_files]

            # Process all files one line at a time using zip(lazy evaluation)
            for idx, lines in enumerate(zip(*file_handlers)):
                # Start with the first file's line as the base
                base_line = lines[0]

                if idx in skip_lines:
                    averaged_file.write(base_line)
                    continue

                # Precompute the presence of '=' and ':' in each line
                eq_flags = [('=' in line) for line in lines]
                colon_flags = [(':' in line) for line in lines]

                # Case 1: single key=value per line.
                if all(eq_flags) and all(len(line.split("=", 1)[1].split()) == 1 for line in lines):
                    key = base_line.split("=", 1)[0].strip()
                    values = []
                    for line in lines:
                        try:
                            value_str = line.split("=", 1)[1].strip()
                            values.append(float(value_str))
                        except (IndexError, ValueError):
                            # Fall back on base_line in case any error occurs
                            values = []
                            break
                    if not values:
                        averaged_file.write(base_line)
                        continue
                    average_value = np.mean(values, axis=0)
                    line_text = f'{key} = {float(average_value):.5E}\n'

                # Case 2: multiple key=value pairs in one line.
                elif all(eq_flags):
                    # Cache regex finds for each line in one go:
                    parsed_lines = [self.var_value_pattern.findall(line) for line in lines]

                    if any(len(matches) > 1 for matches in parsed_lines):
                        # Build the dictionary once
                        var_values_dict = OrderedDict()

                        for matches in parsed_lines:
                            for var, value in matches:
                                var = var.strip()  # remove extra whitespace
                                var_values_dict.setdefault(var, []).append(float(value))

                        # Compute average for each encountered key, preserving order from the first appearance.
                        avg_values = [
                            f"{var} = {float(np.mean(vals)):.5E}"
                            for var, vals in var_values_dict.items()
                        ]
                        line_text = " ".join(avg_values) + "\n"
                    else:
                        line_text = base_line  # If there aren't multiple pairs, fall back to base_line

                # Case 3: Lines with a colon and multiple space-separated values (e.g. "variable: data1 data2 data3 data4")
                elif all(colon_flags):
                    text_part = base_line.split(':', 1)[0].strip() + ': '
                    try:
                        matrix = np.array([list(map(float, line.split(':')[1].split())) for line in lines])
                    except ValueError:
                        matrix = np.array([])
                    if matrix.size:
                        avg_values = np.mean(matrix, axis=0)
                        line_text = text_part + '    '.join(f'{float(val):.5E}' for val in avg_values) + '\n'
                    else:
                        line_text = base_line

                # Case 4: Lines with unnamed values separated by varying spaces
                elif all(self.value_pattern.match(line) for line in lines):
                    try:
                        matrix = np.array([list(map(float, re.split(r'\s+', line.strip()))) for line in lines])
                    except ValueError:
                        matrix = np.array([])
                    if matrix.size:
                        avg_values = np.mean(matrix, axis=0)
                        line_text = '    '.join(f'{float(val):.5E}' for val in avg_values) + '\n'
                    else:
                        line_text = base_line

                # If none of these cases match, use the first line as fallback
                else:
                    line_text = base_line

                # Write the processed line
                averaged_file.write(line_text)


    def HandleNonConvergence(self,
                             ) -> None:
        """
        Average over the last self.SAMPLE_SIZE iterations to determine flowfield variables.
        This method performs the following steps:
            - Creates a subfolder to store output files if it doesn't already exist.
            - Deletes any existing forces output file for the current analysis.
            - Executes a specified number of iterations and generates solver outputs for each.
            - Averages the data from the generated files to estimate the true values.
            - Deletes the individual output files after averaging.

        TODO: Rather than saving the outputs to files in the dump folder, 
              print them to the console output of MTSOL and work directly from memory. 

        Returns
        -------
        None
        """

        # Initialize iteration counter
        iter_counter = 0

        # Initialize watchdog to check when output file has been created
        copied_file = self.forces_template_nonconv.format(self.analysis_name, iter_counter)
        event_handler = FileCreatedHandling(self.filepaths['forces'],
                                            self.dump_folder / copied_file)

        observer = Observer()
        monitor_path = str(self.submodels_path)
        observer.schedule(event_handler,
                          path=monitor_path,
                          recursive=False,
                          )
        observer.start()

        # Keep looping until iter_count exceeds the target value for number of iterations to average
        while iter_counter < self.SAMPLE_SIZE:
            # First check if subprocess is still alive. If the solver has crashed, restart and break the loop.
            if self.process.poll() is not None:
                self.GenerateProcess()
                break

            #Execute iteration
            self.StdinWrite("x1")

            # Wait for current iteration to complete
            self.WaitForCompletion(completion_type=CompletionType.ITERATION)

            # Generate solver outputs
            self.GenerateSolverOutput(output_type=OutputType.FORCES_ONLY)

            # Rename file to indicate the iteration number, and avoid overwriting the same file.
            # Also move the file to the output folder
            # Waits for the file to exist before copying.
            init_time = time.monotonic()
            backoff = 0.01
            max_backoff = 0.1
            timeout = 10
            while not event_handler.is_file_processed() and (time.monotonic() - init_time) <= timeout:
                time.sleep(backoff)
                backoff = min(max_backoff, backoff * 2)

            # Increase iteration counter by step size
            iter_counter += 1
            copied_file = self.forces_template_nonconv.format(self.analysis_name, iter_counter)

            # Re-intialise the event handler for the next iteration with an updated destination
            event_handler.destination = self.dump_folder / copied_file
            event_handler.file_processed = False

        # Wrap up the watchdog
        observer.stop()
        observer.join()

        # Average the data from all the iterations to obtain the assumed true values. This effectively assumes that the iterations are oscillating about the true value.
        # This is a simplification, but it is the best we can do in this case.
        # The average is calculated by summing all the values and dividing by the number of iterations.
        self.GetAverageValues()

        # Remove the now unnecessary output files in the dump folder
        self.CleanupOutputFiles()


    def CleanupOutputFiles(self) -> None:
        """
        A simple method to clean up output files from the dump folder

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # Remove any output file from the dump folder
        for file in self.dump_folder.glob("forces.{}.*".format(self.analysis_name)):
            file.unlink(missing_ok=True)


    def CheckInviscidOutput(self) -> bool:
        """
        Function to check the efficiency of the invisicid output.
        When interpolatation errors occur in the blade forcing field,
        they can yield eta > 1.
        Checking this condition early avoids running the viscous solve,
        speeding up the solution process.

        Returns
        -------
        - valid : bool
            Boolean to indicate if a feasible result was found.
        """

        try:
            # Extract the efficiency from the forces output file
            _, _, eta = output_processing(analysis_name=self.analysis_name
                                          ).GetCTCPEtaP()

            # Return appropriate bool depending on found efficiency
            return 0 < eta < 1.
        except Exception:
            # Treat any exception as non-physical to skip viscous runs but keep
            # any batch analyses alive
            return False


    def GetEtaOutput(self) -> float:
        """
        Function to get the current efficiency.

        Returns
        -------
        - eta: float
            Efficiency
        """

        try:
            _, _, eta = output_processing(analysis_name=self.analysis_name).GetCTCPEtaP()
        except Exception:
            eta = 0
        return eta


    def HandleExitFlag(self,
                       exit_flag: ExitFlag,
                       handle_type : str,
                       update_statefile : bool = False,
                       ) -> None:
        """
        Handle the exit flag of the solver execution.

        Parameters
        ----------
        - exit_flag : ExitFlag
            Exit flag indicating the status of the solver execution.
        - handle_type : str
            A string indicating the type of solve:
                - 'Inviscid'
                - 'Viscous'
        - update_statefile: bool, optional
            A control boolean to determine if the state file should be updated.

        Returns
        -------
        None
        """

        #If solver does not converge, call the non-convergence handler function.
        if exit_flag == ExitFlag.NON_CONVERGENCE:
            if update_statefile:
                self.WriteStateFile()
            if handle_type == 'Inviscid':
                return
            else:
                self.HandleNonConvergence()

        # Else if the solver has crashed, delete all output files except the forces file to keep outputs clean
        elif exit_flag == ExitFlag.CRASH:
            if handle_type == 'Inviscid':
                # For an inviscid crash, cleanup outputs
                self.CleanupOutputFiles()
                return
            else:
                self.GenerateProcess()
                return

        # Else if the solver has finished, update the statefile.
        elif exit_flag in (ExitFlag.COMPLETED, ExitFlag.SUCCESS, ExitFlag.NOT_PERFORMED, ExitFlag.CHOKING):
            if update_statefile:
                self.WriteStateFile()
            return

        else:
            raise OSError(f"Unknown exit flag {exit_flag} encountered!") from None


    def TryExecuteViscousSolver(self,
                                surface_ID: list[int]|int = None,
                                ) -> ExitFlag:
        """
        Try to execute the MTSOL solver for the current analysis on the viscous surface surface_ID.

        Parameters
        ----------
        - surface_ID : list[int]|int, optional
            ID of the surface which is to be toggled. For a ducted fan, the ID should be either 1, 3, or 4.

        Returns
        -------
        - exit_flag : ExitFlag
            Exit flag indicating the status of the solver execution.
        """

        try:
            # Always kill and restart the process for a clean state
            if getattr(self, "process", None) and self.process.poll() is None:
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                # Ensure the old reader thread is stopped before starting a new process
                if getattr(self, "reader", None) and self.reader.is_alive():
                    if hasattr(self, "shutdown_event"):
                        self.shutdown_event.set()
                    try:
                        self.reader.join(timeout=10)
                    except Exception as e:
                        print(f"Error cleaning up reader thread: {e}")
            self.GenerateProcess()

            # Enter the modify menu and set the Reynolds number
            self.StdinWrite("m")
            self.WaitForCompletion(completion_type=CompletionType.PARAM_CHANGE)
            self.StdinWrite(f"R {self.operating_conditions['Inlet_Reynolds']}")
            self.WaitForCompletion(completion_type=CompletionType.PARAM_CHANGE)
            self.StdinWrite("")

            # Set viscous if surface_ID is given
            if surface_ID is not None:
                self.SetViscous(surface_ID, mode="enable")

            # Execute the solver and get the exit flag and iteration count
            exit_flag = self.ExecuteSolver()

            # If solve did not crash, update the statefile
            if exit_flag != ExitFlag.CRASH:
                self.WriteStateFile()

        except (OSError, BrokenPipeError) as e:
            print(e)
            # If the solver crashes, set the exit flag to crash
            exit_flag = ExitFlag.CRASH

        return exit_flag


    def ConvergeIndividualSurfaces(self,
                                   ) -> ExitFlag:
        """
        Should a complete viscous analysis fail and cause an MTSOL crash, a partial run, where each axisymmetric surface is toggled individually,
        may sometimes improve performance and yield (partially) converged results.

        This function executes a consecutive partial run, where the centerbody, outer duct, and inner duct are converged in sequence.

        Returns
        -------
        - total_exit_flag : ExitFlag
            Exit flag indicating the overall status of the convergence.
        """

        # Define initial exit flags and iteration counters
        # Note that a negative iteration count indicates that the solver did not run
        exit_flag_visc_CB = ExitFlag.NOT_PERFORMED
        exit_flag_visc_induct = ExitFlag.NOT_PERFORMED
        exit_flag_visc_outduct = ExitFlag.NOT_PERFORMED
        total_exit_flag = ExitFlag.NOT_PERFORMED

        # Execute the viscous solve for the centerbody
        exit_flag_visc_CB = self.TryExecuteViscousSolver(surface_ID=1)
        if exit_flag_visc_CB == ExitFlag.CRASH:
            # if the viscous CB solve caused a crash, return appropriate exit flag
            return exit_flag_visc_CB

        # Execute the viscous solve for the outside of the duct
        exit_flag_visc_outduct = self.TryExecuteViscousSolver(surface_ID=3)
        if exit_flag_visc_outduct == ExitFlag.CRASH:
            # if the viscous solve caused a crash, return appropriate exit flag
            return exit_flag_visc_outduct

        # Execute the final viscous solve for the inside of the duct
        exit_flag_visc_induct = self.TryExecuteViscousSolver(surface_ID=4)
        if exit_flag_visc_induct == ExitFlag.CRASH:
            # if the viscous solve caused a crash, return appropriate exit flag
            return exit_flag_visc_induct

        total_exit_flag = max([exit_flag_visc_CB, exit_flag_visc_outduct, exit_flag_visc_induct], key=lambda flag: flag.value)

        return total_exit_flag


    def caller(self,
               run_viscous: bool = False,
               generate_output: bool = False,
               output_type: OutputType = OutputType.FORCES_ONLY,
               ) -> ExitFlag:
        """
        Main execution interface of MTSOL.

        All executions of the MTSOL program are wrapped in try... except... finally... blocks to handle crashes of the solver

        Parameters
        ----------
        - run_viscous : bool, optional
            Flag to indicate whether to run a viscous solve. Default is False.
        - generate_output : bool, optional
            Flag to determine if MTFLOW outputs (forces, flowfield, boundary layer) should be generated.
        - output_type : OutputType, optional
            An enum to determine which output files to generate. OutputType.FORCES_ONLY generates only the forces file, while OutputType.ALL_FILES generates all files.

        Returns
        -------
        - total_exit_flag : ExitFlag
            Exit flag indicating the status of the solver execution. Is equal to the maximum value of the inviscid and viscous exit flags, since exit_flag > -1 indicate failed/nonconverging solves.
            This is used as a one-variable status indicator, while the corresponding output list gives more details.
        """

        # Define initial exit flags
        # Note that a negative iteration count indicates that the solver did not run
        exit_flag_invisc = ExitFlag.NOT_PERFORMED
        exit_flag_visc = ExitFlag.NOT_PERFORMED

        try:
            # Generate MTSOL subprocess
            self.GenerateProcess()

            # Write operating conditions
            self.SetOperConditions()

            # Disable all viscous surfaces initially
            self.SetViscous([1, 3, 4], mode="disable")

            if generate_output:
                # Update the statefile with the operating conditions
                self.WriteStateFile()

            # Even if we don't want to generate output, we still need to create the forces file to ensure post-processing of results does not fail
            self.GenerateSolverOutput(output_type=OutputType.FORCES_ONLY)

            # Execute inviscid solve
            try:
                self.ITER_LIMIT = self.ITER_LIMIT_INVISC  # Set the appropriate iteration limit
                exit_flag_invisc = self.ExecuteSolver()
            except (OSError, BrokenPipeError):
                # If the inviscid solve crashes, we need to set the exit flag to crash
                exit_flag_invisc = ExitFlag.CRASH
            finally:
                # Handle solver based on exit flag
                self.HandleExitFlag(exit_flag_invisc,
                                    handle_type='Inviscid',
                                    update_statefile=generate_output)
                total_exit_flag = exit_flag_invisc

                if generate_output:
                    # Generate the requested solver outputs based on output_type
                    self.GenerateSolverOutput(output_type=output_type)

            if not run_viscous:
                # Using handle_type="inviscid" bypasses the handle non-convergence loop.
                # This is intentional for a viscous run, as it speeds up the solution process substantially. However, for an inviscid run, we do need to perform this loop.
                self.HandleExitFlag(total_exit_flag,
                                    handle_type="Viscous",
                                    update_statefile=generate_output)


            # Only run a viscous solve if required by the user
            # Theoretically there is the chance a viscous run may be started on a non-converged inviscid solve.
            # This is acceptable, as we assume a steady state residual case has formed at the end of the inviscid case.
            # There is a probability that by then running a viscous case, convergence to the correct solution may still be obtained.
            # Only runs the viscous case if the exit flag is correct and if the inviscid analysis was physical (i.e. has an efficiency < 100%)
            if run_viscous and total_exit_flag in (ExitFlag.SUCCESS, ExitFlag.COMPLETED, ExitFlag.NON_CONVERGENCE) and self.CheckInviscidOutput():
                # Get the inviscid efficiency
                eta_invisc = self.GetEtaOutput()

                # Toggle viscous on all surfaces
                self.ToggleViscous()  # Set the viscous Reynolds number
                self.SetViscous([1, 3, 4], mode="enable")

                # Update the iteration limit
                self.ITER_LIMIT = self.ITER_LIMIT_VISC

                # First we try to run a complete viscous case. Only if this doesn't work and causes a crash do we try to converge each surface individually
                try:
                    exit_flag_visc = self.ExecuteSolver()

                except (OSError, BrokenPipeError):
                    # If the complete viscous solve crashed, set the exit flag appropriately
                    exit_flag_visc = ExitFlag.CRASH

                finally:
                    # Try converging individual surfaces if the complete solve crashed
                    if exit_flag_visc == ExitFlag.CRASH:
                        exit_flag_visc = self.ConvergeIndividualSurfaces()

                    # Handle the exit flag
                    total_exit_flag = max([exit_flag_visc, exit_flag_invisc], key=lambda flag: flag.value)
                    self.HandleExitFlag(total_exit_flag,
                                        handle_type='Viscous',
                                        update_statefile=generate_output)


                    if generate_output:
                        # Generate the requested solver outputs based on output_type
                        self.GenerateSolverOutput(output_type=output_type)

                    # Get the viscous efficiency and check it is lower than the inviscid efficiency
                    # If not, set the exit flag to crash to enable appropriate downstream handling.
                    eta_visc = self.GetEtaOutput()
                    if eta_visc > eta_invisc:
                        total_exit_flag = ExitFlag.CRASH


            # Close the MTSOL tool
            # If no output is generated, need to write an additional white line to close MTSOL
            # Initial newline char to ensure MTSOL remains in main menu
            if self.process.poll() is None:
                self.StdinWrite("\n")
                self.StdinWrite("Q")
                if not generate_output:
                    self.StdinWrite("\n")

        finally:
            # Check that MTSOL has closed successfully. If not, forcefully closes MTSOL
            if self.process.poll() is None:
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            # Join reader thread if alive
            if getattr(self, "reader", None) and self.reader.is_alive():
                if hasattr(self, "shutdown_event"):
                    self.shutdown_event.set()
                self.reader.join(timeout=5)

        return total_exit_flag


if __name__ == "__main__":
    import time

    analysisName = "0525040913_4804_55f1b299bf34"
    oper = {"Inlet_Mach": 0.10285225,
            "Inlet_Reynolds": 5112279,
            "N_crit": 9,
            }

    start_time = time.monotonic()
    test = MTSOL_call(oper, analysisName).caller(run_viscous=True,
                                                 generate_output=True)
    end_time = time.monotonic()

    print(f"Execution of MTSOL_call took {end_time -  start_time} seconds")