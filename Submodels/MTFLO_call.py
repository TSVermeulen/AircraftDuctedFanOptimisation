"""
MTFLO_call
=============

Description
-----------
This module provides an interface to interact with the MTFLO executable from
Python. It creates a subprocess for the MTFLO executable, loads in the input
file tflow.xxx, and writes the output to the tdat.xxx data file for use within
MTSOL.

Classes
-------
MTFLO_call
    A class to handle the interface between Python and the MTFLO executable.

Examples
--------
>>> analysisName = "test_case"
>>> test = MTFLO_call(analysisName)
>>> test.caller()

Notes
-----
This module is designed to work with the MTFLO executable. Ensure that the
executable and the input file, tflow.xxx, are present in the same directory as
this Python file. When executing the file as a standalone, it uses the inputs
and calls contained within the if __name__ == "__main__" section. This part
also imports the time module to measure the time needed to perform each file
generation call. This is beneficial in runtime optimisation.

References
----------
The required input data, limitations, and structures are documented within the
MTFLOW user manual:
https://web.mit.edu/drela/Public/web/mtflow/mtflow.pdf

Versioning
------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Version: 1.2

Changelog:
- V1.0:   Initial working version
- V1.0.5: Cleaned up inputs, removing file_path and changing it to a constant.
- V1.1:   Added file status check to ensure that the file has been written
          before proceeding. Added stdinwrite function to clean up process
          interactions.
- V1.2:   Updated documentation and removed self.process.returncode return
          from MTFLO_call().caller()
"""

# Import standard libraries
import subprocess
import time
from pathlib import Path


class MTFLO_call:
    """
    Class to handle the interface between Python and the MTFLO executable.
    """

    def __init__(self,
                 analysis_name: str) -> None:
        """
                 Initialize the MTFLO_call instance and locate the MTFLO executable.
                 
                 Sets instance attributes for the analysis name, project parent directory, Submodels path, and the expected mtflo.exe path. Raises FileNotFoundError if mtflo.exe is not found at the computed location.
                 
                 Parameters:
                     analysis_name (str): The name of the analysis case.
                 """

        self.analysis_name = analysis_name

        # Define key paths/directories
        self.parent_dir = Path(__file__).resolve().parent.parent
        self.submodels_path = self.parent_dir / "Submodels"

        # Define filepath of MTFLO as being in the same folder as
        # this Python file
        self.process_path = self.submodels_path / 'mtflo.exe'
        if not self.process_path.exists():
            raise FileNotFoundError(f"MTFLO not found at {self.process_path}")


    def StdinWrite(self,
                   command: str) -> None:
        """
                   Write a single command to the MTFLO subprocess stdin.
                   
                   Parameters:
                       command (str): Command text to send to MTFLO; a newline is appended and the stream is flushed.
                   """

        self.process.stdin.write(f"{command} \n")
        self.process.stdin.flush()


    def GenerateProcess(self) -> None:
        """
        Start the MTFLO subprocess and store the resulting Popen object on self.process.
        
        Launches the executable at self.process_path with the instance's analysis name as an argument. If the subprocess exits immediately, an OSError is raised to indicate failure to start MTFLO.
        
        Raises:
            OSError: If the subprocess fails to start (process exits immediately).
        """

        # Generate the subprocess and write it to self
        self.process = subprocess.Popen([self.process_path, self.analysis_name],
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL,
                                        text=True,
                                        bufsize=1,
                                        )

        # Check if subprocess is started successfully
        if self.process.poll() is not None:
            raise OSError("Error starting MTFLO") from None


    def LoadForcingField(self) -> None:
        """
        Load the tflow.{analysis_name} input into MTFLO, instruct MTFLO to write the tdat output file, and close the MTFLO subprocess.
        
        Sends interactive commands to the MTFLO subprocess to enter the field menu, read the parameter file (accepting the default filename), write the flowfield (tdat) file, and quit. If MTFLO has crashed while reading the input, an ImportError is raised. If MTFLO does not exit within 10 seconds after the quit command, the subprocess is terminated forcefully.
        
        Raises:
            ImportError: If MTFLO crashed while loading the tflow input.
        """

        # Enter field parameter menu
        self.StdinWrite("F")

        # Read parameter text file
        self.StdinWrite("R")

        # Accept default filename
        self.StdinWrite("")

        # Check if file is loaded in successfully.
        # If error occured, MTFLO will have crashed, so we can check success
        # by checking if the subprocess is still alive
        if self.process.poll() is not None:
            raise ImportError("Issue with tflow input file, MTFLO crashed") from None

        # Exit the field parameter menu
        self.StdinWrite("")

        # Write to the flowfield file tdat.xxx and check
        # if writing was successful
        self.StdinWrite("W")

        # Close the MTFLO program
        self.StdinWrite("Q")

        # Check that MTFLO has closed successfully.
        # If not, forcefully closes MTFLO
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()


    def FileStatus(self,
                   fpath: Path) -> bool:
        """
                   Check whether a file is currently accessible (not locked by another process).
                   
                   Parameters:
                       fpath (Path): Path to the file to test.
                   
                   Returns:
                       bool: `True` if the file can be opened for reading (not locked), `False` otherwise.
                   """

        try:
            with open(fpath, "rb"):
                return True
        except OSError:
            return False


    def caller(self) -> None:
        """
        Orchestrates the full interaction with the MTFLO executable: starts the MTFLO subprocess, loads the forcing field, and waits up to 10 seconds for the generated tdat.{analysis_name} output file to become accessible.
        
        The method expects the appropriate tflow input and mtflo executable to be present; it returns after the output file is available or the timeout elapses.
        """

        # Create subprocess for the MTFLO tool
        self.GenerateProcess()

        # Load the numerical grid
        self.LoadForcingField()

        # Wait until file has been processed
        fpath = self.submodels_path / "tdat.{}".format(self.analysis_name)
        start_time = time.time()
        timeout = 10
        while (time.time() - start_time) <= timeout:
            if self.FileStatus(fpath):
                break
            time.sleep(0.01)


if __name__ == "__main__":
    start_time = time.time()
    analysisName = "test_case"
    test = MTFLO_call(analysisName)
    test.caller()
    end_time = time.time()

    print(f"Execution of MTFLO_call({analysisName}).caller() took {end_time - start_time} seconds")