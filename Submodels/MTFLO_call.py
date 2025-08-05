"""
MTFLO_call
=============

Description
-----------
This module provides an interface to interact with the MTFLO executable from Python.
It creates a subprocess for the MTFLO executable, loads in the input file tflow.xxx,
and writes the output to the tdat.xxx data file for use within MTSOL.

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
This module is designed to work with the MTFLO executable. Ensure that the executable and the input file, tflow.xxx,
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
Version: 1.2

Changelog:
- V1.0: Initial working version
- V1.0.5: Cleaned up inputs, removing file_path and changing it to a constant.
- V1.1: Added file status check to ensure that the file has been written before proceeding. Added stdinwrite function to clean up process interactions.
- V1.2: Updated documentation and removed self.process.returncode return from MTFLO_call().caller()
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
                 analysis_name: str,
                 ) -> None:
        """
        Initialize the MTFLO_call class with the file path and analysis name.

        Parameters
        ----------
        - analysis_name : str
            The name of the analysis case.
        """

        self.analysis_name = analysis_name

        # Define key paths/directories
        self.parent_dir = Path(__file__).resolve().parent.parent
        self.submodels_path = self.parent_dir / "Submodels"

        # Define filepath of MTFLO as being in the same folder as this Python file
        self.process_path = self.submodels_path / 'mtflo.exe'
        if not self.process_path.exists():
            raise FileNotFoundError(f"MTFLO executable not found at {self.process_path}")


    def StdinWrite(self,
                   command: str) -> None:
        """
        Simple function to write commands to the subprocess stdin in order to pass commands to MTFLO.

        Parameters
        ----------
        - command : str
            The text-based command to pass to MTFLO.

        Returns
        -------
        None
        """

        self.process.stdin.write(f"{command} \n")
        self.process.stdin.flush()


    def GenerateProcess(self,
                        ) -> None:
        """
        Create MTFLO subprocess

        Requires that the executable, mtflo.exe, and the input file, tflow.xxx are present in the same directory as this
        Python file.
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
            raise ImportError(f"MTFLO or tdat.{self.analysis_name} not found in {self.process_path}") from None


    def LoadForcingField(self,
                         ) -> None:
        """
        Loads the tflow.xxx input file into MTFLO and checks it has been correctly processed
        """

        # Enter field parameter menu
        self.StdinWrite("F")

        # Read parameter text file
        self.StdinWrite("R")

        # Accept default filename
        self.StdinWrite("")

        # Check if file is loaded in successfully.
        # If error occured, MTFLO will have crashed, so we can check success by checking
        # if the subprocess is still alive
        if self.process.poll() is not None:
            raise ImportError(f"Issue with input file tflow.{self.analysis_name}, MTFLO crashed") from None

        # Exit the field parameter menu
        self.StdinWrite("")

        # Write to the flowfield file tdat.xxx and check if writing was successful
        self.StdinWrite("W")

        # Close the MTFLO program
        self.StdinWrite("Q")

        # Check that MTFLO has closed successfully. If not, forcefully closes MTSOL
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()


    def FileStatus(self,
                   fpath: Path,
                   ) -> bool:
        """
        Simple function to check if the file update/write has finished.
        Check occurs based on the presence of a file "lock" on the file if/when it is being used by another process.

        Parameters
        ----------
        - fpath : Path
            Filepath of the file to check

        Returns
        -------
        - status : bool
            A status boolean. True if the file is free, and False if it is still being used by another process.
        """
        try:
            with open(fpath, "rb"):
                return True
        except OSError:
            return False


    def caller(self,
               ) -> None:
        """
        Full interfacing function between Python and MTFLO.

        Requires that the input file, tflow.xxx, has been made and is available together with the mtflo.exe executable in the local directory.

        Returns
        -------
        None
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