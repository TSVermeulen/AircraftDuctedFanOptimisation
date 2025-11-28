"""
checkpoint
====

Description
----------
This module provides a checkpoint functionality to the optimization framework by
extending the Pymoo callback class.

Classes
--------
CheckpointCallBack
    A class implementing a callback method to store the current state of the
    algorithm at regular intervals.

Notes
-----
The checkpoint files are stored in a 'checkpoints' directory relative to this
module. Each checkpoint file is named according to the generation number at
which it was created.

References
----------
For more details on the Pymoo framework, refer to the official documentation:
https://pymoo.org/

Versioning
----------
Author: T.S. Vermeulen
Email: T.S.Vermeulen@tudelft.nl
Version 1.0

Changelog:
- V1.0: Initial implementation.
"""

# Import standard libraries
from pathlib import Path

# Import third-party libraries
import dill
from pymoo.core.callback import Callback
from pymoo.core.algorithm import Algorithm

# Ensure all paths are correctly set up
from utils import ensure_repo_paths
ensure_repo_paths()

class CheckpointCallBack(Callback):
    """
    A callback class to create checkpoints during the optimisation process.

    This class extends the Pymoo Callback class to save the current state of the
    algorithm at specified intervals, allowing for recovery and analysis of the
    optimisation process.
    """

    def __init__(self,
                 interval: int = 10) -> None:
        """
        Initialises the CheckpointCallBack with a specified interval.

        Parameters
        ----------
        interval : int
            The number of generations between each checkpoint save.
        """

        super().__init__()

        # Validate the interval parameter
        if interval <0:
            raise ValueError("Checkpoint interval must be a positive integer.")
        
        self.interval = interval


    def notify(self,
               algorithm: Algorithm) -> None:
        """
        Saves a checkpoint of the algorithm state at specified intervals.

        Parameters
        ----------
        algorithm : Algorithm
            The current state of the optimisation algorithm.
        """

        gen = algorithm.n_gen
        if self.interval == 0:
            # If interval is set to 0, do not save any checkpoints
            return
        elif gen % self.interval == 0:
            # Define the filename for the checkpoint based on the
            # current generation
            filename = f"checkpoint_gen{gen}.dill"

            # Generate the checkpoint storage folder if it does not exist
            # already
            chkpnt_dir = Path(__file__).resolve().parent / "checkpoints"
            chkpnt_dir.mkdir(exist_ok=True,
                             parents=True)

            # Save the current state of the algorithm in a dill file
            try:
                with open(chkpnt_dir / filename, 'wb') as f:
                    dill.dump(algorithm, f)
            except Exception as e:
                # Log the error message to avoid aborting the analysis
                print(f"Error saving checkpoint at generation {gen}: {e}")