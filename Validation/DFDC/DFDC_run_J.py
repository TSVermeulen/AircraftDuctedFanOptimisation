"""
DFDC_run_J
==========

Description
-----------
This code is provided as a courtesy from Bram Meijerink. 
It is fairly simple to understand/use, and hence mostly undocumented. 
Compared to the original code, it has been streamlined, without plotting 
capability, and has some minor documentation/formatting changes to help bring 
it in line to the rest of the codebase.

Versioning
----------
Author: T.S. Vermeulen & B. Meijerink
Email: T.S.Vermeulen@student.tudelft.nl
Student ID: 4995309
Date [dd-mm-yyyy]: [08-06-2025]
Version: 1.1

Changelog:
- V1.0: Initial implementation of B. Meijerink.
- V1.1: Updated script with formatting/documentation to match rest of codebase.
"""

# Import standard libraries
import subprocess
import threading
import re
import csv

# Import 3rd party libraries
import numpy as np


def read_output(process) -> None:
    """
    Function to read the outputs from DFDC.

    Parameters
    ----------
    - process
        The DFDC subprocess
    
    Returns
    -------
    None
    """

    i = 1
    for line in process.stdout:
        J_match = re.search(r"J\s*:\s*([\d\.E+-]+)", line)
        CT_match = re.search(r"Ct\s*:\s*([\d\.E+-]+)", line)
        CP_match = re.search(r"Cp\s*:\s*([\d\.E+-]+)", line)
        if J_match:
            i = i +1
            if i % 2 == 0:
                J = float(J_match.group(1))
                CT = float(CT_match.group(1))
                CP = float(CP_match.group(1))
                J_list.append(J)
                CT_list.append(CT)
                CP_list.append(CP)


""" Execute DFDC """
with subprocess.Popen(["dfdc.exe"],
                      stdin=subprocess.PIPE,
                      stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE,
                      text=True,
                      bufsize=1) as process:

    # Start a thread to read output
    thread = threading.Thread(target=read_output, args=(process,), daemon=True)
    thread.start()
    
    # Define empty output lists and range of operating conditions
    rpm_list = np.linspace(500, 4000, 50)
    J_list = []
    CT_list = []
    CP_list = []

    # Send input continuously
    try:
        while process.poll() is None:
            process.stdin.write("load beta_29.case\n")
            process.stdin.write("ppar\n")
            process.stdin.write("\n")
            process.stdin.write("\n")
            process.stdin.write("pane\n")
            process.stdin.write("oper\n")
            process.stdin.write("vinf 30\n")
            for rpm in rpm_list:
                process.stdin.write(f"RPM {rpm}\n")
                process.stdin.write("e\n")
            next = input("Next?")
            if next == "":
                process.terminate()
            else:
                process.stdin.write(f"{next}\n")
    except KeyboardInterrupt:
        print("Exiting interaction...")

    # Write data to CSV file
    data_out=np.transpose([J_list, CT_list, CP_list])
    data_out=np.vstack((np.array(["J", "CT", "CP"]), data_out))
    with open('dfdc_J_range.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data_out)
