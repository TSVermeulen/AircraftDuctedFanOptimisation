# The Unified Ducted Fan Code and Ducted Fan Optimisation Framework

![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/TSVermeulen/Conceptual-Investigation-of-Alternative-Electric-Ducted-Fan-Architectures)

This GitHub repository contains the codebase developed for the Unified Ducted fan Code (UDC) and its implementation into a ducted fan optimisation framework using the Unified Non-dominated Sorting Genetic Algorithm III (U-NSGA-III), as part of the MSc thesis titled "A Framework for Medium-Fidelity Ducted Fan Design Optimisation" by T.S. Vermeulen at Delft University of Technology, Faculty of Aerospace Engineering.

## Description

This thesis builds on the MTFLOW software developed by M. Drela to create a fast, robust, and accurate ducted fan analysis code. This code is implemented in the U-NSGA-III algorithm to enable design explorations for different operating conditions, objectives, and constraints. This repository also contains the validation data used to validate the implementation against experimental wind tunnel data of the X-22A ducted propulsor. This wind tunnel data is reported in NASA-TN-D-4142. The validation data is contained in the validation folder. 

For a detailed description of the developed methods and results, the reader is referred to the thesis, which is publicly available [here](https://repository.tudelft.nl/), and in the misc folder of this repository

As per the License for MTFLOW, the MTFLOW codes cannot be freely distributed.
Should the reader wish to use the developed frameworks in this thesis, they need to request a license for MTFLOW directly from the MIT Technology Licensing Office. This can be done here: https://tlo.mit.edu/industry-entrepreneurs/available-technologies/mtflow-software-multielement-through-flow

The code in this repository is designed to work on Windows. For a Linux/Unix-like system, the MTFLOW executable filenames, filepaths, and file block checking need to be adjusted accordingly.

For best performance, it is recommended to run the optimisation framework on a computer or server with as many CPU cores/threads as possible, since each thread can be used to run one analysis. Testing of the developer shows 16 analyses can be conducted simultaneously on an AMD Ryzen 5xxx 8-core/16-thread CPU. An average design takes between 10--45 seconds to evaluate in the UDC.


## Requirements

To run the UDC or the developed optimisation framework, the following dependencies must be satisfied:

- Pymoo (Optimisation framework only)
- Ambiance
- NumPy
- SciPy
- Matplotlib
- Dill (Optimisation framework only)
- Pandas

The tools and frameworks developed in this thesis were written in Python 3.12.8 using a conda environment. Although the author sees no issues with usage at other Python versions, no guarantees are given. An offline Python install package is available in the misc subfolder of this repository. 

## The UDFDAC

The developed ducted fan analysis code, UDC, in this repository builds on and integrates the existing MTFLOW software in a wrapper to obtain a unified, robust ducted fan design and analysis tool. Simplified diagrams of the UDC are presented below, illustrating both the connections between the various Python modules and the sequential solving strategy employed in the UDC.

<img width="625" height="467" alt="UDC_filediagram" src="https://github.com/user-attachments/assets/6de1bdb7-ba54-4368-bcac-01f99c6120f4" />

<img width="935" height="507" alt="UDC_flowdiagram" src="https://github.com/user-attachments/assets/42adb0ba-d46e-4328-8185-132a23d8eda9" />

<img width="1255" height="395" alt="MTSOL_flowdiagram" src="https://github.com/user-attachments/assets/2b932ffa-78a9-431c-85c8-e7f440e4db9b" />

For more details on the different (sub-)modules of the UDC, please refer to Appendix C.1 of the written thesis, or the numerous documentation present within each file in this repository. 

## The Optimisation Framework

The developed modular ducted fan optimisation framework in this repository integrates the UDC into a customised, mixed-variable (continuous + integer) U-NSGA-III genetic algorithm. A pipeline diagram of the developed optimisation framework is shown below:

<img width="1049" height="681" alt="OptimisationFramework" src="https://github.com/user-attachments/assets/5b679ff5-89d8-4027-bc02-ab6e10ab5329" />


For more details on the different (sub-)modules of the optimisation framework, please refer to Appendix C.2 of the written thesis, or the numerous documentation present within each file in this repository.
