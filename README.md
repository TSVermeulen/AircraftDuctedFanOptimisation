# The Unified Ducted Fan Code and Ducted Fan Optimisation Framework

This GitHub repository contains the codebase developed for the Unified Ducted fan Code (UDC) and its implementation into a ducted fan optimisation framework using the Unified Non-dominated Sorting Genetic Algorithm III (U-NSGA-III), as started in work of the MSc thesis titled "A Framework for Medium-Fidelity Ducted Fan Design Optimisation" by T.S. Vermeulen at Delft University of Technology, Faculty of Aerospace Engineering.

The developed UDC and optimisation framework have been improved and updated by the original author. The updated version has been used to write a paper for the Turbo Expo 2026 conference as paper number GT2026-175354 "A Medium-Fidelity Modelling Framework for Ducted E-Fan Design" and an associated journal publication in the Journal of Engineering for Gas Turbines and Power using the same title with paper number GTP-26-1236.

## License

Two different licenses apply for the UDC and MTFLOW. The MTFLOW software has the following license:

Copyright Notice and Disclaimer. The software [or “portions of the software”] incorporated
herein is MTFLOW Software, © MIT 1997 used with permission. All Rights Reserved.

For the developed UDC, a standard MIT license holds. See also the dedicated LICENSE file in the repository for further details.

## Description

This code builds on the MTFLOW software developed by M. Drela to create a fast, robust, and accurate ducted fan analysis code. This code is implemented in the U-NSGA-III algorithm to enable design explorations for different operating conditions, objectives, and constraints. This repository also contains the validation data used to validate the implementation against experimental wind tunnel data of the X-22A ducted propulsor. This wind tunnel data is reported in NASA-TN-D-4142. The validation data is contained in the validation folder.

Currently, the UDC works on both Windows and Linux, and is capable of solving fixed- and variable-pitch optimisation problems for both single operating conditions (which, by nature are fixed-pitch) and flight profiles defined using multiple operating conditions.
For a detailed description of the developed methods and results, the reader is referred to the thesis, which is publicly in the [TU Delft Repository](https://repository.tudelft.nl/record/uuid:21c8fb7b-f2d5-4c79-a43c-367cc1537764).

As per the license for MTFLOW, the MTFLOW codes cannot be freely distributed.
Should the reader wish to use the developed frameworks in this thesis, they need to request a license for MTFLOW directly from the [MIT Technology Licensing Office](https://tlo.mit.edu/industry-entrepreneurs/available-technologies/mtflow-software-multielement-through-flow).

For best performance, it is recommended to run the optimisation framework on a computer or server with as many CPU cores/threads as possible, since each thread can be used to run one analysis. Testing of the developer shows 16 analyses can be conducted simultaneously on an AMD Ryzen 5xxx 8-core/16-thread CPU. An average design takes between 10--45 seconds to evaluate in the UDC.

## Requirements

To run the UDC or the developed optimisation framework, the dependencies listed below must be satisfied. A requirements.txt file is provided as part of this repository. To improve performance, it is suggested to install pymoo last during the installation process, using following the command, as per the instructions in the pymoo documentation.

```console
pip install -U pymoo==0.6.1.5
```

- Pymoo 0.6.1.5 (Optimisation framework only)
- Ambiance 1.3.1
- NumPy 2.2.1
- SciPy 1.16.3
- Matplotlib 3.10.7
- Dill 0.4.0 (Optimisation framework only)
- Pandas 2.3.3

The tools and frameworks developed in this thesis were written in Python 3.12.8 using a standard conda environment. Although the author sees no issues with usage at other Python versions, no guarantees are given.

## The UDC

The developed ducted fan analysis code, UDC, in this repository builds on and integrates the existing MTFLOW software in a wrapper to obtain a unified, robust ducted fan design and analysis tool. Simplified diagrams of the UDC are presented below, illustrating both the connections between the various Python modules and the sequential solving strategy employed in the UDC.

<img width="625" height="467" alt="UDC_filediagram" src="https://github.com/user-attachments/assets/6de1bdb7-ba54-4368-bcac-01f99c6120f4" />

<img width="1177" height="637" alt="UDC_flowdiagram" src="https://github.com/user-attachments/assets/7f9f31ba-0df9-4ce7-9711-ffdcdbd4d6e0" />

<img width="1255" height="395" alt="MTSOL_flowdiagram" src="https://github.com/user-attachments/assets/171b42dc-8db0-4ad0-9e3e-db6c266d2ac7" />

For more details on the different (sub-)modules of the UDC, please refer to Appendix C.1 of the written thesis, or the numerous documentation present within each file in this repository.

## The Optimisation Framework

The developed modular ducted fan optimisation framework in this repository integrates the UDC into a customised, mixed-variable (continuous + integer) U-NSGA-III genetic algorithm. A pipeline diagram of the developed optimisation framework is shown below:

<img width="1042" height="625" alt="OptimisationFramework" src="https://github.com/user-attachments/assets/1723ee1a-166a-46c8-b4a4-3846485d613e" />

For more details on the different (sub-)modules of the optimisation framework, please refer to Appendix C.2 of the written thesis, or the numerous documentation present within each file in this repository.

## Funding

This research was partially conducted by DASAL (Dutch Aviation Systems Analysis Lab), which is a project within the research and innovation program ‘Luchtvaart in Transitie’. Luchtvaart in Transitie is co-funded by the National Growth Fund.
