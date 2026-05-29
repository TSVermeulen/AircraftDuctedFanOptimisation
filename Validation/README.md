# Unified Ducted Fan Code - Validation

This folder contains the data and summarised results of the validation exercise undertaken for the UDC.
The validation has been done against the X-22A 3-bladed ducted propeller, for which geometry and experimental test results are given in [NASA-TN-D-4142](https://ntrs.nasa.gov/citations/19670025554) 

The structure of this folder is the following:

```bash
└── Validation
    ├── README.md
    ├── Validation Data.xlsx                          # main validation results
    ├── Profiles                                      # airfoil geometry
    │   ├── Dstrut.dat                                # diagonal strut profile
    │   ├── Hstrut.dat                                # horizontal strut profile
    │   ├── X22_02R.dat
    │   ├── X22_03R.dat
    │   ├── X22_04R.dat
    │   ├── X22_05R.dat
    │   ├── X22_06R.dat
    │   ├── X22_07R.dat
    │   ├── X22_08R.dat
    │   ├── X22_09R.dat
    │   └── X22_10R.dat
    ├── DFDC                                         # DFDC related code and files
    │   ├── beta_145.case                            # input case file for 14.5 degrees blade angle
    │   ├── beta_245.case                            # input case file for 24.5 degrees blade angle
    │   ├── dfdc_beta145.csv                         # DFDC results for 14.5 degrees
    │   ├── dfdc_beta245.csv                         # DFDC results for 24.5 degrees
    │   ├── dfdc.exe                                 # DFDC executable
    │   └── DFDC_run_J.py                            # Python DFDC wrapper
    └── X22A Propeller Design Data.xlsx               # X-22A design parameters
```

- `Validation Data.xlsx`: contains the NASA wind tunnel experimental data taken from NASA-TN-D-4142, for the X-22A at 14.5 and 24.5 degree blade tip angles. This corresponds to blade angles of 19 and 29 degrees at the 75% span station, which is the reference datum used in the experimental data. It also contains the output data obtained from the UDC for the same two blade tip angles considered. The comparison shows:

  - OMEGA: Nondimensionalised rotational rate used internally in the UDC/MTFLOW. Defined as RPS * 2 * PI * D / V
  - V: Freestream velocity in m/s
  - RPS: The rotational rate of the blade row in Hz
  - J: Advance ratio
  - CT: Thrust Coefficient, normalised using the rotational rate
  - CP: Power Coefficient, normalised using the rotational rate
  - EtaP: Propulsor Efficiency, CT/CP * J

- `X22A Propeller Design Data.xlsx`: The chord, geometric parameters, and collective blade angle distributions for the X-22A blade.

Additionally, there are 2 subfolders:

- `Profiles`: contains the blade profiles for the X-22A propeller used to evaluate the UDC. There are 9 profiles at different radii. These profiles are provided as input for MTFLOW analysis. The subfolder also contains the diagonal and horizontal strut airfoil profiles.

- `DFDC`: contains the input files, python wrapper, executable, and results to compare DFDC vs MTFLOW.
