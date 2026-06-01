# Case study and validation data of the Unified Ducted fan Code and Optimisation Framework (UDC)

## Description

This dataset contains the data used to demonstrate and validate the Unified Ducted fan Code and Optimisation Framework (UDC). The UDC is a medium-fidelity analysis tool for ducted fan propulsors that implements an optimisation framework using the U-NSGA-III genetic algorithm to automatically design optimal ducted fans for different operating conditions. This UDC was first developed in [T.S. Vermeulen's MSc thesis](https://repository.tudelft.nl/record/uuid:21c8fb7b-f2d5-4c79-a43c-367cc1537764), and subsequently improved and expanded during the corresponding author's PhD project in late 2025/early 2026 to support the writing of a conference paper for the ASME 2026 Turbo Expo [1] and accompanying journal article in the Journal of Engineering for Gas Turbines and Power [2]. The data is made public to aid others in using/experimenting with the UDC and further analysis of the performance of the optimisation framework.

Four sets of data are included in this dataset:

- validation data used to validate the UDC implementation against experimental wind tunnel data of the X-22A ducted propulsor (`Validation`). See `Validation/README.md` for further details.

- Configuration files and UDC output for the optimisations performed in support of the mentioned publications [1] and [2] (`Single-Point Data/` and `Multi-Point Data/`). The optimisation configuration files for the framework (`config.py`) are provided for:
        - The single-point endurance cruise operating condition, where the propulsor efficiency is to be maximized (see more information on the associated operating conditions in `Single-Point Data/config.py`).  
        - The multi-point simplified endurance mission. This mission consists of a climb condition combined with the endurance cruise condition. Here, the total mission energy is to be minimized (see more information on the associated operating conditions in `Multi-Point Data/config.py`).  

- Performance analysis code and results for the multi-point optimised Bell X-22A design presented in the abovementioned papers [1] and [2] (`Performance Curves/`). The code performs a parameter sweep of the rotational rate and tip set angle at a given operating condition, to create curves of the propulsor efficiency as function of the thrust coefficient for different blade tip set angles. The code must be used together with the UDC code, see also the 'Related Code' section of this readme. The `data.csv` contains the performance data with columns:
        - beta_tip: blade tip angle (degrees)
        - RPS: rotations per second
        - TC: thrust coefficient
        - PC: power coefficient
        - EtaP: propulsor efficiency
 The performance data is obtained for the climb condition using the UDC at rotational rates between 20-44 Hz (using steps of 2 Hz). The X-22A blade tip set angle was varied between 5 and 25 degrees using steps of 4 degrees.

## Disclaimer

The optimisation result objects, stored as .dill files, are not included in this repository due to their size. Instead, they are available from the dataset on the [4TU database](https://doi.org/10.4121/02e861af-4062-49ac-a481-32f9e4659b14). These optimisation objects contain both the optimization history and the final optimized population of designs. To load the objects and analyse them in Python, the dill package must be installed together with a copy of the UDC. See also the requirements for the UDC in the parent directory of the UDC codebase. The objects can be loaded using the PostProcessing class of the UDC, part of the GA folder of the UDC. Note that the single- and multi- point optimizations have different configuration files, which must be used appropriately to ensure correct processing of the results objects.

## Dataset Structure

```batch
.
├── Multi-Point Data
│   ├── config.py
├── Performance Curves
│   ├── data.csv
│   └── generate_performance_curves.py
├── README.md
├── Single-Point Data
│   ├── config.py
└── Validation
    ├── README.md
    ├── Uncertainties on Experimental X22A Data.xlsx
    ├── Validation Data.xlsx
    ├── X22A Blade Profiles
    │   ├── Dstrut.dat
    │   ├── Hstrut.dat
    │   ├── X22_02R.dat
    │   ├── X22_03R.dat
    │   ├── X22_04R.dat
    │   ├── X22_05R.dat
    │   ├── X22_06R.dat
    │   ├── X22_07R.dat
    │   ├── X22_08R.dat
    │   ├── X22_09R.dat
    │   └── X22_10R.dat
    ├── X22A DFDC Data
    │   ├── README.md
    │   ├── beta_145.case
    │   ├── beta_245.case
    │   ├── dfdc_beta145.csv
    │   └── dfdc_beta245.csv
    └── X22A Propeller Design Data.xlsx
```
