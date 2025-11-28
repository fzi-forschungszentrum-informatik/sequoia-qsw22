# Evaluation of QuantumArchitecturePatterns

This branch of the repository contains the evaluation scripts for our paper *Detecting and Tolerating Faults in Hybrid Quantum Software Systems Using Architectural Redundancy*, DOI:[10.1109/QSW67625.2025.00028](https://doi.org/10.1109/QSW67625.2025.00028)
```
@inproceedings{Klamroth2025,
    doi       = {10.1109/QSW67625.2025.00028},
    url       = {https://doi.org/10.1109/QSW67625.2025.00028},
    author    = {Klamroth, Jonas and Scheerer, Max Scheerer and Denninger, Oliver},
    title     = {Detecting and Tolerating Faults in Hybrid Quantum Software Systems Using Architectural Redundancy}, 
    booktitle = {2025 IEEE International Conference on Quantum Software (QSW)}, 
    year      = {2025},
    pages     = {162-172}
}
```

This work is part of the SEQUOIA and SEQUOIA End-to-End projects funded by the Ministry of Economic Affairs Baden-Württemberg, Germany.

## Setup
Install the necessary dependencies with ```pip install -r requirements.txt```

## Usage
To reproduce the results from the paper just run the evalution python script for the corresponding pattern from the patterns folder: ```VoterEvaluation.py```, ```SwitchEvaluation.py``` or ```ComparatorEvaluation.py```. 

If you want to run an evaluation on new simulation results run the ```SimulationRunner.py```.