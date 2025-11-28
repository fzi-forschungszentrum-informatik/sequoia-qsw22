# quantumNVersionEval
This repository provides the necessary scripts and files to reproduce the experiments described in our paper *Fault-tolerant Hybrid Quantum Software System*, DOI:[10.1109/QSW55613.2022.00023](https://doi.org/10.1109/QSW55613.2022.00023)
```
@inproceedings{Klamroth2023,
    doi       = {10.1109/QSW55613.2022.00023},
    url       = {https://doi.org/10.1109/QSW55613.2022.00023},
    author    = {Scheerer, Max and Klamroth, Jonas and Denninger, Oliver},
    title     = {Fault-tolerant Hybrid Quantum Software Systems},
    booktitle = {2022 IEEE International Conference on Quantum Software (QSW)},
    year      = {2022},
    pages     = {52-57}
}
```

An extension of this work was published in 2025 in our paper *Detecting and Tolerating Faults in Hybrid Quantum Software Systems using Architectural Redundancy*, for details and scripts see repository branch [qsw25](https://github.com/fzi-forschungszentrum-informatik/sequoia-qsw22/tree/qsw25).

This work is part of the SEQUOIA and SEQUOIA End-To-End projects funded by the Ministry of Economic Affairs Baden-Württemberg, Germany

## Setup
In order to run the experiments the necessary python packages have to be installed. This can be done using ```pip -r install requirements.txt```.

Additionally if the experiments are to be run on the same backend as we did in the paper one has to provide a valid IBM access for the quantum device. This is encapsulated in the file ```activateAcc.py```. Normally this file should contain code similar to the following, where *token* is your IBMQ token.
```
from qiskit import IBMQ

def activateAcc():
    APITOKEN = 'token'
    APIURL = 'https://auth.de.quantum-computing.ibm.com/api'
    IBMQ.enable_account(APITOKEN, APIURL)
```

## Run
The experiments are provided by the python script ```voterPatternEval.py```. To run new experiments call ```python voterPatternEval.py run N``` where N is the number of circuits you would like to use for the experiment. To get a summary of the results presented in the paper run ```python voterPatternEval.py interpret```. Several options are available in order to configure the output and the experiment setup. To view all options run the script with the --help option (note the --help option for the subcommands as well e.g. ```python voterPatternEval.py run --help```).
