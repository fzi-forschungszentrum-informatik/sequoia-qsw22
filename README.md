# quantumNVersionEval
This repository provides the necessary material to reproduce the experiments described in our paper "Fault-tolerant Hybrid Quantum Software System" submitted to QSW 2022 which you can find [here](https://ieeexplore.ieee.org/abstract/document/9860169). 

An extension of this work was published in our paper "Detecting and Tolerating Faults in Hybrid Quantum Software Systems using Architectural Redundancy" available [here](https://ieeexplore.ieee.org/abstract/document/11134344). The evaluation of this paper can be found in the [qsw25-branch](https://github.com/fzi-forschungszentrum-informatik/sequoia-qsw22/tree/qsw25)

This work is part of the SEQUOIA and SEQUOIA End-To-End projects funded by the Ministry of Economic Affairs Baden-Württemberg, Germany

## Setup
In order to run the experiments the necessary python packages have to be installed. This can be done using ```pip -r install requirements.txt```. 

Additionally if the experiments are to be run on the same backend as we did in the paper one has to provide a valid IBM access for the quantum device. This is encapsulated in the file activateAcc.py. Normally this file should contain code similar to the following: 
```
from qiskit import IBMQ

def activateAcc():
    APITOKEN = 'token'
    APIURL = 'https://auth.de.quantum-computing.ibm.com/api'
    IBMQ.enable_account(APITOKEN, APIURL)
```

where token is your IBMQ token. 

## Run
The experiments are provided by the python script "voterPatternEval.py". To run new experiments call "python voterPatternEval.py run N" where N is the number of circuits you would like to use for the experiment. To get a summary of the results presented in the paper run "python voterPatternEval.py interpret". Several options are available in order to configure the output and the experiment setup. To view all options run the script with the --help option (note the --help option for the subcommands as well e.g. "python voterPatternEval.py run --help").
