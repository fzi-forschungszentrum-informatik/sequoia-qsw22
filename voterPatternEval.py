from ast import alias
import json
from math import log2
import sys
from tokenize import String
from tabulate import tabulate
from qiskit import IBMQ, Aer, execute, transpile
from qiskit.circuit.random import random_circuit
import warnings
warnings.filterwarnings('ignore')
from qiskit.ignis.verification import combine_counts
from qiskit.test.mock import FakeProvider
from qiskit.providers.aer.noise import NoiseModel
import numpy as np
import pandas as pd
import random
import argparse
from os import listdir, mkdir
from os.path import isfile, join, exists
from datetime import datetime
from activateAcc import activateAcc


resultsDir = "results"

class ExperimentResult:
    groundTruth = None
    transpilationResults = None
    optimizationResults = None
    backendResults = None

    def __init__(self, groundTruth) -> None:
        self.groundTruth = groundTruth

    def __init__(self, groundTruth, transpilationResults, optimizationResults, backendResults) -> None:
        self.groundTruth = groundTruth
        self.optimizationResults = optimizationResults
        self.transpilationResults = transpilationResults
        self.backendResults = backendResults

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

    def combine(self, other):
        if(len(self.transpilationResults) != 0 and len(self.transpilationResults)/len(self.groundTruth) != len(other.transpilationResults)/len(other.groundTruth)):
            print("Error combining results. Each result must have the same number of transpilation variants")
            sys.exit(1)
        if(len(self.optimizationResults) != 0 and len(self.optimizationResults)/len(self.groundTruth) != len(other.optimizationResults)/len(other.groundTruth)):
            print("Error combining results. Each result must have the same number of optimization variants")
            sys.exit(1)
        if(len(self.backendResults) != 0 and len(self.backendResults)/len(self.groundTruth) != len(other.backendResults)/len(other.groundTruth)):
            print("Error combining results. Each result must have the same number of backend variants")
            sys.exit(1)
        self.groundTruth.extend(other.groundTruth)
        self.transpilationResults.extend(other.transpilationResults)
        self.optimizationResults.extend(other.optimizationResults)
        self.backendResults.extend(other.backendResults)
        return self
        


def tryTranspilations(backend, circuits, N=9):
    print("Creating transpilation seed experiment job")
    transpiledCircuits = []
    for circuit in circuits:
        for i in range(N):
            transpiledCircuits.append(transpile(circuit, backend=backend, seed_transpiler=random.randrange(0, 10000)))

    job = execute(transpiledCircuits, backend, shots=4096)

    return job

def runOnBackend(backend, circuit):
    simulator = Aer.get_backend('qasm_simulator')
    noise_model = NoiseModel.from_backend(backend, warnings=False)

    # Execute the circuit on the simulator with error characteristics of the given backend
    job = execute(circuit, simulator, shots=1024, noise_model=noise_model)

    # Grab results from the job
    result = job.result()
    return result.get_counts()


def tryBackends(circuits):
    print("Running backend experiment batch")
    provider = FakeProvider()
    backends = [ b.name() for b in provider.backends() if b.configuration().n_qubits >= 10] 
    counts = [{} for _ in range(len(circuits)*len(backends))] 
    for i in range(len(backends)):
        be = provider.get_backend(backends[i])
        counts[i::len(backends)] = runOnBackend(be, circuits)

    return counts


def tryOptimizations(backend, circuits):
    print("Creating optimization experiment job")
    transpiledCircuits = []
    for circuit in circuits:
        for i in range(4):
            transpiledCircuits.append(transpile(circuit, backend=backend, optimization_level=i, seed_transpiler=123))

    job = execute(transpiledCircuits, backend, shots=4096)

    return job

def runOnSimulator(circuits):
    print("Running on simulator to get ground truth")
    simulator = Aer.get_backend('statevector_simulator')
    res = []
    for circuit in circuits:
        job = execute(circuit, simulator)
        result = job.result()
        stv = result.get_statevector(circuit, decimals=3)
        probs = stv.probabilities()
        bestIdxs = np.argwhere(probs == np.amax(probs)).flatten().tolist()
        n = (int)(log2(len(probs)))
        getbinary = lambda x, n: format(x, 'b').zfill(n)
        bestIdxsBin = [getbinary(i, n) for i in bestIdxs]
        res.append(bestIdxsBin)
    return res


def saveResult(result, fileName=None):
    jsonString = result.toJSON()
    if fileName is None:
        fileName = join(resultsDir, "results" + str(datetime.now()) +  ".json")
    if not exists(resultsDir):
        mkdir(resultsDir)
    with open(fileName, "w") as f:
      f.write(jsonString)
    print("Writing result file: " + fileName)
    return fileName

def loadResult(fileName):
    with open(fileName, "r") as f:
      res = f.read()
      o = json.loads(res)
      return ExperimentResult(groundTruth=o['groundTruth'], transpilationResults=o['transpilationResults'], optimizationResults=o['optimizationResults'], backendResults=o['backendResults'])

def aggregateResults(counts):
    res = {}
    for c in counts:
        res = combine_counts(res, c)
    return res

def getPosForResult(result, truth):
    truthCount = 0
    for t in truth:
        if t in result:
            truthCount = max(result[t], truthCount)
    res = 0
    for k in result.keys():
        if result[k] > truthCount:
            res += 1
    return res

def getNumCorrect(results, truth):
    return sum(1 for r in results if getPosForResult(r, truth) == 0)

def getNumTopTenPercent(results, truth, numPossibleResults):
    fraction = (int)(numPossibleResults/10)
    fraction = max(3, fraction)
    return sum(1 for r in results if getPosForResult(r, truth) < fraction)

def getAveragePos(results, truth):
    return sum(getPosForResult(r, truth) for r in results)/len(results)

def runExperiments(args):
    resultsDir = args.outputDir
    N = args.N
    #activate your IBMQ account here. You need that to run circuits on real backends
    #if you want to run on a simulator provide this here
    activateAcc()
    provider = IBMQ.get_provider(project="ticket")
    backend = provider.get_backend(args.backend)
    if args.simulate:
        provider = FakeProvider()
        if args.backend != "ibmq_ehningen":
            backend = provider.get_backend(args.backend)
        else:
            backend = provider.get_backend("fake_boeblingen")
    res = []
    optJobs = []
    transJobs = []
    results = []
    allCircuits = []
    numBatches = 0
    while N > 0:
        #number of randomized circuits to consider
        numCircuits = 25 if N >= 25 else N 
        N -= numCircuits
        numBatches += 1
        maxNumQubits = 10 
        maxDepth = 40

        circuits = []
        for i in range(numCircuits):
            circuit = random_circuit(random.randint(2, maxNumQubits), random.randint(5, maxDepth), measure=True)
            circuits.append(circuit)

        groundTruth = runOnSimulator(circuits)
        result = ExperimentResult(groundTruth, None, None, None)
        results.append(result)
        optJobs.append(tryTranspilations(backend, circuits))
        transJobs.append(tryOptimizations(backend, circuits))
        allCircuits.append(circuits)
        
    for i in range(numBatches):
        results[i].backendResults = tryBackends(allCircuits[i])

    for i in range(numBatches):
        print("Trying to retrieve results from backend.")
        try:
            results[i].transpilationResults = transJobs[i].result().get_counts()
            results[i].optimizationResults = optJobs[i].result().get_counts()
            res.append(saveResult(results[i]))
        except Exception:
            print("retrieving one experiment result was unsucessful. Trying next.")
            continue
    return res

def processResults(results, groundTruth, string=""):
    numCircuits = len(groundTruth)
    numResults = (int)(len(results)/len(groundTruth))
    correct = [0 for i in range(numResults)]
    topTens = [0 for i in range(numResults)]
    combPos = [0 for i in range(numCircuits)]
    allPos = [[0 for i in range(numResults)] for i in range(numCircuits)]
    notFiltered = []
    numBetterAverage = 0
    numWorseAverage = 0
    numAvgCorrect = 0
    numCombCorrect = 0
    numAvgTT = 0
    numCombTT = 0
    numFiltered = 0
    for i in range(numCircuits):
        truth = groundTruth[i]
        numPossibleResults = 2**len(truth[0])
        resultSegment = results[i*numResults:i*numResults+numResults:]
        numCorrect = getNumCorrect(resultSegment, truth)
        numTopThree = getNumTopTenPercent(resultSegment, truth, numPossibleResults)
        #if numCorrect > 0:
        #if numTopThree > 0:
        if args.filter == None or (args.filter == "correct" and numCorrect > 0) or (args.filter == "T10" and numTopThree > 0):
            notFiltered.append(i)
            for j in range(numResults):
                correct[j] += getNumCorrect([resultSegment[j]], truth)
                topTens[j] += getNumTopTenPercent([resultSegment[j]], truth, numPossibleResults)
                p = getPosForResult(resultSegment[j], truth)
                allPos[i][j] = p
            if numTopThree > numResults / 2:
                numAvgTT += 1
            if numCorrect > numResults / 2:
                numAvgCorrect += 1
            avgPos = getAveragePos(resultSegment, truth)

            combinedResult = aggregateResults(resultSegment)
            isTopThree = getNumTopTenPercent([combinedResult], truth, numPossibleResults) == 1 
            if isTopThree:
                numCombTT += 1
            isCorrect = getNumCorrect([combinedResult], truth) == 1 
            if isCorrect:
                numCombCorrect += 1
            combinedPos = getPosForResult(combinedResult, truth)
            combPos[i] = combinedPos
            numFiltered += 1
            if combinedPos < avgPos:
                numBetterAverage += 1
            if combinedPos > avgPos:
                numWorseAverage += 1

    allPos = np.array(allPos)
    idx = np.argmax(correct)
    numBetterBest = sum([1 for i in range(numCircuits) if combPos[i] < allPos[i][idx] and i in notFiltered])/numFiltered
    numWorseBest = sum([1 for i in range(numCircuits) if combPos[i] > allPos[i][idx] and i in notFiltered])/numFiltered
    numEqualBest = 1.0 - numBetterBest - numWorseBest
    numBetterAverage /= numFiltered
    numWorseAverage /= numFiltered
    numEqualAverage = 1.0 - numBetterAverage - numWorseAverage 
    print("number of Experiments for " + string + " = " + str(numFiltered))
            
    return [[string, "Avg", numAvgCorrect, numAvgTT, tuple(np.round((numWorseAverage, numEqualAverage, numBetterAverage), decimals=2))],
            [string, "Agg", numCombCorrect, numCombTT, "-"],
            [string, "Best", correct[idx], topTens[idx], tuple(np.round((numWorseBest, numEqualBest, numBetterBest), decimals=2))]]
            
def interpretFolder(args):
    resultsDir = args.dir
    print("Interpreting existing results in " + str(resultsDir))
    resultFiles = [join(resultsDir, f) for f in listdir(resultsDir) if isfile(join(resultsDir, f)) and "results" == f[:7] and ".json" == f[-5:]]
    if len(resultFiles) == 0:
        print("Did not find any result files. Aborting.")
    else: 
        print("Found " + str(len(resultFiles)) + " files with results.")
        latex = interpretResults(resultFiles)
        if args.printLatex:
            print(latex)


def interpretResults(fileNames):
    res = ExperimentResult([], [], [], [])
    for fileName in fileNames:
        res = res.combine(loadResult(fileName))

    df = []

    df.extend(processResults(res.transpilationResults, res.groundTruth, "seed"))
    df.extend(processResults(res.optimizationResults, res.groundTruth, "opt"))
    df.extend(processResults(res.backendResults, res.groundTruth, "back"))

    df = pd.DataFrame(df, columns=["Appr.", "View", "numT1", "numT10%", "comparison"])
    df = df.round(decimals=1)
    print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))
    return df.to_latex(index=False)

def showBackends(args):
    provider = FakeProvider()
    backends = [ b.name() for b in provider.backends() if b.configuration().n_qubits >= 10]
    print(backends)

def runAndShowExperiments(args):
    if(args.N % 25 == 1):
        print("Due to technical details runs with circuits % 25 == 1 are not allowed")
        sys.exit(1)
    latex = interpretResults(runExperiments(args))
    if(args.printLatex):
        print(latex)


parser = argparse.ArgumentParser(description="Script to run and interpret experiments for the evaluation of the N-version pattern for quantum computing.")
subparsers = parser.add_subparsers(help="sub-command help")

parser_int = subparsers.add_parser('interpret', help="summarizing existing result files")
parser_int.add_argument('--dir', '-d', default="results", help="the folder which contains the result files (default = ./results)")
parser_int.add_argument('--filter', '-f', help="filter results, available options = [none (default, show all), correct (at least one correct variant), T10 (at least one correct variant with T10 result)", choices=[None, "correct", "T10"], default=None)
parser_int.add_argument('--printLatex', '-pl', help="print a latex table representing the results", action='store_true')
parser_int.set_defaults(func=interpretFolder)

parser_run = subparsers.add_parser('run', help="running experiments")
parser_run.add_argument('N', type=int, default=25, help="number of circuits to run experiments on")
parser_run.add_argument('--filter', '-f', help="filter results, available options = [none (default, show all), correct (at least one correct variant), T10 (at least one correct variant with T10 result)", choices=[None, "correct", "T10"], default=None)
parser_run.add_argument('--outputDir', '-o', help="the folder to save results to")
parser_run.add_argument('--simulate', '-sim', help="simulate all experiments instead of running on real backend", action='store_true')
parser_run.add_argument('--backend', '-b', help="run on certain backend (has to be fake-backend if used in combination with --simulate)", default="ibmq_ehningen")
parser_run.add_argument('--printLatex', '-pl', help="print a latex table representing the results", action='store_true')
parser_run.set_defaults(func=runAndShowExperiments)

parser_showBackends = subparsers.add_parser('showBackends', help="show available fake backends suited for the experiments")
parser_showBackends.set_defaults(func=showBackends)

if len(sys.argv) < 2:
    parser.print_help()
else:
    args = parser.parse_args(sys.argv[1:])
    args.func(args)
