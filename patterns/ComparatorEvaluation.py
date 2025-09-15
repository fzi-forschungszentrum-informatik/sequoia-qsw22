import numpy as np
import math
import os
import json
from tqdm import tqdm
from itertools import combinations
from QuantumPatterns import QuantumComparator
from ErrorDetection import MaxComparator, DistComparator, hellinger, UniformRater, DiracRater
from tabulate import tabulate  # Add this import at the top if not present


considered_benchmarks = ["dj", "grover-noancilla", "grover-v-chain", "ae", "ghz", "graphstate", "qaoa", "qft", "qftentangled", "twolocalrandom", "vqe"]

def get_all_results(path):
    files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    res = []
    for file in files:
        with open(file) as f:
            res.append(json.load(f))
    return np.array(res)

def get_best_results(truth):
    probs = truth
    epsilon = 0.0001
    getbinary = lambda x, n: format(x, 'b').zfill(n)
    maxProb = np.amax(probs)
    bestIdxs = np.argwhere(probs > maxProb - epsilon).flatten().tolist()
    n = (int)(math.log2(len(probs)))
    return [getbinary(i, n) for i in bestIdxs]

def get_best(counts):
    if counts is None:
        return None
    return max(counts, key=counts.get)


def evalDistComparator(res1, res2, truth):
    best_result = get_best_results(truth)
    max1 = get_best(res1)
    max2 = get_best(res2)
    maxComp = get_best(QuantumComparator(DistComparator(hellinger, 0.4), None).execute_on_results([res1, res2]))
    if max1 in best_result and max2 in best_result and maxComp is None:
        return [1, 0, 0]
    if not (max1 in best_result) and not (max2 in best_result) and maxComp is not None:
        return [0, 1, 0]
    return [0, 0, 1]

def evalMaxComparator(res1, res2, truth):
    best_result = get_best_results(truth)
    max1 = get_best(res1)
    max2 = get_best(res2)
    maxComp = get_best(QuantumComparator(MaxComparator(), None).execute_on_results([res1, res2]))
    if max1 in best_result and max2 in best_result and maxComp is None:
        return [1, 0, 0]
    if not (max1 in best_result) and not (max2 in best_result) and maxComp is not None:
        return [0, 1, 0]
    return [0, 0, 1]

def eval(evalFunc, results, truth):
    res = []
    for (backend1_results, backend2_results) in tqdm(list(combinations(results, 2))[:]):
        combination_result = []
        for circuit_result_idx in range(len(backend2_results)):
            combination_result.append(evalFunc(backend1_results[circuit_result_idx][0], backend2_results[circuit_result_idx][0], truth[circuit_result_idx]))
        res.append(combination_result)

    res = np.array(res)
    altRes = res[:, np.r_[0:40, 64:88], :]
    res1 = np.sum(res, axis=0)
    res1 = np.sum(res1, axis=0)
    total = np.sum(res1)
    percentages = res1/total
    print("\nFull Range Results:")
    print(tabulate([res1, percentages], headers=["FP", "FN", "Correct"], tablefmt="pretty"))
    res1 = np.sum(altRes, axis=0)
    res1 = np.sum(res1, axis=0)
    total = np.sum(res1)
    percentages = res1/total
    print("\nLimited Benchmark Results:")
    print(tabulate([res1, percentages], headers=["FP", "FN", "Correct"], tablefmt="pretty"))

def printConfiguration(pattern, comparison_metric, num_channels, variant_creation_method):
    config_data = [
        ["Pattern", pattern],
        ["Comparison Metric", comparison_metric],
        ["Number of Channels", num_channels],
        ["Variant Creation Method", variant_creation_method]
    ]
    print("\nRun evalution for the following configuration:")
    print(tabulate(config_data, tablefmt="pretty"))



if __name__ == "__main__":
    with open('simulations/optimal.json') as f:
        truth = json.load(f)

    printConfiguration("ComparatorPattern", "Hellinger(0.4)", 2, "backends")
    results = get_all_results('simulations/backends')
    eval(evalDistComparator, results, truth)

    printConfiguration("ComparatorPattern", "Hellinger(0.4)", 2, "seeds")
    results = get_all_results('simulations/seeds')
    eval(evalDistComparator, results, truth)

    printConfiguration("ComparatorPattern", "Hellinger(0.4)", 2, "optimization levels")
    results = get_all_results('simulations/optimizations')
    eval(evalDistComparator, results, truth)

    printConfiguration("ComparatorPattern", "Max", 2, "backends")
    results = get_all_results('simulations/backends')
    eval(evalMaxComparator, results, truth)
    
    printConfiguration("ComparatorPattern", "Max", 2, "seeds")
    results = get_all_results('simulations/seeds')
    eval(evalMaxComparator, results, truth)

    printConfiguration("ComparatorPattern", "Max", 2, "optimization levels")
    results = get_all_results('simulations/optimizations')
    eval(evalMaxComparator, results, truth)
