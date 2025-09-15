import numpy as np
import math
import os
import json
from tabulate import tabulate
from tqdm import tqdm
from itertools import combinations
from QuantumPatterns import QuantumComparator, QuantumSwitch
from ErrorDetection import MaxComparator, DistComparator, hellinger, UniformRater, DiracRater, UniformErrorDetector, DiracErrorDetector


considered_benchmarks = ["dj", "grover-noancilla", "grover-v-chain", "ae", "ghz", "graphstate", "qaoa", "qft", "qftentangled", "twolocalrandom", "vqe"]
selected_benchmarks = ["dj", "grover-noancilla", "grover-v-chain", "ae", "ghz", "vqe"]

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

def get_position(counts, correct_states):
    if counts is None:
        return None
    max_count = 0
    for correct_state in correct_states:
        if correct_state not in counts:
            count = 0
        else:
            count = counts[correct_state]
        if count >= max_count:
            max_count = count
    
    if (max_count == 0):
        return 2 ** len(next(iter(counts)))
    sorted_counts = list(reversed(sorted(list(set(counts.values())))))
    return sorted_counts.index(max_count)

def evalDistComparator(results, truth):
    best_result = get_best_results(truth)
    comb = QuantumSwitch(None, UniformErrorDetector(hellinger, 0.4)).execute_on_results(results)
    maxComb = get_best(comb)
    max1 = get_best(results[0])
    max2 = get_best(results[1])
    max3 = get_best(results[2])

    if (max1 in best_result or max2 in best_result or max3 is best_result) and maxComb is None:
        return [1, 0, 0]
    if (max1 not in best_result and max2 not in best_result and max3 not in best_result) and maxComb is not None:
        return [0, 1, 0]
    return [0, 0, 1]
    
    
def eval(evalFunc, results, truth):
    res = []
    for (backend1_results, backend2_results, backend3_results) in tqdm(list(combinations(results, 3))[:]):
        combination_result = []
        for circuit_result_idx in range(len(backend2_results)):
            combination_result.append(evalFunc([backend3_results[circuit_result_idx][0], backend2_results[circuit_result_idx][0], backend1_results[circuit_result_idx][0]], truth[circuit_result_idx]))
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


def select_benchmarks(selected_benchmarks, all_benchmarks):
    res = []
    slices = []
    for b in selected_benchmarks:
        idx = considered_benchmarks.index(b)
        slices.append(slice(idx * 8, idx*8 + 8))
    for b in all_benchmarks:
        res.append(b[np.r_[tuple(slices)]])
    return np.array(res)
    
def printConfiguration(pattern, comparison_metric, num_channels, variant_creation_method):
    config_data = [
        ["Pattern", pattern],
        ["Error Detection", comparison_metric],
        ["Number of Channels", num_channels],
        ["Variant Creation Method", variant_creation_method]
    ]
    print("\nRun evalution for the following configuration:")
    print(tabulate(config_data, tablefmt="pretty"))


if __name__ == "__main__":
    with open('simulations/optimal.json') as f:
        truth = json.load(f)
    
    printConfiguration("Switch", "UniformDetection(0.4)", 3, "backends")
    results = get_all_results('simulations/backends')
    eval(evalDistComparator, results, truth)

    printConfiguration("Switch", "UniformDetection(0.4)", 3, "seeds")
    results = get_all_results('simulations/seeds')
    eval(evalDistComparator, results, truth)

    printConfiguration("Switch", "UniformDetection(0.4)", 3, "optimizations levels")
    results = get_all_results('simulations/optimizations')
    eval(evalDistComparator, results, truth)

