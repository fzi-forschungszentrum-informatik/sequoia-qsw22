import numpy as np
import math
import os
import json
from tabulate import tabulate
from tqdm import tqdm
from itertools import combinations
from QuantumPatterns import QuantumComparator, QuantumSwitch, QuantumVoter, UniformQuantumCombiner
from ErrorDetection import hellinger, get_probabilities


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
    trueProbabilities = get_probabilities(truth)
    best_result = get_best_results(truth)
    poss = list(map(lambda x: get_position(x, best_result), results))
    hells = list(map(lambda x: hellinger(get_probabilities(x), trueProbabilities), results))
    avgHell = np.average(hells)
    avg = np.average(poss)
    comb = QuantumVoter(UniformQuantumCombiner(), None).execute_on_results(results)
    posComp = get_position(comb, best_result) 

    res = []
    res.append(np.min(poss))
    res.append(avg)
    res.append(posComp)
    res.append(posComp <= avg)
    res.append(np.min(hells))
    res.append(avgHell)
    hell = hellinger(get_probabilities(comb), trueProbabilities)
    res.append(hell)
    res.append(hell <= avgHell)
    res.append(1)
    return res
    
def eval(evalFunc, results, truth):
    res = []
    for (backend1_results, backend2_results, backend3_results) in tqdm(list(combinations(results, 3))[:]):
        combination_result = []
        for circuit_result_idx in range(len(backend2_results)):
            combination_result.append(evalFunc([backend3_results[circuit_result_idx][0], backend2_results[circuit_result_idx][0], backend1_results[circuit_result_idx][0]], truth[circuit_result_idx]))
        res.append(combination_result)

    res = np.array(res)
    relevenat = res
    res1 = np.sum(relevenat, axis=0)
    res1 = np.sum(res1, axis=0)
    res1 = res1[1:-1]
    res1 = np.delete(res1, 3)
    percentages = res1 / res1[-1]
    print(tabulate([res1, percentages], headers=["P(v)", "P(c)", "%P(c) < P(v)", "H(v)", "H(c)", "%H(c)<H(v)"], tablefmt="pretty"))


def select_benchmarks(selected_benchmarks, all_benchmarks):
    res = []
    slices = []
    for b in selected_benchmarks:
        idx = considered_benchmarks.index(b)
        slices.append(slice(idx * 8, idx*8 + 8))
    for b in all_benchmarks:
        res.append(b[np.r_[tuple(slices)]])
    return np.array(res)
    
def printConfiguration(pattern, weighting_scheme, num_channels, variant_creation_method, benchmark="limited"):
    config_data = [
        ["Pattern", pattern],
        ["Channel weighting", weighting_scheme],
        ["Number of Channels", num_channels],
        ["Variant Creation Method", variant_creation_method],
        ["Benchmark", benchmark]
    ]
    print("\nRun evalution for the following configuration:")
    print(tabulate(config_data, tablefmt="pretty"))


if __name__ == "__main__":
    with open('simulations/optimal.json') as f:
        truth = json.load(f)
    printConfiguration("Combiner Pattern", "uniform", 3, "backends", "full")
    results = get_all_results('simulations/backends')
    eval(evalDistComparator, results, truth)

    printConfiguration("Combiner Pattern", "uniform", 3, "backends", "limited")
    results = select_benchmarks(selected_benchmarks, results)
    eval(evalDistComparator, results, truth)

    printConfiguration("Combiner Pattern", "uniform", 3, "seeds", "full")
    results = get_all_results('simulations/seeds')
    eval(evalDistComparator, results, truth)

    printConfiguration("Combiner Pattern", "uniform", 3, "seeds", "limited")
    results = select_benchmarks(selected_benchmarks, results)
    eval(evalDistComparator, results, truth)

    printConfiguration("Combiner Pattern", "uniform", 3, "optimization levels", "full")
    results = get_all_results('simulations/optimizations')
    eval(evalDistComparator, results, truth)

    printConfiguration("Combiner Pattern", "uniform", 3, "optimization levels", "limited")
    results = select_benchmarks(selected_benchmarks, results)
    eval(evalDistComparator, results, truth)

