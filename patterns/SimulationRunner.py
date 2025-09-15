from tqdm import tqdm
from mqt.bench import get_benchmark
from QuantumPatterns import RunConfiguration 
from qiskit_ibm_runtime.fake_provider import FakeBoeblingenV2, FakeProviderForBackendV2
from qiskit.quantum_info import Statevector
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def get_circuits():
    considered_benchmarks = ["dj", "grover-noancilla", "grover-v-chain", "ae", "ghz", "graphstate", "qaoa", "qft", "qftentangled", "twolocalrandom", "vqe"]
    #considered_benchmarks = ["dj", "grover-noancilla"]
    min_qubits = 3
    max_qubits = 10 
    circuits = []
    for b in considered_benchmarks:
        for size in range(min_qubits, max_qubits + 1):
            circuits.append(get_benchmark(b, "alg", size))
    return circuits

def run_single_experiment(configurations, circuits, seed=None, optimization_level=None):
    results = []
    for circuit in tqdm(circuits, position=1, desc="Circuits: "):
        r = []
        for configuration in configurations:
            r.append(configuration.execute(circuit, seed=seed, optimization_level=optimization_level))
        #circuit.remove_final_measurements(inplace=True)
        #probs = Statevector(circuit).probabilities()
        #r.append(probs)
        results.append(r)
    
    return results

def backendSimulations():
    #config_A = RunConfiguration(FakeSherbrooke())
    #config_B = RunConfiguration(FakeAlmadenV2())
    #config_A = DefaultRunConfiguration()
    #config_B = DefaultRunConfiguration()

    #config_comparator = QuantumComparator(UniformComparator(hellinger), [config_A, config_B])
    fake_provider = FakeProviderForBackendV2()
    backends = fake_provider.backends(min_num_qubits=10)

    for backend in tqdm(backends, position=0, desc="Backends: "):
        try:
            results = run_single_experiment([RunConfiguration(backend)], get_circuits())
            with open('simulations/backends/' + backend.backend_name + '.json', 'w') as f:
                f.write(json.dumps(results, cls=NumpyEncoder))
        except Exception as e:
            print("skipped backend " + backend.name + " because of error: " + str(e))

def optimizationSimulations():
    run_config  = RunConfiguration(FakeBoeblingenV2())
    
    for opt in tqdm(range(4)):
        try:
            results = run_single_experiment([run_config], get_circuits(), optimization_level=opt, seed=42)
            with open('simulations/optimizations/' + str(opt) + '.json', 'w') as f:
                f.write(json.dumps(results, cls=NumpyEncoder))
        except Exception as e:
            print("skipped backend " + str(opt) + " because of error: " + str(e))



def seedSimulations():
    run_config  = RunConfiguration(FakeBoeblingenV2())
    
    for seed in tqdm(range(1, 51)):
        try:
            results = run_single_experiment([run_config], get_circuits(), seed=seed)
            with open('simulations/seeds/' + str(seed) + '.json', 'w') as f:
                f.write(json.dumps(results, cls=NumpyEncoder))
        except Exception as e:
            print("skipped backend " + str(seed) + " because of error: " + str(e))



def get_measured_qbits(circuit):
    l = []
    for o in circuit.data:
        if o.name == 'measure':
            for qb in o.qubits:
                l.append(circuit.find_bit(qb).index)
    return l

def optimalSimulation():
    circuits = get_circuits()
    results = []
    for circuit in tqdm(circuits):
        measured_qbits = get_measured_qbits(circuit)
        circuit.remove_final_measurements(inplace=True)
        probs = Statevector(circuit).probabilities(measured_qbits)
        results.append(probs)
    with open('simulations/optimal.json', 'w') as f:
        f.write(json.dumps(results, cls=NumpyEncoder))
    return results

if __name__ == "__main__":
    print("Running simulations for patterns with different seeds")
    seedSimulations()
    print("Running simulations for patterns with different backends")
    backendSimulations()
    print("Running simulations for patterns with different optimization levels")
    optimizationSimulations()
    print("Running simulations for error free backend to obtain ground truth")
    optimalSimulation()

