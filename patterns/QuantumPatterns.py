from itertools import combinations
from qiskit.providers.basic_provider import BasicSimulator
from qiskit import transpile
from ErrorDetection import QuantumComparatorMetric, QuantumRater
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeBoeblingenV2, FakeAlmadenV2

class RunConfiguration:
    def __init__(self, backend) -> None:
        self.backend = backend

    def execute(self, circuit, shots=1024, seed=None, optimization_level=None):
        circuit = transpile(circuit, self.backend, seed_transpiler=seed, optimization_level=optimization_level)
        job = self.backend.run(circuit, shots=shots)
        return job.result().get_counts()

class DefaultRunConfiguration(RunConfiguration):
    def __init__(self) -> None:
        self.backend = BasicSimulator()


class QuantumSwitch(RunConfiguration):
    def __init__(self, variants, fault_detector) -> None:
        self.fault_detector = fault_detector
        self.variants = variants

    def execute(self, circuit):
        results = list(map(lambda x: x.execute(circuit), self.channels))
        return self.execute_on_results(results)


    def execute_on_results(self, results):
        counts = results[0]
        i = 1
        while (i < len(results) and self.fault_detector.reject(counts)):
            counts = results[i]
            i += 1
        if self.fault_detector.reject(counts):
            return None
        return counts

    def get_probabilities(self, counts):
        total = sum(counts.values())
        return np.fromiter(counts.values(), dtype=float)/total

class QuantumSwitch(RunConfiguration):
    def __init__(self, variants, fault_detector) -> None:
        self.fault_detector = fault_detector
        self.variants = variants

    def execute(self, circuit):
        results = list(map(lambda x: x.execute(circuit), self.channels))
        return self.execute_on_results(results)


    def execute_on_results(self, results):
        counts = results[0]
        i = 1
        while (i < len(results) and self.fault_detector.reject(counts)):
            counts = results[i]
            i += 1
        if self.fault_detector.reject(counts):
            return None
        return counts

    def get_probabilities(self, counts):
        total = sum(counts.values())
        return np.fromiter(counts.values(), dtype=float)/total


class QuantumCombiner:
    def combine(self, results):
        raise NotImplementedError("This should be implemented by a subclass")

class UniformQuantumCombiner(QuantumCombiner):
    def combine(self, measurements):
        combined_measurements = {}

        for m in measurements:
            for measurement in m:
                combined_measurements[measurement] = combined_measurements.get(measurement, 0) + m.get(measurement)                
        return combined_measurements

class QuantumVoter(RunConfiguration):
    def __init__(self, combiner : QuantumCombiner, channels) -> None:
        self.combiner = combiner
        self.channels = channels

    def execute(self, circuit):
        results = list(map(lambda x: x.execute(circuit), self.channels))
        return self.execute_on_results(results)

    def execute_on_results(self, results):
        result = self.combiner.combine(results)
        return result 

class QuantumComparator(RunConfiguration):
    def __init__(self, comparator: QuantumComparatorMetric, channels) -> None:
        self.comparator = comparator
        self.channels = channels

    def execute(self, circuit):
        results = list(map(lambda x: x.execute(circuit), self.channels))
        return self.execute_on_results(results)

    def execute_on_results(self, results):
        pairs = combinations(results, 2)
    
        for a, b in pairs:
            if not self.comparator.compare(a, b):
                return None
        
        return results[0]

