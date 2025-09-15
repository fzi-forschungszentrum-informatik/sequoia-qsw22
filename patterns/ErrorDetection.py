import numpy as np
from math import log, sqrt

def get_probabilities(counts):
    if isinstance(counts, list):
        res = np.array(counts)
        return res/np.sum(counts)
    total = sum(counts.values())
    size = 2 ** len(next(iter(counts)))
    res = np.zeros(size)
    for l in counts:
        res[int(l, 2)] = counts[l]/total
    return res

class Distribution():
    def __init__(self) -> None:
        pass

class EquiDist(Distribution):
    def __init__(self, m) -> None:
        super().__init__()
        self.probs = np.full((m, ), 1/m)

class DiracDist(Distribution):
    def __init__(self):
        super().__init__()
        self.probs = np.array([1.0])
    
    def __init__(self, n):
        self.probs = np.zeros(n)
        self.probs[0] = 1.0

class QuantumComparatorMetric:
    def compare(self, measurementsA, measurementsB):
        raise NotImplemented("This is an abstract comparator and has to be implemented")


class MaxComparator(QuantumComparatorMetric):
    def compare(self, measurementsA, measurementsB):
        max_keyA = max(measurementsA, key=measurementsA.get)
        max_keyB = max(measurementsB, key=measurementsB.get)
        return max_keyA == max_keyB

class DistComparator(QuantumComparatorMetric):
    def __init__(self, metric, threshold) -> None:
        super().__init__()
        self.metric = metric
        self.threshold = threshold

    def compare(self, measurementsA, measurementsB):
        measurementsA = get_probabilities(measurementsA)
        measurementsB = get_probabilities(measurementsB)
        return self.metric(measurementsA, measurementsB) < self.threshold

class UniformComparator(QuantumComparatorMetric):
    def __init__(self, metric) -> None:
        super().__init__()
        self.metric = metric

    def compare(self, measurementsA, measurementsB):
        measurementsA = get_probabilities(measurementsA)
        measurementsB = get_probabilities(measurementsB)
        n = len(measurementsA)
        dist_A = self.metric(measurementsA, EquiDist(n).probs)
        dist_B = self.metric(measurementsB, EquiDist(n).probs)
        if dist_A > dist_B:
            return dist_B
        else: 
            return dist_A

class QuantumRater:
    def rate(self, measurementsA):
        raise NotImplemented("This is an abstract comparator and has to be implemented")

class DiracRater(QuantumRater):
    def __init__(self, metric) -> None:
        super().__init__()
        self.metric = metric

    def rate(self, measurements):
        measurements = get_probabilities(measurements)
        n = len(measurements)
        measurements = np.sort(measurements)
        return self.metric(measurements, DiracDist(n).probs)

class UniformRater(QuantumRater):
    def __init__(self, metric) -> None:
        super().__init__()
        self.metric = metric

    def rate(self, measurements):
        measurements = get_probabilities(measurements)
        n = len(measurements)
        return 1.0 - self.metric(measurements, EquiDist(n).probs)

class QuantumErrorDetector:
    def reject(counts):
        raise NotImplemented("This should be implemented by a subclass")

class DiracErrorDetector(QuantumErrorDetector):
    def __init__(self, metric, threshold):
        super().__init__()
        self.threshold = threshold
        self.metric = metric
    
    def reject(self, counts):
        counts = get_probabilities(counts)
        n = len(counts)
        return self.metric(counts, DiracDist(n).probs) < self.threshold

class UniformErrorDetector(QuantumErrorDetector):
    def __init__(self, metric, threshold):
        super().__init__()
        self.threshold = threshold
        self.metric = metric
    
    def reject(self, counts):
        counts = get_probabilities(counts)
        n = len(counts)
        return self.metric(counts, EquiDist(n).probs) > self.threshold


def shannon_entropy(p_dist):
    return sum(p * log(p) for p in p_dist) * (-1)

'''Assuming that the probabilities of p_dist and q_dist are ordered according to the values'''
def hellinger(p_dist, q_dist):
    assert len(p_dist) == len(q_dist)
    
    return sum([(sqrt(t[0])-sqrt(t[1]))*(sqrt(t[0])-sqrt(t[1]))\
                for t in zip(p_dist, q_dist)])/sqrt(2.)

'''Assuming that the probabilities of p_dist and q_dist are ordered according to the values'''
def kl_divergence(p_dist, q_dist):
    assert len(p_dist) == len(q_dist)

    return sum(p_dist[i] * log(p_dist[i] / q_dist[i]) for i in range(len(p_dist)))

'''Assuming that the probabilities of p_dist and q_dist are ordered according to the values'''
def cross_entropy(p_dist, q_dist):
    assert len(p_dist) == len(q_dist)

    return sum(p_dist[i] * log(q_dist[i]) for i in range(len(p_dist))) * (-1)

'''Assuming that the probabilities of p_dist and q_dist are ordered according to the values'''
def jensen_shannon_divergence(p_dist, q_dist):
    assert len(p_dist) == len(q_dist)

    m = [(p_dist[i] * q_dist[i]) / 2 for i in range(len(p_dist))]
    return 0.5 * kl_divergence(p_dist, m) + 0.5 * kl_divergence(q_dist, m)

'''Assuming that the probabilities of p_dist and q_dist are ordered according to the values'''
def bhattacharyya(p_dist, q_dist):
    assert len(p_dist) == len(q_dist)

    bc = sum(sqrt(p_dist[i] * q_dist[i]) for i in range(len(p_dist)))
    return log(bc) * (-1)