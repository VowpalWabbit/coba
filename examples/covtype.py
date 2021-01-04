"""
This is an example script that creates a ClassificationSimulation using the covertype dataset.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

from coba.simulations import JsonSimulation
from coba.learners import RandomLearner, EpsilonLearner, VowpalLearner, UcbTunedLearner
from coba.benchmarks import Benchmark

if __name__ == '__main__':
    simulation = JsonSimulation('{ "type":"classification", "from": { "format":"openml", "id":150 } }')
    benchmark  = Benchmark([simulation], batch_size=2, take=5000, seeds=list(range(3)))

    learners = [
        RandomLearner(seed=10),
        EpsilonLearner(0.025,seed=10),
        UcbTunedLearner(seed=10),
        VowpalLearner(epsilon=0.025,seed=10),
        VowpalLearner(bag=5,seed=10),
        VowpalLearner(softmax=3.5,seed=10)
    ]

    benchmark.evaluate(learners).standard_plot()