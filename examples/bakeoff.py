"""
This is an example script that creates a Benchmark that matches the bandit bakeoff paper.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

from coba.learners import RandomLearner, EpsilonLearner, VowpalLearner, UcbTunedLearner
from coba.benchmarks import Benchmark
from coba.analysis import Plots

if __name__ == '__main__':
    benchmark = Benchmark.from_file("./examples/benchmark_short.json")

    learners = [
        RandomLearner(seed=10),
        EpsilonLearner(epsilon=0.025, seed=10),
        UcbTunedLearner(seed=10),
        VowpalLearner(bag=5, seed=10),
    ]

    result = benchmark.evaluate(learners)

    Plots.standard_plot(result)