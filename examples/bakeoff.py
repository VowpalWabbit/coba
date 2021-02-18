"""
This is an example script that creates a Benchmark that matches the bandit bakeoff paper.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

from coba.learners import RandomLearner, EpsilonLearner, VowpalLearner, UcbTunedLearner, CorralLearner
from coba.benchmarks import Benchmark

if __name__ == '__main__':
    learners = [
        RandomLearner(),
        EpsilonLearner(epsilon=0.025),
        UcbTunedLearner(),
        VowpalLearner(bag=5, seed=10),
        CorralLearner([VowpalLearner(bag=5, seed=10), UcbTunedLearner()], eta=.075, T=40000, seed=10),
    ]

    Benchmark.from_file("./examples/benchmark_long.json").evaluate(learners, seed=10).standard_plot()