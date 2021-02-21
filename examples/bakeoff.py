"""
This is an example script that creates a Benchmark that matches the bandit bakeoff paper.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

from coba.learners import RandomLearner, EpsilonBanditLearner, VowpalLearner, UcbBanditLearner, CorralLearner
from coba.benchmarks import Benchmark

if __name__ == '__main__':

    learners = [
        RandomLearner(),
        UcbBanditLearner(),
        EpsilonBanditLearner(epsilon=0.025),
        VowpalLearner(bag=5, seed=10),
        VowpalLearner(epsilon=.1, seed=10),
        CorralLearner([VowpalLearner(bag=5, seed=10), VowpalLearner(epsilon=.1, seed=10)], eta=.075, T=40000, seed=10),
    ]

    Benchmark.from_file("./examples/benchmark_short.json").evaluate(learners, './examples/bakeoff.log', seed=10).standard_plot()