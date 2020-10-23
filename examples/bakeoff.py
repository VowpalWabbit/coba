"""
This is an example script that creates a Benchmark that matches the bandit bakeoff paper.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

from coba.learners import RandomLearner, EpsilonLearner, VowpalLearner, UcbTunedLearner
from coba.benchmarks import UniversalBenchmark, Factory
from coba.analysis import Plots
from coba.execution import ExecutionContext

if __name__ == '__main__':
    benchmark = UniversalBenchmark.from_file("./examples/benchmark_short.json")

    learner_factories = [
        Factory(RandomLearner,seed=10),
        Factory(EpsilonLearner,0.025, seed=10),
        Factory(UcbTunedLearner,seed=10),
        Factory(VowpalLearner,epsilon=0.025, seed=10),
        Factory(VowpalLearner,bag=5, seed=10),
        Factory(VowpalLearner,softmax=1, seed=10)
    ]

    with ExecutionContext.Logger.log("evaluating learners..."):
        results = benchmark.evaluate(learner_factories)

    Plots.standard_plot(results, show_err=False)