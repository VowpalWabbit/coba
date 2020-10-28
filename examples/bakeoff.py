"""
This is an example script that creates a Benchmark that matches the bandit bakeoff paper.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

from coba.learners import RandomLearner, EpsilonLearner, VowpalLearner, UcbTunedLearner
from coba.benchmarks import UniversalBenchmark, LearnerFactory
from coba.analysis import Plots
from coba.execution import ExecutionContext

if __name__ == '__main__':
    benchmark = UniversalBenchmark.from_file("./examples/benchmark_long.json")

    learner_factories = [
        LearnerFactory(RandomLearner,seed=10),
        LearnerFactory(EpsilonLearner,0.025, seed=10),
        LearnerFactory(UcbTunedLearner,seed=10),
        LearnerFactory(VowpalLearner,epsilon=0.025, seed=10),
        LearnerFactory(VowpalLearner,bag=5, seed=10),
        LearnerFactory(VowpalLearner,softmax=1, seed=10)
    ]

    with ExecutionContext.Logger.log("evaluating learners..."):
        results = benchmark.evaluate(learner_factories, 'bakeoff.log')

    Plots.standard_plot(results, show_err=False)