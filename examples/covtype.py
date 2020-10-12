"""
This is an example script that creates a ClassificationSimulation using the covertype dataset.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

from build.lib.coba.benchmarks import SizeBatcher
from coba.simulations import ClassificationSimulation, LazySimulation
from coba.learners import RandomLearner, EpsilonLearner, VowpalLearner, UcbTunedLearner
from coba.benchmarks import UniversalBenchmark, SizeBatcher
from coba.analysis import Plots

simulation = LazySimulation(lambda:ClassificationSimulation.from_openml(150))
benchmark  = UniversalBenchmark([simulation], SizeBatcher(1, max_interactions=5000), shuffle_seeds=list(range(10)))

learner_factories = [
    lambda: RandomLearner(),
    lambda: EpsilonLearner(0.025),
    lambda: UcbTunedLearner(),
    lambda: VowpalLearner(epsilon=0.025),
    lambda: VowpalLearner(bag=5),
    lambda: VowpalLearner(softmax=3.5)
]

results = benchmark.evaluate(learner_factories)

Plots.standard_plot(results)