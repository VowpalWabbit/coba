"""
This is an example script that creates a ClassificationSimulation using the covertype dataset.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

from coba.simulations import ClassificationSimulation, ShuffleSimulation
from coba.learners import RandomLearner, EpsilonLearner, VowpalLearner, UcbTunedLearner
from coba.benchmarks import LambdaBatcher, UniversalBenchmark
from coba.analysis import Plots

simulation = ShuffleSimulation(ClassificationSimulation.from_openml(150))
benchmark  = UniversalBenchmark([simulation], LambdaBatcher(lambda i: 100 + i*100))

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