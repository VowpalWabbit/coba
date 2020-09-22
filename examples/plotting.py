"""
This is an example script that creates a simple game and plots benchmark results.
This script requires that the matplotlib package be installed.
"""
import random

from coba.simulations import LambdaSimulation
from coba.learners import RandomLearner, EpsilonLearner, UcbTunedLearner
from coba.benchmarks import UniversalBenchmark
from coba.analysis import Plots

#define a simulation
simulation = LambdaSimulation(50, lambda i: None, lambda s: [0,1,2,3,4], lambda s,a: random.uniform(a-2, a+2))

#define a benchmark: this benchmark replays the simulation 30 times
benchmark = UniversalBenchmark([simulation]*30, batch_size=1)

#create three learner factories
learner_factories = [lambda: RandomLearner(), lambda: EpsilonLearner(1/10), lambda: UcbTunedLearner()]

#benchmark all three learner factories
results = benchmark.evaluate(learner_factories)

Plots.standard_plot(results, show_err=False)