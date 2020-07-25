"""
This is an example script that benchmarks a vowpal wabbit bandit learner.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

import random

from coba.simulations import LambdaSimulation
from coba.learners import RandomLearner, EpsilonLearner, VowpalLearner, UcbTunedLearner
from coba.benchmarks import UniversalBenchmark
from coba.analysis import Plots

import matplotlib.pyplot as plt

#define a simulation
simulation = LambdaSimulation(900, lambda i: None, lambda s: [0,1,2,3,4], lambda s,a: random.uniform(a-2, a+2))

#define a benchmark: this benchmark replays the simulation 15 times
benchmark = UniversalBenchmark([simulation]*15, batch_size=1)

#create the learner factories
learner_factories = [ lambda: RandomLearner(), lambda: EpsilonLearner(1/10), lambda: UcbTunedLearner(), lambda: VowpalLearner() ]

#benchmark all three learners
results = benchmark.evaluate(learner_factories)

Plots.standard_plot(results)