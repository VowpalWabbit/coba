"""
This is an example script that benchmarks a vowpal wabbit bandit learner.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

import random

from coba.simulations import LambdaSimulation
from coba.learners import RandomLearner, EpsilonLearner, VowpalLearner, UcbTunedLearner
from coba.benchmarks import UniversalBenchmark
from coba.analysis import Plots

#define a simulation
simulation = LambdaSimulation(900, lambda i: None, lambda s: [0,1,2,3,4], lambda s,a: random.uniform(a-2, a+2))

#define a benchmark: this benchmark replays the simulation 15 times
benchmark = UniversalBenchmark([simulation]*15, batch_size=1)

#create the learner factories
learner_factories = [
    lambda: RandomLearner(),
    lambda: EpsilonLearner(.025),
    lambda: UcbTunedLearner(),
    lambda: VowpalLearner(epsilon=0.025),
    lambda: VowpalLearner(bag=3),
    lambda: VowpalLearner(softmax=1)
]

#benchmark all learners
results = benchmark.evaluate(learner_factories)

#plot the learners
Plots.standard_plot(results, show_err=False)