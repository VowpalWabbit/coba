"""
This is an example script that benchmarks a vowpal wabbit bandit learner.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

import coba.random
from coba.simulations import LambdaSimulation
from coba.learners import RandomLearner, EpsilonLearner, VowpalLearner, UcbTunedLearner
from coba.benchmarks import SizeBatcher, UniversalBenchmark
from coba.analysis import Plots

#make sure the simulation is repeatable
random = coba.random.Random(10)

#define a simulation
simulation = LambdaSimulation(300, lambda i: None, lambda s: [0,1,2,3,4], lambda s,a: random.randint(a-2,a+2))

#define a benchmark: this benchmark replays the simulation 15 times
benchmark = UniversalBenchmark([simulation]*10, SizeBatcher(1))

#create the learner factories
learner_factories = [
    lambda: RandomLearner(seed=10),
    lambda: EpsilonLearner(.025, seed=10),
    lambda: UcbTunedLearner(seed=10),
    lambda: VowpalLearner(epsilon=0.025, seed=10),
    lambda: VowpalLearner(bag=3, seed=10),
    lambda: VowpalLearner(softmax=3.5, seed=10)
]

#benchmark all learners
results = benchmark.ignore_raise(False).evaluate(learner_factories)

#plot the learners
Plots.standard_plot(results, show_err=False)