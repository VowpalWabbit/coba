"""
This is an example script that benchmarks a vowpal wabbit bandit learner.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

import coba.random
from coba.simulations import LambdaSimulation
from coba.learners import RandomLearner, EpsilonLearner, VowpalLearner, UcbTunedLearner
from coba.benchmarks import UniversalBenchmark, Factory
from coba.preprocessing import SizeBatcher
from coba.analysis import Plots
from coba.execution import ExecutionContext

if __name__ == '__main__':
    #make sure the simulation is repeatable
    random = coba.random.Random(10)

    #define a simulation
    simulations = [
        LambdaSimulation(300, lambda i: None, lambda s: [0,1,2,3,4], lambda s,a: random.randint(a-2,a+2)),
        LambdaSimulation(300, lambda i: None, lambda s: [0,1,2,3,4], lambda s,a: random.randint(a-2,a+2)),
        LambdaSimulation(300, lambda i: None, lambda s: [0,1,2,3,4], lambda s,a: random.randint(a-2,a+2)),
    ]

    #define a benchmark: this benchmark replays the simulation 15 times
    benchmark = UniversalBenchmark(simulations, SizeBatcher(1), shuffle_seeds=list(range(10)))

    #create the learner factories
    learner_factories = [
        Factory(RandomLearner, seed=10),
        Factory(EpsilonLearner, .025, seed=10),
        Factory(UcbTunedLearner, seed=10),
        Factory(VowpalLearner, epsilon=0.025, seed=10),
        Factory(VowpalLearner, bag=3, seed=10),
        Factory(VowpalLearner, softmax=3.5, seed=10)
    ]

    with ExecutionContext.Logger.log("RUNNING"):
        results = benchmark.ignore_raise(False).evaluate(learner_factories) #type: ignore

    Plots.standard_plot(results, show_sd=True)