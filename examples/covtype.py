"""
This is an example script that creates a ClassificationSimulation using the covertype dataset.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

from coba.simulations import JsonSimulation
from coba.learners import RandomLearner, EpsilonLearner, VowpalLearner, UcbTunedLearner
from coba.benchmarks import UniversalBenchmark, LearnerFactory
from coba.analysis import Plots
from coba.execution import ExecutionContext

if __name__ == '__main__':
    simulation = JsonSimulation('{ "type":"classification", "from": { "format":"openml", "id":150 } }')
    benchmark  = UniversalBenchmark([simulation], batch_size=2, max_interactions=5000, shuffle_seeds=list(range(10)))

    learner_factories = [
        LearnerFactory(RandomLearner,seed=10),
        LearnerFactory(EpsilonLearner,0.025,seed=10),
        LearnerFactory(UcbTunedLearner,seed=10),
        LearnerFactory(VowpalLearner,epsilon=0.025,seed=10),
        LearnerFactory(VowpalLearner,bag=5,seed=10),
        LearnerFactory(VowpalLearner,softmax=3.5,seed=10)
    ]

    with ExecutionContext.Logger.log("evaluating learners..."):
        results = benchmark.evaluate(learner_factories)

    Plots.standard_plot(results)