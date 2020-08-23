"""
This is an example script that creates a Benchmark that matches the bandit bakeoff paper.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

from coba.learners import RandomLearner, EpsilonLearner, VowpalLearner, UcbTunedLearner
from coba.benchmarks import UniversalBenchmark
from coba.analysis import Plots
from coba.execution import ExecutionContext


benchmark = UniversalBenchmark.from_file("./examples/benchmark_short.json")

learner_factories = [
    lambda: RandomLearner(),
    lambda: EpsilonLearner(0.025),
    lambda: UcbTunedLearner(),
    lambda: VowpalLearner(epsilon=0.025),
    lambda: VowpalLearner(bag=5),
    lambda: VowpalLearner(softmax=1)
]

with ExecutionContext.Logger.log("evaluating learners..."):
    result = benchmark.evaluate(learner_factories)

print(result.to_pandas()[2].groupby('simulation_id').aggregate(len))
print(result.to_pandas()[2].groupby('learner_id').aggregate(len))

#Plots.standard_plot(result)