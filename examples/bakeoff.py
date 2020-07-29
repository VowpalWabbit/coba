"""
This is an example script that creates a Benchmark that matches the bandit bakeoff paper.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

from coba.learners import RandomLearner, EpsilonLearner, VowpalLearner, UcbTunedLearner
from coba.benchmarks import UniversalBenchmark
from coba.analysis  import Plots

with open("./examples/data/short_bakeoff.json") as fs:
    json = fs.read()

print(" - creating benchmark...")
benchmark = UniversalBenchmark.from_json(json)

print(" - creating learners...")
learner_factories = [ lambda: RandomLearner(), lambda: EpsilonLearner(1/10), lambda: UcbTunedLearner(), lambda: VowpalLearner(bag=5) ]

print(" - evaluating learners...")    
results = benchmark.evaluate(learner_factories)

Plots.standard_plot(results)