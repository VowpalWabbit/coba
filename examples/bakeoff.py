"""
This is an example script that creates a Benchmark that matches the bandit bakeoff paper.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

from coba.learners import RandomLearner, EpsilonLearner, VowpalLearner, UcbTunedLearner
from coba.benchmarks import UniversalBenchmark

import matplotlib.pyplot as plt

with open("./examples/data/bakeoff.json") as fs:
    json = fs.read()

#create the benchmark
print("creating benchmark...")
benchmark = UniversalBenchmark.from_json(json)

#create the learner to benchmark
print("creating learners...")
learner_factories = [ lambda: RandomLearner(), lambda: EpsilonLearner(1/10), lambda: UcbTunedLearner(), lambda: VowpalLearner(bag=5) ]

print("evaluating the learners...")
(random_result, epsilon_result, ucb_result, vowpal_result) = benchmark.evaluate(learner_factories)

#plot the benchmark results
fig = plt.figure()

ax1 = fig.add_subplot(1,1,1) #type: ignore

ax1.plot([ i.mean for i in random_result .cumulative_batch_stats][50::], label="Random")
ax1.plot([ i.mean for i in epsilon_result.cumulative_batch_stats][50::], label="Epsilon-greedy")
ax1.plot([ i.mean for i in ucb_result    .cumulative_batch_stats][50::], label="UCB")
ax1.plot([ i.mean for i in vowpal_result .cumulative_batch_stats][50::], label="Vowpal")

ax1.set_title("Progressive Validation Loss")
ax1.set_xlabel("Batch Index")

scale = 0.25
box2 = ax1.get_position()
ax1.set_position([box2.x0, box2.y0 + box2.height * scale, box2.width, box2.height * (1-scale)])

# Put a legend below current axis
fig.legend(*ax1.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .175), fancybox=True, ncol=2) #type: ignore

plt.show()