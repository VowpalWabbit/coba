"""
This is an example script that creates a Benchmark that matches the bandit bakeoff paper.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

from coba.learners import RandomLearner, EpsilonLearner, VowpalLearner, UcbTunedLearner
from coba.benchmarks import UniversalBenchmark

import matplotlib.pyplot as plt

with open("./examples/bakeoff.json") as fs:
    json = fs.read()

print("creating benchmark...")
benchmark = UniversalBenchmark.from_json(json)

print("creating learners...")
learner_factories = [ lambda: RandomLearner(), lambda: EpsilonLearner(1/10), lambda: UcbTunedLearner(), lambda: VowpalLearner(bag=5) ]

print("evaluating the learners...")
(random_result, epsilon_result, ucb_result, vowpal_result) = benchmark.evaluate(learner_factories)

#plot the benchmark results
fig = plt.figure()

ax1 = fig.add_subplot(1,2,1) #type: ignore
ax2 = fig.add_subplot(1,2,2) #type: ignore

ax1.plot([ i.mean for i in random_result .batch_stats], label="Random")
ax1.plot([ i.mean for i in epsilon_result.batch_stats], label="Epsilon-greedy")
ax1.plot([ i.mean for i in ucb_result    .batch_stats], label="UCB")
ax1.plot([ i.mean for i in vowpal_result .batch_stats], label="Vowpal")

ax1.set_title("Mean Reward by Batch Index")
ax1.set_ylabel("Mean Reward")
ax1.set_xlabel("Batch Index")

ax2.plot([ i.mean for i in random_result .cumulative_batch_stats], label="Random")
ax2.plot([ i.mean for i in epsilon_result.cumulative_batch_stats], label="Epsilon-greedy")
ax2.plot([ i.mean for i in ucb_result    .cumulative_batch_stats], label="UCB")
ax2.plot([ i.mean for i in vowpal_result .cumulative_batch_stats], label="Vowpal")

ax2.set_title("Progressive Validation Reward")
ax2.set_xlabel("Batch Index")

(bot1, top1) = ax1.get_ylim()
(bot2, top2) = ax2.get_ylim()

ax1.set_ylim(min(bot1,bot2), max(top1,top2))
ax2.set_ylim(min(bot1,bot2), max(top1,top2))

scale = 0.25
box1 = ax1.get_position()
box2 = ax2.get_position()
ax1.set_position([box1.x0, box1.y0 + box1.height * scale, box1.width, box1.height * (1-scale)])
ax2.set_position([box2.x0, box2.y0 + box2.height * scale, box2.width, box2.height * (1-scale)])

# Put a legend below current axis
fig.legend(*ax1.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .175), fancybox=True, ncol=2) #type: ignore

plt.show()