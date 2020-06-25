"""
This is an example script that creates a ClassificationSimulation using the covertype dataset.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

import itertools
import random

from bbench.simulations import ClassificationSimulation, ShuffleSimulation
from bbench.learners import RandomLearner, EpsilonLookupLearner, VowpalLearner, UcbTunedLearner
from bbench.benchmarks import UniversalBenchmark

import matplotlib.pyplot as plt

csv_path   = "./examples/data/covtype.data"
label_col  = 54
csv_stater = lambda row: tuple([int(v) for v in row])

#define a simulation
print("loading covtype dataset")
simulation = ClassificationSimulation.from_csv_path(csv_path, label_col, csv_stater=csv_stater)
simulation = ShuffleSimulation(simulation)

#create three different learner factories
random_factory = lambda: RandomLearner()
lookup_factory = lambda: EpsilonLookupLearner(1/10)
ucb_factory    = lambda: UcbTunedLearner()
vowpal_factory = lambda: VowpalLearner()

#define a benchmark
benchmark = UniversalBenchmark([simulation], None, lambda i: 500 + i*1000)

#benchmark all three learners
print("random started...")
random_result = benchmark.evaluate(random_factory)

print("epsilon started...")
lookup_result = benchmark.evaluate(lookup_factory)

print("ucb started...")
ucb_result = benchmark.evaluate(ucb_factory)

print("vowpal started...")
vowpal_result = benchmark.evaluate(vowpal_factory)

#plot the benchmark results
fig = plt.figure()

ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.plot([ i.mean for i in random_result.batch_stats], label="random")
ax1.plot([ i.mean for i in lookup_result.batch_stats], label="epsilon-greedy")
ax1.plot([ i.mean for i in ucb_result   .batch_stats], label="ucb")
ax1.plot([ i.mean for i in vowpal_result.batch_stats], label="vowpal")

ax1.set_title("Mean Reward by Batch Index")
ax1.set_ylabel("Mean Reward")
ax1.set_xlabel("Batch Index")

ax2.plot([ i.mean for i in random_result.sweep_stats], label="random")
ax2.plot([ i.mean for i in lookup_result.sweep_stats], label="epsilon-greedy")
ax2.plot([ i.mean for i in ucb_result   .sweep_stats], label="ucb")
ax2.plot([ i.mean for i in vowpal_result.sweep_stats], label="vowpal")

ax2.set_title("Mean Reward by Sweep Index")
ax2.set_xlabel("Sweep Index")

box1 = ax1.get_position()
box2 = ax2.get_position()

(bot1, top1) = ax1.get_ylim()
(bot2, top2) = ax2.get_ylim()

ax1.set_ylim(min(bot1,bot2), max(top1,top2))
ax2.set_ylim(min(bot1,bot2), max(top1,top2))

scale = 0.25

ax1.set_position([box1.x0, box1.y0 + box1.height * scale, box1.width, box1.height * (1-scale)])
ax2.set_position([box2.x0, box2.y0 + box2.height * scale, box2.width, box2.height * (1-scale)])

# Put a legend below current axis
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .175), fancybox=True, ncol=2)

plt.show()