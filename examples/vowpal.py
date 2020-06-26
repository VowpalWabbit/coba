"""
This is an example script that benchmarks a vowpal wabbit bandit learner.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

import itertools
import random

from bbench.simulations import LambdaSimulation
from bbench.learners import RandomLearner, EpsilonLookupLearner, VowpalLearner, UcbTunedLearner
from bbench.benchmarks import UniversalBenchmark

import matplotlib.pyplot as plt

#define a simulation
simulation = LambdaSimulation(900, lambda i: None, lambda s: [0,1,2,3,4], lambda s,a: random.uniform(a-2, a+2))

#create three different learner factories
random_factory = lambda: RandomLearner()
lookup_factory = lambda: EpsilonLookupLearner(1/10)
vowpal_factory = lambda: VowpalLearner()
ucb_factory    = lambda: UcbTunedLearner()

#define a benchmark
#  the benchmark replays the simulation 15 times in order to average
#  out when a learner randomly guesses the right answer early
benchmark = UniversalBenchmark([simulation]*15, None, 1)

#benchmark all three learners
random_result = benchmark.evaluate(random_factory)
lookup_result = benchmark.evaluate(lookup_factory)
ucb_result    = benchmark.evaluate(ucb_factory)
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

ax2.set_title("Progressive Validation Loss")
ax2.set_xlabel("Batch Index")

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