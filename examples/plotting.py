"""
This is an example script that creates a simple game and plots benchmark results.
This script requires that the matplotlib package be installed.
"""
import random
import matplotlib.pyplot as plt

from coba.simulations import LambdaSimulation
from coba.learners import RandomLearner, EpsilonLearner, UcbTunedLearner
from coba.benchmarks import UniversalBenchmark

#define a simulation
simulation = LambdaSimulation(50, lambda i: None, lambda s: [0,1,2,3,4], lambda s,a: random.uniform(a-2, a+2))

#define a benchmark: this benchmark replays the simulation 30 times
benchmark = UniversalBenchmark([simulation]*30, batch_size=1)

#create three learner factories
learner_factories = [lambda: RandomLearner(), lambda: EpsilonLearner(1/10), lambda: UcbTunedLearner()]

#benchmark all three learner factories
(random_result, epsilon_result, ucb_result) = benchmark.evaluate(learner_factories)

#plot the results
fig = plt.figure()

ax1 = fig.add_subplot(1,2,1) #type: ignore
ax2 = fig.add_subplot(1,2,2) #type: ignore

ax1.plot([ i.mean for i in random_result .batch_stats], label="random")
ax1.plot([ i.mean for i in epsilon_result.batch_stats], label="epsilon-greedy")
ax1.plot([ i.mean for i in ucb_result    .batch_stats], label="UCB")

ax1.set_title("Reward by Batch Index")
ax1.set_ylabel("Mean Reward")
ax1.set_xlabel("Batch Index")

ax2.plot([ i.mean for i in random_result .cumulative_batch_stats], label="Random")
ax2.plot([ i.mean for i in epsilon_result.cumulative_batch_stats], label="Epsilon-greedy")
ax2.plot([ i.mean for i in ucb_result    .cumulative_batch_stats], label="UCB")

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
fig.legend(*ax1.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .175), fancybox=True, ncol=3) #type: ignore

plt.show()