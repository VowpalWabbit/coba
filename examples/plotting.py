"""
This is an example script that creates a simple game and plots benchmark results.
This script requires that the matplotlib package be installed.
"""

import random
import matplotlib.pyplot as plt

from bbench.games import LambdaGame
from bbench.solvers import RandomSolver, EpsilonLookupSolver
from bbench.benchmarks import UniversalBenchmark

#define a game
game = LambdaGame(50, lambda i: None, lambda s: [0,1,2,3,4], lambda s,a: random.uniform(a-2, a+2))

#create three different solver factories
randomsolver_factory   = lambda: RandomSolver()
lookupsolver_factory1 = lambda: EpsilonLookupSolver(1/10, 0)
lookupsolver_factory2 = lambda: EpsilonLookupSolver(1/10, 10)

#define a benchmark
#  the benchmark replays the game 30 times to average 
#  out a solver randomly guessing the right answer
benchmark = UniversalBenchmark([game]*30, 50, 1)

#benchmark all three solvers
random_result   = benchmark.evaluate(randomsolver_factory)
average_result1 = benchmark.evaluate(lookupsolver_factory1)
average_result2 = benchmark.evaluate(lookupsolver_factory2)

#plot the results

fig = plt.figure()

ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.plot([ i.mean for i in random_result  .batch_stats], label="random")
ax1.plot([ i.mean for i in average_result1.batch_stats], label="pessimistic epsilon-greedy")
ax1.plot([ i.mean for i in average_result2.batch_stats], label="optimistic epsilon-greedy")

ax1.set_title("Mean Reward by Batch Index")
ax1.set_ylabel("Mean Reward")
ax1.set_xlabel("Batch Index")

ax2.plot([ i.mean for i in random_result  .sweep_stats], label="random")
ax2.plot([ i.mean for i in average_result1.sweep_stats], label="pessimistic epsilon-greedy")
ax2.plot([ i.mean for i in average_result2.sweep_stats], label="optimistic epsilon-greedy")

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