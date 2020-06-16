"""
This is an example script that creates a simple game and plots benchmark results.
This script requires that the matplotlib package be installed.
"""

import random
import matplotlib.pyplot as plt

from bbench.games import LambdaGame
from bbench.solvers import RandomSolver, EpsilonAverageSolver
from bbench.benchmarks import UniversalBenchmark

#define a game
game = LambdaGame(lambda i: None, lambda s: [0,1,2,3,4], lambda s,a: random.uniform(a-2, a+2))

#create three different solver factories
randomsolver_factory   = lambda: RandomSolver()
averagesolver_factory1 = lambda: EpsilonAverageSolver(1/10, lambda a: 0)
averagesolver_factory2 = lambda: EpsilonAverageSolver(1/10, lambda a: 10)

#define a benchmark
#  the benchmark replays the game 30 times to average 
#  out a solver randomly guessing the right answer
benchmark = UniversalBenchmark([game]*30, lambda i: 1, 300)

#benchmark all three solvers
random_result   = benchmark.evaluate(randomsolver_factory)
average_result1 = benchmark.evaluate(averagesolver_factory1)
average_result2 = benchmark.evaluate(averagesolver_factory2)

fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.plot([ i.mean for i in random_result.progressive_stats]  , label="random")
ax.plot([ i.mean for i in average_result1.progressive_stats], label="pessimistic epsilon-greedy")
ax.plot([ i.mean for i in average_result2.progressive_stats], label="optimistic epsilon-greedy")

ax.set_title("Mean Observed Reward for Progressive Iterations")
ax.set_ylabel("Mean Reward")
ax.set_xlabel("Progressive Iteration")

ax.legend()
plt.show()