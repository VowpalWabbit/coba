"""
This is an example script that benchmarks a vowpal wabbit bandit solver.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

import itertools
import random

from bbench.games import LambdaGame
from bbench.solvers import RandomSolver, EpsilonAverageSolver, Solver
from bbench.benchmarks import UniversalBenchmark

import matplotlib.pyplot as plt
from vowpalwabbit import pyvw

class VowpalSolver(Solver):
    def __init__(self):
        self._vw = pyvw.vw("--cb_explore 5 --epsilon 0.1 --quiet")
        self._prob = {}

    def choose(self, state, actions):
        pmf = self._vw.predict("| " + self._vw_format(state))

        cdf   = list(itertools.accumulate(pmf))
        rng   = random.random()
        index = [ rng < c for c in cdf].index(True)

        self._prob[self._key(state, actions[index])] = pmf[index]

        return index

    def learn(self, state, action, reward):
        
        prob  = self._prob[self._key(state,action)]
        state = self._vw_format(state)
        cost  = -reward

        self._vw.learn(str(action+1) + ":" + str(cost) + ":" + str(prob) + " | " + state)

    def _vw_format(self, state):
        
        if state is None:  return ""

        try:
            iter(state)
        except:
            return str(state)
        else:
            return " ". join(str(feature) for feature in state)

    def _key(self, state, action):
        return self._tuple(state) + self._tuple(action)

    def _tuple(self, value):

        if value is None or isinstance(value, (int,str)):
            return tuple([value]) 

        return tuple(value)

#define a game
game = LambdaGame(lambda i: None, lambda s: [0,1,2,3,4], lambda s,a: random.uniform(a-2, a+2))

#create three different solver factories
randomsolver_factory   = lambda: RandomSolver()
averagesolver_factory1 = lambda: EpsilonAverageSolver(1/10, lambda a: 0)
averagesolver_factory2 = lambda: EpsilonAverageSolver(1/10, lambda a: 10)
vowpalsolver_factory   = lambda: VowpalSolver()

#define a benchmark
#  the benchmark replays the game 15 times in order to average
#  out when a solver randomly guesses the right answer early
benchmark = UniversalBenchmark([game]*15, 900, 1)

#benchmark all three solvers
random_result   = benchmark.evaluate(randomsolver_factory)
average_result1 = benchmark.evaluate(averagesolver_factory1)
average_result2 = benchmark.evaluate(averagesolver_factory2)
vowpal_result   = benchmark.evaluate(vowpalsolver_factory)

#plot the benchmark results
fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.plot([ i.mean for i in random_result  .sweep_stats], label="random")
ax.plot([ i.mean for i in average_result1.sweep_stats], label="pessimistic epsilon-greedy")
ax.plot([ i.mean for i in average_result2.sweep_stats], label="optimistic epsilon-greedy")
ax.plot([ i.mean for i in vowpal_result  .sweep_stats], label="vowpal")

ax.set_title("Mean Observed Reward for Progressive Iterations")
ax.set_ylabel("Mean Observed Reward")
ax.set_xlabel("Progressive Iteration")

ax.legend()
plt.show()