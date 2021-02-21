"""
This is an example script that benchmarks a vowpal wabbit bandit learner.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

import coba.random

from coba.simulations import LambdaSimulation
from coba.learners import RandomLearner, EpsilonBanditLearner, VowpalLearner, UcbBanditLearner
from coba.benchmarks import Benchmark

if __name__ == '__main__':
    
    def dot(v1,v2): return sum(mul(v1,v2))
    def mul(v1,v2): return [s1*s2 for s1,s2 in zip(v1,v2)]

    context    = lambda i  : coba.random.randoms(3)
    actions    = lambda i,c: [(1,0,0), (0,1,0), (0,0,1)]

    lin_reward = lambda i,c,a: dot([dot(c,[1,2,4]), dot(c,[4,1,2]), dot(c,[2,4,1])], a)
    pol_reward = lambda i,c,a: dot([dot(c+mul(c,c),[1,2,3,1,2,3]), dot(c+mul(c,c),[3,1,2,3,1,2]), dot(c+mul(c,c),[2,3,1,2,3,1])], a)
    rng_reward = lambda i,c,a: coba.random.randint(dot([0,1,2],a), dot([0,1,2],a)+2)

    simulations = [
        LambdaSimulation(2000, context, actions, lin_reward, seed=10),
        LambdaSimulation(2000, context, actions, rng_reward, seed=10),
        LambdaSimulation(2000, context, actions, pol_reward, seed=10),
    ]

    learners = [
        RandomLearner(),
        EpsilonBanditLearner(epsilon=0.1),
        VowpalLearner(bag=5, seed=10),
    ]

    Benchmark(simulations, shuffle=list(range(5))).evaluate(learners).standard_plot()