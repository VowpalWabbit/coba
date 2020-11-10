"""
This is an example script that benchmarks a vowpal wabbit bandit learner.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

from coba.random import CobaRandom
from coba.simulations import LambdaSimulation
from coba.learners import RandomLearner, EpsilonLearner, VowpalLearner, UcbTunedLearner
from coba.benchmarks import UniversalBenchmark, LearnerFactory
from coba.analysis import Plots
from coba.execution import ExecutionContext

if __name__ == '__main__':
    #make sure the simulation is repeatable
    random = CobaRandom(10)

    def dot(v1,v2): return sum(mul(v1,v2))
    def mul(v1,v2): return [s1*s2 for s1,s2 in zip(v1,v2)]

    no_contexts = lambda t: None
    contexts    = lambda t: random.randoms(3)
    actions     = lambda t: [(1,0,0), (0,1,0), (0,0,1)]

    linear_rewards_1 = lambda c,a: dot([dot(c,[1,2,4]), dot(c,[4,1,2]), dot(c,[2,4,1])], a)
    linear_rewards_2 = lambda c,a: dot([dot(c,[1,0,0]), dot(c,[0,1,0]), dot(c,[0,0,1])], a)

    polynomial_reward_1 = lambda c,a: dot([dot(c+mul(c,c),[1,2,3,1,2,3]), dot(c+mul(c,c),[3,1,2,3,1,2]), dot(c+mul(c,c),[2,3,1,2,3,1])], a)

    random_rewards_1 = lambda c,a: random.randint(dot([0,1,2],a), dot([0,1,2],a)+2)
    random_rewards_2 = lambda c,a: random.random()*dot([1,2,3],a)
    random_rewards_3 = lambda c,a: dot([0,1,2],a) + random.random()

    linear_plus_random_rewards_1 = lambda c,a: linear_rewards_1(c,a) + random_rewards_1(c,a)
    linear_plus_random_rewards_2 = lambda c,a: linear_rewards_2(c,a) + random_rewards_2(c,a)

    #define a simulation
    simulations = [
        LambdaSimulation(1000, contexts, actions, linear_rewards_1),
        LambdaSimulation(1000, contexts, actions, linear_rewards_2),
        LambdaSimulation(1000, contexts, actions, random_rewards_1),
        LambdaSimulation(1000, contexts, actions, random_rewards_2),
        LambdaSimulation(1000, contexts, actions, random_rewards_3),
        LambdaSimulation(1000, no_contexts, actions, random_rewards_1),
        LambdaSimulation(1000, no_contexts, actions, random_rewards_2),
        LambdaSimulation(1000, no_contexts, actions, random_rewards_3),
        LambdaSimulation(1000, contexts, actions, linear_plus_random_rewards_1),
        LambdaSimulation(1000, contexts, actions, linear_plus_random_rewards_2),
        LambdaSimulation(1000, contexts, actions, polynomial_reward_1),
    ]

    #define a benchmark: this benchmark replays the simulation 15 times
    benchmark = UniversalBenchmark(simulations, batch_size = 1, shuffle_seeds=list(range(5)))

    #create the learner factories
    learner_factories = [
        LearnerFactory(RandomLearner  , seed=10),
        LearnerFactory(EpsilonLearner , epsilon=0.025, seed=10),
        LearnerFactory(UcbTunedLearner, seed=10),
        LearnerFactory(VowpalLearner  , epsilon=0.025, seed=10),
        LearnerFactory(VowpalLearner  , epsilon=0.025, is_adf=False, seed=10),
        LearnerFactory(VowpalLearner  , bag=5, seed=10),
    ]

    with ExecutionContext.Logger.log("RUNNING"):
        results = benchmark.ignore_raise(False).evaluate(learner_factories) #type: ignore

    Plots.standard_plot(results)