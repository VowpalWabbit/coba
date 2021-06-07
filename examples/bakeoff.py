"""
This is an example script that creates a Benchmark.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

from pathlib import Path

from coba.learners import RandomLearner, EpsilonBanditLearner, VowpalLearner
from coba.simulations import ValidationSimulation
from coba.benchmarks import Benchmark

#this line is required by Python in order to use multi-processing
if __name__ == '__main__':

    #This existence check is only needed to provide a fail safe against different execution environments
    result_file = "bakeoff.log" if Path("bakeoff.py").exists() else "./examples/bakeoff.log"

    #First, we define the learners that we want to test
    learners = [
        RandomLearner(),
        EpsilonBanditLearner(epsilon=0.025),
        VowpalLearner(epsilon=.1, seed=10), #This learner requires that VowpalWabbit be installed
    ]

    #Then we define the simulations that we want to test our learners on
    simulations = [ ValidationSimulation(5000, context_features=True, action_features=True) ]

    #And also define a collection of seeds used to shuffle our simulations
    seeds = [0,1,2,3,4,5]

    #We then create our benchmark using our simulations and seeds
    benchmark = Benchmark(simulations, shuffle=seeds)

    #Finally we evaluate our learners on our benchmark (the results will be saved in `result_file`).
    result = benchmark.evaluate(learners, result_file)

    #After evaluating can create a quick summary plot to get a sense of how the learners performed
    result.plot_learners()

    #We can also create a plot examining how one specific learner did across each shuffle of a simulation
    result.plot_shuffles("*Validation*True*True*", "*vw*epsilon*")