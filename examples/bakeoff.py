"""
This is an example script that creates a Benchmark.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

from coba.learners import RandomLearner, EpsilonBanditLearner, VowpalLearner, UcbBanditLearner, CorralLearner
from coba.benchmarks import Benchmark

#this line is necessary to use multi-processing
if __name__ == '__main__':

    #First, we define our learners that we wish to test
    learners = [
        RandomLearner(),
        UcbBanditLearner(),
        EpsilonBanditLearner(epsilon=0.025),
        VowpalLearner(bag=5, seed=10),      #This learner requires that VowpalWabbit be installed
        VowpalLearner(epsilon=.1, seed=10), #This learner requires that VowpalWabbit be installed
        CorralLearner([VowpalLearner(bag=5, seed=10), VowpalLearner(epsilon=.1, seed=10)], eta=.075, T=300, seed=10),
    ]

    #Then we create our benchmark from the benchmark configuration file
    benchmark = Benchmark.from_file("./examples/bakeoff_short.json")

    #Next we evaluate our learners given our benchmark. 
    #The provided log file is where results will be written and restored on evaluation.
    result = benchmark.evaluate(learners, './examples/bakeoff_short.log', seed=10)

    #We can create a quick sanity plot to get a sense of how the results looked
    #For more in-depth analysis it is useful to load the result into a Jupyter Notebook
    result.standard_plot() #This line requires that Matplotlib be installed