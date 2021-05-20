"""
This is an example script that creates a Benchmark.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

from pathlib import Path

from coba.learners import RandomLearner, EpsilonBanditLearner, VowpalLearner, UcbBanditLearner, CorralLearner
from coba.benchmarks import Benchmark

#this line is required by Python in order to use multi-processing
if __name__ == '__main__':

    #The existence check is only needed to provide a failsafe against different execution environments
    benchmark_file   = "bakeoff_short.json" if Path("bakeoff_short.json").exists() else "./examples/bakeoff_short.json"
    transaction_file = "bakeoff_short.log"  if Path("bakeoff_short.json").exists() else "./examples/bakeoff_short.log"

    #First, we define the learners that we want to test
    learners = [
        RandomLearner(),
        UcbBanditLearner(),
        EpsilonBanditLearner(epsilon=0.025),
        VowpalLearner(bag=5, seed=10),      #This learner requires that VowpalWabbit be installed
        VowpalLearner(epsilon=.1, seed=10), #This learner requires that VowpalWabbit be installed
        CorralLearner([VowpalLearner(bag=5, seed=10), VowpalLearner(epsilon=.1, seed=10)], eta=.075, T=300, seed=10),
    ]

    #Then we create our benchmark from the benchmark configuration file
    benchmark = Benchmark.from_file(benchmark_file)

    #Next we evaluate our learners given our benchmark. 
    #The provided log file is where results will be written and restored on evaluation.
    result = benchmark.evaluate(learners, transaction_file, seed=10)

    #We can create a quick summary plot to get a sense of how the results looked
    #For more in-depth analysis it is useful to load the result into a Jupyter Notebook
    result.standard_plot() #This line requires that Matplotlib be installed