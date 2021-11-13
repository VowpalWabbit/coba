"""
This is an example script that creates a Benchmark.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

from coba.learners     import RandomLearner, EpsilonBanditLearner, VowpalLearner
from coba.experiments  import Experiment
from coba.environments import Environments

#this line is required by Python in order to use multi-processing
if __name__ == '__main__':

    #First, we define the learners that we want to test
    learners = [
        RandomLearner(),
        EpsilonBanditLearner(epsilon=.1),
        VowpalLearner(epsilon=.1), #This learner requires that VowpalWabbit be installed
        VowpalLearner(squarecb="all")
    ]

    environments = Environments.from_debug(5000,3,2,2).shuffle([0,1,2]).binary()

    #We then create our benchmark using our simulations and seeds
    result = Experiment(environments,learners).evaluate()

    #After evaluating can create a quick summary plot to get a sense of how the learners performed
    result.plot_learners(err='sd')

    #We can also create a plot examining how one specific learner did across each shuffle of our simulation
    result.filter_lrn(full_name="vw").plot_learners(err='sd',each=True)
