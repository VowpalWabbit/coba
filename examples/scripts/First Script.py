"""
This is an example script that creates and execuates an Experiment.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

from coba.learners     import RandomLearner, VowpalEpsilonLearner
from coba.experiments  import Experiment
from coba.environments import Environments

#this line is required by Python in order to use multi-processing
if __name__ == '__main__':

    #This isn't required but it is recommended with environments
    #that are created from online data repositories like openml.
    #The two synthetic environments below do not use openml.
    Environments.set_caching_directory('./.coba_cache')

    #First, we define the learners that we want to test
    learners = [ VowpalEpsilonLearner(), RandomLearner() ]

    #Next we create the environments we'd like to evaluate against
    environments  = Environments.from_linear_synthetic(1000, n_action_features=0).shuffle([1,2,3,4])
    environments += Environments.from_kernel_synthetic(1000, n_action_features=0).shuffle([1,2,3,4])

    #We then create and run our experiment from our environments and learners
    result = Experiment(environments,learners).run()

    #After evaluating can create a quick summary plot to get a sense of how the learners performed
    result.plot_learners(y='reward',err='se')

    #We can then filter down the plot down to get a closer look at a region of interest
    result.filter_lrn(family="vw").plot_learners(y='reward', err='se', xlim=(700,1000))

    #We can also directly contrast two learners to see exactly when one learner out-performs 
    #another with 95% confidence (given that the 95% CI assumptions are appropriate for the data)
    result.plot_contrast(0, 1, "index", "reward", mode="diff", err='se', labels=["VWEpsilon","VWBag"])

    #Finally, we can group our environments to see which environments our two learners perform better on    
    result.plot_contrast(0, 1, "type", "reward" , mode="scat", err='se', labels=["VWEpsilon","VWBag"])
