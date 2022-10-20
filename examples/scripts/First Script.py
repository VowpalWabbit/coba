"""
This is an example script that creates and execuates an Experiment.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

import coba as cb

#First, we define the learners that we want to test
learners = [ cb.VowpalEpsilonLearner(), cb.RandomLearner() ]

#Next we create an environment we'd like to evaluate against
environments = cb.Environments.from_linear_synthetic(1000, n_action_features=0).shuffle([1,2,3])

#We then create and run our experiment from our environments and learners
result = cb.Experiment(environments,learners).run()

#After evaluating can create a quick summary plot to get a sense of how the learners performed
result.plot_learners(y='reward',err='se',xlim=(10,1000))
