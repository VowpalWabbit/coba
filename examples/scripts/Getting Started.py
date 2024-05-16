"""
This is an example script that creates and executes an Experiment.
This script depends on the matplotlib and vowpalwabbit packages.
"""

import coba as cb

#First, we define the learners that we wish to evaluate
learners = [ cb.VowpalEpsilonLearner(), cb.RandomLearner() ]

#Next, we create an environment we'd like to evaluate against
environments = cb.Environments.from_linear_synthetic(1000, n_action_features=0).shuffle([1,2,3])

#We then create and run an experiment using our environments and learners
result = cb.Experiment(environments,learners).run()

#Finally, we can plot the results of our experiment
result.plot_learners(y='reward',err='se',xlim=(10,None))
