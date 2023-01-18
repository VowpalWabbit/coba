"""
This is an example script that creates and execuates an Experiment.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

import coba as cb
from coba.learners import VowpalMediator
from typing import List

# this mediator splits the 'x' namespace into multiple namespaces
class CustomVWMediator(VowpalMediator):
    def __init__(self, namespaces: List):
        self.namespaces = namespaces
        super().__init__()

    def transform_example(self, vw_shared, vw_uniques, labels):
        DEFAULT_NS = 'x'
        num_groups = min(len(self.namespaces), len(vw_shared[DEFAULT_NS]))
        num_feat_each_group = len(vw_shared[DEFAULT_NS]) // num_groups

        for k, v in vw_shared[DEFAULT_NS].items():
            new_ns = min(int(k) // num_feat_each_group, num_groups-1)
            vw_shared[self.namespaces[new_ns]] = vw_shared.get(self.namespaces[new_ns], {})
            vw_shared[self.namespaces[new_ns]][k] = v
        
        del vw_shared[DEFAULT_NS]

#First, we define the learners that we want to test
learners = [ cb.VowpalLearner(args="-q :: --cb_explore_adf --epsilon 0.05", vw=CustomVWMediator(['b', 'c'])), cb.VowpalLearner(vw=CustomVWMediator(['b','c']))]

#Next we create an environment we'd like to evaluate against
environments = cb.Environments.from_linear_synthetic(1000, n_action_features=0).shuffle([1])

#We then create and run our experiment from our environments and learners
result = cb.Experiment(environments,learners).run()

# this is None if we use shuffle with multiple values
assert learners[0]._vw._vw != None

# get_weighted_examples returns both labeled and unlabeled examples
# we multiply by 2 since we divided by double the number of examples actually learned on
# note: only learnable examples have labels
loss_with_qcolcol = learners[0]._vw._vw.get_sum_loss() / learners[0]._vw._vw.get_weighted_examples() * 2
loss_without_qcolcol = learners[1]._vw._vw.get_sum_loss() / learners[1]._vw._vw.get_weighted_examples() * 2

if loss_with_qcolcol > loss_without_qcolcol:
    raise("Something is totally wrong")

for learner in learners:
    learner.finish()

#After evaluating can create a quick summary plot to get a sense of how the learners performed
# result.plot_learners(y='reward',err='se',xlim=(10,None))
