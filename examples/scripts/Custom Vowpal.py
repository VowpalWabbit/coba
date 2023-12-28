"""
This is an example script that creates and execuates an Experiment.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

import math
import collections
import coba as cb

# this mediator splits the 'x' namespace into multiple namespaces
class CustomVWMediator(cb.VowpalMediator):

    def __init__(self, target: str, partitions: list):
        self.target     = target
        self.partitions = partitions
        super().__init__()

    @property
    def params(self):
        return {"target": self.target, "parts": self.partitions}

    def custom_ns(self, ns: dict):
        if self.target not in ns:
            return ns

        feats  = ns.pop(self.target)
        new_ns = collections.defaultdict(dict,ns)

        n_parts   = len(self.partitions)
        n_feats   = len(feats)
        part_size = math.ceil(n_feats/min(n_parts, n_feats))

        for i, k in enumerate(feats):
            g = i//part_size
            new_ns[self.partitions[g]][k] = feats[k]

        return new_ns

# First, we define the learners that we want to test
learners = [
    cb.VowpalLearner(args="-q :: --cb_explore_adf --epsilon 0.05", vw=CustomVWMediator('x',["b", "c"])),
    cb.VowpalLearner(vw=CustomVWMediator('x',["b", "c"])),
]

# Next we create an environment we'd like to evaluate against
environments = cb.Environments.from_linear_synthetic(1000, n_action_features=0).shuffle([1])

# We then create and run our experiment from our environments and learners
result = cb.Experiment(environments, learners).run()

# this is None if we use shuffle with multiple values
assert learners[0]._vw._vw != None

# get_weighted_examples returns both labeled and unlabeled examples
# we multiply by 2 since we divided by double the number of examples actually learned on
# note: only learnable examples have labels
loss_with_qcolcol    = learners[0]._vw._vw.get_sum_loss() / learners[0]._vw._vw.get_weighted_examples() * 2
loss_without_qcolcol = learners[1]._vw._vw.get_sum_loss() / learners[1]._vw._vw.get_weighted_examples() * 2

if loss_with_qcolcol > loss_without_qcolcol:
    raise ("Something is totally wrong")

for learner in learners:
    learner.finish()

# After evaluating can create a quick summary plot to get a sense of how the learners performed
result.plot_learners(y='reward',err='se',xlim=(10,None))
