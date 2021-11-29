import pytest

from coba.config       import CobaConfig
from coba.learners     import RandomLearner, EpsilonBanditLearner, VowpalLearner
from coba.experiments  import Experiment
from coba.environments import Environments

def test_does_not_crash():
    # These configuration changes aren't ever required. 
    # They are simply here to serve as an example of what can be changed.
    CobaConfig.cacher.cache_directory = './.coba_cache'
    CobaConfig.experiment.processes   = 1
    CobaConfig.experiment.chunk_by    = 'task'

    #First, we define the learners that we want to test
    learners = [
        RandomLearner(),
        EpsilonBanditLearner(epsilon=.1),
        VowpalLearner(epsilon=.1),
        VowpalLearner(squarecb="all")
    ]

    #Next we create the environments we'd like evaluate against
    environments = Environments.from_debug().shuffle([0,1,2]).binary()

    #We then create and evaluate our experiment from our environments and learners 
    result = Experiment(environments,learners).evaluate()

    assert result != None
