"""
This is an example script that creates and execuates an Experiment.
This script requires that the matplotlib and vowpalwabbit packages be installed.
"""

from coba.contexts     import CobaContext
from coba.learners     import RandomLearner, VowpalEpsilonLearner, VowpalSquarecbLearner
from coba.experiments  import Experiment
from coba.environments import Environments

#this line is required by Python in order to use multi-processing
if __name__ == '__main__':

    # These configuration changes aren't ever required. 
    # They are simply here to serve as an example.
    # These can also be set automatically by creating a .coba file your project root. 
    CobaContext.cacher.cache_directory = './.coba_cache'
    CobaContext.experiment.processes   = 1
    CobaContext.experiment.chunk_by    = 'task'

    #First, we define the learners that we want to test
    learners = [
        RandomLearner(),
        VowpalEpsilonLearner(),
        VowpalSquarecbLearner()
    ]

    #Next we create the environments we'd like evaluate against
    environments = Environments.from_linear_synthetic(1000).shuffle([0,1,2]).binary()

    #We then create and evaluate our experiment from our environments and learners 
    result = Experiment(environments,learners).evaluate()

    #After evaluating can create a quick summary plot to get a sense of how the learners performed
    result.plot_learners(err='sd',xlim=(10,1000))

    #We can also create a plot examining how specific learners did across each shuffle of our environments
    result.filter_lrn(full_name="vw").plot_learners(err='sd',each=True,xlim=(10,1000))
