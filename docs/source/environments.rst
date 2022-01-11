====================
Environments
====================

Within ``Coba`` environments are one of the core building blocks for experiments. An environment is a sequence
of interactions with the world. Interactions represent situations where our Learner must make a decision. They
are defined in terms of a context, actions and action rewards. 

``Environments`` come in one of three flavors: ``SimulatedEnvironment``, ``LoggedEnvironment``, and ``WarmStartEnvironment``. 
``SimulatedEnvironments`` contain reward information for each actio, ``LoggedEnvironments`` contain reward information for a single action, 
and ``WarmStartEnvironments`` start as ``LoggedEnvironments`` in the "warm-start" phase and then turn into ``SimulatedEnvironments``.

The Role of Environments
~~~~~~~~~~~~~~~~~~~~~~~~

Environments are important in ``Coba`` experiments because they allow us to both learn and evaluate how well we are learning. This
dual purpose is possible because of the online structure in ``Coba`` experiments where we evaluate one interaction at a time rather
than a batch of interactions. For anyone familiar with this kind of setting ``SimulatedEnvironments`` can generally be thought of as being
most appropriate for on-policy experiments while ``LoggedEnvironments`` are most appropriate for off-policy experiments (i.e., logged as
in logged bandit data).

Friendly Environments API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``coba.environments`` module contains all functionality pertaining to creating and modifying Environments. This functionality can be
a little overwhelming initially so a more simplified interface has been provided to help one get started called Environments. This can
be imported as shown below. In the following examples we will demonstrate how to use the friendly API to create and modify environments

.. code-block:: python

    from coba.environments import Environments

Creating Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are four ways to create SimulatedEnvironments. Below we briefly cover each of these. For more information about environments than is 
contained here the API reference is a good next step.

From Openml Supervised Datasets
-----------------------------------
The simplest method for creating simulated environments is by downloading Openml supervised datasets. This can be done either through the
environments interface or by directly creating an OpenmlSimulation. In either case we strongly recommend enabling local caching when working
with Openml derived simulations. Otherwise ``Coba`` will download the dataset from the internet each time the Environment is used in an experiment.

.. code-block:: python

    from coba.environments import Environments, OpenmlSimulation # Only one of these is needed
    from coba.contexts import CobaContext # Only needed to enable disk caching
    
    #Enable local caching by providing a cache_directory,
    #The directory will becreated if it does not exist.
    CobaContext.cacher.cache_directory = './.coba_cache'
    
    
    #Create a simulation from covertype https://www.openml.org/d/180 through the Environments interface
    env = Environments.from_openml(180)
    
    #Create a simulation from covertype https://www.openml.org/d/180 by direct instantiation
    env = [ OpenmlSimulation(180) ]

Often times Openml datasets will be larger than we want. In order to deal with this OpenmlSimulations can be told how many interactions to take 
when creating the simulation. When the take parameter is provided a resevoir sampler will be used to reduce to the given number of examples. We 
may also want to create multiple openml simulations at once to see how a Learner performs in general. Both of these usedcases are demonstrated below.
   
.. code-block:: python

    from coba.environments import Environments, OpenmlSimulation #Only one of these is needed
    
    #Create 5 simulations by selecting 3000 random examples from eacah dataset
    env = Environments.from_openml([3, 44, 46, 151, 180], take=3000)
    
    #Create 5 simulations by selecting 3000 random examples from eacah dataset
    env = [ OpenmlSimulation(id,take=3000) for id in [3, 44, 46, 151, 180] ]


From Local Supervised Datasets
-----------------------------------
In the case that you have an existing supervised dataset that you'd like to turn into a SimulatedEnvironment you can use a SupervisedSimulation.
SupervisedSimulation can be used on both sparse and dense examples as well as both regression and classification labels. Coba natively supports
four data formats: CSV, ARFF, Libsvm, and Manik. An example is shown creating a SupervisedSimulation through the two available interfaces. Only
one of the interfaces needs to be used in practice.

.. code-block:: python

    from coba.pipes import ArffReader
    from coba.environments import Environments, SupervisedSimulation # Only one of these is needed
        
    #Create a simulation from an arff data set with regression labels at header "my_label"
    #The take=1000 is an optional parameter that tells the simulation to only use 1000 randomly
    #selected examples from the data set.
    env = Environments.from_supervised("path/file.arff", ArffReader(), label_type="R", label_col="my_label", take=1000)
        
    #This is equivalent to the Environments interface code immediately above
    env = [ SupervisedSimulation("path/file.arff", ArffReader(), label_type="R", label_col="my_label", take=1000) ]


From Synthetic Generation
-----------------------------------
For the case where one wants to have complete control over the characteristics of a SimulatedEnvironment used in an Experiment Coba provides two 
synethic environments: LinearSyntheticSimulation and LocalSyntheticSimulation. The linear synthetic simulation follows traditional linear contextual 
bandit assumptions where each action's expected reward has a linear relationship to the action and context features. Local synthetic on the other
hand creates local exemplars and calculates reward based on the locaion of a context and action feature set with respect to the exemplars. As above
we demonstrate below the two interfaces for working with these.

.. code-block:: python

    from coba.environments import Environments, LinearSyntheticSimulation, LocalSyntheticSimulation
        
    #cross_terms controls the parameterization of the reward function where this example uses action features, action and context features, and action and context^2 featurs.
    env = Environments.from_linear_synthetic(n_interactions=1000, n_actions=10, n_context_features=20, n_action_features=2, cross_terms = ["a", "ax", "axx"], r_noise_var=.1)
    
    #n_contexts indicate the numbe of "exemplar" contexts to create and asign reward values to within the generated space
    env = Environments.from_local_synthetic(n_interactions=1000, n_actions=10, n_context_features=20, n_contexts=200, seed=3)
    
    #As we've demonstrated above here is the alternative interface for creating the synthetic environments
    env = [ LinearSyntheticSimulation(n_interactions=1000, n_actions=10, n_context_features=20, n_action_features=2, cross_terms = ["a", "ax", "axx"], r_noise_var=.1) ]
    env = [ LocalSyntheticSimulation(n_interactions=1000, n_actions=10, n_context_features=20, n_contexts=200, seed=3)]
    
An additional simulation, called LambdaSimulation, is also available if even more control is needed when generating synthetic datasets. The LambdaSimulation is the base class
of the two synthetic environments mentioned above. The LambdaSimulation allows one to define an environment in terms of three generative functions: a context generator, an
action generator given contexts, and a reward generator given contexts and actions. LambdaSimulation is also available through the class interface given its more advanced nature.

.. code-block:: python

    from coba.environments import LambdaSimulation
    
    #Here is an example of a deterministic simulation
    
    contexts = [[1,2],[3,4],[5,6]]
    actions  = [1,4,7]
    
    #index increments from 0...n, it's provided for convenience and can be used or ignored.
    def context_generator(index):
        return contexts[index]
        
    def action_generator(index, context):
        return actions
        
    def reward_generator(index, context, action):
        return action * context[0] - action * context[1]
        
    env = [ LambdaSimulation(n_interactions=1000, context_generator, action_generator, reward_generator) ]

It is also possible to create a stochastic LambdaSimulation.
    
.. code-block:: python
    
   #Here is an example of a stochastic simulation, note the aditional rng parameter provided to the generators
   #The flag to indicate that the LambdaSimulation is stochastic is the seed parameter when created below
    
    contexts = [[1,2],[3,4],[5,6]]
    actions  = [1,4,7]
    
    def context_generator(index, rng):
        return rng.randoms(3)
        
    def action_generator(index, context, rng):
        return [ rng.randoms(3) for _ in range(4) ] 
        
    def reward_generator(index, context, action, rng):
        return sum([ c*a for c,a in zip(context,action ]) + rng.random()/100
        
    env = [ LambdaSimulation(n_interactions=1000, context_generator, action_generator, reward_generator, seed=2) ]

From Scratch
-----------------------------------

Finally if all the provide simulations above still do not meet the needs of your research you can easily create your own SimulatedEnvironment
from scratch. Coba uses duck typing for SimulatedEnvironments so no inheritence or dependencies are needed. One only needs to implement the
protocol. Below is a very simple example.

.. code-block:: python

    from coba.environments import SimulatedInteraction

    class MyScratchSimulation:
    
        def read(self):
            yield SimulatedInteraction(context=1, actions=['a','b','c'], rewards=[1,2,3])
            yield SimulatedInteraction(context=2, actions=['a','b','c'], rewards=[2,-2,6])
            yield SimulatedInteraction(context=3, actions=['a','b','c'], rewards=[1,2,3])
            yield SimulatedInteraction(context=4, actions=['a','b','c'], rewards=[2,-2,6])
            yield SimulatedInteraction(context=5, actions=['a','b','c'], rewards=[1,2,3])

        @property
        def params(self):
            return { "key": "data describing my simulation" } # this will be written to results and can be used for sorting and filtering

    env = [ MyScratchSimulation() ]
    
Filtering Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once an environment has been created we often want to modify it in some way. In ``Coba`` a modification to an environment is called a "filter".
This language is adopted because applying a series of modifications to an environment is viewed as a pipeline. Using pipelines many environments
can be made very quickly from a handful of base environments. Modifying environments is where the Environments API really shines. We share a few
examples below. All available filters can be seen in the API Reference.

.. code-block:: python
    
    from coba.environments import Environments

    #this single lines takes a single synthetic environment and turns it into three 
    #environments with the same three interactions shuffled into different orders.
    Environments.from_linear_synthetic(n_interactions=1000).shuffle([1,2,3])
    
    #This builds on the above example but creates 30 environments via shuffling and then turns
    #the continuous rewards of the linear environment into binary rewards where the max reward
    #in each interaction has a value of 1 and all other rewards have a value of 0. Binarizing
    #rewards is useful for interpretting performance as the % of times the best action is picked.
    Environments.from_linear_synthetic(n_interactions=1000).shuffle(range(30)).binary()
    
    #when working with real world data sets often times we have features on wildly different scales
    #or we may have to deal with missing data. When these are our problems we can impute and scale.
    Environments.from_supervised("my_data.arff.gz").impute("mean",1000).scale(shift="med",scale="iqr",using=1000).shuffle([1,2,3]).take(3000)

    #For very lage datasets shuffling and then taking can be problematic because shuffle requires all data to be loaded into memory.
    #To help with this Coba also provides resevoir sampling. This technique is a combination of take and shuffle and doesn't require
    #full data sets to be loaded into memory.
    Environments.from_supervised("my_data.arff.gz").impute("mean",1000).scale(shift="med",scale="iqr",using=1000).reservoir(3000,[1,2,3]) 
    
    
Conclusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Above we've shown several ways to create and modify environments. On its own an environment isn't incredibly useful. When combined with Experiment
though they become powerful tools to understand how various algorithms perform. Therefore, if you haven't already, we suggest you visit the page
about Experiments to see how to use the Environments you create.
