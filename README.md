# Coba

### What is it?

 Coba is a powerful benchmarking framework built specifically for contextual bandit (CB) research.

### How do you benchmark?

Think for a second about the last time you benchmarked an algorithm or dataset and ask yourself

 1. Was it easy to add new data sets?
 2. Was it easy to add new algorithms?
 3. Was it easy to create, run and share benchmarks?

### The Coba Way
 
 Coba was built from the ground up to do all that and more.
 
 Coba is...
 
 * ... *light-weight* (it has no dependencies to get started)
 * ... *distributed* (it was built to work across the web with caching, api-key support, checksums and more)
 * ... *verbose* (it has customizable, hierarchical logging for meaningful, readable feedback on log running jobs)
 * ... *robust* (benchmarks write every action to file so they can always be resumed whenever your system crashes)
 * ... *just-in-time* (no resources are loaded until needed, and they are released immediately to keep memory small)
 * ... *a duck?* (Coba relies only on duck-typing so no inheritance is needed to implement our interfaces)
 
 But don't take our word for it. We encourage you to look at the code yourself or read more below.
 
 ## Workflow
 
 Coba is architected around a simple workflow: Environments -> Learners -> Experiment -> Results.
 
 Environments represent unique CB problems that need to be solved. Learners are the CB algorithms that we can use to learn policies. Experiments are combinations of Environments and Learners that we want to evaluate. And Results are the outcome of an Experiment, containing all the data from the Experiment.
 
 ## Environments
 
 Environments are the core unit of evaluation in Coba. They are nothing more than a sequence of interactions with contexts, actions and rewards. A number of tools have been built into Coba to make simulation creation easier. All these tools are defined in the `coba.environments` module. We describe a few of these tools here.
 
 ### Creating Environments From Classification Data Sets
 
 Classification data sets are the easiest way to create Environments in Coba. Coba natively supports: 
 
 * Binay, multiclass and multi-label problems
 * Dense and sparse representations
 * Openml, Csv, Arff, Libsvm, and the extreme classification (Manik) format
 * Local files and files over http (with local caching)
 
 The classification environments built into Coba are `OpenmlSimulation`, `CsvSimulation`, `ArffSimulation`, `LibsvmSimulation`, and `ManikSimulation`.

 ### Creating Environments From Generative Functions
 
 Sometimes we have well defined models that an agent has to make decisions within but no data. To support evaluation in these domains one can use `LambdaSimulation` to define generative functions for that will create an Environment. 
 
 ### Creating Environments From Scratch
 
 If more customization is needed beyond what is offered above then you can easily create your own simulation by implementing Coba's simple `SimulatedEnvironment` interface.
 
 ## Learners
 
 Learners are algorithms which are able to improve their action selection through interactions with simulations.
 
 A number of algorithms are provided natively with Coba for quick comparsions. These include:
 
 * All contextual bandit learners in VowpalWabbit
 * UCB1-Tuned Bandit Learner by Auer et al. 2002
 * LinUCB by Chu et al. 2011
 * Corral by Agarwal et al. 2017

 
## Experiments
 
 The `Experiment` class contains all the logic for learner performance evaluation. This includes execution logic such as how many processors to use and where to write results. There is only one `Experiment` implementation in Coba and it can be found in the `coba.experiments` module.
  
 
 ## Examples
 
 An examples directory is included in the repository with a number of code and experiment demonstrations. These examples show how to create experiments, evaluate learners against them and plot the results.
