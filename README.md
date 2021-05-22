# Coba

### What is it?

 Coba is a powerful benchmarking framework built specifically for research of contextual bandit algorithms.

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
 
 Coba is architected around a simple workflow: Simulations -> Benchmark -> Learners -> Results.
 
 Simulations are the core unit of evaluation in Coba. A simulation contains all the necessary logic to produce interactions and rewards. With a collection of simulations we can then define a Benchmark. Benchmarks package up all rules for robust and repeatable evaluation. Finally, once we have a benchmark we can then apply that benchmark to learners to see how a learner is able to perform on the given benchmark.
 
 ## Simulations
 
 Simulations are the core unit of evaluation in Coba. They are nothing more than a collection of interactions with an environment and potential rewards. A number of tools have been built into Coba to make simulation creation easier. All these tools are defined in the `coba.simulations` module. We describe these tools in more detail below.
 
 ### Importing Simulations From Classification Data Sets
 
 Classification data sets are the easiest way to quickly evaluate CB algorithms with Coba. Coba natively supports: 
 
 * Binay, multiclass and multi-label problems
 * Dense and sparse representations
 * Openml, Csv, Arff, Libsvm, and the extreme classification (Manik) format
 * Local files and files over http (with local caching)
 
 The classification simulations built into Coba are `OpenmlSimulation`, `CsvSimulation`, `ArffSimulation`, `LibsvmSimulation`, and `ManikSimulation`.

 ### Generating Simulations From Generative Functions
 
 Sometimes we have well defined models that an agent has to make decisions within. To support evaluation in these domains one can use `LambdaSimulation` to define generative functions for . 
 
 ### Creating Simulations From Scratch
 
 If more customization is needed beyond what is offered above then you can easily create your own simulation by implementing Coba's simple `Simulation` interface.
 
 ## Benchmarks
 
 The `Benchmark` class contains all the logic for learner performance evaluation. This includes both evaluation logic (e.g., which simulations and how many interactions) and execution logic (e.g., how many processors to use and where to write results).
 
 There is only one `Benchmark` implementation in Coba and it can be found in the `coba.benchmarks` module.
 
 ## Learners
 
 Learners are algorithms which are able to improve their action selection through interactions with Simulations.
 
 A number of algorithms have been provided out of the box for quick comparsions. These include:
 
 * All contextual bandit learners in VowpalWabbit
 * UCB1-Tuned Bandit Learner by Auer et al. 2002
 * Corral by Agarwal et al. 2017
  
 ## Examples
 
 An examples directory is included in the repository with a number of code demonstrations and benchmark demonstrations. These examples show how to create benchmarks, evaluate learners against them and plot the results.
