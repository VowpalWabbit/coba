[![codecov](https://codecov.io/gh/VowpalWabbit/coba/branch/master/graph/badge.svg?token=885XLZJ2D4)](https://codecov.io/gh/VowpalWabbit/coba)
[![doc](https://readthedocs.org/projects/coba-docs/badge/?version=latest)](https://coba-docs.readthedocs.io/?badge=v8.0.3)
[![pypi](https://img.shields.io/badge/pypi-v8.0.3-blue)](https://pypi.org/project/coba/)

# Coba

 ### What is it?

 Coba is a powerful framework built to facilitate research with online contextual bandit (CB) algorithms.

 ### Documentation

 To start using Coba right away we recommend visiting our [documentation](https://coba-docs.readthedocs.io/).

 ### How do you research?

 Think for a second about the last time you considered evaluating CB methods for a research project

 1. Was it easy to incorporate your own data sets during CB analysis?
 2. Was it easy to incorporate the wide array of available CB algorithms?
 3. Was it easy to create, run and share the results of CB experiments?

 Coba was built from the ground up to do these three things and more.

 ### The Coba Way

 Coba is...

 * ... *light-weight* (there are minimal project dependencies to get started)
 * ... *verbose* (sophisticated logging is built-in for meaningful, readable feedback on log running jobs)
 * ... *robust* (results are frequently saved to disk so that no work is lost if research is interrupted for any reason)
 * ... *lazy* (resources are loaded just-in-time and then released quickly to minimize resource requirements)
 * ... *a duck?* (all code uses duck-typing so that inheritance is a non-issue when extending built-in functionality)

 But don't take our word for it. We encourage you to look at the code yourself or read more below.

 ## Workflow

 Coba is architected around a simple workflow: (Learners, Environments) -> Experiments -> Results -> Analysis.

 Environments represent unique CB problems that need to be solved. Learners are the CB algorithms that we can use to solve these problems. Experiments are combinations of Environments and Learners that we want to evaluate. And Results are the outcome of an Experiment, containing all the data from the Experiment.

 ## Learners

 Learners are algorithms which are able to improve their action selection through interactions with environments.

 A number of algorithms are provided natively with Coba for quick comparsions. These include:

 * All contextual bandit learners in [Vowpal Wabbit](https://vowpalwabbit.org/)
 * UCB1-Tuned Bandit Learner (Auer et al., 2002)
 * LinUCB (Li and Chu et al., 2011)
 * Linear Thompson Sampling (Agrawal and Goyal et al., 2013)
 * Corral (Agarwal et al., 2017)

 ## Environments

 Environments are the core unit of evaluation in Coba. They are nothing more than a sequence of interactions with contexts, actions and rewards. A number of tools have been built into Coba to make simulation creation easier. All these tools are defined in the `coba.environments` module. We describe a few of these tools here.

 ### Creating Environments From Classification Data Sets

 Classification data sets are the easiest way to create Environments in Coba. Coba natively supports:

 * Binay, multiclass and multi-label problems
 * Dense and sparse representations
 * Openml, Csv, Arff, Libsvm, and the extreme classification (Manik) format
 * Local files and files over http (with local caching)

 ### Creating Environments From Scratch

 If more customization is needed beyond what is offered above then you can easily create your own simulation by implementing Coba's `Environment` interface and generating interactions.

 ## Experiments

 Experiments are combinations of Learners and Environments. The Experiment class orchestrates the complex tasks of evaluating learners on environments in a resource efficient, repeatable, and robust manner. Experiments can be run on any number of processes (we often run on 96 core machines) and writes all results to file as it runs so that it can be restarted if it gets interrupted.

 ## Results

 The end result of an Experiment is a Result object that supports easy analysis. The result class provides a number of plots as well as filtering capabilities out of the box. If the built in functionality isn't advanced enough data can be exported to pandas dataframes where all manner of advanced filtering and plotting can be performed.

 ## Examples

 To learn more, many working code examples are provided in the [documentation](https://coba-docs.readthedocs.io/).

