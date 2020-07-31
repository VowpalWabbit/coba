# Bandit Benchmarking

This repository contains python modules designed to benchmark the performance of contextual bandit algorithms.

# Core Modules

The package has three core modules:
 1. simulations -- This module contains contextual bandit simulations with context, actions and rewards for those actions.
 2. learners -- This module contains common contextual bandit algorithms such as epsilon-greedy and upper confidence bound.
 3. benchmarks -- This module uses simulations to determine how well a learner performs and compare to other learners.

# Core Interfaces
 
In order to make the package testable and extensible it has been built around three simple interfaces.

 1. The simulation interface
 2. The learner interface
 3. The benchmark interface
 
For most users the only interface that they will ever need to implement on their own is the learner interface. Implementing the learner interface on a custom algorithm allows one to leverage all simulations and benchmarks provided by the package to collect performance metrics.

# Learning More

For those who wish to learn more the source code is perhaps the best place to start. All modules and classes have detailed docstrings and unittests.