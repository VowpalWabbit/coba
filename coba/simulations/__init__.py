"""The simulations module contains core classes and types for defining contextual bandit simulations.

This module contains the abstract interface expected for bandit simulations along with the 
class defining an Interaction within a bandit simulation. Additionally, this module also contains 
the type hints for Context, Action and Reward. These type hints don't contain any functionality. 
Rather, they simply make it possible to use static type checking for any project that desires 
to do so.

TODO Add RegressionSimulation
"""

from coba.simulations.core import (
    Context, Action, Interaction, Simulation, MemorySimulation, 
    LambdaSimulation, ClassificationSimulation, CsvSimulation, ArffSimulation, 
    LibsvmSimulation, ManikSimulation, ValidationSimulation, RegressionSimulation
)
from coba.simulations.openml  import OpenmlSource, OpenmlSimulation
from coba.simulations.filters import SimulationFilter, Shuffle, Take, Sort, Scale, CycleNoise
from coba.simulations.pipes   import SimSourceFilters

__all__ = [
    'Context',
    'Action',
    'Interaction',
    'Simulation',
    'OpenmlSource',
    'MemorySimulation',
    'LambdaSimulation',
    'ClassificationSimulation',
    'RegressionSimulation',
    'CsvSimulation',
    'ArffSimulation',
    'LibsvmSimulation',
    'ManikSimulation',
    'OpenmlSimulation',
    'ValidationSimulation',
    "SimulationFilter",
    'Shuffle',
    'Take',
    'Sort',
    'Scale',
    'CycleNoise',
    'SimSourceFilters'
]
