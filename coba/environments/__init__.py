"""The simulations module contains core classes and types for defining contextual bandit simulations.

This module contains the abstract interface expected for bandit simulations along with the 
class defining an Interaction within a bandit simulation. Additionally, this module also contains 
the type hints for Context, Action and Reward. These type hints don't contain any functionality. 
Rather, they simply make it possible to use static type checking for any project that desires 
to do so.

TODO Add RegressionSimulation
"""

from coba.environments.core         import Context, Action, SimulatedInteraction, LoggedInteraction, Simulation
from coba.environments.openml       import OpenmlSource, OpenmlSimulation
from coba.environments.filters      import SimulationFilter, Identity, Shuffle, Take, Sort, Scale, Cycle, Impute
from coba.environments.pipes        import EnvironmentPipe
from coba.environments.environments import Environments

from coba.environments.simulations import (
    MemorySimulation, LambdaSimulation, ClassificationSimulation, CsvSimulation, 
    ArffSimulation, LibsvmSimulation, ManikSimulation, ValidationSimulation, RegressionSimulation
)

__all__ = [
    'Context',
    'Action',
    'SimulatedInteraction',
    'LoggedInteraction',
    'Simulation',
    'Environments',
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
    'SimulationFilter',
    'Identity',
    'Shuffle',
    'Take',
    'Sort',
    'Scale',
    'Cycle',
    'Impute',
    'EnvironmentPipe',
]
