"""The simulations module contains core classes and types for defining contextual bandit simulations.

This module contains the abstract interface expected for bandit simulations along with the 
class defining an Interaction within a bandit simulation. Additionally, this module also contains 
the type hints for Context, Action and Reward. These type hints don't contain any functionality. 
Rather, they simply make it possible to use static type checking for any project that desires 
to do so.

TODO Add RegressionSimulation
"""

from coba.simulations.core import (
    Context, Action, Key, Interaction, Simulation, Reward, MemorySimulation, 
    LambdaSimulation, ClassificationSimulation, BatchedSimulation, MemoryReward, ClassificationReward
)
from coba.simulations.openml  import OpenmlSource, OpenmlSimulation
from coba.simulations.filters import Shuffle, Take, Batch, PCA, Sort

__all__ = [
    'Context',
    'Action',
    'Key',
    'Interaction',
    'Simulation',
    'OpenmlSource',
    'MemorySimulation',
    'LambdaSimulation',
    'ClassificationSimulation',
    'Reward',
    'MemoryReward',
    'ClassificationReward',
    'OpenmlSimulation',
    'BatchedSimulation',
    'Shuffle',
    'Take',
    'Batch',
    'PCA',
    'Sort',
]
