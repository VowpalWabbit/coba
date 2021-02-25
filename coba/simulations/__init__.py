"""The simulations module contains core classes and types for defining contextual bandit simulations.

This module contains the abstract interface expected for bandit simulations along with the 
class defining an Interaction within a bandit simulation. Additionally, this module also contains 
the type hints for Context, Action and Reward. These type hints don't contain any functionality. 
Rather, they simply make it possible to use static type checking for any project that desires 
to do so.

TODO Add RegressionSimulation
"""

from typing import (
    Optional, Hashable, Sequence
)
Context = Optional[Hashable]
Action  = Hashable
Key     = int

from coba.simulations.edits import Shuffle, Take, Batch, PCA, Sort
from coba.simulations.rewards import Reward, MemoryReward, ClassificationReward
from coba.simulations.simulations import Interaction, Simulation, OpenmlSource, LambdaSource, MemorySimulation, LambdaSimulation, ClassificationSimulation, OpenmlSimulation, BatchedSimulation


__all__ = [
    'Shuffle',
    'Take',
    'Batch',
    'PCA',
    'Sort',
    'Interaction',
    'Reward',
    'MemoryReward',
    'ClassificationReward',
    'Context',
    'Action',
    'Key',
    'Simulation',
    'OpenmlSource',
    'LambdaSource',
    'MemorySimulation',
    'LambdaSimulation',
    'ClassificationSimulation',
    'OpenmlSimulation',
    'BatchedSimulation'
]
