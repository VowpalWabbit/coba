"""The environments module contains core classes and types for defining contextual bandit environments.

This module contains the abstract interface expected for bandit environments along with the 
class defining an Interaction within a bandit environment. Additionally, this module also contains 
the type hints for Context, Action and Reward. These type hints don't contain any functionality. 
They simply make it possible to use static type checking for any project that desires 
to do so.
"""

from coba.environments.simulated.primitives import SimulatedInteraction, SimulatedEnvironment
from coba.environments.simulated.primitives import MemorySimulation
from coba.environments.simulated.supervised import SupervisedSimulation
from coba.environments.simulated.openml     import OpenmlSimulation, OpenmlSource
from coba.environments.simulated.synthetics import LambdaSimulation, LinearSyntheticSimulation, LocalSyntheticSimulation

__all__ = [
    'SimulatedInteraction',
    'SimulatedEnvironment',
    'MemorySimulation',
    'LambdaSimulation',
    'OpenmlSimulation',
    'OpenmlSource',
    'LinearSyntheticSimulation',
    'LocalSyntheticSimulation',
    'SupervisedSimulation'
]
