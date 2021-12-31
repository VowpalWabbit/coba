"""The environments module contains core classes and types for defining contextual bandit environments.

This module contains the abstract interface expected for bandit environments along with the 
class defining an Interaction within a bandit environment. Additionally, this module also contains 
the type hints for Context, Action and Reward. These type hints don't contain any functionality. 
They simply make it possible to use static type checking for any project that desires 
to do so.
"""

from coba.environments.simulated.primitives import SimulatedInteraction, SimulatedEnvironment
from coba.environments.simulated.primitives import MemorySimulation, LambdaSimulation
from coba.environments.simulated.primitives import ClassificationSimulation, RegressionSimulation
from coba.environments.simulated.readers    import ReaderSimulation, CsvSimulation, ArffSimulation, LibsvmSimulation, ManikSimulation
from coba.environments.simulated.openml     import OpenmlSimulation, OpenmlSource
from coba.environments.simulated.synthetics import LinearSyntheticSimulation, LocalSyntheticSimulation

__all__ = [
    'SimulatedInteraction',
    'SimulatedEnvironment',
    'MemorySimulation',
    'LambdaSimulation',
    'ClassificationSimulation',
    'RegressionSimulation',
    'ReaderSimulation',
    'CsvSimulation',
    'ArffSimulation',
    'LibsvmSimulation',
    'ManikSimulation',
    'OpenmlSimulation',
    'OpenmlSource',
    'LinearSyntheticSimulation',
    'LocalSyntheticSimulation'
]
