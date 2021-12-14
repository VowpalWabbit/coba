"""The environments module contains core classes and types for defining contextual bandit environments.

This module contains the abstract interface expected for bandit environments along with the 
class defining an Interaction within a bandit environment. Additionally, this module also contains 
the type hints for Context, Action and Reward. These type hints don't contain any functionality. 
They simply make it possible to use static type checking for any project that desires 
to do so.
"""

from coba.environments.primitives   import Context, Action
from coba.environments.primitives   import Interaction, SimulatedInteraction, LoggedInteraction
from coba.environments.primitives   import Environment, SimulatedEnvironment, LoggedEnvironment, WarmStartEnvironment
from coba.environments.primitives   import EnvironmentFilter
from coba.environments.openml       import OpenmlSource, OpenmlSimulation
from coba.environments.filters      import Shuffle, Take, Identity, Sort, Scale, Cycle, Impute, Binary, ToWarmStart, Sparse
from coba.environments.pipes        import EnvironmentPipe
from coba.environments.core         import Environments

from coba.environments.simulations import MemorySimulation, LambdaSimulation
from coba.environments.simulations import CsvSimulation, ArffSimulation, LibsvmSimulation, ManikSimulation
from coba.environments.simulations import ClassificationSimulation, RegressionSimulation
from coba.environments.simulations import LinearSyntheticSimulation, LocalSyntheticSimulation

__all__ = [
    'Context',
    'Action',
    'Interaction',
    'SimulatedInteraction',
    'LoggedInteraction',
    'Environment',
    'SimulatedEnvironment',
    'LoggedEnvironment',
    'WarmStartEnvironment',
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
    'LinearSyntheticSimulation',
    'LocalSyntheticSimulation',
    'EnvironmentFilter',
    'Sort',
    'Scale',
    'Cycle',
    'Impute',
    'Binary',
    'ToWarmStart',
    'EnvironmentPipe',
    'Shuffle', 
    'Take', 
    'Identity',
    'Sparse'
]
