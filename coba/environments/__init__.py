"""This module contains core functionality for working with contextual bandit environments.

This module contains the abstract interfaces for common types of bandit environments, several 
concrete implementations of these environments for use in experiments, and various filters that 
can be applied to environments to modify them in useful ways (e.g., shuffling, scaling, and imputing).
"""

from coba.environments.core       import Environments
from coba.environments.primitives import Context, Action, Interaction, Environment
from coba.environments.filters    import Shuffle, Take, Identity, Reservoir
from coba.environments.filters    import Sort, Scale, Cycle, Impute
from coba.environments.filters    import Binary, WarmStart, Sparse, Where
from coba.environments.filters    import EnvironmentFilter

from coba.environments.simulated import SimulatedInteraction, SimulatedEnvironment
from coba.environments.simulated import MemorySimulation, LambdaSimulation
from coba.environments.simulated import LinearSyntheticSimulation, NeighborsSyntheticSimulation
from coba.environments.simulated import OpenmlSimulation, OpenmlSource
from coba.environments.simulated import SupervisedSimulation, CsvSource, ArffSource, LibsvmSource, ManikSource
from coba.environments.simulated import SerializedSimulation

from coba.environments.logged.primitives import LoggedInteraction, LoggedEnvironment
from coba.environments.warmstart.primitives import WarmStartEnvironment

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
    'MemorySimulation',
    'LambdaSimulation',
    'OpenmlSimulation',
    'OpenmlSource',
    'SupervisedSimulation',
    'CsvSource',
    'ArffSource',
    'LibsvmSource',
    'ManikSource',
    'LinearSyntheticSimulation',
    'NeighborsSyntheticSimulation',
    'SerializedSimulation',
    'EnvironmentFilter',
    'Sort',
    'Scale',
    'Cycle',
    'Impute',
    'Binary',
    'Where',
    'WarmStart',
    'Shuffle', 
    'Take',
    'Reservoir',
    'Identity',
    'Sparse'
]
