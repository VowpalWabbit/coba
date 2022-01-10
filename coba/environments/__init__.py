"""This module contains core functionality for working with contextual bandit environments.

This module contains the abstract interfaces for common types of bandit environments, several 
concrete implementations of these environments for use in experiments, and various filters that 
can be applied to environments to modify them in useful ways (e.g., shuffling, scaling, and imputing).
"""

from coba.environments.core       import Environments
from coba.environments.primitives import Context, Action, Interaction, Environment
from coba.environments.filters    import Shuffle, Take, Identity, Reservoir
from coba.environments.filters    import Sort, Scale, Cycle, Impute
from coba.environments.filters    import Binary, WarmStart, Sparse
from coba.environments.filters    import FilteredEnvironment, EnvironmentFilter

from coba.environments.simulated import SimulatedInteraction, SimulatedEnvironment
from coba.environments.simulated import MemorySimulation, LambdaSimulation
from coba.environments.simulated import LinearSyntheticSimulation, LocalSyntheticSimulation
from coba.environments.simulated import OpenmlSimulation, SupervisedSimulation

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
    'SupervisedSimulation',
    'LinearSyntheticSimulation',
    'LocalSyntheticSimulation',
    'EnvironmentFilter',
    'Sort',
    'Scale',
    'Cycle',
    'Impute',
    'Binary',
    'WarmStart',
    'FilteredEnvironment',
    'Shuffle', 
    'Take',
    'Reservoir',
    'Identity',
    'Sparse'
]
