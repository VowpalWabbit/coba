"""This module contains core functionality for working with contextual bandit environments.

This module contains the abstract interfaces for common types of contextual bandit environments,
several concrete implementations of these environments for use in experiments, and various filters that
can be applied to environments to modify them in useful ways (e.g., shuffling, scaling, and imputing).
"""

from coba.environments.core       import Environments
from coba.environments.primitives import Context, Action, Interaction, Environment, SafeEnvironment
from coba.environments.filters    import Shuffle, Take, Identity, Reservoir, Riffle
from coba.environments.filters    import Sort, Scale, Cycle, Impute, Flatten
from coba.environments.filters    import Binary, Warm, Sparse, Where, Noise
from coba.environments.filters    import EnvironmentFilter

from coba.environments.simulated.primitives import SimulatedInteraction, SimulatedEnvironment
from coba.environments.simulated.synthetics import MemorySimulation, LambdaSimulation
from coba.environments.simulated.synthetics import LinearSyntheticSimulation, NeighborsSyntheticSimulation
from coba.environments.simulated.synthetics import KernelSyntheticSimulation, MLPSyntheticSimulation
from coba.environments.simulated.openml     import OpenmlSimulation, OpenmlSource
from coba.environments.simulated.supervised import SupervisedSimulation, CsvSource, ArffSource, LibSvmSource, ManikSource
from coba.environments.simulated.serialized import SerializedSimulation

from coba.environments.logged.primitives import LoggedInteraction, LoggedEnvironment
from coba.environments.warmstart.primitives import WarmStartEnvironment

__all__ = [
    'Context',
    'Action',
    'Interaction',
    'SimulatedInteraction',
    'LoggedInteraction',
    'Environment',
    'SafeEnvironment',
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
    'LibSvmSource',
    'ManikSource',
    'LinearSyntheticSimulation',
    'NeighborsSyntheticSimulation',
    'KernelSyntheticSimulation',
    'MLPSyntheticSimulation',
    'SerializedSimulation',
    'EnvironmentFilter',
    'Sort',
    'Scale',
    'Cycle',
    'Impute',
    'Binary',
    'Where',
    'Warm',
    'Shuffle',
    'Take',
    'Reservoir',
    'Noise',
    'Identity',
    'Sparse',
    'Riffle',
    'Flatten'
]
