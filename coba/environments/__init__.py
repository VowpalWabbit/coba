"""This module contains core functionality for working with contextual bandit environments.

This module contains the abstract interfaces for common types of contextual bandit environments,
several concrete implementations of these environments for use in experiments, and various filters that
can be applied to environments to modify them in useful ways (e.g., shuffling, scaling, and imputing).
"""

from coba.environments.core       import Environments
from coba.environments.primitives import Interaction, Environment, SafeEnvironment, EnvironmentFilter
from coba.environments.primitives import SimulatedInteraction, LoggedInteraction, GroundedInteraction
from coba.environments.filters    import Shuffle, Take, Identity, Reservoir, Riffle, Cache
from coba.environments.filters    import Sort, Scale, Cycle, Impute, Flatten, Params
from coba.environments.filters    import Binary, Sparse, Where, Noise, Grounded
from coba.environments.filters    import Repr, Finalize, Batch, BatchSafe, Chunk, Logged

from coba.environments.synthetics import LambdaSimulation, LinearSyntheticSimulation, NeighborsSyntheticSimulation
from coba.environments.synthetics import KernelSyntheticSimulation, MLPSyntheticSimulation
from coba.environments.openml     import OpenmlSimulation, OpenmlSource
from coba.environments.supervised import SupervisedSimulation, CsvSource, ArffSource, LibSvmSource, ManikSource

__all__ = [
    'Interaction',
    'SimulatedInteraction',
    'LoggedInteraction',
    'GroundedInteraction',
    'Environment',
    'SafeEnvironment',
    'Environments',
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
    'EnvironmentFilter',
    'Sort',
    'Scale',
    'Cycle',
    'Impute',
    'Binary',
    'Where',
    'Shuffle',
    'Take',
    'Reservoir',
    'Noise',
    'Identity',
    'Sparse',
    'Riffle',
    'Flatten',
    'Params',
    'Cache',
    'Grounded',
    'Finalize',
    'Repr',
    'Batch',
    'BatchSafe',
    'Chunk',
    'Logged'
]
