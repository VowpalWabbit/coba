"""This module contains core functionality for conducting and analyzing CB experiments.

This module contains the abstract interfaces for common types of bandit environments, several 
concrete implementations of these environments for use in experiments, and various filters that 
can be applied to environments to modify them in useful ways (e.g., shuffling, scaling, and imputing).
"""

"""The experiments module contains core experiment functionality and protocols.

This module contains the abstract interface expected for Experiment implementations.
"""

from coba.experiments.core    import Experiment
from coba.experiments.results import Result, Table
from coba.experiments.tasks   import LearnerTask, SimpleLearnerTask
from coba.experiments.tasks   import EnvironmentTask, SimpleEnvironmentTask, ClassEnvironmentTask
from coba.experiments.tasks   import EvaluationTask, OnlineOnPolicyEvalTask, OnlineOffPolicyEvalTask, OnlineWarmStartEvalTask

__all__ = [
    'Result',
    'Table',
    'Experiment',
    'LearnerTask',
    'EnvironmentTask',
    'EvaluationTask',
    'SimpleLearnerTask',
    'ClassEnvironmentTask',
    'SimpleEnvironmentTask',
    'OnlineOnPolicyEvalTask',
    'OnlineOffPolicyEvalTask',
    'OnlineWarmStartEvalTask'
]