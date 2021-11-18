"""The experiments module contains core experiment functionality and protocols.

This module contains the abstract interface expected for Experiment implementations.
"""

from coba.experiments.core    import Experiment
from coba.experiments.results import Result
from coba.experiments.tasks   import (
    LearnerTask, EnvironmentTask, EvaluationTask,
    SimpleLearnerTask, SimpleEnvironmentTask, ClassEnvironmentTask,
    OnPolicyEvaluationTask, OffPolicyEvaluationTask, WarmStartEvaluationTask
)

__all__ = [
    'Result',
    'Experiment',
    'LearnerTask',
    'EnvironmentTask',
    'EvaluationTask',
    'SimpleLearnerTask',
    'ClassEnvironmentTask',
    'SimpleEnvironmentTask',
    'OnPolicyEvaluationTask',
    'OffPolicyEvaluationTask',
    'WarmStartEvaluationTask'    
]