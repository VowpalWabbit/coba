"""This module contains core experiment and result functionality.

This module contains the Experiment evaluator, implementations of specific tasks that are performed
during an experiment evaluation, and the Result datastructures returned after experiment evalution.
"""

from coba.experiments.core    import Experiment
from coba.experiments.results import Result, Table
from coba.experiments.tasks   import LearnerTask, SimpleLearnerInfo
from coba.experiments.tasks   import EnvironmentTask, SimpleEnvironmentInfo, ClassEnvironmentInfo
from coba.experiments.tasks   import EvaluationTask, SimpleEvaluation

__all__ = [
    'Result',
    'Table',
    'Experiment',
    'LearnerTask',
    'EnvironmentTask',
    'EvaluationTask',
    'SimpleLearnerInfo',
    'ClassEnvironmentInfo',
    'SimpleEvaluation',
    'SimpleEnvironmentInfo',
]