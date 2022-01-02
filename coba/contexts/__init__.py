"""This module contains classes and states scoped to various execution contexts.

This module should not be confused with the idea of "context" within Contextual Bandit problems.
This is simply context relevant to framework functions at various scopes. For developers who have
experience with other frameworks these are similar in kind to ThreadContext, DbContext, or ServerContext.
"""

from coba.contexts.cachers import Cacher, NullCacher, MemoryCacher, DiskCacher, ConcurrentCacher
from coba.contexts.loggers import Logger, NullLogger, BasicLogger, IndentLogger, ExceptLog, NameLog, StampLog, DecoratedLogger
from coba.contexts.core    import CobaContext, LearnerContext, ExperimentConfig

__all__ =[
    'CobaContext',
    'LearnerContext',
    'ExperimentConfig',
    'NullCacher',
    'MemoryCacher',
    'DiskCacher',
    'ConcurrentCacher',
    'Cacher',
    'NullLogger',
    'BasicLogger',
    'IndentLogger',
    'ExceptLog',
    'NameLog',
    'StampLog',
    'DecoratedLogger',
    'Logger'
]