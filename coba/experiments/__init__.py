"""The benchmarks module contains core benchmark functionality and protocols.

This module contains the abstract interface expected for Benchmark implementations. This 
module also contains several Benchmark implementations and Result data transfer class.
"""

from coba.experiments.core    import Experiment
from coba.experiments.results import Result

__all__ = [
    'Result',
    'Experiment',
]