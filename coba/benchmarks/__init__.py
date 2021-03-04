"""The benchmarks module contains core benchmark functionality and protocols.

This module contains the abstract interface expected for Benchmark implementations. This 
module also contains several Benchmark implementations and Result data transfer class.
"""

from coba.benchmarks.core import Result, Benchmark
from coba.benchmarks.formats import BenchmarkFileFmtV1, BenchmarkFileFmtV2 

__all__ = [
    'Result',
    'Benchmark',
    'BenchmarkFileFmtV1',
    'BenchmarkFileFmtV2'
]