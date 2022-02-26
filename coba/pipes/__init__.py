"""This module contains core functionality for defining data and workflows within Coba.

This module contains core functionality for defining jobs and working with datasets in Coba.
One does not need any understanding of this module to use coba for research. That being said,
a good understanding of the patterns in coba.pipes will help one understand coba and how to
best take advantage what it has to offer. 
"""

from coba.pipes.primitives import Filter, Source, Sink
from coba.pipes.multiprocessing import PipeMultiprocessor

from coba.pipes.filters import Take, Shuffle, Drop, Structure, Identity, Flatten, Default, Reservoir
from coba.pipes.filters import Encode, JsonDecode, JsonEncode

from coba.pipes.readers import ManikReader, LibsvmReader, CsvReader, ArffReader
from coba.pipes.sources import NullSource, DiskSource, ListSource, QueueSource, HttpSource, LambdaSource, UrlSource
from coba.pipes.sinks   import NullSink, ConsoleSink, DiskSink, ListSink, QueueSink, LambdaSink

from coba.pipes.core import Pipes, Foreach, CsvSource, ArffSource, LibsvmSource, ManikSource, QueueIO

__all__ = [
    "Filter",
    "Source",
    "Sink",
    "PipeMultiprocessor",
    "Pipes",
    "Foreach",
    "CsvSource",
    "ArffSource",
    "LibsvmSource",
    "ManikSource",
    "JsonEncode",
    "JsonDecode",
    "CsvReader",
    "ArffReader",
    "LibsvmReader",
    "ManikReader",
    "UrlSource",
    "NullSource",
    "DiskSource",
    "ListSource",
    "QueueSource",
    "HttpSource",
    "LambdaSource",
    "Encode",
    "Flatten",
    "Default",
    "Drop",
    "Structure",
    "Identity",
    "Take",
    "Reservoir",
    "Shuffle",
    "NullSink",
    "ConsoleSink",
    "DiskSink",
    "ListSink",
    "QueueSink",
    "LambdaSink",
    "QueueIO"
]