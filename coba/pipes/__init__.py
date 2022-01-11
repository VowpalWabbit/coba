"""This module contains core functionality for defining data and workflows within Coba.

This module contains core functionality for defining jobs and working with datasets in Coba.
One does not need any understanding of this module to use coba for research. That being said,
a good understanding of the patterns in coba.pipes will help one understand coba and how to
best take advantage what it has to offer. 
"""

from coba.pipes.primitives import Filter, Source, Sink
from coba.pipes.core import Pipe, Foreach, SourceFilters, FiltersFilter, FiltersSink
from coba.pipes.multiprocessing import PipeMultiprocessor

from coba.pipes.filters import Take, Shuffle, Drop, Structure, Identity, Flatten, Default, Reservoir
from coba.pipes.filters import Encode, JsonDecode, JsonEncode

from coba.pipes.readers import Reader, ManikReader, LibSvmReader, CsvReader, ArffReader

from coba.pipes.io import NullIO, ConsoleIO, DiskIO, ListIO, QueueIO, HttpIO, IdentityIO, IO

__all__ = [
    "Pipe",
    "Filter",
    "Source",
    "Sink",
    "Foreach",
    "JsonEncode",
    "JsonDecode",
    "ArffReader",
    "CsvReader",
    "LibSvmReader",
    "ManikReader",
    "Encode",
    "Flatten",
    "Default",
    "Drop",
    "Structure",
    "Identity",
    "Take",
    "Reservoir",
    "Shuffle",
    "PipeMultiprocessor",
    "NullIO",
    "ConsoleIO",
    "DiskIO",
    "ListIO",
    "QueueIO",
    "HttpIO",
    "IdentityIO",
    "Foreach",
    "IO",
    "SourceFilters", 
    "FiltersFilter", 
    "FiltersSink",
    "Reader"
]