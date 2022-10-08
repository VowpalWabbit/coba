"""This module contains functionality for defining data and workflows within Coba.

This module is primarily intended for internal use within Coba. However, it is documented
and made public for advanced use cases where existing Environment and Experiment creation
functionality is not sufficient. That being said, a good understanding of the patterns
in coba.pipes can help one understand how to best take advantage what Coba has to offer.
"""

from coba.pipes.primitives import Filter, Source, Sink
from coba.pipes.multiprocessing import Multiprocessor

from coba.pipes.filters import Take, Shuffle, Drop, Structure, Identity, Flatten, Default, Reservoir
from coba.pipes.filters import Encode, JsonDecode, JsonEncode

from coba.pipes.rows    import DenseRow, SparseRow, IDenseRow, ISparseRow, EncodeRow, DropRow, IndexRow, LabelRow
from coba.pipes.readers import ManikReader, LibsvmReader, CsvReader, ArffReader
from coba.pipes.sources import NullSource, DiskSource, IterableSource, QueueSource, HttpSource, LambdaSource, UrlSource
from coba.pipes.sinks   import NullSink, ConsoleSink, DiskSink, ListSink, QueueSink, LambdaSink

from coba.pipes.core import Pipes, Foreach, QueueIO, SourceFilters

__all__ = [
    "Filter",
    "Source",
    "Sink",
    "Multiprocessor",
    "Pipes",
    "Foreach",
    "JsonEncode",
    "JsonDecode",
    "CsvReader",
    "ArffReader",
    "LibsvmReader",
    "ManikReader",
    "UrlSource",
    "NullSource",
    "DiskSource",
    "IterableSource",
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
    "QueueIO",
    "SourceFilters",
    "DenseRow",
    "SparseRow",
    "IDenseRow",
    "ISparseRow",
    "EncodeRow",
    "DropRow",
    "IndexRow",
    "LabelRow"
]