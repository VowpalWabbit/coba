"""This module contains functionality for defining data and workflows within Coba.

This module is primarily intended for internal use within Coba. However, it is documented
and made public for advanced use cases where existing Environment and Experiment creation
functionality is not sufficient. That being said, a good understanding of the patterns
in coba.pipes can help one understand how to best take advantage what Coba has to offer.
"""

from coba.pipes.primitives import Filter, Source, Sink, Foreach, SourceFilters, StopPipe
from coba.pipes.multiprocessing import Multiprocessor

from coba.pipes.filters import Take, Shuffle, Structure, Identity, Flatten, Default, Reservoir
from coba.pipes.filters import Encode, JsonDecode, JsonEncode, Cache, Slice, Insert

from coba.pipes.rows    import LabelRows, EncodeRows, HeadRows, DropRows, EncodeCatRows
from coba.pipes.rows    import LazyDense, EncodeDense, LabelDense, KeepDense, HeadDense
from coba.pipes.rows    import LazySparse, EncodeSparse, LabelSparse, DropSparse
from coba.pipes.readers import ManikReader, LibsvmReader, CsvReader, ArffReader
from coba.pipes.sources import NullSource, IdentitySource, DiskSource, IterableSource
from coba.pipes.sources import QueueSource, HttpSource, LambdaSource, UrlSource, ListSource
from coba.pipes.sinks   import NullSink, ConsoleSink, DiskSink, ListSink, QueueSink, LambdaSink

from coba.pipes.core import Pipes

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
    "IdentitySource",
    "UrlSource",
    "NullSource",
    "DiskSource",
    "ListSource",
    "IterableSource",
    "QueueSource",
    "HttpSource",
    "LambdaSource",
    "Encode",
    "Flatten",
    "Default",
    "Structure",
    "Identity",
    "Take",
    "Slice",
    "Reservoir",
    "Shuffle",
    "Insert",
    "NullSink",
    "ConsoleSink",
    "DiskSink",
    "ListSink",
    "QueueSink",
    "LambdaSink",
    "SourceFilters",
    "Cache",
    "LazyDense",
    "LazySparse",
    "EncodeDense",
    "LabelDense",
    "KeepDense",
    "HeadDense",
    "EncodeSparse",
    "LabelSparse",
    "DropSparse",
    "LabelRows", 
    "EncodeRows", 
    "HeadRows",
    "DropRows",
    "EncodeCatRows",
    "StopPipe",
]