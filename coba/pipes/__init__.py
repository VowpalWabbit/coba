"""This module contains functionality for defining dataflows and workflows within Coba.

This module is primarily intended for internal use within Coba. However, it is documented
and made public for advanced use cases where existing Environment and Experiment creation
functionality is not sufficient. That being said, a good understanding of the patterns
in coba.pipes can help one understand how to best take advantage what Coba has to offer.
"""

from coba.pipes.multiprocessing import Multiprocessor

from coba.pipes.filters import Take, Shuffle, Structure, Identity, Flatten, Default, Reservoir
from coba.pipes.filters import Encode, Cache, Slice, Insert, FiltersFilter

from coba.pipes.rows    import LabelRows, EncodeRows, HeadRows, DropRows, EncodeCatRows
from coba.pipes.rows    import LazyDense, EncodeDense, LabelDense, KeepDense, HeadDense
from coba.pipes.rows    import LazySparse, EncodeSparse, LabelSparse, DropSparse, SparseDense
from coba.pipes.readers import ManikReader, LibsvmReader, CsvReader, ArffReader
from coba.pipes.sources import NullSource, IdentitySource, DiskSource, IterableSource, DataFrameSource
from coba.pipes.sources import QueueSource, HttpSource, LambdaSource, UrlSource, ListSource, NextSource
from coba.pipes.sources import SourceFilters, DelimSource
from coba.pipes.sinks   import NullSink, ConsoleSink, DiskSink, ListSink, QueueSink, LambdaSink, FiltersSink
from coba.pipes.lines   import SourceSink, ThreadLine, ProcessLine

from coba.pipes.core import Pipes, Foreach
