"""The data module contains core classes and types for reading and writing data sources.

TODO: Add docstrings for Pipe
TODO: Add unit tests for all Pipe

TODO: Add docstrings for all Sinks
TODO: Add unit tests for all Sinks

TODO: Add docstrings for all Sources
TODO: Add unit tests for all Sources
"""

import collections

from multiprocessing import Manager, Process, Pool
from abc import abstractmethod, ABC
from typing import Any, List, Iterable, Sequence, Dict, Hashable, overload

from coba.utilities import check_pandas_support
from coba.json import CobaJsonEncoder, CobaJsonDecoder

class Source(ABC):
    @abstractmethod
    def read(self) -> Iterable[Any]:
        ...

class Sink(ABC):

    @abstractmethod
    def write(self, items:Iterable[Any]) -> None:
        ...

class Filter(ABC):
    @abstractmethod
    def filter(self, items:Iterable[Any]) -> Iterable[Any]:
        ...

class StopPipe(Exception):
    pass

class Pipe:

    class SourceFilters(Source):
        def __init__(self, source: Source, filters: Iterable[Filter]) -> None:
            self._source = source
            self._filters = filters
        
        def read(self) -> Iterable[Any]:
            items = self._source.read()

            for filt in self._filters:
                items = filt.filter(items)

            return items

    class FiltersSink(Sink):
        def __init__(self, filters: Iterable[Filter], sink: Sink) -> None:
            self._filters = filters
            self._sink    = sink

        def write(self, items: Iterable[Any]):
            
            for filt in self._filters:
                items = filt.filter(items)

            self._sink.write(items)

    @overload
    @staticmethod
    def join(source: Source, filters: Iterable[Filter]) -> Source:
        ...
    
    @overload
    @staticmethod
    def join(filters: Iterable[Filter], sink: Sink) -> Sink:
        ...

    @overload
    @staticmethod
    def join(source: Source, sink: Sink) -> 'Pipe':
        ...

    @overload
    @staticmethod
    def join(source: Source, filters: Iterable[Filter], sink: Sink) -> 'Pipe':
        ...

    @staticmethod #type: ignore
    def join(*args):
        if len(args) == 3:
            return Pipe(*args)

        if len(args) == 2:
            if isinstance(args[1], collections.Sequence):
                return Pipe.SourceFilters(args[0], args[1])
            elif isinstance(args[0], collections.Sequence):
                return Pipe.FiltersSink(args[0], args[1])
            else:
                return Pipe(args[0], [], args[1])

        raise Exception("An unknown pipe was joined.")

    def __init__(self, source: Source, filters: Sequence[Filter], sink: Sink) -> None:
        self._source  = source
        self._filters = filters
        self._sink    = sink

    def run(self, max_processes=1) -> None:

        if max_processes == 1:
            try:
                items = self._source.read()

                for filt in self._filters:
                    items = filt.filter(items)

                self._sink.write(items)

            except StopPipe:
                pass
        else:

            if len(self._filters) == 0:
                raise Exception("There was nothing to multi-process within the pipe.")

            with Pool(max_processes) as pool, Manager() as manager:

                out_queue = manager.Queue() #type: ignore

                out_sink   = QueueSink(out_queue)
                out_source = QueueSource(out_queue)

                filters_sink = Pipe.join(self._filters, out_sink)
                remerge_pipe = Pipe.join(out_source, self._sink)

                is_writing_to_memory = isinstance(self._sink, MemorySink) or isinstance(self._sink, Pipe.FiltersSink) and isinstance(self._sink._sink, MemorySink) 

                if is_writing_to_memory:
                    pool.map(filters_sink.write, map(lambda i: [i], self._source.read()))

                    out_queue.put(None)
                    remerge_pipe.run()
                else:
                    remerge_process = Process(target = remerge_pipe.run)

                    remerge_process.start()
                    pool.map(filters_sink.write, map(lambda i: [i], self._source.read()))

                    out_queue.put(None)
                    remerge_process.join()

class JsonEncode(Filter):
    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        encoder = CobaJsonEncoder()
        for item in items: yield encoder.encode(item)

class JsonDecode(Filter):
    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        decoder = CobaJsonDecoder()
        for item in items: yield decoder.decode(item)

class DiskSource(Source):
    def __init__(self, filename:str):
        self.filename = filename

    def read(self) -> Iterable[str]:
        with open(self.filename, "r+") as f:
            for line in f:
                yield line

class DiskSink(Sink):
    def __init__(self, filename:str, mode:str='a+'):
        self.filename = filename
        self._mode    = mode

    def write(self, items: Iterable[str]) -> None:
        with open(self.filename, self._mode) as f:
            for item in items: f.write(item + '\n')

class MemorySource(Source):
    def __init__(self, source: Iterable[Any] = None):
        self._source = source if source is not None else []

    def read(self) -> Iterable[Any]:
        return self._source

class MemorySink(Sink):
    def __init__(self, sink: List[Any] = None):
        self.items = sink if sink is not None else []

    def write(self, items: Iterable[Any]) -> None:
        for item in items:
            self.items.append(item)

class QueueSource(Source):
    def __init__(self, source: Any, poison=None) -> None:
        self._queue  = source
        self._poison = None

    def read(self) -> Iterable[Any]:
        while True:
            item = self._queue.get()

            if item == self._poison:
                raise StopPipe()

            yield item

class QueueSink(Sink):
    def __init__(self, sink: Any) -> None:
        self._queue = sink

    def write(self, items:Iterable[Any]) -> None:
        for item in items: self._queue.put(item)

class Table:
    """A container class for storing tabular data."""

    def __init__(self, name:str, primary: Sequence[str], default=float('nan')):
        """Instantiate a Table.
        
        Args:
            name: The name of the table.
            default: The default values to fill in missing values with
        """
        self._name    = name
        self._primary = primary
        self._columns = list(primary)
        self._default = default

        self.rows: Dict[Hashable, Sequence[Any]] = {}

    def add_row(self, *row, **kwrow) -> None:
        """Add a row of data to the table. The row must contain all primary columns."""

        if kwrow:
            self._columns.extend([col for col in kwrow if col not in self._columns])
            
        row = row + tuple( kwrow.get(col, self._default) for col in self._columns[len(row):] )
        self.rows[row[0] if len(self._primary) == 1 else tuple(row[0:len(self._primary)])] = row

    def get_row(self, key: Hashable) -> Dict[str,Any]:
        row = self.rows[key]
        row = list(row) + [self._default] * (len(self._columns) - len(row))

        return {k:v for k,v in zip(self._columns,row)}

    def rmv_row(self, key: Hashable) -> None:
        self.rows.pop(key, None)

    def get_where(self, **kwrow) -> Iterable[Dict[str,Any]]:
        idx_val = [ (self._columns.index(col), val) for col,val in kwrow.items() ]

        for key,row in self.rows.items():
            if all( row[i]==v for i,v in idx_val):
                yield {k:v for k,v in zip(self._columns,row)}

    def rmv_where(self, **kwrow) -> None:

        idx_val = [ (self._columns.index(col), val) for col,val in kwrow.items() ]
        rmv_keys  = []

        for key,row in self.rows.items():
            if all( row[i]==v for i,v in idx_val):
                rmv_keys.append(key)

        for key in rmv_keys: 
            del self.rows[key] 

    def to_tuples(self) -> Sequence[Any]:
        """Convert a table into a sequence of namedtuples."""
        return list(self.to_indexed_tuples().values())

    def to_indexed_tuples(self) -> Dict[Hashable, Any]:
        """Convert a table into a mapping of keys to tuples."""

        my_type = collections.namedtuple(self._name, self._columns) #type: ignore #mypy doesn't like dynamic named tuples
        my_type.__new__.__defaults__ = (self._default, ) * len(self._columns) #type: ignore #mypy doesn't like dynamic named tuples
        
        return { key:my_type(*value) for key,value in self.rows.items() } #type: ignore #mypy doesn't like dynamic named tuples

    def to_pandas(self) -> Any:
        """Convert a table into a pandas dataframe."""

        check_pandas_support('Table.to_pandas')
        import pandas as pd #type: ignore #mypy complains otherwise

        return pd.DataFrame(self.to_tuples())

    def __contains__(self, primary) -> bool:

        if isinstance(primary, collections.Mapping):
            primary = list(primary.values())[0] if len(self._primary) == 1 else tuple([primary[col] for col in self._primary])

        return primary in self.rows

    def __str__(self) -> str:
        return str({"Table": self._name, "Columns": self._columns, "Rows": len(self.rows)})

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return len(self.rows)