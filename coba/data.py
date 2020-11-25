"""The data module contains core classes and types for reading and writing data sources.

TODO: Add docstrings for Pipe

TODO: Add docstrings for all Sinks
TODO: Add unit tests for all Sinks

TODO: Add docstrings for all Sources
TODO: Add unit tests for all Sources
"""

import collections

from hashlib import md5
from multiprocessing import Manager, Pool
from threading import Thread
from abc import abstractmethod, ABC
from typing import Any, List, Iterable, Sequence, Dict, Hashable, TypeVar, overload, Generic

import requests

from coba.execution import ExecutionContext, UniversalLogger
from coba.utilities import check_pandas_support
from coba.json import CobaJsonEncoder, CobaJsonDecoder

_T_out = TypeVar("_T_out", bound=Any, covariant=True)
_T_in  = TypeVar("_T_in", bound=Any, contravariant=True)

class Source(ABC, Generic[_T_out]):
    @abstractmethod
    def read(self) -> _T_out:
        ...

class Sink(ABC, Generic[_T_in]):

    @abstractmethod
    def write(self, items: _T_in) -> None:
        ...

class Filter(ABC, Generic[_T_in, _T_out]):
    @abstractmethod
    def filter(self, items:_T_in) -> _T_out:
        ...

class StopPipe(Exception):
    pass

class Pipe:

    class SourceFilters(Source):
        def __init__(self, source: Source, filters: Sequence[Filter]) -> None:
            self._source = source
            self._filters = filters
        
        def read(self) -> Any:
            items = self._source.read()

            for filt in self._filters:
                items = filt.filter(items)

            return items

    class FiltersSink(Sink):
        def __init__(self, filters: Sequence[Filter], sink: Sink) -> None:
            self._filters = filters
            self._sink    = sink

        def final_sink(self) -> Sink:
            if isinstance(self._sink, Pipe.FiltersSink):
                return self._sink.final_sink()
            else:
                return self._sink

        def write(self, items: Iterable[Any]):
            for filt in self._filters:
                items = filt.filter(items)
            self._sink.write(items)

    class FiltersFilter(Filter):
        def __init__(self, filters: Sequence[Filter]):
            self._filters = filters

        def filter(self, items: Any) -> Any:
            for filter in self._filters:
                items = filter.filter(items)

            return items

    @overload
    @staticmethod
    def join(source: Source, filters: Sequence[Filter]) -> Source:
        ...
    
    @overload
    @staticmethod
    def join(filters: Sequence[Filter], sink: Sink) -> Sink:
        ...

    @overload
    @staticmethod
    def join(source: Source, sink: Sink) -> 'Pipe':
        ...

    @overload
    @staticmethod
    def join(source: Source, filters: Sequence[Filter], sink: Sink) -> 'Pipe':
        ...

    @overload
    @staticmethod
    def join(filters: Sequence[Filter]) -> Filter:
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
        
        if len(args) == 1:
            return Pipe.FiltersFilter(args[0])

        raise Exception("An unknown pipe was joined.")

    def __init__(self, source: Source, filters: Sequence[Filter], sink: Sink) -> None:
        self._source  = source
        self._filters = filters
        self._sink    = sink

    def run(self, processes: int = 1, maxtasksperchild=None) -> None:
        try:
            if processes == 1 and maxtasksperchild is None:
                filter = Pipe.join(self._filters)
            else:
                filter = MultiProcessFilter(self._filters, processes, maxtasksperchild)

            self._sink.write(filter.filter(self._source.read()))
        except StopPipe as e:
            pass

class JsonEncode(Filter[Iterable[Any], Iterable[str]]):
    def filter(self, items: Iterable[Any]) -> Iterable[str]:
        encoder = CobaJsonEncoder()
        for item in items: yield encoder.encode(item)

class JsonDecode(Filter[Iterable[str], Iterable[Any]]):
    def filter(self, items: Iterable[str]) -> Iterable[Any]:
        decoder = CobaJsonDecoder()
        for item in items: yield decoder.decode(item)

class DiskSource(Source[Iterable[str]]):
    def __init__(self, filename:str):
        self.filename = filename

    def read(self) -> Iterable[str]:
        with open(self.filename, "r+") as f:
            for line in f:
                yield line

class DiskSink(Sink[Iterable[str]]):
    def __init__(self, filename:str, mode:str='a+'):
        self.filename = filename
        self._mode    = mode

    def write(self, items: Iterable[str]) -> None:
        with open(self.filename, self._mode) as f:
            for item in items: f.write(item + '\n')

class MemorySource(Source[_T_out]):
    def __init__(self, source: _T_out): #type:ignore
        self._source = source

    def read(self) -> _T_out:
        return self._source

class MemorySink(Sink[_T_in]):
    def __init__(self):
        self.items: List[_T_in] = []

    def write(self, items: _T_in) -> None:
        try:
            self.items.extend(items)
        except TypeError as e:
            if "not iterable" not in str(e): raise
            self.items.append(items)

class QueueSource(Source[Iterable[Any]]):
    def __init__(self, source: Any, poison=None) -> None:
        self._queue  = source
        self._poison = poison

    def read(self) -> Iterable[Any]:
        while True:
            item = self._queue.get()

            if item == self._poison:
                return

            yield item

class QueueSink(Sink[Iterable[Any]]):
    def __init__(self, sink: Any) -> None:
        self._queue = sink

    def write(self, items:Iterable[Any]) -> None:
        for item in items: self._queue.put(item)

class SinkLogger(UniversalLogger):
    def __init__(self, sink: Sink) -> None:
        super().__init__(lambda msg,end: sink.write([(msg[20:],end)]))

class LoggerSink(Sink[Iterable[Any]]):
    def write(self, items: Iterable[Any]) -> None:
        for msg,end in items:
            ExecutionContext.Logger.log(msg,end)

class HttpSource(Source[Iterable[str]]):
    def __init__(self, url: str, file_extension: str = None, checksum: str = None, desc: str = "") -> None:
        self._url       = url
        self._checksum  = checksum
        self._desc      = desc
        self._cachename = f"{md5(self._url.encode('utf-8')).hexdigest()}{file_extension}"

    def read(self) -> Iterable[str]:
        bites = self._get_bytes()

        if self._checksum is not None and md5(bites).hexdigest() != self._checksum:
            message = (
                f"The dataset at {self._url} did not match the expected checksum. This could be the result of "
                "network errors or the file becoming corrupted. Please consider downloading the file again "
                "and if the error persists you may want to manually download and reference the file.")
            raise Exception(message) from None

        if self._cachename not in ExecutionContext.FileCache: ExecutionContext.FileCache.put(self._cachename, bites)

        return bites.decode('utf-8').splitlines()
    
    def _get_bytes(self) -> bytes:
        if self._cachename in ExecutionContext.FileCache:
            with ExecutionContext.Logger.log(f'loading {self._desc} from cache... '.replace('  ', ' ')):
                return ExecutionContext.FileCache.get(self._cachename)
        else:
            with ExecutionContext.Logger.log(f'loading {self._desc} from http... '):
                response = requests.get(self._url)

                if response.status_code == 412 and 'openml' in self._url:
                    if 'please provide api key' in response.text:
                        message = (
                            "An API Key is needed to access openml's rest API. A key can be obtained by creating an "
                            "openml account at openml.org. Once a key has been obtained it should be placed within "
                            "~/.coba as { \"openml_api_key\" : \"<your key here>\", }.")
                        raise Exception(message) from None

                    if 'authentication failed' in response.text:
                        message = (
                            "The API Key you provided no longer seems to be valid. You may need to create a new one"
                            "longing into your openml account and regenerating a key. After regenerating the new key "
                            "should be placed in ~/.coba as { \"openml_api_key\" : \"<your key here>\", }.")
                        raise Exception(message) from None

                return response.content

class MultiProcessFilter(Filter):

    class Processor:

        def __init__(self, filters: Sequence[Filter], stdout: Sink, stderr: Sink, stdlog:Sink) -> None:
            self._filter = Pipe.join(filters)
            self._stdout = stdout
            self._stderr = stderr
            self._stdlog = stdlog

        def process(self, item) -> None:

            ExecutionContext.Config.processes        = 1
            ExecutionContext.Config.maxtasksperchild = None
            ExecutionContext.Logger = SinkLogger(self._stdlog)

            try:
                self._stdout.write(self._filter.filter([item]))

            except Exception as e:
                self._stderr.write([e])
                raise

            except KeyboardInterrupt:
                # if you are here because keyboard interrupt isn't working for multiprocessing
                # or you want to improve it in some way I can only say good luck. After many hours
                # of spelunking through stackoverflow and the python stdlib I still don't understand
                # why it works the way it does. I arrived at this solution not based on understanding
                # but based on experimentation. This seemed to fix my problem. As best I can tell 
                # KeyboardInterrupt is a very special kind of exception that propogates up everywhere
                # and all we have to do in our child processes is make sure they don't become zombified.
                pass

    def __init__(self, filters: Sequence[Filter], processes=1, maxtasksperchild=None) -> None:
        self._filters          = filters
        self._processes        = processes
        self._maxtasksperchild = maxtasksperchild

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:

        # It does seem like this could potentially be made faster...
        # I'm not sure how or why, but my original thread implementation
        # within Pool seemed to complete the full job about a minute and
        # a half faster... See commit 7fb3653 for that implementation.
        # My best guess is that 7fb3653 doesn't rely on a generator.

        if len(self._filters) == 0:
            return items

        with Pool(self._processes, maxtasksperchild=self._maxtasksperchild) as pool, Manager() as manager:

            std_queue = manager.Queue() #type: ignore
            err_queue = manager.Queue() #type: ignore
            log_queue = manager.Queue() #type: ignore

            stdout_writer, stdout_reader = QueueSink(std_queue), QueueSource(std_queue)
            stderr_writer, stderr_reader = QueueSink(err_queue), QueueSource(err_queue)
            stdlog_writer, stdlog_reader = QueueSink(log_queue), QueueSource(log_queue)

            log_thread = Thread(target=Pipe.join(stdlog_reader, [], LoggerSink()).run)
            processor  = MultiProcessFilter.Processor(self._filters, stdout_writer, stderr_writer, stdlog_writer)

            def finished_callback(result):
                std_queue.put(None)
                err_queue.put(None)
                log_queue.put(None)

            def error_callback(error):
                std_queue.put(None) #not perfect but I'm struggling to think of a better way. May result in lost work.
                err_queue.put(None) #not perfect but I'm struggling to think of a better way. May result in lost work.
                log_queue.put(None) #not perfect but I'm struggling to think of a better way. May result in lost work.
            
            log_thread.start()

            result = pool.map_async(processor.process, items, callback=finished_callback, error_callback=error_callback) 

            # When items is empty finished_callback will not be called and we'll get stuck waiting for the poison pill.
            # When items is empty ready() will be true immediately and this check will place the poison pill into the queues.
            if result.ready():
                std_queue.put(None)
                err_queue.put(None)
                log_queue.put(None)

            #this structure is necessary to make sure we don't exit the context before we're done
            for item in stdout_reader.read():
                yield item

            # if an error occured within map_async this will cause it to re-throw 
            # in the main thread allowing us to capture it and handle it appropriately 
            try:
                result.get()
            except Exception as e:
                if "Can't pickle" in str(e) or "Pickling" in str(e):
                    message = (
                            "Learners are required to be picklable in order to evaluate a Benchmark in multiple processes. "
                            "To help with this learner's have an optional `def init(self) -> None` that is only called "
                            "after pickling has occured. Any non-picklable objects can be created within `init()` safely.")
                    raise Exception(message) from e


            # in the case where an exception occurred in one of our processes
            # we will have poisoned the std and err queue even though the pool
            # isn't finished yet, so we need to kill it here. We are unable to
            # kill it in the error_callback because that will cause a hang.
            pool.terminate()
            log_thread.join()

            for err in stderr_reader.read():
                if not isinstance(err, StopPipe):
                    raise err

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

        row = row + tuple( kwrow.get(col, self._default) for col in self._columns[len(row):] ) #type:ignore
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