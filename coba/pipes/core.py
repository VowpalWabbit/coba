from queue import Queue
from collections import Counter
from typing import Iterable, Any, Union, Dict, Callable, Optional, Mapping, Sequence, Iterator

from coba.backports import Literal
from coba.exceptions import CobaException
from coba.primitives import SafeProcess, SafeThread

from coba.pipes.sinks      import QueueSink
from coba.pipes.sources    import QueueSource
from coba.pipes.primitives import Pipe,Filter, Source, Sink

def resolve_params(pipes:Sequence[Pipe]):
    
    params = [p.params for p in pipes if hasattr(p,'params')]
    keys   = [ k for p in params for k in p.keys() ]
    counts = Counter(keys)
    index  = {}

    def resolve_key_conflicts(key):
        if counts[key] == 1:
            return key
        else:
            index[key] = index.get(key,0)+1
            return f"{key}{index[key]}"
    
    return { resolve_key_conflicts(k):v for p in params for k,v in p.items() }

class SourceFilters(Source):
    def __init__(self, *pipes: Union[Source,Filter]) -> None:
        if isinstance(pipes[0], SourceFilters):
            self._source = pipes[0]._source
            self._filter = FiltersFilter(*pipes[0]._filter._filters, *pipes[1:])
        else:
            self._source = pipes[0]
            self._filter = FiltersFilter(*pipes[1:])

    @property
    def params(self) -> Dict[str,Any]:
        return resolve_params(list(self))

    def read(self) -> Any:
        return self._filter.filter(self._source.read())

    def __str__(self) -> str:
        return ",".join(map(str,[self._source, self._filter]))

    def __getitem__(self, index:int) -> Union[Source,Filter]:
        if index == 0 or index == -len(self):
            return self._source
        elif index < 0:
            return self._filter[index]
        else:
            return self._filter[index-1]

    def __iter__(self) -> Iterator[Union[Source,Filter]]:
        yield self._source
        yield from self._filter

    def __len__(self) -> int:
        return len(self._filter)+1

class FiltersFilter(Filter):

    def __init__(self, *pipes: Filter):
        self._filters = sum([f._filters if isinstance(f, FiltersFilter) else [f] for f in pipes ],[])

    @property
    def params(self) -> Dict[str,Any]:
        return resolve_params(list(self))

    def filter(self, items: Any) -> Any:
        for filter in self._filters:
            items = filter.filter(items)
        return items

    def __str__(self) -> str:
        return ",".join(map(str,self._filters))

    def __getitem__(self, index: int) -> Filter:
        return self._filters[index]

    def __iter__(self) -> Iterator[Filter]:
        return iter(self._filters)
    
    def __len__(self) -> int:
        return len(self._filters)

class FiltersSink(Sink):

    def __init__(self, *pipes: Union[Filter,Sink]) -> None:

        filters = list(pipes[:-1])
        sink    = pipes[-1 ]

        if isinstance(sink, FiltersSink):
            filters += sink._filter._filters
            sink     = sink._sink

        self._filter = FiltersFilter(*filters)
        self._sink   = sink

    @property
    def params(self) -> Dict[str,Any]:
        return resolve_params(list(self))

    def write(self, items: Iterable[Any]):
        self._sink.write(self._filter.filter(items))

    def __str__(self) -> str:
        return ",".join(map(str,[self._filter, self._sink]))

    def __getitem__(self, index: int) -> Union[Filter,Sink]:
        if index == -1 or index == len(self._filter):
            return self._sink
        elif index < 0:
            return self._filter[index+1] 
        else:
            return self._filter[index]

    def __iter__(self) -> Iterator[Filter]:
        yield from self._filter
        yield self._sink

    def __len__(self) -> int:
        return len(self._filter)+1

class Pipeline:
    def __init__(self, *pipes: Union[Source,Filter,Sink]) -> None:
        self._pipes = list(pipes)
    
    def run(self) -> None:
        """Run the pipeline."""

        source  = self._pipes[0   ]
        filters = self._pipes[1:-1]
        sink    = self._pipes[-1  ]

        item = source.read()

        for filter in filters:
            item = filter.filter(item)

        sink.write(item)

    def run_async(self, callback:Callable[[Optional[Exception]],None]=None, mode:Literal['process','thread']='process') -> Union[SafeThread,SafeProcess]:
        """Run the pipeline asynchronously."""
        mode = mode.lower()

        if mode == "process":
            worker = SafeProcess(target=self.run, daemon=True, callback=callback)
        elif mode == "thread":
            worker = SafeThread(target=self.run, callback=callback)
        else:
            raise CobaException(f"Unrecognized pipe async mode {mode}. Valid values are 'process' and 'thread'.")

        worker.start()
        return worker

    @property
    def params(self) -> Dict[str, Any]:
        return resolve_params(list(self))

    def __str__(self) -> str:
        return ",".join(filter(None,map(str,self._pipes)))

    def __len__(self) -> int:
        return len(self._pipes)

    def __iter__(self) -> Iterator[Pipe]:
        yield from self._pipes

    def __getitem__(self, index:int) -> Union[Source,Filter,Sink]:
        return self._pipes[index]

class Pipes:

    @staticmethod
    def join(*pipes: Union[Source, Filter, Sink]) -> Union[Source, Filter, Sink, Pipeline]:
        """Join a sequence of pipes into a single pipe.

        Args:
            pipes: a sequence of pipes.

        Returns:
            A single pipe that is a composition of the given pipes. The type of pipe returned
            is determined by the sequence given. A sequence of Filters will return a Filter. A
            sequence that begins with a Source and is followed by Filters will return a Source.
            A sequence that starts with Filters and ends with a Sink will return a Sink. A
            sequence that begins with a Source and ends with a Sink will return a completed pipe.
        """
        if len(pipes) == 0:
            raise CobaException("No pipes were passed to join.")

        if len(pipes) == 1 and any(hasattr(pipes[0],attr) for attr in ['read','filter','write']):
            return pipes[0]

        first = pipes[0 ] if not isinstance(pipes[0 ], Foreach) else pipes[0 ]._pipe
        last  = pipes[-1] if not isinstance(pipes[-1], Foreach) else pipes[-1]._pipe

        if hasattr(first,'read') and hasattr(last,'write'):
            return Pipeline(*pipes)

        if hasattr(first,'read') and hasattr(last,'filter'):
            return SourceFilters(*pipes)

        if hasattr(first,'filter') and hasattr(last,'filter'):
            return FiltersFilter(*pipes)

        if hasattr(first,'filter') and hasattr(last,'write'):
            return FiltersSink(*pipes)

        raise CobaException("An unknown pipe was passed to join.")

class Foreach(Filter[Iterable[Any], Iterable[Any]], Sink[Iterable[Any]]):
    """A pipe that wraps an inner pipe and passes items to it one at a time."""

    def __init__(self, pipe: Union[Filter[Any,Any],Sink[Any]]):
        """Instantiate a Foreach pipe.

        Args:
            pipe: The pipe that we wish to pass a sequence of items one at a time.
        """
        self._pipe = pipe

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        """Filter the items using the inner pipe. This method only works if the inner pipe is a Filter."""
        for item in items:
            yield self._pipe.filter(item)

    def write(self, items: Iterable[Any]):
        """Write the items using the inner pipe. This method only works if the inner pipe is a Sink."""
        for item in items:
            self._pipe.write(item)

    @property
    def params(self) -> Mapping[str,Any]:
        return self._pipe.params

    def __str__(self):
        return str(self._pipe)

class QueueIO(Source[Iterable[Any]], Sink[Any]):
    def __init__(self, queue:Queue, block:bool=True, poison:Any=None) -> None:
        self._queue  = queue
        self._sink   = QueueSink(queue)
        self._source = QueueSource(queue, block, poison)

    def write(self, item: Any) -> None:
        self._sink.write(item)

    def read(self) -> Iterable[Any]:
        return self._source.read()