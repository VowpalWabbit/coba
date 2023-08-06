from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, TypeVar, Generic, Mapping, Sequence, Iterator, Iterable

_T_out = TypeVar("_T_out", bound=Any, covariant    =True)
_T_in  = TypeVar("_T_in" , bound=Any, contravariant=True)

class Pipe:

    @property
    def params(self) -> Mapping[str,Any]:
        """Parameters describing the pipe."""
        return { }

    def __str__(self) -> str:
        return str(self.params)

class Source(ABC, Pipe, Generic[_T_out]):
    """A pipe that can be read."""

    @abstractmethod
    def read(self) -> _T_out:
        """Read the item."""
        ...

class Filter(ABC, Pipe, Generic[_T_in, _T_out]):
    """A pipe that can modify an item."""

    @abstractmethod
    def filter(self, item: _T_in) -> _T_out:
        """Filter the item."""
        ...

class Sink(ABC, Pipe, Generic[_T_in]):
    """A pipe that writes item."""

    @abstractmethod
    def write(self, item: _T_in) -> None:
        """Write the item."""
        ...

class Line(ABC, Pipe):
    """A pipe that can be run."""

    @abstractmethod
    def run(self):
        """Run the pipe."""
        ...

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
    def __init__(self, *pipes: Source|Filter) -> None:
        if isinstance(pipes[0], SourceFilters):
            self._source = pipes[0]._source
            self._filter = FiltersFilter(*pipes[0]._filter._filters, *pipes[1:])
        else:
            self._source = pipes[0]
            self._filter = FiltersFilter(*pipes[1:])

    @property
    def params(self) -> Mapping[str,Any]:
        return resolve_params(list(self))

    def read(self) -> Any:
        return self._filter.filter(self._source.read())

    def __str__(self) -> str:
        return ",".join(map(str,[self._source, self._filter]))

    def __getitem__(self, index:int) -> Source|Filter:
        if index == 0 or index == -len(self):
            return self._source
        elif index < 0:
            return self._filter[index]
        else:
            return self._filter[index-1]

    def __iter__(self) -> Iterator[Source|Filter]:
        yield self._source
        yield from self._filter

    def __len__(self) -> int:
        return len(self._filter)+1

class FiltersFilter(Filter):

    def __init__(self, *pipes: Filter):
        self._filters = sum([f._filters if isinstance(f, FiltersFilter) else [f] for f in pipes ],[])

    @property
    def params(self) -> Mapping[str,Any]:
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

    def __init__(self, *pipes: Filter|Sink) -> None:

        filters = list(pipes[:-1])
        sink    = pipes[-1 ]

        if isinstance(sink, FiltersSink):
            filters += sink._filter._filters
            sink     = sink._sink

        self._filter = FiltersFilter(*filters)
        self._sink   = sink

    @property
    def params(self) -> Mapping[str,Any]:
        return resolve_params(list(self))

    def write(self, item: Any):
        self._sink.write(self._filter.filter(item))

    def __str__(self) -> str:
        return ",".join(map(str,[self._filter, self._sink]))

    def __getitem__(self, index: int) -> Filter|Sink:
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

class SourceSink(Line):
    def __init__(self, *pipes: Source|Filter|Sink) -> None:
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

    @property
    def params(self) -> Mapping[str, Any]:
        return resolve_params(list(self))

    def __str__(self) -> str:
        return ",".join(filter(None,map(str,self._pipes)))

    def __len__(self) -> int:
        return len(self._pipes)

    def __iter__(self) -> Iterator[Pipe]:
        yield from self._pipes

    def __getitem__(self, index:int) -> Source|Filter|Sink:
        return self._pipes[index]

class Foreach(Filter[Iterable[Any], Iterable[Any]], Sink[Iterable[Any]]):
    """A pipe that wraps an inner pipe and passes items to it one at a time."""

    def __init__(self, pipe: Filter|Sink):
        """Instantiate a Foreach pipe.

        Args:
            pipe: The pipe that we wish to pass a sequence of items one at a time.
        """
        self._pipe = pipe

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        """Filter the items using the inner pipe. This method only works if the inner pipe is a Filter."""
        yield from map(self._pipe.filter,items)

    def write(self, items: Iterable[Any]):
        """Write the items using the inner pipe. This method only works if the inner pipe is a Sink."""
        write = self._pipe.write
        for item in items: 
            write(item)

    @property
    def params(self) -> Mapping[str,Any]:
        return self._pipe.params

    def __str__(self):
        return str(self._pipe)

class StopPipe(BaseException):
    pass
