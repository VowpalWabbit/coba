import collections.abc
from typing import Sequence, Iterable, Any, overload, Union

from coba.exceptions import CobaException
from coba.pipes.primitives import Filter, Source, Sink

class Foreach(Filter[Iterable[Any], Iterable[Any]], Sink[Iterable[Any]]):

    def __init__(self, pipe: Union[Source[Any],Filter[Any,Any],Sink[Any]], poison=None):
        self._pipe = pipe
        self._poison = poison

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        for item in items:
            yield self._pipe.filter(item)

    def write(self, items: Iterable[Any]):
        for item in items:
            self._pipe.write(item)
    
    def __str__(self):
        return str(self._pipe)

class FiltersFilter(Filter):

    def __init__(self, filters: Sequence[Filter]):

        def flat_filters(filters: Sequence[Filter]) -> Iterable[Filter]:
            for filter in filters:
                if isinstance(filter, FiltersFilter):
                    for filter in flat_filters(filter._filters):
                        yield filter
                else:
                    yield filter

        self._filters = list(flat_filters(filters))

    def filter(self, items: Any) -> Any:
        for filter in self._filters:
            items = filter.filter(items)
        return items

    def __str__(self) -> str:
        return ",".join(map(str,self._filters))

class SourceFilters(Source):
    def __init__(self, source: Source, filters: Sequence[Filter]) -> None:
        
        if isinstance(source, SourceFilters):
            self._source = source._source
            self._filter = FiltersFilter(source._filter._filters + filters)
        else:
            self._source = source
            self._filter = FiltersFilter(filters)

    def read(self) -> Any:
        return self._filter.filter(self._source.read())

    def __str__(self) -> str:
        return ",".join(map(str,[self._source, self._filter]))

class FiltersSink(Sink):
    def __init__(self, filters: Sequence[Filter], sink: Sink) -> None:

        self._filter = FiltersFilter(filters)

        if isinstance(sink, FiltersSink):
            self._filter = FiltersFilter([self._filter, sink._filter])
            self._sink = sink._sink
        else:
            self._sink = sink

    def write(self, items: Iterable[Any]):
        self._sink.write(self._filter.filter(items))

    def __str__(self) -> str:
        return ",".join(map(str,[self._filter, self._sink]))

class Pipeline:
    
    def __init__(self, source: Source, filters: Sequence[Filter], sink: Sink) -> None:
        self._source = source
        self._filter = Pipe.join(filters)
        self._sink   = sink

    def run(self) -> None:
        self._sink.write(self._filter.filter(self._source.read()))

    def __str__(self) -> str:
        return ",".join(filter(None,map(str,[self._source, self._filter, self._sink])))

class Pipe:

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
    def join(source: Source, sink: Sink) -> Pipeline:
        ...

    @overload
    @staticmethod
    def join(source: Source, filters: Sequence[Filter], sink: Sink) -> Pipeline:
        ...

    @overload
    @staticmethod
    def join(filters: Sequence[Filter]) -> Filter:
        ...

    @staticmethod #type: ignore
    def join(*args) -> Union[Source, Filter, Sink, Pipeline]:

        if len(args) == 3:
            return Pipeline(*args)

        if len(args) == 2:
            if isinstance(args[1], collections.abc.Sequence):
                return SourceFilters(args[0], args[1])
            elif isinstance(args[0], collections.abc.Sequence):
                return FiltersSink(args[0], args[1])
            else:
                return Pipeline(args[0], [], args[1])
        
        if len(args) == 1:
            return FiltersFilter(args[0])

        raise CobaException("An unknown pipe was joined.")
