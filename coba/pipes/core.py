"""The data.pipes module contains core classes for creating data pipelines."""

import collections.abc

from typing import Sequence, Iterable, Any, overload, Union

from coba.pipes.primitives import Filter, Source, Sink, StopPipe
from coba.pipes.filters import Identity 

class Pipe:

    class FiltersFilter(Filter):
        def __init__(self, filters: Sequence[Filter]):
            
            def flat_filters(filters: Sequence[Filter]) -> Iterable[Filter]:
                for filter in filters:
                    if isinstance(filter, Pipe.FiltersFilter):
                        for filter in flat_filters(filter._filters):
                            yield filter
                    else:
                        yield filter

            self._filters = list(flat_filters(filters))

        def filter(self, items: Any) -> Any:
            for filter in self._filters:
                items = filter.filter(items)
            return items
 
        def __repr__(self) -> str:
            return ",".join(map(str,self._filters))

    class SourceFilters(Source):
        def __init__(self, source: Source, filters: Sequence[Filter]) -> None:
            
            if isinstance(source, Pipe.SourceFilters):
                self._source = source._source
                self._filter = Pipe.FiltersFilter(source._filter._filters + filters)
            else:
                self._source = source
                self._filter = Pipe.FiltersFilter(filters)

        def read(self) -> Any:
            return self._filter.filter(self._source.read())

        def __repr__(self) -> str:
            return ",".join(map(str,[self._source, self._filter]))

    class FiltersSink(Sink):
        def __init__(self, filters: Sequence[Filter], sink: Sink) -> None:
            self._filter = Pipe.FiltersFilter(filters)
            self._sink   = sink

        def final_sink(self) -> Sink:
            if isinstance(self._sink, Pipe.FiltersSink):
                return self._sink.final_sink()
            else:
                return self._sink

        def write(self, items: Iterable[Any]):
            self._sink.write(self._filter.filter(items))

        def __repr__(self) -> str:
            return ",".join(map(str,[self._filter, self._sink]))

    @overload
    @staticmethod
    def join(source: Source, *filters: Filter) -> Source:
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
    def join(*args) -> Union[Source, Filter, Sink, 'Pipe']:

        #Source, *Filter
        #Source,  Sink
        #Source, *Filter, Sink
        #*Filter, Sink
        #*Filter

        if len(args) == 3:
            return Pipe(*args)

        if len(args) == 2:
            if isinstance(args[1], collections.abc.Sequence):
                return Pipe.SourceFilters(args[0], args[1])
            elif isinstance(args[0], collections.abc.Sequence):
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

    def run(self) -> None:
        try:
            self._sink.write(Pipe.join(self._filters).filter(self._source.read()))
        except StopPipe:
            pass

    def __repr__(self) -> str:
        return ",".join(map(str,[self._source, *self._filters, self._sink]))
