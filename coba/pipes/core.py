from queue import Queue
from typing import Sequence, Iterable, Any, overload, Union, Dict

from coba.exceptions import CobaException

from coba.pipes.sinks      import QueueSink
from coba.pipes.sources    import UrlSource, QueueSource
from coba.pipes.readers    import ArffReader, ManikReader, CsvReader, LibsvmReader
from coba.pipes.primitives import Filter, Source, Sink

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

    @property
    def params(self) -> Dict[str,Any]:
        return { k:v for f in self._filters for k,v in f.params.items() }

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

    @property
    def params(self) -> Dict[str,Any]:
        return { **self._source.params, **self._filter.params }

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

class Pipes:

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
    def join(source: Source, sink: Sink) -> 'Pipes':
        ...

    @overload
    @staticmethod
    def join(source: Source, filters: Sequence[Filter], sink: Sink) -> 'Pipes':
        ...

    @overload
    @staticmethod
    def join(filters: Sequence[Filter]) -> Filter:
        ...

    @staticmethod #type: ignore
    def join(*args) -> Union[Source, Filter, Sink, 'Pipes']:

        if len(args) == 3:
            return Pipes(*args)

        if len(args) == 2:
            if hasattr(args[1], '__len__'):
                return SourceFilters(args[0], args[1])
            elif hasattr(args[0], '__len__'):
                return FiltersSink(args[0], args[1])
            else:
                return Pipes(args[0], [], args[1])

        if len(args) == 1:
            return FiltersFilter(args[0])

        raise CobaException("An unknown pipe was joined.")

    def __init__(self, source: Source, filters: Sequence[Filter], sink: Sink) -> None:
        self._source = source
        self._filter = Pipes.join(filters)
        self._sink   = sink

    def run(self) -> None:
        self._sink.write(self._filter.filter(self._source.read()))

    def __str__(self) -> str:
        return ",".join(filter(None,map(str,[self._source, self._filter, self._sink])))

class Foreach(Filter[Iterable[Any], Iterable[Any]], Sink[Iterable[Any]]):

    def __init__(self, pipe: Union[Source[Any],Filter[Any,Any],Sink[Any]]):
        self._pipe = pipe

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        for item in items:
            yield self._pipe.filter(item)

    def write(self, items: Iterable[Any]):
        for item in items:
            self._pipe.write(item)

    @property
    def params(self) -> Dict[str,Any]:
        return self._pipe.params

    def __str__(self):
        return str(self._pipe)

class CsvSource(SourceFilters):

    def __init__(self, source: Union[str,Source[Iterable[str]]], has_header:bool=False, **dialect) -> None:
        source = UrlSource(source) if isinstance(source,str) else source
        reader = CsvReader(has_header, **dialect)
        super().__init__(source, [reader])

class ArffSource(SourceFilters):

    def __init__(self, 
        source: Union[str,Source[Iterable[str]]], 
        cat_as_str: bool = False, 
        skip_encoding: bool = False, 
        lazy_encoding: bool = True, 
        header_indexing: bool = True) -> None:

        source = UrlSource(source) if isinstance(source,str) else source
        reader = ArffReader(cat_as_str, skip_encoding, lazy_encoding, header_indexing)

        super().__init__(source, [reader])

class LibsvmSource(SourceFilters):

    def __init__(self, source: Union[str,Source[Iterable[str]]]) -> None:
        source = UrlSource(source) if isinstance(source,str) else source
        reader = LibsvmReader()
        super().__init__(source, [reader])

class ManikSource(SourceFilters):

    def __init__(self, source: Union[str,Source[Iterable[str]]]) -> None:
        source = UrlSource(source) if isinstance(source,str) else source
        reader = ManikReader()
        super().__init__(source, [reader])

class QueueIO(Source[Iterable[Any]], Sink[Any]):
    def __init__(self, queue:Queue=None, poison:Any=None, block:bool=True) -> None:
        self._queue  = queue or Queue()
        self._sink   = QueueSink(queue)
        self._source = QueueSource(queue, poison, block)

    def write(self, item: Any) -> None:
        self._sink.write(item)

    def read(self) -> Iterable[Any]:
        return self._source.read()
