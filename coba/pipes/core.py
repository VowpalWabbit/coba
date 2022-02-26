from queue import Queue
from typing import Iterable, Any, Union, Dict

from coba.exceptions import CobaException

from coba.pipes.sinks      import QueueSink
from coba.pipes.sources    import UrlSource, QueueSource
from coba.pipes.readers    import ArffReader, ManikReader, CsvReader, LibsvmReader
from coba.pipes.primitives import Filter, Source, Sink

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
        return { **self._source.params, **self._filter.params }

    def read(self) -> Any:
        return self._filter.filter(self._source.read())

    def __str__(self) -> str:
        return ",".join(map(str,[self._source, self._filter]))

class FiltersFilter(Filter):

    def __init__(self, *pipes: Filter):
        self._filters = sum([f._filters if isinstance(f, FiltersFilter) else [f] for f in pipes ],[])

    @property
    def params(self) -> Dict[str,Any]:
        return { k:v for f in self._filters for k,v in f.params.items() }

    def filter(self, items: Any) -> Any:
        for filter in self._filters:
            items = filter.filter(items)
        return items

    def __str__(self) -> str:
        return ",".join(map(str,self._filters))

class FiltersSink(Sink):

    def __init__(self, *pipes: Union[Filter,Sink]) -> None:

        filters = list(pipes[:-1])
        sink    = pipes[-1 ]

        if isinstance(sink, FiltersSink):
            filters += sink._filter._filters
            sink     = sink._sink

        self._filter = FiltersFilter(*filters)
        self._sink   = sink

    def write(self, items: Iterable[Any]):
        self._sink.write(self._filter.filter(items))

    def __str__(self) -> str:
        return ",".join(map(str,[self._filter, self._sink]))

class Pipes:

    class Line:

        def __init__(self, *pipes: Union[Source,Filter,Sink]) -> None:
            self.pipes = list(pipes)

        def run(self) -> None:
            """Run the pipeline."""

            source  = self.pipes[0   ]
            filters = self.pipes[1:-1]
            sink    = self.pipes[-1  ]

            item = source.read()

            for filter in filters:
                item = filter.filter(item)

            sink.write(item)

        @property
        def params(self) -> Dict[str, Any]:
            return { k:v for p in self.pipes for k,v in p.params.items() }

        def __str__(self) -> str:
            return ",".join(filter(None,map(str,self.pipes)))

    @staticmethod
    def join(*pipes: Union[Source, Filter, Sink]) -> Union[Source, Filter, Sink, Line]:

        if len(pipes) == 0:
            raise CobaException("No pipes were passed to join.")

        if len(pipes) == 1 and any(hasattr(pipes[0],attr) for attr in ['read','filter','write']):
            return pipes[0]

        first = pipes[0 ] if not isinstance(pipes[0 ], Foreach) else pipes[0 ]._pipe
        last  = pipes[-1] if not isinstance(pipes[-1], Foreach) else pipes[-1]._pipe

        if hasattr(first,'read') and hasattr(last,'write'):
            return Pipes.Line(*pipes)

        if hasattr(first,'read') and hasattr(last,'filter'):
            return SourceFilters(*pipes)

        if hasattr(first,'filter') and hasattr(last,'filter'):
            return FiltersFilter(*pipes)
        
        if hasattr(first,'filter') and hasattr(last,'write'):
            return FiltersSink(*pipes)

        raise CobaException("An unknown pipe was passed to join.")

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
        super().__init__(source, reader)

class ArffSource(SourceFilters):

    def __init__(self, 
        source: Union[str,Source[Iterable[str]]], 
        cat_as_str: bool = False, 
        skip_encoding: bool = False, 
        lazy_encoding: bool = True, 
        header_indexing: bool = True) -> None:

        source = UrlSource(source) if isinstance(source,str) else source
        reader = ArffReader(cat_as_str, skip_encoding, lazy_encoding, header_indexing)

        super().__init__(source, reader)

class LibsvmSource(SourceFilters):

    def __init__(self, source: Union[str,Source[Iterable[str]]]) -> None:
        source = UrlSource(source) if isinstance(source,str) else source
        reader = LibsvmReader()
        super().__init__(source, reader)

class ManikSource(SourceFilters):

    def __init__(self, source: Union[str,Source[Iterable[str]]]) -> None:
        source = UrlSource(source) if isinstance(source,str) else source
        reader = ManikReader()
        super().__init__(source, reader)

class QueueIO(Source[Iterable[Any]], Sink[Any]):
    def __init__(self, queue:Queue=None, poison:Any=None, block:bool=True) -> None:
        self._queue  = queue or Queue()
        self._sink   = QueueSink(queue)
        self._source = QueueSource(queue, poison, block)

    def write(self, item: Any) -> None:
        self._sink.write(item)

    def read(self) -> Iterable[Any]:
        return self._source.read()
