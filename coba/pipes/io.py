import gzip

from typing import Iterable, TypeVar, List, Any

import requests

from coba.pipes.core import Sink, Source

_T_in  = TypeVar("_T_in", bound=Any, contravariant=True)
_T_out = TypeVar("_T_out", bound=Any, covariant=True)

class NoneSink(Sink[Iterable[_T_in]]):
    def write(self, items: Iterable[_T_in]) -> None:
        pass

class ConsoleSink(Sink[Iterable[_T_in]]):
    def write(self, items: Iterable[_T_in]) -> None:
        for item in items: print(item)

class DiskSink(Sink[Iterable[str]]):
    def __init__(self, filename:str, mode:str='a+'):
        
        #If you are using the gzip functionality of disk sink
        #then you should note that this implementation isn't optimal
        #in terms of compression since it compresses one line at a time.
        #see https://stackoverflow.com/a/18109797/1066291 for more info.
        
        self.filename = filename
        self._mode    = mode
        self._open    = open if not filename.endswith(".gz") else gzip.open
        self._encode  = (lambda s: s) if not filename.endswith(".gz") else (lambda s: s.encode("utf-8"))

    def write(self, items: Iterable[str]) -> None:
        for item in items: 
            with self._open(self.filename, self._mode) as f:
                f.write(self._encode(item + '\n'))
                f.flush()

class MemorySink(Sink[_T_in]):
    def __init__(self):
        self.items: List[_T_in] = []

    def write(self, items: _T_in) -> None:
        try:
            self.items.extend(items)
        except TypeError as e:
            if "not iterable" not in str(e): raise
            self.items.append(items)

class QueueSink(Sink[Iterable[Any]]):
    def __init__(self, sink: Any) -> None:
        self._queue = sink

    def write(self, items:Iterable[Any]) -> None:
        try:
            for item in items: self._queue.put(item)
        except (EOFError,BrokenPipeError):
            pass

class DiskSource(Source[Iterable[str]]):
    def __init__(self, filename:str):
        self.filename = filename
        self._open    = open if not filename.endswith(".gz") else gzip.open
        self._decode  = (lambda s: s) if not filename.endswith(".gz") else (lambda s: s.decode("utf-8"))

    def read(self) -> Iterable[str]:
        with self._open(self.filename, "r+") as f:
            for line in f:
                yield self._decode(line).rstrip('\n')

class MemorySource(Source[_T_out]):
    def __init__(self, item: _T_out, __repr__: str = None): #type:ignore
        self._item = item

    def read(self) -> _T_out:
        return self._item

    def __repr__(self) -> str:
        return repr(self._item)

class QueueSource(Source[Iterable[Any]]):
    def __init__(self, source: Any, poison=None) -> None:
        self._queue  = source
        self._poison = poison

    def read(self) -> Iterable[Any]:
        try:
            while True:
                item = self._queue.get()

                if item == self._poison:
                    return

                yield item
        except (EOFError,BrokenPipeError):
            pass

class HttpSource(Source[requests.Response]):
    def __init__(self, url: str) -> None:
        self._url = url

    def read(self) -> requests.Response:
        return requests.get(self._url)