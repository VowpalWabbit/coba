import gzip

from queue import Queue
from collections.abc import Iterator
from typing import Callable, Any, List

from coba.pipes.primitives import Sink

class NullSink(Sink[Any]):
    def write(self, item: Any) -> None:
        pass

class ConsoleSink(Sink[Any]):
    def write(self, item: Any) -> None:
        print(item)

class DiskSink(Sink[str]):

    def __init__(self, filename:str, mode:str='a+') -> None:
        
        #If you are using the gzip functionality of disk sink
        #then you should note that this implementation isn't optimal
        #in terms of compression since it compresses one line at a time.
        #see https://stackoverflow.com/a/18109797/1066291 for more info.

        self._filename   = filename
        self._open_func  = self._gzip_open if ".gz" in filename else self._text_open
        self._open_file  = None
        self._open_count = 0
        self._given_mode = mode is not None
        self._mode       = mode

    def _gzip_open(self, filename:str, mode:str):
        return gzip.open(filename, mode, compresslevel=6)

    def _text_open(self, filename:str, mode:str):
        return open(filename, mode)

    def __enter__(self) -> 'DiskSink':
        self._open_count += 1

        if self._open_file is None:
            self._open_file = self._open_func(self._filename, f"{self._mode}b")

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._open_count -= 1
        if self._open_count == 0 and self._open_file is not None:
            self._open_file.close()
            self._open_file = None

    def write(self, item: str) -> None:

        with self:
            self._open_file.write((item + '\n').encode('utf-8'))
            self._open_file.flush()

class ListSink(Sink[Any]):

    def __init__(self, items: List[Any] = None) -> None:
        self.items = items if items is not None else []

    def write(self, item) -> None:
        self.items.append(list(item) if isinstance(item, Iterator) else item)

class QueueSink(Sink[Any]):

    def __init__(self, queue:Queue=None) -> None:
        self._queue  = queue or Queue()

    def write(self, item: Any) -> None:
        try:
            self._queue.put(item)
        except (EOFError,BrokenPipeError):
            pass

class LambdaSink(Sink[Any]):

    def __init__(self, write: Callable[[Any],None]):
        self._write = write

    def write(self, item: Any) -> None:
        return self._write(item)
