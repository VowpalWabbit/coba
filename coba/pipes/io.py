import requests
import gzip

from collections.abc import Iterator
from queue import Queue
from typing import Iterable, Sequence, TypeVar, Any, Generic, Union

from coba.pipes.primitives import Sink, Source

_T_in  = TypeVar("_T_in", bound=Any, contravariant=True)
_T_out = TypeVar("_T_out", bound=Any, covariant=True)

class IO(Source[_T_out], Sink[_T_in], Generic[_T_out, _T_in]):
    pass

class NullIO(IO[None,Any]):
    def read(self) -> None:
        return None

    def write(self, item: Any) -> None:
        pass

class ConsoleIO(IO[str,Any]):
    
    def read(self) -> str:
        return input()

    def write(self, item: Any) -> None:
        print(item)

class DiskIO(IO[Iterable[str],str]):

    def __init__(self, filename:str, mode:str=None):
        
        #If you are using the gzip functionality of disk sink
        #then you should note that this implementation isn't optimal
        #in terms of compression since it compresses one line at a time.
        #see https://stackoverflow.com/a/18109797/1066291 for more info.

        gzip_open = lambda filename, mode: gzip.open(filename,mode, compresslevel=6)
        text_open = lambda filename, mode: open(filename,mode)

        self._filename   = filename
        self._open_func  = gzip_open if ".gz" in filename else text_open
        self._open_file  = None
        self._open_count = 0
        self._given_mode = mode is not None
        self._mode       = mode

    def __enter__(self) -> 'DiskIO':
        self._open_count += 1
        
        if self._open_file is None:
            self._open_file = self._open_func(self._filename, f"{self._mode}b")
        
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._open_count -= 1
        if self._open_count == 0 and self._open_file is not None:
            self._open_file.close()
            self._open_file = None

    def read(self) -> Iterable[str]: #read all non-destuctive
        
        if not self._given_mode and self._open_count == 0:
            self._mode = 'r+'

        with self:
            #self._open_file.seek(0)
            for line in self._open_file.__enter__():
                yield line.decode('utf-8').rstrip('\n')

    def write(self, item: str) -> None: #write one at a time
        
        if not self._given_mode and self._open_count == 0:
            self._mode = 'a+'

        with self:
            self._open_file.write((item + '\n').encode('utf-8'))
            self._open_file.flush()

class MemoryIO(IO[Iterable[Any], Any]):
    def __init__(self, initial_memory: Sequence[Any] = []):
        self.items =  list(initial_memory)

    def read(self) -> Sequence[Any]: #read all nondestructive
        return self.items

    def write(self, item: Any) -> None: #write one at a time
        self.items.append(list(item) if isinstance(item, Iterator) else item)

class QueueIO(IO[Iterable[Any], Any]):
    
    def __init__(self, queue: Queue = Queue(), poison:Any=None, block:bool=True) -> None:
        self._queue  = queue
        self._poison = poison
        self._block  = block

    def read(self) -> Iterable[Any]: #read all, destructive
        try:
            while self._block or self._queue.qsize() > 0:
                item = self._queue.get()

                if item == self._poison:
                    break

                yield item
        except (EOFError,BrokenPipeError):
            pass

    def write(self, item: Any) -> None: #write one at a time
        try:
            self._queue.put(item)
        except (EOFError,BrokenPipeError):
            pass
        
class HttpIO(IO[requests.Response, Any]):
    def __init__(self, url: str) -> None:
        self._url = url

    def read(self) -> requests.Response:
        return requests.get(self._url, stream=True) #by default sends accept-encoding gzip and deflate

    def write(self, item: Any) -> None:
        raise NotImplementedError()
