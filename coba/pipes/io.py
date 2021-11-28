import requests
import gzip

from itertools import chain
from queue import Queue
from typing import Iterable, Sequence, TypeVar, Any, Generic, Union

from coba.pipes.core import Sink, Source

_T_in  = TypeVar("_T_in", bound=Any, contravariant=True)
_T_out = TypeVar("_T_out", bound=Any, covariant=True)

class IO(Source[_T_out], Sink[_T_in], Generic[_T_out, _T_in]):
    pass

class NullIO(IO[Any,Any]):
    def read(self) -> Any:
        return None

    def write(self, item: Any) -> None:
        pass

class ConsoleIO(IO[str,Union[Any,Iterable[Any]]]):
    def read() -> str:
        return input()

    def write(self, item: Union[Any,Iterable[Any]]) -> None:
        try:
            if isinstance(item,str):
                print(item)
            else:
               for it in item:
                   print(it)
        except TypeError as e:
            if "not iterable" in str(e):
                print(item)
            else:
                raise

class DiskIO(IO[Iterable[str],Union[str,Iterable[str]]]):
    
    def __init__(self, filename:str, append=True):
        
        #If you are using the gzip functionality of disk sink
        #then you should note that this implementation isn't optimal
        #in terms of compression since it compresses one line at a time.
        #see https://stackoverflow.com/a/18109797/1066291 for more info.
        
        gzip_open = lambda filename, mode: gzip.open(filename,mode, compresslevel=6)
        text_open = lambda filename, mode: open(filename,mode)

        self.filename = filename
        self._open    = gzip_open if ".gz" in filename else text_open
        self._append  = append

    def read(self) -> Iterable[str]:
        with self._open(self.filename, "rb+") as f:
            for line in f:
                yield line.decode('utf-8').rstrip('\n')

    def write(self, item: Union[str,Iterable[str]]) -> None:

        items = [item] if isinstance(item,str) else item
        mode  = 'ab+' if self._append else 'wb+'

        try:
            #by peeking at the first item we can prevent the file
            #from being opened/created when there are no items to write
            items      = iter(items)
            first_item = next(items)
            items      = chain([first_item], items)

            if self._append:
                for item in items:
                    with self._open(self.filename, mode) as f:
                        f.write((item + '\n').encode('utf-8'))
                        f.flush()
            else:
                with self._open(self.filename, mode) as f:
                    for item in items:
                        f.write((item + '\n').encode('utf-8'))
                        f.flush()
        
        except StopIteration:
            pass

class MemoryIO(IO[Sequence[Any], Any]):
    def __init__(self, initial_memory: Sequence[Any] = []):
        self.items =  list(initial_memory)

    def read(self) -> Sequence[Any]:
        return self.items

    def write(self, item: Any) -> None:
        try:
            if isinstance(item,str):
                self.items.append(item)
            else:
                for item in item:
                    self.items.append(item)
        except TypeError as e:
            if "not iterable" in str(e):
                self.items.append(item)
            else:
                raise

class QueueIO(IO[Iterable[Any], Union[Any,Iterable[Any]]]):
    
    def __init__(self, queue: Queue = Queue(), poison_pill:Any=None, blocking_get:bool=True) -> None:
        self._queue        = queue
        self._poison_pill  = poison_pill
        self._blocking_get = blocking_get

    def read(self) -> Iterable[Any]:
        try:
            while self._blocking_get or self._queue.qsize() > 0:
                item = self._queue.get()

                if item == self._poison_pill:
                    return

                yield item
        except (EOFError,BrokenPipeError):
            pass

    def write(self, items: Union[Any,Iterable[Any]]) -> None:
        try:
            if isinstance(items,str):
                self._queue.put(items)
            else:
                for item in items: self._queue.put(item)
        except TypeError as e:
            if "not iterable" in str(e):
                self._queue.put(items)
            else:
                raise 
        except (EOFError,BrokenPipeError):
            pass

class HttpIO(IO[requests.Response, Any]):
    def __init__(self, url: str) -> None:
        self._url = url

    def read(self) -> requests.Response:
        return requests.get(self._url, stream=True) #by default sends accept-encoding gzip and deflate

    def write(self, item: Any) -> None:
        requests.put(self._url, item)