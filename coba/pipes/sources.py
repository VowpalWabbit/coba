import requests
import gzip

from queue import Queue
from typing import Callable, Iterable, Sequence, Any, Union, Dict
from coba.backports import Literal

from coba.exceptions import CobaException
from coba.pipes.primitives import Source

class NullSource(Source[Any]):
    def read(self) -> Iterable[Any]:
        return []

class DiskSource(Source[Iterable[str]]):

    def __init__(self, filename:str, mode:str='r+'):
        
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

    def __enter__(self) -> 'DiskSource':
        self._open_count += 1
        
        if self._open_file is None:
            self._open_file = self._open_func(self._filename, f"{self._mode}b")
        
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._open_count -= 1
        if self._open_count == 0 and self._open_file is not None:
            self._open_file.close()
            self._open_file = None

    def read(self) -> Iterable[str]:
        with self:
            for line in self._open_file.__enter__():
                yield line.decode('utf-8').rstrip('\r\n')

class QueueSource(Source[Iterable[Any]]):

    def __init__(self, queue:Queue, poison:Any=None, block:bool=True) -> None:
        self._queue  = queue or Queue()
        self._poison = poison
        self._block  = block

    def read(self) -> Iterable[Any]:
        try:
            while self._block or self._queue.qsize() > 0:
                item = self._queue.get()

                if item == self._poison:
                    break

                yield item
        except (EOFError,BrokenPipeError):
            pass

class HttpSource(Source[Union[requests.Response, Iterable[str]]]):
    def __init__(self, url: str, mode: Literal["response","lines"] = "response") -> None:
        self._url = url
        self._mode = mode

    def read(self) -> Union[requests.Response, Iterable[str]]:
        response = requests.get(self._url, stream=True) #by default this includes the header accept-encoding gzip and deflate
        return response if self._mode == "response" else response.iter_lines(decode_unicode=True)

class ListSource(Source[Iterable[Any]]):

    def __init__(self, items: Sequence[Any]=None):
        self.items = [] if items is None else items

    def read(self) -> Iterable[Any]:
        for item in self.items:
            yield item

class LambdaSource(Source[Any]):

    def __init__(self, read: Callable[[],Any]):
        self._read = read

    def read(self) -> Iterable[Any]:
        return self._read()

class UrlSource(Source[Iterable[str]]):

    def __init__(self, url:str) -> None:
        self._url = url

        if url.startswith("http://") or url.startswith("https://"):
            self._source = HttpSource(url, mode='lines') 
        elif url.startswith("file://"):
            self._source = DiskSource(url[7:])
        elif "://" not in url:
            self._source = DiskSource(url)
        else:
            raise CobaException("Unrecognized scheme, supported schemes are: http, https or file.")

    def read(self) -> Iterable[str]:
        return self._source.read()
