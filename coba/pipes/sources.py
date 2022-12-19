import requests
import gzip
import zipfile

from queue import Queue
from typing import Callable, Iterable, Any, Union, Mapping
from coba.backports import Literal

from coba.exceptions import CobaException
from coba.pipes.primitives import Source

class NullSource(Source[Any]):
    """A source which always returns an empty list."""

    def read(self) -> Iterable[Any]:
        return []

class IdentitySource(Source[Any]):
    """A source which reads from an iterable."""

    def __init__(self, item: Any, params: Mapping[str,Any] = None):
        """Instantiate an IterableSource.

        Args:
            item: The item to return from read.
            params: Teh params descirbing the source.
        """
        self._item = item
        self._params = params or {}

    @property
    def params(self) -> Mapping[str, Any]:
        return self._params

    def read(self) -> Iterable[Any]:
        return self._item

class DiskSource(Source[Iterable[str]]):
    """A source which reads a file from disk.

    This source supports reading both plain text files as well gz compressed file.
    In order to make this distinction gzip files must end with a gz extension.
    """

    def __init__(self, filename:str, mode:str='r+'):
        """Instantiate a DiskSource.

        Args:
            filename: The path to the file to read.
            mode: The mode with which the file should be read.
        """

        self._filename = filename
        self._file     = None
        self._count    = 0
        self._mode     = mode

    def __enter__(self) -> 'DiskSource':
        self._count += 1

        if self._file is None:
            if ".gz" in self._filename:
                self._file = gzip.open(self._filename, f"{self._mode}b", compresslevel=6)
            else:
                self._file = open(self._filename, f"{self._mode}b")

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._count -= 1
        if self._count == 0 and self._file is not None:
            self._file.close()
            self._file = None

    def read(self) -> Iterable[str]:
        with self:
            for line in self._file:
                yield line.decode('utf-8').rstrip('\r\n')

class QueueSource(Source[Iterable[Any]]):
    """A source which reads from a queue."""

    def __init__(self, queue:Queue, block:bool=True, poison:Any=None) -> None:
        """Instantiate a QueueSource.

        Args:
            queue: The queue that should be read.
            block: Indicates if the queue should block when it is empty.
            poison: The poison pill that indicates when to stop blocking (if blocking).
        """
        self._queue  = queue or Queue()
        self._poison = poison
        self._block  = block

    def read(self) -> Iterable[Any]:
        try:
            while self._block or self._queue.qsize() > 0:
                item = self._queue.get()

                if self._block and item == self._poison:
                    break

                yield item
        except (EOFError,BrokenPipeError,TypeError):
            pass

class HttpSource(Source[Union[requests.Response, Iterable[str]]]):
    """A source which reads from a web URL."""

    def __init__(self, url: str, mode: Literal["response","lines"] = "response") -> None:
        """Instantiate an HttpSource.

        Args:
            url: url that we should request an HTTP response from.
            mode: Return the response object if mode=`response` otherwise just return the response's lines.
        """
        self._url = url
        self._mode = mode

    def read(self) -> Union[requests.Response, Iterable[str]]:
        response = requests.get(self._url, stream=True) #by default this includes the header accept-encoding gzip and deflate
        return response if self._mode == "response" else response.iter_lines(decode_unicode=True)

class IterableSource(Source[Iterable[Any]]):
    """A source which reads from an iterable."""

    def __init__(self, iterable: Iterable[Any]=None):
        """Instantiate an IterableSource.

        Args:
            iterable: The iterable we should read from.
        """
        self.iterable = [] if iterable is None else iterable

    def read(self) -> Iterable[Any]:
        return self.iterable

class LambdaSource(Source[Any]):
    """A source which reads from a callable method."""

    def __init__(self, read: Callable[[],Any]):
        """Instantiate a LambdaSource.

        Args:
            read: A function to call for a return value when reading.
        """
        self._read = read

    def read(self) -> Iterable[Any]:
        return self._read()

class UrlSource(Source[Iterable[str]]):
    """A source which reads from a url.

    If the given url uses a file scheme or is a local path then a DiskSource is used internally.
    If the given url uses an http or https scheme then an HttpSource is used internally.
    """

    def __init__(self, url:str) -> None:
        """Instantiate a UrlSource.

        Args:
            url: The url to a resource. Can be either a web request or a local path.
        """
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

class ZipMemberSource(Source[Iterable[bytes]]):
    def __init__(self, path: str, member: str):
        self._path = path
        self._member = member

    def read(self) -> Iterable[bytes]:
        with zipfile.ZipFile(self._path) as zip:
            with zip.open(self._member) as f:
                yield from (l.rstrip(b"\r\n") for l in f.readlines())
