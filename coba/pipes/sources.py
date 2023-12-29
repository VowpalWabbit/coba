import requests
import gzip

from queue import Queue
from typing import Any, Callable, Iterable, Union, Mapping, Sequence, Literal, Tuple, Iterator

from coba.exceptions import CobaException
from coba.primitives import Source, Filter, resolve_params
from coba.utilities  import try_else

class SourceFilters(Source):
    def __init__(self, *pipes: Union[Source,Filter]) -> None:
        self._pipes = sum((try_else(lambda: list(p),[p]) for p in pipes),[])

    @property
    def params(self) -> Mapping[str,Any]:
        return resolve_params(list(self))

    def read(self) -> Any:
        item = self._pipes[0].read()
        for filter in self._pipes[1:]:
            item = filter.filter(item)
        return item

    def __str__(self) -> str:
        return ",".join(map(str,self._pipes))

    def __getitem__(self, index:int) -> Union[Source,Filter]:
        return self._pipes[index]

    def __iter__(self) -> Iterator[Union[Source,Filter]]:
        return iter(self._pipes)

    def __len__(self) -> int:
        return len(self._pipes)

class NullSource(Source[Any]):
    """A source which always returns an empty list."""

    def read(self) -> Iterable[Any]:
        return []

class IdentitySource(Source[Any]):
    """A source that reads from an iterable."""

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

class NextSource(Source[Any]):

    def __init__(self, source: Source[Iterable[Any]]) -> None:
        self._source = source

    def read(self) -> Any:
        items = self._source.read()
        try:
            return next(iter(items))
        finally:
            if hasattr(items,'close') and callable(items.close):
                items.close()

class DiskSource(Source[Iterable[str]]):
    """A source that reads a file from disk.

    This source supports reading both plain text files as well gz compressed file.
    In order to make this distinction gzip files must end with a gz extension.
    """

    def __init__(self, path:str, mode:str='rt+', start_loc:int = 0, include_loc: bool = False):
        """Instantiate a DiskSource.

        Args:
            filename: The path to the file to read.
            mode: The mode with which the file should be read.
        """

        self._path        = path
        self._mode        = mode
        self._start_loc   = start_loc
        self._include_loc = include_loc

    def read(self) -> Union[Iterable[str],Iterable[Tuple[int,str]]]:
        #this stackoverflow question raises a potential problem
        #with this implementation https://stackoverflow.com/q/15353220/1066291

        #and this stackoverflow question points out a shortcoming
        #with this implementation #https://stackoverflow.com/a/59168992/1066291

        opener = gzip.open if ".gz" in self._path else open
        with opener(self._path, self._mode) as f:
            f.seek(self._start_loc)
            loc,line = f.tell(), f.readline()
            while line != '':
                line = line.rstrip('\r\n')
                yield (loc,line) if self._include_loc else line
                loc,line = f.tell(), f.readline()

class QueueSource(Source[Iterable[Any]]):
    """A source that reads from a queue."""

    def __init__(self, queue:Queue, block:bool=True, poison:Any=None) -> None:
        """Instantiate a QueueSource.

        Args:
            queue: The queue that should be read.
            block: Indicates if the queue should block when it is empty.
            poison: The poison pill that indicates when to stop blocking (if blocking).
        """
        self._queue    = queue or Queue()
        self._poison   = poison
        self._block    = block
        self._poisoned = False

    def read(self) -> Iterable[Any]:
        try:
            while self._block or self._queue.qsize() > 0:
                item = self._queue.get()

                if self._block and item == self._poison:
                    self._poisoned = True
                    break

                yield item
        except (EOFError,BrokenPipeError,TypeError):
            pass

class HttpSource(Source[Union[requests.Response, Iterable[str]]]):
    """A source that reads from a web URL."""

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
    """A source that reads from an iterable."""

    def __init__(self, iterable: Iterable[Any]=None):
        """Instantiate an IterableSource.

        Args:
            iterable: The iterable we should read from.
        """
        self.iterable = [] if iterable is None else iterable

    def read(self) -> Iterable[Any]:
        return self.iterable

class ListSource(Source[Sequence[Any]]):
    """A source that reads from a list."""

    def __init__(self, sequence: Sequence[Any]=None):
        """Instantiate a ListSource.

        Args:
            list: The sequence we should read from.
        """
        self.items = [] if sequence is None else sequence

    def read(self) -> Iterable[Any]:
        return self.items

class LambdaSource(Source[Any]):
    """A source that reads from a callable method."""

    def __init__(self, read: Callable[[],Any]):
        """Instantiate a LambdaSource.

        Args:
            read: A function to call for a return value when reading.
        """
        self._read = read

    def read(self) -> Iterable[Any]:
        return self._read()

class UrlSource(Source[Iterable[str]]):
    """A source that reads from a url.

    If the given url uses a file scheme or is a local path then
    a DiskSource is used internally.If the given url uses an http
    or https scheme then an HttpSource is used internally.
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

class DataFrameSource(Source[Iterable[Mapping[str,Any]]]):

    def __init__(self, df) -> None:
        self._df = df

    def read(self) -> Iterable[Mapping[str,Any]]:
        "Iterate over DataFrame rows as dictionaries."
        yield from self._df.to_dict(orient='records')
