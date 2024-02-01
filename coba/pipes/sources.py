import gzip
import zlib

from io import StringIO, BytesIO, TextIOWrapper
from queue import Queue
from urllib import request
from typing import Any, Callable, Iterable, Union, Mapping, Sequence, Tuple, Iterator

from coba.exceptions import CobaException
from coba.primitives import Source, Filter
from coba.utilities  import try_else
from coba.pipes.utilities import resolve_params

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
        return " | ".join(map(str,self._pipes))

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

from http.client import HTTPResponse

class HttpSource(Source[Union[str,Iterable[str]]]):
    """Get content from a web URL."""

    def __init__(self, url: str, chunk_size=None, timeout:float=None) -> None:
        """Instantiate an HttpSource.

        Args:
            url: Location we should get an HTTP response from.
            chunk_size: The number of bytes to read at a time.
            timeout: Seconds to wait for a response before failing.
        """
        self._url        = url
        self._chunk_size = chunk_size
        self._timeout    = timeout

    @staticmethod
    def _byte_io_(encoding:str, charset:str, bites: BytesIO) -> StringIO:
        # In my testing TextIOWrapper is a little slower than
        # codecs.getreader(charset). That said the format of
        # new lines in codecs is inconsistent across platforms
        # making unit testing problematic. Additionally, Python,
        # sees TextIOWrapper as a pure upgrade codecs.getreader.
        # https://peps.python.org/pep-0400/
        bites = gzip.open(bites) if encoding else bites
        return TextIOWrapper(bites,encoding=charset)

    @staticmethod
    def _resp_io_(resp: HTTPResponse) -> StringIO:
        encoding = resp.headers.get('Content-Encoding')
        charset  = next(iter(resp.info().get_charsets()),None) or 'utf-8'
        return HttpSource._byte_io_(encoding,charset,resp)

    @staticmethod
    def _byte_it_(encoding:str, charset:str, chunk:int, bites: BytesIO) -> Union[str,Iterable[str]]:
        # this is faster than the _byte_io_ pattern but we don't have
        # as many guardrails because we're rolling a lot of our own code
        if encoding == 'deflate':
            decomp = zlib.decompressobj(-zlib.MAX_WBITS).decompress
        elif encoding == "gzip":
            decomp = zlib.decompressobj(16+zlib.MAX_WBITS).decompress
        else:
            decomp = lambda x: x

        if not chunk:
            with bites as b:
                return decomp(b.read()).decode(charset)
        else:
            def chunks(decomp,charset,size,bites):
                with bites as b:
                    while chunk := b.read(size):
                        yield decomp(chunk).decode(charset)

            return DelimSource(IterableSource(chunks(decomp,charset,chunk,bites))).read()

    @staticmethod
    def _resp_it_(resp: HTTPResponse, chunk:int = None) -> Union[str,Iterable[str]]:
        encoding = resp.headers.get('Content-Encoding')
        charset  = next(iter(resp.info().get_charsets()),None) or 'utf-8'
        return HttpSource._byte_it_(encoding,charset,chunk,resp)

    def read(self) -> Union[str,Iterable[str]]:
        try:
            req  = request.Request(self._url,headers={'Accept-Encoding':'gzip, deflate'})
            return self._resp_it_(request.urlopen(req, timeout=self._timeout), self._chunk_size)
        except request.HTTPError as e:
            raise request.HTTPError(e.url, e.code, e.msg, e.hdrs, self._resp_io_(e.fp))

class DelimSource(Source[Iterable[str]]):
    """Chunk an Iterable[str] by delimeter."""

    def __init__(self, source: Source[Iterable[str]], delimeter:str = None) -> None:
        self._source = source
        self._delim  = delimeter

    def read(self) -> Iterable[str]:
        pending     = None
        split_lines = not self._delim
        delim       = self._delim

        if split_lines:
            for text in filter(None,self._source.read()):
                lines = text.splitlines()
                if pending:
                    lines[0] = pending + lines[0]
                    pending = None
                if text[-1] not in '\r\n':
                    pending = lines.pop()
                yield from lines
        else:
            for text in filter(None,self._source.read()):
                lines = text.split(delim)
                if pending:
                    lines[0] = pending + lines[0]
                    pending = None
                if text[-1] != delim:
                    pending = lines.pop()
                else:
                    lines.pop() #this makes the delim case behave similar to splitlines

                yield from lines

        if pending is not None:
            yield pending

class IterableSource(Source[Iterable[Any]]):
    """A source that reads from an iterable."""

    def __init__(self, iterable: Iterable[Any]=None):
        """Instantiate an IterableSource.

        Args:
            iterable: The iterable we should read from.
        """
        self.iterable = [] if iterable is None else iterable

    def read(self) -> Iterable[Any]:
        yield from self.iterable

class ListSource(Source[Sequence[Any]]):
    """A source that reads from a list."""

    def __init__(self, sequence: Sequence[Any]=None):
        """Instantiate a ListSource.

        Args:
            list: The sequence to read.
        """
        self._items = [] if sequence is None else sequence

    @property
    def items(self) -> Sequence[Any]:
        return self._items

    def read(self) -> Sequence[Any]:
        return self._items

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
            self._source = HttpSource(url)
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
