import gzip

from queue import Queue
from collections.abc import Iterator
from itertools import islice
from typing import Callable, Any, List, Union, Sequence, Iterable

from coba.pipes.primitives import Sink

class NullSink(Sink[Any]):
    """A sink which does nothing with written items."""
    def write(self, item: Any) -> None:
        pass

class ConsoleSink(Sink[Any]):
    """A sink which prints written items to console."""

    def write(self, item: Any) -> None:
        print(item)

class DiskSink(Sink[Union[str,Sequence[str]]]):
    """A sink which writes to a file on disk.

    This sink supports writing in either plain text or as a gz compressed file.
    In order to make this distinction gzip files must end with a gz extension.
    """

    def __init__(self, filename:str, mode:str='a+', batch:int=None) -> None:
        """Instantiate a DiskSink.

        Args:
            filename: The path to the file to write.
            mode: The mode with which the file should be written.
            batch: The number of lines to write before closing and reopening the file.
        """

        #Gzip compression performance is relative to batch size
        #see https://stackoverflow.com/a/18109797/1066291 for more info.

        self._filename = filename
        self._count    = 0
        self._file     = None
        self._mode     = mode
        self._batch    = batch

    def __enter__(self) -> 'DiskSink':
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

    def write(self, lines: Union[str,Iterable[str]]) -> None:
        if isinstance(lines,str):
            lines = [lines]

        lines = iter(lines)
        batch = None

        while self._unfinished(batch):
            batch = self._get_batch(lines)
            with self:
                for line in batch:
                    self._file.write((line + '\n').encode('utf-8'))
                    self._file.flush()

    def _get_batch(self, lines: Iterable[str]) -> Iterable[str]:
        batch = islice(lines,self._batch)
        if self._batch: batch = list(batch)
        return batch

    def _unfinished(self, batch:Any) -> bool:
        not_started  = batch is None
        not_finished = isinstance(batch,list) and len(batch) == self._batch
        return not_started or not_finished

class ListSink(Sink[Any]):
    """A sink which appends written items to a list."""

    def __init__(self, items: List[Any] = None, foreach: bool=False) -> None:
        """Instantiate a ListSink.

        Args:
            items: The list we wish to write to.
        """
        self.items = items if items is not None else []
        self._foreach = foreach

    def write(self, item) -> None:
        item = (item if self._foreach else [item])
        for i in item:
            self.items.append(list(i) if isinstance(i, Iterator) else i)

class QueueSink(Sink[Any]):
    """A sink which puts written items into a Queue."""

    def __init__(self, queue:Queue=None, foreach:bool=False) -> None:
        """Instantiate a QueueSink.

        Args:
            queue: The queue to put written items into.
            foreach: Indicates whether to foreach over given items when writing.
        """
        self._queue   = queue or Queue()
        self._foreach = foreach

    def write(self, item: Any) -> None:
        try:
            item = (item if self._foreach else [item])
            for i in item:
                self._queue.put(i)
        except (EOFError,BrokenPipeError,AssertionError):
            pass

class LambdaSink(Sink[Any]):
    """A sink which passes written items to a callable function."""

    def __init__(self, write: Callable[[Any],None]):
        """Instantiate a LambdaSink.

        Args:
            write: A callable function written items will be passed to.
        """
        self._write = write

    def write(self, item: Any) -> None:
        return self._write(item)
