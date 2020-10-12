"""The data module contains core classes and types for reading and writing data sources."""

from queue import Queue
from threading import Thread
from typing import Optional, IO

class AsyncFileWriter():
    """A file writing class that marshalls all writes to a single thread for thread safety."""

    def __init__(self, *args) -> None:
        """Instantiate an AsyncFileWriter.
        
        Args:
            *args: A sequence of that will be passed to `open` to open a file for writing.
        """
        self._args                         = args
        self._open_file: Optional[IO[str]] = None
        self._sentinel                     = None

    def __enter__(self) -> 'AsyncFileWriter':
        self.open()
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        self.close()

    def open(self) -> None:
        """Open the AsyncFileWriter."""

        if self._open_file is not None:
            raise Exception("The AsyncFileWriter has already been opened.")

        self._open_file = open(*self._args)
        self._queue: 'Queue[Optional[str]]' = Queue()
        Thread(name = "AsyncFileWriter", target=self._internal_write).start() 

    def close(self) -> None:
        """Close the AsyncFileWriter."""

        self._queue.put(self._sentinel)
        
        self.flush()

        if self._open_file is not None:
            self._open_file.close()
            self._open_file = None

    def async_write(self, data:str) -> None:
        """Write data to file asynchronously.
        
        Args:
            data: The data we'd like to write to file.
        """

        if self._open_file is None:
            raise Exception("The AsyncFileWriter is not currently open for writing")

        self._queue.put(data)

    def flush(self) -> None:
        self._queue.join()

    def _internal_write(self):
        while True:
            
            data = self._queue.get(block=True)

            if data == self._sentinel:
                self._queue.task_done()
                break
            
            if data is not None:
                self._open_file.write(data)
            
            self._queue.task_done()
