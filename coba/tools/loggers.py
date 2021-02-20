"""Basic logging implementation and interface."""

import time
import collections
import traceback

from multiprocessing import current_process
from contextlib import contextmanager
from abc import ABC, abstractmethod
from datetime import datetime
from typing import ContextManager, List, cast, Iterator, Iterable

from coba.data.sinks import Sink

class Logger(ABC):
    """A more advanced logging interface allowing different types of logs to be written."""

    @property
    @abstractmethod
    def sink(self) -> Sink[Iterable[str]]:
        ...

    @abstractmethod
    def log(self, message: str) -> None:
        ...

    @abstractmethod
    def time(self, message: str) -> 'ContextManager[Logger]':
        ...

    @abstractmethod
    def log_exception(self, message: str, exception:Exception) -> None:
        ...

class BasicLogger(Logger):
    """A PowerLogger with context indentation, exception tracking and a consistent preamble."""

    def __init__(self, sink: Sink[Iterable[str]], with_stamp: bool = True, with_name: bool = False):
        """Instantiate a CobaLogger.

        Args:
            print_function: The function that will be called to 'print' any message
                given to the logger.
        """
        self._indent_cnt  = 0
        self._is_newline  = True
        self._sink        = sink
        self._start_times = cast(List[float],[])
        self._with_stamp  = with_stamp
        self._with_name   = with_name
        self._bullets     = collections.defaultdict(lambda: '~', enumerate(['','*','>','-','+']))

    @contextmanager
    def _timing_context(self) -> 'Iterator[BasicLogger]':
        try:
            self._indent_cnt += 1
            self._start_times.append(time.time())

            yield self

            if not self._is_newline: self.log('')
            self.log(f"finished after {round(time.time() - self._start_times[-1], 2)} seconds")

        except KeyboardInterrupt:
            raise

        except Exception as e:
            # we don't want to mask any information so we're not using the formally
            # defined exception chaining syntax (e.g., `raise LoggedException from e`)
            # instead we add our own dunder attribute to indicate that the exception has
            # been logged. This is friendlier when debugging since some Python debuggers
            # don't ever look at the __cause__ attribute set by the explicit syntax
            self.log_exception(f"exception after {round(time.time() - self._start_times[-1], 2)} seconds:", e)
            raise

        finally:
            self._start_times.pop()
            self._indent_cnt -= 1

    @property
    def sink(self) -> Sink[Iterable[str]]:
        return self._sink

    def log(self, message: str) -> None:
        """Log a message.
        
        Args:
            message: The message that should be logged.
        """

        indent = '  ' * self._indent_cnt
        bullet = self._bullets[self._indent_cnt] + (' ' if self._indent_cnt else '')
        stamp  = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' if self._with_stamp else ''
        name   = "-- " + current_process().name + " -- " if self._with_name else ''

        self._sink.write([stamp + name + indent + bullet + message])

    def time(self, message:str) -> 'ContextManager[BasicLogger]':
        """Log a message and the time it takes to exit the returned context manager.
        
        Args:
            message: The message that should be logged.

        Returns:
            A ContextManager that maintains the indentation level of the logger.
            Calling `__enter__` on the manager increases the loggers indentation 
            while calling `__exit__` decreases the logger's indentation.
        """

        self.log(message)
        return self._timing_context()

    def log_exception(self, message:str, ex: Exception) -> None:
        """log an exception if it hasn't already been logged."""

        if not hasattr(ex, '__logged__'):
            setattr(ex, '__logged__', True)
            
            if not self._is_newline: self.log('')

            tb = ''.join(traceback.format_tb(ex.__traceback__))
            msg = ''.join(traceback.TracebackException.from_exception(ex).format_exception_only())

            self.log(f"{message}\n\n{tb}\n  {msg}")