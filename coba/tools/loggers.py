"""Basic logging implementation and interface."""

import time
import collections
import traceback

from multiprocessing import current_process
from contextlib import contextmanager
from abc import ABC, abstractmethod
from datetime import datetime
from typing import ContextManager, List, cast, Iterator, Iterable, Dict

from coba.data.sinks import Sink, NoneSink

class Logger(ABC):
    """A more advanced logging interface allowing different types of logs to be written."""

    @property
    @abstractmethod
    def sink(self) -> Sink[Iterable[str]]:
        ...

    @abstractmethod
    def log(self, message: str) -> 'ContextManager[Logger]':
        ...

    @abstractmethod
    def time(self, message: str) -> 'ContextManager[Logger]':
        ...

    @abstractmethod
    def log_exception(self, message: str, exception:Exception) -> None:
        ...

class NoneLogger(Logger):

    @contextmanager
    def _context(self) -> 'Iterator[Logger]':
        yield self

    @property
    def sink(self) -> Sink[Iterable[str]]:
        return NoneSink()

    def log(self, message: str) -> 'ContextManager[Logger]':
        return self._context()

    def time(self, message: str) -> 'ContextManager[Logger]':
        return self._context()

    def log_exception(self, message: str, exception: Exception) -> None:
        pass

class BasicLogger(Logger):
    """A Logger that writes in real time and indicates time with start/end messages."""

    def __init__(self, sink: Sink[Iterable[str]], with_stamp: bool = True, with_name: bool = False):
        """Instantiate a BasicLogger."""
        self._sink        = sink
        self._with_stamp  = with_stamp
        self._with_name   = with_name

        self._starts = cast(List[float], [])

        self._indent_lvl  = 0
        self._bullets     = collections.defaultdict(lambda: '~', enumerate(['','*','>','-','+']))

    @contextmanager
    def _log_context(self, message:str) -> 'Iterator[Logger]':
        yield self
        self.log(message + " (finish)")
    
    @contextmanager
    def _time_context(self, message:str) -> 'Iterator[Logger]':
        self.log(message)
        self._starts.append(time.time())
        yield self
        self.log(message + f" ({round(time.time()-self._starts.pop(),2)} seconds)")
        
    @property
    def sink(self) -> Sink[Iterable[str]]:
        return self._sink

    def log(self, message: str) -> 'ContextManager[Logger]':
        """Log a message with an optional begin and end context.
        
        Args:
            message: The message that should be logged.

        Returns:
            A ContextManager that while write a finish message on exit
        """

        indent = '  ' * self._indent_lvl
        bullet = self._bullets[self._indent_lvl] + (' ' if self._indent_lvl else '')
        stamp  = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' if self._with_stamp else ''
        name   = "-- " + current_process().name + " -- " if self._with_name else ''

        self._sink.write([stamp + name + indent + bullet + message])
        
        return self._log_context(message)

    def time(self, message:str) -> 'ContextManager[Logger]':
        """Log a message's start and end time.

        Args:
            message: The message that should be logged.

        Returns:
            A ContextManager that while write a finish message on exit
        """

        return self._time_context(message)

    def log_exception(self, message:str, ex: Exception) -> None:
        """log an exception if it hasn't already been logged."""

        # we don't want to mask any information so we're not using the formally
        # defined exception chaining syntax (e.g., `raise LoggedException from e`)
        # instead we add our own dunder attribute to indicate that the exception has
        # been logged. This is friendlier when debugging since some Python debuggers
        # don't ever look at the __cause__ attribute set by the explicit syntax

        if not hasattr(ex, '__logged__'):
            setattr(ex, '__logged__', True)

            tb = ''.join(traceback.format_tb(ex.__traceback__))
            msg = ''.join(traceback.TracebackException.from_exception(ex).format_exception_only())

            self.log(f"{message}\n\n{tb}\n  {msg}")

class IndentLogger(Logger):
    """A Logger with context indentation, exception tracking and a consistent preamble."""

    def __init__(self, sink: Sink[Iterable[str]], with_stamp: bool = True, with_name: bool = False):
        """Instantiate an IndentLogger."""
        self._sink        = sink
        self._with_stamp  = with_stamp
        self._with_name   = with_name

        self._messages  = cast(List[str],[])
        self._starts    = cast(List[float],[])
        self._indexes   = cast(List[int],[])
        self._durations = cast(Dict[int,float],{})

        self._indent_lvl = 0
        self._bullets    = collections.defaultdict(lambda: '~', enumerate(['','*','>','-','+']))

    @property
    def _in_time_context(self) -> bool:
        return len(self._starts) > 0

    def _unwind_time_context(self) -> None:
        for index, message in enumerate(self._messages):
            time = f' ({self._durations[index]} seconds)' if index in self._durations else ''            
            
            self._sink.write([f"{message}{time}"])

        self._messages = []
        self._durations.clear()

    @contextmanager
    def _indent_context(self) -> 'Iterator[Logger]':
        try:
            self._indent_lvl += 1
            yield self
        finally:
            self._indent_lvl -= 1

    @contextmanager
    def _time_context(self, message:str) -> 'Iterator[Logger]':

        self._starts.append(time.time())
        self._indexes.append(len(self._messages))
        self.log(message)

        with self._indent_context():
            try:
                yield self
                self._durations[self._indexes[-1]] = round(time.time() - self._starts[-1],2)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                self.log_exception(f"Exception after {round(time.time() - self._starts[-1], 2)} seconds:", e)
                raise
            finally:
                self._starts.pop()
                self._indexes.pop()

                if len(self._starts) == 0:
                    self._unwind_time_context()

    @property
    def sink(self) -> Sink[Iterable[str]]:
        return self._sink

    def log(self, message: str) -> 'ContextManager[Logger]':
        """Log a message.
        
        Args:
            message: The message that should be logged.
        """

        indent = '  ' * self._indent_lvl
        bullet = self._bullets[self._indent_lvl] + (' ' if self._indent_lvl else '')
        stamp  = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' if self._with_stamp else ''
        name   = "-- " + current_process().name + " -- " if self._with_name else ''

        final_message = stamp + name + indent + bullet + message

        if self._in_time_context:
            self._messages.append(final_message)
        else:            
            self._sink.write([final_message])

        return self._indent_context()

    def time(self, message:str) -> 'ContextManager[Logger]':
        """Log a message and the time it takes to exit the returned context manager.
        
        Args:
            message: The message that should be logged.

        Returns:
            A ContextManager that maintains the indentation level of the logger.
            Calling `__enter__` on the manager increases the loggers indentation 
            while calling `__exit__` decreases the logger's indentation.
        """

        return self._time_context(message)

    def log_exception(self, message:str, ex: Exception) -> None:
        """log an exception if it hasn't already been logged."""

        # we don't want to mask any information so we're not using the formally
        # defined exception chaining syntax (e.g., `raise LoggedException from e`)
        # instead we add our own dunder attribute to indicate that the exception has
        # been logged. This is friendlier when debugging since some Python debuggers
        # don't ever look at the __cause__ attribute set by the explicit syntax

        if not hasattr(ex, '__logged__'):
            setattr(ex, '__logged__', True)

            tb = ''.join(traceback.format_tb(ex.__traceback__))
            msg = ''.join(traceback.TracebackException.from_exception(ex).format_exception_only())

            self.log(f"{message}\n\n{tb}\n  {msg}")