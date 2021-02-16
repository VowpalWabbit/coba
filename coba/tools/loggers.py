"""Various logger implementations.

TODO Figure out real logging that works on multiple threads
"""

import collections
import time
import traceback

from contextlib import contextmanager
from abc import ABC, abstractmethod
from datetime import datetime
from typing import (Callable, ContextManager, Optional, List, cast, Iterator)

class LogInterface(ABC):
    """The interface for a Logger"""
    
    @abstractmethod
    def log(self, message: str, end:str = None) -> 'ContextManager[LogInterface]':
        ...

    @abstractmethod
    def log_exception(self, exception:Exception, preamble: str = '') -> None:
        ...

class UniversalLog(LogInterface):
    """A simple implementation of the LoggerInterface.
    
    This logger allows for its print_function to be overriden. This logger also supports
    logging levels via a context returned with the log command. All logs that occur within
    that context will be indented and written as sublists.

    """

    def __init__(self, print_function: Callable[[str,Optional[str]],None]):
        """Instantiate a UniversalLogger.

        Args:
            print_function: The function that will be called to 'print' any message
                given to the logger.
        """
        self._indent_cnt  = 0
        self._is_newline  = True
        self._print       = print_function
        self._bullets     = collections.defaultdict(lambda: '~', enumerate(['','*','>','-','+','~'])) 
        self._start_times = cast(List[float],[])

    @contextmanager
    def _with(self) -> Iterator[LogInterface]:
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
            self.log_exception(e, f"exception after {round(time.time() - self._start_times[-1], 2)} seconds:")
            raise

        finally:
            self._start_times.pop()
            self._indent_cnt -= 1

    def _prefix(self) -> str:
        indent = '  ' * self._indent_cnt
        bullet = self._bullets[self._indent_cnt]

        return datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' + indent + bullet + (' ' if bullet != '' else '')

    def log(self, message: str, end: str = None) -> ContextManager[LogInterface]:
        """Log a message.
        
        Args:
            message: The message that should be logged.
            end: The string that should be written at the end of the given message.

        Returns:
            A ContextManager that maintains the indentation level of the logger.
            Calling `__enter__` on the manager increases the indentation the loggers 
            indentation while calling `__exit__` decreases the logger's indentation.
        """
        if self._is_newline:
            message = self._prefix() + message

        self._print(message, end)

        self._is_newline = (end is None or end == '\n')

        return self._with()

    def log_exception(self, ex: Exception, preamble:str = "") -> None:
        """log an exception if it hasn't already been logged."""

        if not hasattr(ex, '__logged__'):
            setattr(ex, '__logged__', True)
            
            if not self._is_newline: self.log('')

            tb = ''.join(traceback.format_tb(ex.__traceback__))
            msg = ''.join(traceback.TracebackException.from_exception(ex).format_exception_only())

            self.log(f"{preamble}\n\n{tb}\n  {msg}")

class ConsoleLog(UniversalLog):
    """An implementation of the UniversalLogger that writes to console."""
    def __init__(self) -> None:
        super().__init__(print_function=lambda m,e: print(m,end=e))

class NoneLog(UniversalLog):
    """An implementation of the UniversalLogger that writes to nowhere."""
    def __init__(self) -> None:
        super().__init__(print_function=lambda m,e: None)

class LoggedException(Exception):
    """An exception that has been logged but not handled."""