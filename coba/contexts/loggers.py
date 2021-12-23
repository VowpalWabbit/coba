"""Basic logging implementation and interface."""

import time
import traceback

from abc import abstractmethod, ABC
from multiprocessing import current_process
from contextlib import contextmanager
from datetime import datetime

from coba.typing import ContextManager, Iterator, Sequence, Union
from coba.pipes import Pipe, Filter, Sink, NullIO, ConsoleIO
from coba.exceptions import CobaException

class Logger(ABC):
    """A more advanced logging interface allowing different types of logs to be written."""

    @property
    @abstractmethod
    def sink(self) -> Sink[str]:
        ...

    @sink.setter
    @abstractmethod
    def sink(self, sink: Sink[str]):
        ...

    @abstractmethod
    def log(self, message: Union[str,Exception]) -> 'ContextManager[Logger]':
        ...

    @abstractmethod
    def time(self, message: str) -> 'ContextManager[Logger]':
        ...

class NullLogger(Logger):

    def __init__(self) -> None:
        self._sink = NullIO()

    @contextmanager
    def _context(self) -> 'Iterator[Logger]':
        yield self

    @property
    def sink(self) -> Sink[str]:
        return self._sink

    @sink.setter
    def sink(self, sink: Sink[str]):
        self._sink = sink

    def log(self, message: Union[str,Exception]) -> 'ContextManager[Logger]':
        return self._context()

    def time(self, message: str) -> 'ContextManager[Logger]':
        return self._context()

class BasicLogger(Logger):
    """A Logger that writes in real time and indicates time with start/end messages."""

    def __init__(self, sink: Sink[str] = ConsoleIO()):
        """Instantiate a BasicLogger."""
        self._sink = sink
        self._starts = []

    @contextmanager
    def _log_context(self, message:str) -> 'Iterator[Logger]':
        try:
            yield self
        except KeyboardInterrupt:
            self.log(message + " (interrupt)")
            raise
        except Exception as e:
            self.log(message + " (exception)")
            raise
        else:
            self.log(message + " (completed)")
    
    @contextmanager
    def _time_context(self, message:str) -> 'Iterator[Logger]':
        self.log(message)
        self._starts.append(time.time())
        try:
            yield self
        except KeyboardInterrupt:
            self.log(message + f" ({round(time.time()-self._starts.pop(),2)} seconds) (interrupt)")
            raise
        except Exception:
            self.log(message + f" ({round(time.time()-self._starts.pop(),2)} seconds) (exception)")
            raise
        else:
            self.log(message + f" ({round(time.time()-self._starts.pop(),2)} seconds) (completed)")
        
    @property
    def sink(self) -> Sink[str]:
        return self._sink

    @sink.setter
    def sink(self, sink: Sink[str]):
        self._sink = sink

    def log(self, message: str) -> 'ContextManager[Logger]':
        """Log a message with an optional begin and end context.
        
        Args:
            message: The message or exception that should be logged.

        Returns:
            A ContextManager that will write a finish message on exit.
        """

        self._sink.write(message)
        return self._log_context(message)

    def time(self, message: str) -> 'ContextManager[Logger]':
        """Log a message's start and end time.

        Args:
            message: The message that should be logged to describe what is timed.

        Returns:
            A ContextManager that will write the total execution time on exit.
        """

        return self._time_context(message)

class IndentLogger(Logger):
    """A Logger with context indentation, exception tracking and a consistent preamble."""

    def __init__(self, sink: Sink[str] = ConsoleIO()):
        """Instantiate an IndentLogger."""
        self._sink = sink

        self._messages = []
        self._level    = 0
        self._bullets  = dict(enumerate(['','* ','> ','- ','+ ']))

    @contextmanager
    def _indent_context(self) -> 'Iterator[Logger]':
        try:
            self._level += 1
            yield self
        finally:
            self._level -= 1

    @contextmanager
    def _time_context(self, message:str) -> 'Iterator[Logger]':

        # we don't have all the information we need to write 
        # our message but we want to save our place in line
        # we also level our message before entering context
        place_in_line = len(self._messages)
        self._messages.append("Placeholder")
        message = self._level_message(message)

        with self._indent_context():
            try:
                start = time.time()
                yield self
                outcome = "(completed)"
            except KeyboardInterrupt:
                outcome = "(interrupt)"
                raise
            except Exception:
                outcome = "(exception)"
                raise
            finally:

                self._messages[place_in_line] = message + f" ({round(time.time()-start,2)} seconds) {outcome}"

                if place_in_line == 0:
                    while self._messages:
                        self._sink.write(self._messages.pop(0))

    def _level_message(self, message: str) -> str:
        indent = '  ' * self._level
        bullet = self._bullets.get(self._level,'~')

        return indent + bullet + message

    @property
    def sink(self) -> Sink[str]:
        return self._sink

    @sink.setter
    def sink(self, sink: Sink[str]):
        self._sink = sink

    def log(self, message: Union[str,Exception]) -> 'ContextManager[Logger]':
        """Log a message.

        Args:
            message: The message that should be logged.
        """

        if self._messages:
            self._messages.append(self._level_message(message))
        else:
            self._sink.write(self._level_message(message))

        return self._indent_context()

    def time(self, message: str) -> 'ContextManager[Logger]':
        """Log a message and the time it takes to exit the returned context manager.
        
        Args:
            message: The message that should be logged to describe what is timed.

        Returns:
            A ContextManager that maintains the timing context and writes execution time on exit. 
        """

        return self._time_context(message)

class DecoratedLogger(Logger):

    def __init__(self, pre_decorators: Sequence[Filter], logger: Logger, post_decorators: Sequence[Filter]):

        self._pre_decorator   = Pipe.join(pre_decorators)
        self._post_decorators = post_decorators
        self._logger          = logger
        self._original_sink   = self._logger.sink
        self._logger.sink     = Pipe.join(post_decorators, self._original_sink)

    @property
    def sink(self) -> Sink[str]:
        return self._original_sink

    @sink.setter
    def sink(self, sink: Sink[str]):
        self._original_sink = sink
        self._logger.sink   = Pipe.join(self._post_decorators, sink)

    def log(self, message: Union[str,Exception]) -> 'ContextManager[Logger]':
        return self._logger.log(self._pre_decorator.filter(message))

    def time(self, message: str) -> 'ContextManager[Logger]':
        return self._logger.time(self._pre_decorator.filter(message))

    def undecorate(self) -> Logger:
        self._logger.sink = self._original_sink
        return self._logger

class NameLog(Filter[str,str]):
    def filter(self, log: str) -> str:
        return f"pid-{current_process().pid:<6} -- {log}"

class StampLog(Filter[str,str]):
    def filter(self, log: str) -> str:
        return f"{self._now().strftime('%Y-%m-%d %H:%M:%S')} -- {log}"

    def _now(self)-> datetime:
        return datetime.now()

class ExceptLog(Filter[Union[str,Exception],str]):
    def filter(self, log: Union[str,Exception]) -> str:
        if isinstance(log, str):
            return log
        elif isinstance(log, CobaException):
            return str(log)
        else: 
            tb  = ''.join(traceback.format_tb(log.__traceback__))
            msg = ''.join(traceback.TracebackException.from_exception(log).format_exception_only())
            return f"Unexpected exception:\n\n{tb}\n  {msg}"