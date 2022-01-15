import time
import traceback

from abc import abstractmethod, ABC
from multiprocessing import current_process
from contextlib import contextmanager
from datetime import datetime
from typing import ContextManager, Iterator, Sequence, Union

from coba.pipes import Pipe, Filter, Sink, NullIO, ConsoleIO
from coba.exceptions import CobaException

class Logger(ABC):
    """The interface for a logger."""

    @property
    @abstractmethod
    def sink(self) -> Sink[str]:
        """The sink the logger writes to."""
        ...

    @sink.setter
    @abstractmethod
    def sink(self, sink: Sink[str]):
        ...

    @abstractmethod
    def log(self, message: Union[str,Exception]) -> 'ContextManager[Logger]':
        """Log a message or exception to the sink.
        
        Args:
            message: The message or exception that should be logged.

        Returns:
            A ContextManager that can be used to communicate log hierarchy.
        """
        ...

    @abstractmethod
    def time(self, message: str) -> 'ContextManager[Logger]':
        """Log a timed message to the sink.

        Args:
            message: The message that should be logged.

        Returns:
            A ContextManager that indicates when to stop timing.
        """
        ...

class NullLogger(Logger):
    """A logger which writes nothing."""

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
    """A Logger with flat hierarchy and separate begin/end messages."""

    def __init__(self, sink: Sink[str] = ConsoleIO()):
        """Instantiate a BasicLogger.
        
        Args:
            sink: The sink to write to (by default console).
        """
        self._sink = sink
        self._starts = []

    @contextmanager
    def _log_context(self, message:str) -> 'Iterator[Logger]':
        outcome = "(error)"
        try:
            yield self
            outcome = "(completed)"
        except KeyboardInterrupt:
            outcome = "(interrupt)"
            raise
        except (Exception, TypeError) as e:
            outcome = "(exception)"
            raise
        finally:
            self.log(message + f" {outcome}")
    
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
        self._sink.write(message)
        return self._log_context(message)

    def time(self, message: str) -> 'ContextManager[Logger]':
        return self._time_context(message)

class IndentLogger(Logger):
    """A Logger with indentation hierarchy and a single timed log with total runtime."""

    def __init__(self, sink: Sink[str] = ConsoleIO()):
        """Instantiate an IndentLogger.
        
        Args:
            sink: The sink to write the logs to (by default console).
        """
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
            outcome = "(error)"
            try:
                start = time.time()
                yield self
                outcome = "(completed)"
            except KeyboardInterrupt:
                outcome = "(interrupt)"
                raise
            except (Exception, TypeError):
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
        if self._messages:
            self._messages.append(self._level_message(message))
        else:
            self._sink.write(self._level_message(message))

        return self._indent_context()

    def time(self, message: str) -> 'ContextManager[Logger]':
        return self._time_context(message)

class DecoratedLogger(Logger):
    """A Logger which decorates a base logger."""

    def __init__(self, pre_decorators: Sequence[Filter], logger: Logger, post_decorators: Sequence[Filter]):
        """Instantiate DecoratedLogger.
        
        Args:
            pre_decorators: A sequence of decorators to be applied before the base logger.
            logger: The base logger we are decorating.
            post_decorators: A sequence of decorators to be applied after the base logger.
        """

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
    """A log decorator that names the process writing the log."""

    def filter(self, log: str) -> str:
        return f"pid-{current_process().pid:<6} -- {log}"

class StampLog(Filter[str,str]):
    """A log decorator that adds a timestamp to logs."""

    def filter(self, log: str) -> str:
        return f"{self._now().strftime('%Y-%m-%d %H:%M:%S')} -- {log}"

    def _now(self)-> datetime:
        return datetime.now()

class ExceptLog(Filter[Union[str,Exception],str]):
    """A Log decorator that turns exceptions into messages."""

    def filter(self, log: Union[str,Exception]) -> str:
        if isinstance(log, str):
            return log
        elif isinstance(log, CobaException):
            return str(log)
        else: 
            tb  = ''.join(traceback.format_tb(log.__traceback__))
            msg = ''.join(traceback.TracebackException.from_exception(log).format_exception_only())
            return f"Unexpected exception:\n\n{tb}\n  {msg}"