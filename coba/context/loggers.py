import time
import traceback

from abc import abstractmethod, ABC
from multiprocessing import current_process
from contextlib import contextmanager, nullcontext
from datetime import datetime
from typing import ContextManager, Iterator, Sequence, Union
from copy import copy

from coba.primitives import Filter, Sink
from coba.exceptions import CobaException
from coba.pipes import Pipes, NullSink, ConsoleSink, Identity

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
            A ContextManager that controls hierarchy.
        """
        ...

    @abstractmethod
    def time(self, message: str) -> 'ContextManager[Logger]':
        """Log a timed message to the sink.

        Args:
            message: The message that should be logged.

        Returns:
            A ContextManager that controls when timing stops.
        """
        ...

class NullLogger(Logger):
    """A logger which writes nothing."""

    def __init__(self, sink: Sink[str] = NullSink()) -> None:
        """Instantiate a NullLogger.

        Args:
            sink: The sink to write to (by default null).
        """
        self._sink = sink

    @property
    def sink(self) -> Sink[str]:
        return self._sink

    @sink.setter
    def sink(self, sink: Sink[str]):
        self._sink = sink

    def log(self, message: Union[str,Exception]) -> 'ContextManager[Logger]':
        return nullcontext(self)

    def time(self, message: str) -> 'ContextManager[Logger]':
        return nullcontext(self)

class BasicLogger(Logger):
    """A Logger with flat hierarchy."""

    def __init__(self, sink: Sink[str] = ConsoleSink()):
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
            self.log(f"{message} {outcome}")

    @contextmanager
    def _time_context(self, message:str) -> 'Iterator[Logger]':
        self.log(message)
        self._starts.append(time.time())
        try:
            yield self
        except KeyboardInterrupt:
            self.log(f"{message} ({round(time.time()-self._starts.pop(),2)} seconds) (interrupt)")
            raise
        except Exception:
            self.log(f"{message} ({round(time.time()-self._starts.pop(),2)} seconds) (exception)")
            raise
        else:
            self.log(f"{message} ({round(time.time()-self._starts.pop(),2)} seconds) (completed)")

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
    """A Logger with indented hierarchy."""

    def __init__(self, sink: Sink[str] = ConsoleSink()):
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

class ExceptionLogger(Logger):
    """A Logger that only logs exceptions."""

    def __init__(self, sink: Sink[str] = ConsoleSink()):
        """Instantiate an ExceptionLogger.

        Args:
            sink: The sink to write the logs to (by default console).
        """
        self._sink = sink
        self._filter = ExceptLog()

    @property
    def sink(self) -> Sink[str]:
        return self._sink

    @sink.setter
    def sink(self, sink: Sink[str]):
        self._sink = sink

    def log(self, message: Union[str,Exception]) -> 'ContextManager[Logger]':
        if isinstance(message,Exception):
            self._sink.write(self._filter.filter(message))
        return nullcontext(self)

    def time(self, message: str) -> 'ContextManager[Logger]':
        return nullcontext(self)

class DecoratedLogger(Logger):
    """A Logger which decorates a logger."""

    def __init__(self, pre_decorators: Sequence[Filter], logger: Logger, post_decorators: Sequence[Filter]):
        """Instantiate DecoratedLogger.

        Args:
            pre_decorators: A sequence of decorators to be applied before the base logger.
            logger: The base logger we are decorating.
            post_decorators: A sequence of decorators to be applied after the base logger.
        """

        self._pre_decorator    = Pipes.join(*pre_decorators) if pre_decorators else Identity()
        self._post_decorators  = post_decorators
        self._original_logger  = logger
        self._copy_logger      = copy(logger)
        self._copy_logger.sink = Pipes.join(*post_decorators, logger.sink)

    @property
    def sink(self) -> Sink[str]:
        return self._original_logger.sink

    @sink.setter
    def sink(self, sink: Sink[str]):
        self._original_logger.sink = sink
        self._copy_logger.sink     = Pipes.join(*self._post_decorators, sink)

    def log(self, message: Union[str,Exception]) -> 'ContextManager[Logger]':
        return self._copy_logger.log(self._pre_decorator.filter(message))

    def time(self, message: str) -> 'ContextManager[Logger]':
        return self._copy_logger.time(self._pre_decorator.filter(message))

    def undecorate(self) -> Logger:
        """Remove the decorator.

        Returns:
            The original logger without a decorator.
        """
        return self._original_logger

class NameLog(Filter[str,str]):
    """Add process name to logs."""

    def filter(self, log: str) -> str:
        return f"pid-{current_process().pid:<6} -- {log}"

class StampLog(Filter[str,str]):
    """Add timestamp to logs."""

    def filter(self, log: str) -> str:
        return f"{self._now().strftime('%Y-%m-%d %H:%M:%S')} -- {log}"

    def _now(self)-> datetime:
        return datetime.now()

class ExceptLog(Filter[Union[str,Exception],str]):
    """Add stack traces to logs."""

    def filter(self, log: Union[str,Exception]) -> str:
        if isinstance(log, str):
            return log
        elif isinstance(log, CobaException):
            return f"EXCEPTION: {log}"
        else:
            return f'Unexpected exception:\n\n{self.format_exception(log)}'

    def format_exception(self, ex: Exception):

        out = []

        if ex.__cause__ is not None:
            out.append(self.format_exception(ex.__cause__))
            out.append("The above exception was the direct cause of the following exception:\n")

        msg = ''.join(traceback.TracebackException.from_exception(ex).format_exception_only())

        #we manualy format to provide more helpful messages for strange edge conditions
        #the alternative method is tb  = ''.join(traceback.format_tb(ex.__traceback__))
        tbs  = []
        for frame in traceback.extract_tb(ex.__traceback__):

            code   = frame.line or '<unknown code, this is likely due to the code being in a Jupyter cell>'
            fname  = frame.filename or '<unknown file>'
            mname  = frame.name or '<unknown method>'
            lineno = frame.lineno or '<unknown line>'

            tbs.append(f'  File "{fname}", line {lineno}, in {mname}\n    {code}\n')
        tb = ''.join(tbs)

        if tb : out.append(tb)
        if msg: out.append(f'  {msg}')

        return '\n'.join(out)
