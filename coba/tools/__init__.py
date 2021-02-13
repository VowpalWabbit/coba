"""Simple one-off utility methods with no clear home.

TODO Figure out real logging that works on multiple threads
TODO Add unittests for CobaConfig.
"""

import json
import collections
import time
import sys
import os
import traceback

from importlib_metadata import entry_points #type: ignore
from io import UnsupportedOperation
from contextlib import contextmanager
from gzip import compress, decompress
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import (Callable, ContextManager, Union, Generic, TypeVar, Dict, IO, Optional, List, cast, Iterator, Any)

_K = TypeVar("_K")
_V = TypeVar("_V")

registry: Dict[str, Any] = {}

@contextmanager
def redirect_stderr(to: IO[str]):
    """Redirect stdout for both C and Python.

    Remarks:
        This code comes from https://stackoverflow.com/a/17954769/1066291. Because this modifies
        global pointers this code is not "thread-safe". This limitation is also true of the built-in
        Python modules such as `contextlib.redirect_stdout` and `contextlib.redirect_stderr`. See
        https://docs.python.org/3/library/contextlib.html#contextlib.redirect_stdout for more info.
    """
    try:
        #we assume that this fd is the same
        #one that is used by our C library
        stderr_fd = sys.stderr.fileno()

        def _redirect_stderr(redirect_stderr_fd):
            
            #first we flush Python's stderr. It should be noted that this
            #doesn't close the file descriptor (i.e., sys.stderr.fileno())
            #or Python's wrapper around the stderr_fd.
            sys.stderr.flush()
        
            # next we change the stderr_fd to point to the
            # file contained in the redirect_stderr_fd.
            # If C has anything buffered for stderr it
            # will now go to the new fd. There do appear
            # to be ways to flush C buffers from Python 
            # but I'm not sure it is worth it given the
            # amount of complexity it adds to the code.
            # This change also means that sys.stderr now
            # points to a new file since sys.stderr points
            # to whatever file is at stderr_fd
            os.dup2(redirect_stderr_fd, stderr_fd)

        # when we dup there are now two fd's
        # pointing to the same file. Closing
        # one of these doesn't close the other.
        # therefore it is on us to close the
        # duplicate fd we make here before ending.
        old_stderr_fd = os.dup(stderr_fd)
        new_stderr_fd = to.fileno()

        try:
            _redirect_stderr(new_stderr_fd)
            yield # allow code to be run with the redirected stderr
        finally:
            _redirect_stderr(old_stderr_fd) 
            os.close(old_stderr_fd)
    except UnsupportedOperation:
        #if for some reason we weren't able to redirect
        #then simply move on. No reason to stop working.
        yield

def register_class(name: str, cls: Any) -> None:
    registry[name] = cls

def retrieve_class(name:str) -> Any:

    if len(registry) == 0:
        for eps in entry_points()['coba.register']:
            eps.load()

    return registry[name]

def create_class(recipe: Any) -> Any:

    name   = ""
    args   = []
    kwargs = {}

    if isinstance(recipe, str):
        name = recipe
 
    if isinstance(recipe, collections.Mapping):
        mutable_recipe = dict(recipe)

        name   = mutable_recipe.pop("name"  , "")
        args   = mutable_recipe.pop("args"  , [])
        kwargs = mutable_recipe.pop("kwargs", {})

        if ( 
            len(mutable_recipe) > 1 or 
            len(mutable_recipe) == 1 and (name != "" or args != []) or 
            len(mutable_recipe) == 0 and name == ""
        ):
            raise Exception(f"Invalid recipe {str(recipe)}")

        if len(mutable_recipe) == 1:
            name,args = list(mutable_recipe.items())[0]

    if not isinstance(args, list):
        args = [args]

    try:
        return retrieve_class(name)(*args, **kwargs)
    except KeyError:
        raise Exception(f"Unknown recipe {str(recipe)}")

def check_matplotlib_support(caller_name: str) -> None:
    """Raise ImportError with detailed error message if matplotlib is not installed.

    Functionality requiring matplotlib should call this helper and then lazily import.

    Args:    
        caller_name: The name of the caller that requires matplotlib.

    Remarks:
        This pattern borrows heavily from sklearn. As of 6/20/2020 sklearn code could be found
        at https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/__init__.py
    """
    try:
        import matplotlib # type: ignore
    except ImportError as e:
        raise ImportError(
            caller_name + " requires matplotlib. You can "
            "install matplotlib with `pip install matplotlib`."
        ) from e

def check_vowpal_support(caller_name: str) -> None:
    """Raise ImportError with detailed error message if vowpalwabbit is not installed.

    Functionality requiring vowpalwabbit should call this helper and then lazily import.

    Args:    
        caller_name: The name of the caller that requires matplotlib.

    Remarks:
        This pattern was inspired by sklearn (see coba.tools.check_matplotlib_support for more information).
    """
    try:
        import vowpalwabbit # type: ignore
    except ImportError as e:
        raise ImportError(
            caller_name + " requires vowpalwabbit. You can "
            "install vowpalwabbit with `pip install vowpalwabbit`."
        ) from e

def check_pandas_support(caller_name: str) -> None:
    """Raise ImportError with detailed error message if pandas is not installed.

    Functionality requiring pandas should call this helper and then lazily import.

    Args:
        caller_name: The name of the caller that requires pandas.

    Remarks:
        This pattern was inspired by sklearn (see coba.tools.check_matplotlib_support for more information).
    """
    try:
        import pandas # type: ignore
    except ImportError as e:
        raise ImportError(
            caller_name + " requires pandas. You can "
            "install pandas with `pip install pandas`."
        ) from e

def check_numpy_support(caller_name: str) -> None:
    """Raise ImportError with detailed error message if numpy is not installed.

    Functionality requiring numpy should call this helper and then lazily import.

    Args:
        caller_name: The name of the caller that requires numpy.

    Remarks:
        This pattern was inspired by sklearn (see coba.tools.check_matplotlib_support for more information).
    """
    try:
        import numpy # type: ignore
    except ImportError as e:
        raise ImportError(
            caller_name + " requires numpy. You can "
            "install numpy with `pip install numpy`."
        ) from e

class CacheInterface(Generic[_K, _V], ABC):
    """The interface for a cacher."""
    
    @abstractmethod
    def __contains__(self, key: _K) -> bool:
        ...

    @abstractmethod
    def get(self, key: _K) -> _V:
        ...

    @abstractmethod
    def put(self, key: _K, value: _V) -> None:
        ...

    @abstractmethod
    def rmv(self, key: _K) -> None:
        ...

class NoneCache(CacheInterface[_K, _V]):
    def __init__(self) -> None:
        self._cache: Dict[_K,_V] = {}

    def __contains__(self, key: _K) -> bool:
        return False

    def get(self, key: _K) -> _V:
        raise Exception("the key didn't exist in the cache")

    def put(self, key: _K, value: _V) -> None:
        pass

    def rmv(self, key: _K):
        pass

class MemoryCache(CacheInterface[_K, _V]):
    def __init__(self) -> None:
        self._cache: Dict[_K,_V] = {}

    def __contains__(self, key: _K) -> bool:
        return key in self._cache

    def get(self, key: _K) -> _V:
        return self._cache[key]

    def put(self, key: _K, value: _V) -> None:
        self._cache[key] = value

    def rmv(self, key: _K) -> None:
        del self._cache[key]

class DiskCache(CacheInterface[str, bytes]):
    """A cache that writes bytes to disk.
    
    The DiskCache compresses all values before storing in order to conserve space.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        """Instantiate a DiskCache.
        
        Args:
            path: The path to the directory where all files will be cached
        """
        self._cache_dir = path if isinstance(path, Path) else Path(path).expanduser()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def __contains__(self, filename: str) -> bool:
        return self._cache_path(filename).exists()

    def get(self, filename: str) -> bytes:
        """Get a filename from the cache.

        Args:
            filename: Requested filename to retreive from the cache.
        """

        return decompress(self._cache_path(filename).read_bytes())

    def put(self, filename: str, value: bytes):
        """Put a filename and its bytes into the cache.
        
        Args:
            filename: The filename to store in the cache.
            value: The bytes that should be cached for the given filename.
        """

        self._cache_path(filename).touch()
        self._cache_path(filename).write_bytes(compress(value))

    def rmv(self, filename: str) -> None:
        """Remove a filename from the cache.

        Args:
            filename: The filename to remove from the cache.
        """

        if self._cache_path(filename).exists(): self._cache_path(filename).unlink()

    def _cache_name(self, filename: str) -> str:
        return filename + ".gz"

    def _cache_path(self, filename: str) -> Path:
        return self._cache_dir/self._cache_name(filename)

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

class CobaConfig_meta(type):
    """To support class properties before python 3.9 we must implement our properties directly 
       on a meta class. Using class properties rather than class variables is done to allow 
       lazy loading. Lazy loading moves errors to execution time instead of import time where
       they are easier to debug.
    """

    def __init__(cls, *args, **kwargs):
        cls._api_keys  = None
        cls._cache     = None
        cls._log       = None
        cls._benchmark = None

    @staticmethod
    def _load_config() -> Dict[str,Any]:
        search_paths = [Path("./.coba"), Path.home() / ".coba"]

        config = {
            "api_keys"  : collections.defaultdict(lambda:None),
            "cache"     : "NoneCache",
            "log"       : "ConsoleLog",
            "benchmark" : {"processes": 1, "maxtasksperchild": None, "file_fmt": "BenchmarkFileV1"}
        }

        for potential_path in search_paths:
            if potential_path.exists():
                with open(potential_path) as fs:
                    for key,value in json.load(fs).items():
                        if isinstance(config[key], collections.MutableMapping):
                            config[key].update(value)
                        else:
                            config[key] = value
                break

        return config

    @property
    def Api_Keys(cls):
        if cls._api_keys is None:
            cls._api_keys = cls._load_config()['api_keys']
        return cls._api_keys

    @Api_Keys.setter
    def Api_Keys(cls, value):
        cls._api_keys = value

    @property
    def Cacher(cls):
        if cls._cache is None:
            cls._cache = create_class(cls._load_config()['cache'])
        return cls._cache
    
    @Cacher.setter
    def Cacher(cls, value):
        cls._cache = value

    @property
    def Logger(cls):
        if cls._log is None:
            cls._log = create_class(cls._load_config()['log'])
        return cls._log
    
    @Logger.setter
    def Logger(cls, value):
        cls._log = value

    @property
    def Benchmark(cls):
        if cls._benchmark is None:
            cls._benchmark = cls._load_config()['benchmark']
        return cls._benchmark

    @Benchmark.setter
    def Benchmark(cls, value):
        cls._benchmark = value

class CobaConfig(metaclass=CobaConfig_meta):
    """Create a global configuration context to allow easy mocking and customization.

    In short, So long as the same modulename is always used to import and the import
    always occurs on the same thread I'm fairly confident this pattern will always work.

    In long, I'm somewhat unsure about this pattern for the following reasons:
        > While there seems concensus that multi-import doesn't repeat [1] it may if different modulenames are used [2]
            [1] https://stackoverflow.com/a/19077396/1066291
            [2] https://stackoverflow.com/q/13392038/1066291

        > Python 3.7 added in explicit context management in [3,4] but this implementation is thread local only
            [3] https://www.python.org/dev/peps/pep-0567/
            [4] https://docs.python.org/3/library/contextvars.html
    """
    pass