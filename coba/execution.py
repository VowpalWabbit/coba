"""The contexts module contains classes determining the context for shared functionality.

TODO Add unittests for CobaConfig.
TODO Add unittests for all CacheInterface implementations.
"""

import json
import copy
import collections
import time

from typing import Callable, ContextManager, Union, Generic, TypeVar, Dict, IO, Mapping, Any, Optional, List, MutableMapping, cast, Iterator

from contextlib import contextmanager
from itertools import repeat
from gzip import compress, decompress
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from datetime import datetime


_K = TypeVar("_K")
_V = TypeVar("_V")

class TemplatingEngine():

    @staticmethod
    def parse(json_val:Union[str, Any]):

        root = json.loads(json_val) if isinstance(json_val, str) else json_val

        if "templates" in root:

            templates: Dict[str,Dict[str,Any]] = root.pop("templates")
            nodes    : List[Any]               = [root]
            scopes   : List[Dict[str,Any]]     = [{}]

            def materialize_template(document: MutableMapping[str,Any], template: Mapping[str,Any]):

                for key in template:
                    if key in document:
                        if isinstance(template[key], collections.Mapping) and isinstance(template[key], collections.Mapping):
                            materialize_template(document[key],template[key])
                    else:
                        document[key] = template[key]

            def materialize_variables(document: MutableMapping[str,Any], variables: Mapping[str,Any]):
                for key in document:
                    if isinstance(document[key],str) and document[key] in variables:
                        document[key] = variables[document[key]]

            while len(nodes) > 0:
                node  = nodes.pop()
                scope = scopes.pop().copy()  #this could absolutely be made more memory-efficient if needed

                if isinstance(node, collections.MutableMapping):

                    if "template" in node and node["template"] not in templates:
                        raise Exception(f"We were unable to find template '{node['template']}'.")

                    keys      = list(node.keys())
                    template  = templates[node.pop("template")] if "template" in node else cast(Dict[str,Any], {})
                    variables = { key:node.pop(key) for key in keys if key.startswith("$") }

                    template = copy.deepcopy(template)
                    scope.update(variables)

                    materialize_template(node, template)
                    materialize_variables(node, scope)

                    for child_node, child_scope in zip(node.values(), repeat(scope)):
                        nodes.append(child_node)
                        scopes.append(child_scope)

                if isinstance(node, collections.Sequence) and not isinstance(node, str):
                    for child_node, child_scope in zip(node, repeat(scope)):
                        nodes.append(child_node)
                        scopes.append(child_scope)

        return root

class CobaConfig():
    def __init__(self):
        search_paths = [Path("./.coba"), Path.home() / ".coba"]

        config = {}

        for potential_path in search_paths:
            if potential_path.exists():
                with open(potential_path) as fs:
                    config = json.load(fs)
                break

        self._config = config

    @property
    def openml_api_key(self) -> Optional[str]:
        return self._config.get("openml_api_key", None)

    @property
    def file_cache(self) -> Mapping[str,Any]:
        return self._config.get("file_cache", {"type":"none"})        

class CacheInterface(Generic[_K, _V], ABC):

    @abstractmethod
    def __contains__(self, key: _K) -> bool:
        ...

    @abstractmethod
    def get(self, key: _K) -> _V:
        ...

    @abstractmethod
    def put(self, key: _K, value: _V) -> _V:
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

    def put(self, key: _K, value: _V) -> _V:
        return value

    def rmv(self, key: _K):
        pass

class MemoryCache(CacheInterface[_K, _V]):
    def __init__(self) -> None:
        self._cache: Dict[_K,_V] = {}

    def __contains__(self, key: _K) -> bool:
        return key in self._cache

    def get(self, key: _K) -> _V:
        return self._cache[key]

    def put(self, key: _K, value: _V) -> _V:
        self._cache[key] = value

        return value

    def rmv(self, key: _K):
        del self._cache[key]

class DiskCache(CacheInterface[str, IO[bytes]]):
    def __init__(self, path: Union[str, Path]) -> None:
        self._cache_dir = path if isinstance(path, Path) else Path(path).expanduser()

    def __contains__(self, filename: str) -> bool:
        gzip_filename = filename if filename.endswith(".gz") else (filename + ".gz")

        return (self._cache_dir/gzip_filename).exists()

    def get(self, filename: str) -> IO[bytes]:
        is_gzip = filename.endswith(".gz")

        gzip_filename = filename + ("" if is_gzip else ".gz")
        gzip_bytes    = (self._cache_dir/gzip_filename).read_bytes()

        return BytesIO(gzip_bytes if is_gzip else decompress(gzip_bytes))

    def put(self, filename: str, value: IO[bytes]) -> IO[bytes]:
        is_gzip = filename.endswith(".gz")

        gzip_filename = filename + ("" if is_gzip else ".gz")
        gzip_bytes    = value.read() if is_gzip else compress(value.read())

        (self._cache_dir/gzip_filename).parent.mkdir(parents=True, exist_ok=True)
        (self._cache_dir/gzip_filename).touch()
        (self._cache_dir/gzip_filename).write_bytes(gzip_bytes)

        return self.get(filename)

    def rmv(self, filename: str) -> None:
        is_gzip = filename.endswith(".gz")

        gzip_filename = filename + ("" if is_gzip else ".gz")

        if (self._cache_dir/gzip_filename).exists():
            (self._cache_dir/gzip_filename).unlink()

class LoggerInterface(ABC):
    @abstractmethod
    def log(self, message: str, end:str = None) -> 'ContextManager[LoggerInterface]':
        ...

class UniversalLogger(LoggerInterface):

    def __init__(self, print_function: Callable[[str,Optional[str]],None]):
        self._indent_cnt  = 0
        self._is_newline  = True
        self._print       = print_function
        self._bullets     = collections.defaultdict(lambda: '~', enumerate(['','*','>','-','+','~'])) 
        self._start_times = []

    @contextmanager
    def _with(self) -> Iterator[LoggerInterface]:
        try:
            self._indent_cnt += 1
            self._start_times.append(time.time())

            yield self

            if not self._is_newline: self.log('')
            self.log(f"finished after {round(time.time() - self._start_times.pop(), 2)} seconds")

        except LoggedException as e:
            raise #simply pass it along, no need to log it again

        except Exception as e:
            if not self._is_newline: self.log('')
            self.log(f"unhandeled exception after {round(time.time() - self._start_times.pop(), 2)} seconds: {e}")
            raise LoggedException from e

        finally:
            self._indent_cnt -= 1

    def _prefix(self) -> str:
        indent = '  ' * self._indent_cnt
        bullet = self._bullets[self._indent_cnt]

        return datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' + indent + bullet + (' ' if bullet != '' else '')

    def log(self, message: str, end: str = None) -> ContextManager[LoggerInterface]:
        
        if self._is_newline:
            message = self._prefix() + message

        self._print(message, end)

        self._is_newline = (end is None or end == '\n')

        return self._with()

class ConsoleLogger(UniversalLogger):
    def __init__(self) -> None:
        super().__init__(print_function=lambda m,e: print(m,end=e))

class NoneLogger(UniversalLogger):
    def __init__(self) -> None:
        super().__init__(print_function=lambda m,e: None)

class LoggedException(Exception):
    """An exception that has been logged but not handled."""

class ExecutionContext:
    """Create a global execution context to allow easy mocking and modification.

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

    TemplatingEngine: TemplatingEngine = TemplatingEngine()
    CobaConfig      : CobaConfig       = CobaConfig()
    FileCache       : CacheInterface   = NoneCache()
    Logger          : LoggerInterface  = ConsoleLogger()

    if CobaConfig.file_cache["type"] == "disk":
        FileCache = DiskCache(CobaConfig.file_cache["directory"])

    if CobaConfig.file_cache["type"] == "memory":
        FileCache  = MemoryCache()