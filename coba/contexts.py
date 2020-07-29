"""The contexts module contains classes determining the context for shared functionality."""

import json

from gzip import compress, decompress
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Union, Generic, TypeVar, Dict, IO, Mapping, Any, Optional
from pathlib import Path

_K = TypeVar("_K")
_V = TypeVar("_V")    

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
    def put(self, key: _K, value: _V) -> None:
        ...

    @abstractmethod
    def rmv(self, key: _K, value: _V) -> None:
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

    def rmv(self, key: _K):
        del self._cache[key]

class DiskCache(CacheInterface[str, IO[bytes]]):
    def __init__(self, path: Union[str, Path]) -> None:
        self._cache_dir = path if isinstance(path, Path) else Path(path).expanduser()
    
    def __contains__(self, filename: str) -> bool:
        is_gzip = filename.endswith(".gz")

        gzip_filename = filename + ("" if is_gzip else ".gz")

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

        (self._cache_dir/gzip_filename).unlink()

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
    
    CobaConfig: CobaConfig    = CobaConfig()
    FileCache: CacheInterface = NoneCache()

    if CobaConfig.file_cache["type"] == "disk":
        FileCache = DiskCache(CobaConfig.file_cache["directory"])
    
    if CobaConfig.file_cache["type"] == "memory":
        FileCache  = MemoryCache()