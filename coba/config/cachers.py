"""Various caching implementations."""

from hashlib import md5
from gzip import compress, decompress
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Generic, Dict, TypeVar

_K = TypeVar("_K")
_V = TypeVar("_V")

class Cacher(Generic[_K, _V], ABC):
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

class NoneCacher(Cacher[_K, _V]):
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

class MemoryCacher(Cacher[_K, _V]):
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

class DiskCacher(Cacher[str, bytes]):
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

    def __contains__(self, key: str) -> bool:
        return self._cache_path(key).exists()

    def get(self, key: str) -> bytes:
        """Get a key from the cache.

        Args:
            filename: Requested filename to retreive from the cache.
        """

        return decompress(self._cache_path(key).read_bytes())

    def put(self, key: str, value: bytes):
        """Put a key and its bytes into the cache.
        
        In the case of a key collision this will overwrite the existing key

        Args:
            key: The key to store in the cache.
            value: The bytes that should be cached for the given filename.
        """

        self._cache_path(key).touch()
        self._cache_path(key).write_bytes(compress(value))

    def rmv(self, key: str) -> None:
        """Remove a key from the cache.

        Args:
            key: The key to remove from the cache.
        """

        if self._cache_path(key).exists(): self._cache_path(key).unlink()

    def _cache_name(self, key: str) -> str:
        return md5(key.encode('utf-8')).hexdigest() + ".gz"

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir/self._cache_name(key)