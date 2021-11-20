"""Various caching implementations."""

import gzip

from hashlib import md5
from pathlib import Path
from typing import Union, Dict, TypeVar, Iterable, Optional

from coba.config.core import Cacher, CobaConfig

_K = TypeVar("_K")
_V = TypeVar("_V")

class NullCacher(Cacher[_K, _V]):
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

class DiskCacher(Cacher[str, Iterable[bytes]]):
    """A cache that writes bytes to disk.

    The DiskCache compresses all values before storing in order to conserve space.
    """

    def __init__(self, cache_dir: Union[str, Path] = None) -> None:
        """Instantiate a DiskCache.

        Args:
            cache_dir: The directory path where all given keys will be cached as files
        """
        self.cache_directory = cache_dir

    @property
    def cache_directory(self) -> Optional[str]:
        return str(self._cache_dir) if self._cache_dir is not None else None

    @cache_directory.setter
    def cache_directory(self,value:Union[Path,str,None]) -> None:
        self._cache_dir = value if isinstance(value, Path) else Path(value).expanduser() if value else None
        if self._cache_dir is not None: self._cache_dir.mkdir(parents=True, exist_ok=True)

    def __contains__(self, key: str) -> bool:
        return self._cache_dir is not None and self._cache_path(key).exists()

    def get(self, key: str) -> Iterable[bytes]:
        """Get a key from the cache.

        Args:
            key: Requested key to retreive from the cache.
        """

        if key not in self: return []

        try:
            with gzip.open(self._cache_path(key), 'rb') as f:
                for line in f:
                    yield line.rstrip(b'\r\n')
        except:
            #do we want to clear the cache here if something goes wrong?
            #it seems reasonable since this would indicate the cache is corrupted...
            raise

    def put(self, key: str, value: Iterable[bytes]):
        """Put a key and its bytes into the cache. In the case of a key collision nothing will be put.

        Args:
            key: The key to store in the cache.
            value: The bytes that should be cached for the given filename.
        """

        #I'm not crazy about this... This means we can only put one thing at a time...
        #What we really want is a lock on `key` though I haven't been able to find a good
        #way to do this. A better method might be to create a manager.List() and then only
        #lock long enough to add and remove keys from the manager.List() rather than locking
        #for the entire time it takes to put something (which could be a considerable amount)
        #of time.
        if CobaConfig.store.get("cachelck"): CobaConfig.store.get("cachelck").acquire()

        try:
            if key in self: return

            if isinstance(value,bytes): value = [value]

            with gzip.open(self._cache_path(key), 'wb+') as f:
                for line in value:
                    f.write(line.rstrip(b'\r\n') + b'\r\n')
        except:
            if key in self: self.rmv(key)
            raise
        finally:
            if CobaConfig.store.get("cachelck"): CobaConfig.store.get("cachelck").release()

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