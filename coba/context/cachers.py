import gzip
import time

from hashlib import blake2b
from threading import Lock, current_thread
from pathlib import Path
from contextlib import nullcontext, contextmanager
from collections.abc import Iterator
from collections import defaultdict
from abc import abstractmethod, ABC
from typing import Union, Dict, TypeVar, Iterable, Optional, Callable, Generic, Sequence, ContextManager

from coba.exceptions import CobaException

_K = TypeVar("_K")
_V = TypeVar("_V")

class Cacher(Generic[_K, _V], ABC):
    """The interface for a cacher."""

    @abstractmethod
    def __contains__(self, key: _K) -> bool:
        """Determine if key is in cache."""
        ...

    @abstractmethod
    def rmv(self, key: _K) -> None:
        """Remove a key from the cache.

        Args:
            key: The key to remove from the cache.
        """
        ...

    @abstractmethod
    def get_set(self, key: _K, getter: Union[Callable[[], _V],_V]) -> ContextManager[_V]:
        """Get a key from the cache.

        If the key is not in the cache put it first using getter.

        Args:
            key: The key to get from the cache.
            getter: A method for getting the value if necessary.
        """
        ...

class NullCacher(Cacher[_K, _V]):
    """A cacher which does not cache any data."""

    def __init__(self) -> None:
        """Instantiate a NullCacher."""
        pass

    def __contains__(self, key: _K) -> bool:
        return False

    def rmv(self, key: _K):
        pass

    def get_set(self, key: _K, getter: Union[Callable[[], _V],_V]) -> ContextManager[_V]:
        return nullcontext(getter())

class MemoryCacher(Cacher[_K, _V]):
    """A cacher that caches in memory."""

    def __init__(self) -> None:
        """Instantiate a MemoryCacher."""
        self._cache: Dict[_K,_V] = {}

    def __contains__(self, key: _K) -> bool:
        return key in self._cache

    def rmv(self, key: _K) -> None:
        if key in self:
            del self._cache[key]

    def get_set(self, key: _K, getter: Union[Callable[[], _V],_V]) -> ContextManager[_V]:
        if key not in self:
            value = getter() if callable(getter) else getter
            value = list(value) if isinstance(value,Iterator) else value
            self._cache[key] = value
            return nullcontext(value)
        return nullcontext(self._cache[key])

class DiskCacher(Cacher[str, Iterable[str]]):
    """A cacher that writes to disk.

    The DiskCacher compresses all values before writing to conserve disk space.
    """

    def __init__(self, cache_dir: Union[str, Path]) -> None:
        """Instantiate a DiskCacher.

        Args:
            cache_dir: The directory path where all given keys will be cached as files
        """
        self.cache_directory = cache_dir

    @property
    def cache_directory(self) -> Optional[str]:
        """The directory where the cache will write to disk."""
        return str(self._cache_dir)

    @cache_directory.setter
    def cache_directory(self, cache_dir:Union[Path,str]) -> None:
        if not isinstance(cache_dir,(str,Path)):
            raise CobaException(f"An invalid cache directory was supplied: {cache_dir}.")
        self._cache_dir = str(cache_dir)

    def __contains__(self, key: str) -> bool:
        return self._cache_path(key).exists()

    def rmv(self, key: str) -> None:
        if self._cache_path(key).exists(): self._cache_path(key).unlink()

    def get_set(self, key: str, getter: Union[Callable[[], Iterable[str]],Iterable[str]]) -> ContextManager[Iterable[str]]:

        if key not in self:
            try:
                lines = getter() if callable(getter) else getter
                if isinstance(lines,str): lines = [lines]
                Path(self._cache_dir).expanduser().mkdir(parents=True, exist_ok=True)

                with gzip.open(self._cache_path(key), "wt+", 6, "utf-8") as f:
                    for line in lines:
                        f.write(line.rstrip('\r\n'))
                        f.write('\n')
            except:
                if key in self: 
                    self.rmv(key)
                raise
        
        return gzip.open(self._cache_path(key), 'rt', 'utf-8')

    def _cache_name(self, key: str) -> str:
        if not all(c.isalnum() or c in (' ','.','_') for c in key):
            raise CobaException(f"A key was given to DiskCacher which couldn't be made into a file, {key}")
        return f"{key}.gz"

    def _cache_path(self, key: str) -> Path:
        return Path(self._cache_dir).expanduser()/self._cache_name(key)

class ConcurrentCacher(Cacher[_K, _V]):
    """A cacher that is multi-process safe."""

    def __init__(self, cache:Cacher[_K, _V], list: Sequence = None, lock: Lock = None):
        """Instantiate a ConcurrentCacher.

        Args:
            cache: The base cacher that we wish to make multi-process safe.
            list: A shared memory object which allows us to track read and write locks
            lock: The memory synchronization object to be used to ensure read/write safety
        """
        self._digest_size = 2

        self._cache = cache
        self._lock  = lock or Lock()
        self._array = list or [0]*2**(8*self._digest_size)
        
        self._write_waits = 0 # for testing purposes only. won't be accurate in production.
        self._read_waits  = 0 # for testing purposes only. won't be accurate in production.

        self._locks = defaultdict(int) # for safety to make sure our current process/thread isn't waiting on itself

        assert len(self._array) >= 2**(8*self._digest_size)

    def __contains__(self, key: _K) -> bool:
        return key in self._cache

    def rmv(self, key: _K):
        lock = None
        try:
            if key in self:
                self._acquire_write_lock(key)
                lock='write'
                self._cache.rmv(key)
                lock = None
                self._release_write_lock(key)
        except:
            if lock == 'write': self._release_write_lock(key)
            raise

    def get_set(self, key: _K, getter: Union[Callable[[],_V],_V]) -> ContextManager[_V]:

        try:
            self._acquire_read_lock(key)
            if key in self._cache:
                return self._release_read_on_exit(key,self._cache.get_set(key,None))
            self._release_read_lock(key)

            self._acquire_write_lock(key)
            if key in self:#pragma: no cover; this is super hard to isolate so I'm just going to trust it...
                self._switch_write_to_read_lock(key)
                return self._release_read_on_exit(key,self._cache.get_set(key,None))
            else:
                item = self._cache.get_set(key, getter)
                self._switch_write_to_read_lock(key)
                return self._release_read_on_exit(key,item)
        except Exception as e:
            if self._has_read_lock(key): self._release_read_lock(key)
            if self._has_write_lock(key): self._release_write_lock(key)
            raise

    @contextmanager
    def _release_read_on_exit(self,key:str,item:ContextManager[_V]) -> _V:
        try:
            with item as out:
                yield out
        finally:
            self._release_read_lock(key)

    def _acquire_read_lock(self, key):
        if self._has_write_lock(key):
            raise CobaException("The concurrent cacher was asked to enter an unrecoverable state.")
        
        index = self._index(key)
        self._read_waits += 1
        while True:
            with self._lock:
                if self._array[index] >= 0:
                    self._locks[(current_thread().ident,key)] += 1
                    self._array[index] += 1
                    self._read_waits -= 1
                    break
            time.sleep(1)
    
    def _release_read_lock(self, key):
        index = self._index(key)
        self._array[index] -= 1
        self._locks[(current_thread().ident,key)] -= 1

    def _acquire_write_lock(self, key) -> ContextManager:
        if self._has_write_lock(key) or self._has_read_lock(key):
            raise CobaException("The concurrent cacher was asked to enter an unrecoverable state.")
        index = self._index(key)
        self._write_waits += 1
        while True:
            with self._lock:
                if self._array[index] == 0:
                    self._locks[(current_thread().ident,key)] = -1
                    self._array[index] = -1
                    self._write_waits -= 1
                    break
            time.sleep(1)

    def _release_write_lock(self, key):
        index = self._index(key)
        self._array[index] = 0
        self._locks[(current_thread().ident,key)] = 0

    def _switch_write_to_read_lock(self, key) -> None:
        index = self._index(key)
        assert self._array[index] == -1, "You don't have write permissions"
        assert self._locks[(current_thread().ident,key)] == -1, "You don't have write permissions"
        self._array[index] = 1
        self._locks[(current_thread().ident,key)] = 1

    def _has_read_lock(self, key) -> bool:
        return self._locks[(current_thread().ident,key)] > 0

    def _has_write_lock(self, key) -> bool:
        return self._locks[(current_thread().ident,key)] == -1

    def _index(self, key) -> int:
        return int.from_bytes(blake2b(str(key).encode('utf-8'),digest_size=self._digest_size).digest(),"big")