"""The data module contains core classes and types for reading and writing data sources."""

from abc import abstractmethod, ABC
from build.lib.coba.json import CobaJsonDecoder
from pathlib import Path
from typing import Any, List, Iterable

from coba.json import CobaJsonEncoder

class ReadWrite(ABC):
    @abstractmethod
    def write(self, obj:Any) -> None:
        ...

    @abstractmethod
    def read(self) -> Iterable[Any]:
        ...

class DiskReadWrite(ReadWrite):
    
    def __init__(self, filename:str):
        self._json_encoder = CobaJsonEncoder()
        self._json_decoder = CobaJsonDecoder()
        self._filepath     = Path(filename)
        self._filepath.touch()

    def write(self, obj: Any) -> None:
        with open(self._filepath, "a") as f:
            f.write(self._json_encoder.encode(obj))
            f.write("\n")
    
    def read(self) -> Iterable[Any]:
        with open(self._filepath, "r") as f:
            for line in f.readlines():
                yield self._json_decoder.decode(line)

class MemoryReadWrite(ReadWrite):
    def __init__(self, memory: List[Any] = None):
        self._memory = memory if memory else []

    def write(self, obj:Any) -> None:
        self._memory.append(obj)

    def read(self) -> Iterable[Any]:
        return self._memory

class QueueReadWrite(ReadWrite):
    def __init__(self, queue: Any) -> None:
        self._queue = queue

    def write(self, obj:Any) -> None:
        self._queue.put(obj)

    def read(self) -> Iterable[Any]:
        while not self._queue.empty():
            yield self._queue.get()