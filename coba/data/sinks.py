"""The data.sinks module contains core classes for sinks used in data pipelines.

TODO: Add docstrings for all Sinks
TODO: Add unit tests for all Sinks
"""

from abc import ABC, abstractmethod

from typing import Generic, Iterable, TypeVar, List, Any

_T_in  = TypeVar("_T_in", bound=Any, contravariant=True)

class Sink(ABC, Generic[_T_in]):

    @abstractmethod
    def write(self, items: _T_in) -> None:
        ...

class NoneSink(Sink[Iterable[_T_in]]):
    def write(self, items: Iterable[_T_in]) -> None:
        pass

class ConsoleSink(Sink[Iterable[_T_in]]):
    def write(self, items: Iterable[_T_in]) -> None:
        for item in items: print(item)

class DiskSink(Sink[Iterable[str]]):
    def __init__(self, filename:str, mode:str='a+'):
        self.filename = filename
        self._mode    = mode

    def write(self, items: Iterable[str]) -> None:
        with open(self.filename, self._mode) as f:
            for item in items: f.write(item + '\n')

class MemorySink(Sink[_T_in]):
    def __init__(self):
        self.items: List[_T_in] = []

    def write(self, items: _T_in) -> None:
        try:
            self.items.extend(items)
        except TypeError as e:
            if "not iterable" not in str(e): raise
            self.items.append(items)

class QueueSink(Sink[Iterable[Any]]):
    def __init__(self, sink: Any) -> None:
        self._queue = sink

    def write(self, items:Iterable[Any]) -> None:
        for item in items: self._queue.put(item)