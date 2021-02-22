"""The data.sources module contains core classes for sources used in data pipelines.

TODO: Add docstrings for all Sources
TODO: Add unit tests for all Sources
"""

import requests

from abc import ABC, abstractmethod
from typing import Generic, Iterable, TypeVar, Any

_T_out = TypeVar("_T_out", bound=Any, covariant=True)

class Source(ABC, Generic[_T_out]):
    @abstractmethod
    def read(self) -> _T_out:
        ...

class DiskSource(Source[Iterable[str]]):
    def __init__(self, filename:str):
        self.filename = filename

    def read(self) -> Iterable[str]:
        with open(self.filename, "r+") as f:
            for line in f:
                yield line

class MemorySource(Source[_T_out]):
    def __init__(self, item: _T_out): #type:ignore
        self._item = item

    def read(self) -> _T_out:
        return self._item

class QueueSource(Source[Iterable[Any]]):
    def __init__(self, source: Any, poison=None) -> None:
        self._queue  = source
        self._poison = poison

    def read(self) -> Iterable[Any]:
        while True:
            item = self._queue.get()

            if item == self._poison:
                return

            yield item

class HttpSource(Source[requests.Response]):
    def __init__(self, url: str) -> None:
        self._url = url

    def read(self) -> requests.Response:
        return requests.get(self._url)