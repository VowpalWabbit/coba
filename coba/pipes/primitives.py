from abc import ABC, abstractmethod

from typing import Any, TypeVar, Generic
from coba.backports import Protocol

_T_out = TypeVar("_T_out", bound=Any, covariant=True    )
_T_in  = TypeVar("_T_in" , bound=Any, contravariant=True)

class Source(ABC, Generic[_T_out]):
    @abstractmethod
    def read(self) -> _T_out:
        ...

class Filter(ABC, Generic[_T_in, _T_out]):
    @abstractmethod
    def filter(self, item: _T_in) -> _T_out:
        ...

class Sink(ABC, Generic[_T_in]):

    @abstractmethod
    def write(self, item: _T_in) -> None:
        ...
class MutableMap(Protocol):
    def __getitem__(self, key: Any) -> Any: pass
    def __setitem__(self, key: Any, val: Any) -> None: pass
    def pop(key:Any) -> Any: pass