from abc import ABC, abstractmethod

from coba.typing import Any, TypeVar, Generic

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
