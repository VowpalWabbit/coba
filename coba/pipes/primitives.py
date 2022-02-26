from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic, Dict

_T_out = TypeVar("_T_out", bound=Any, covariant    =True)
_T_in  = TypeVar("_T_in" , bound=Any, contravariant=True)

class Pipe:

    @property
    def params(self) -> Dict[str,Any]:
        """Parameters describing the pipe."""
        return { }

    def __str__(self) -> str:
        return str(self.params)

class Source(ABC, Pipe, Generic[_T_out]):
    @abstractmethod
    def read(self) -> _T_out:
        """Read the item."""
        ...

class Filter(ABC, Pipe, Generic[_T_in, _T_out]):    
    @abstractmethod
    def filter(self, item: _T_in) -> _T_out:
        """Filter the item."""
        ...

class Sink(ABC, Pipe, Generic[_T_in]):
    @abstractmethod
    def write(self, item: _T_in) -> None:
        """Write the item."""
        ...
