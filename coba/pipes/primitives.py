from operator import eq
from abc import ABC, abstractmethod
from collections import abc
from typing import Any, TypeVar, Generic, Mapping, Iterable, Iterator

_T_out = TypeVar("_T_out", bound=Any, covariant    =True)
_T_in  = TypeVar("_T_in" , bound=Any, contravariant=True)

class Pipe:

    @property
    def params(self) -> Mapping[str,Any]:
        """Parameters describing the pipe."""
        return { }

    def __str__(self) -> str:
        return str(self.params)

class Source(ABC, Pipe, Generic[_T_out]):
    """A pipe that can be read."""

    @abstractmethod
    def read(self) -> _T_out:
        """Read the item."""
        ...

class Filter(ABC, Pipe, Generic[_T_in, _T_out]):
    """A pipe that can modify an item."""

    @abstractmethod
    def filter(self, item: _T_in) -> _T_out:
        """Filter the item."""
        ...

class Sink(ABC, Pipe, Generic[_T_in]):
    """A pipe that writes item."""

    @abstractmethod
    def write(self, item: _T_in) -> None:
        """Write the item."""
        ...

class Dense(ABC):
    __slots__ = ()
    
    @abstractmethod
    def __getitem__(self, key) -> Any:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __iter__(self) -> Iterator:
        ...

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._row, attr)

    def __eq__(self, o) -> bool:
        try:
            return len(self) == len(o) and all(map(eq, self, o))
        except:
            return False

class Sparse(ABC):

    @abstractmethod
    def __getitem__(self, key) -> Any:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __iter__(self) -> Iterator:
        ...

    @abstractmethod
    def keys(self) -> abc.KeysView:
        ...

    @abstractmethod
    def items(self) -> Iterable:
        ...

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._row, attr)

    def __eq__(self, o: object) -> bool:
        try:
            return set(self.items()) == set(o.items())
        except:
            return False

Sparse.register(abc.Mapping)
Dense.register(list)
Dense.register(tuple)
