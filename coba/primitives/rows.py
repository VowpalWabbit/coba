from operator import eq
from collections import abc
from abc import ABC, abstractmethod
from typing import Sequence, Iterator, Iterable, Any

class Dense(ABC):
    __slots__=()

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

    def copy(self) -> list:
        return list(iter(self))

class Dense_:
    ##Instantiating classes which inherit from Dense is moderately expensive due to the ABC checks.
    ##Therefore we keep Dense around for public API checks but internally we use Dense_ for inheritance.
    __slots__=('_row')

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._row, attr)

    def __eq__(self, o) -> bool:
        try:
            return len(self) == len(o) and all(map(eq, self, o))
        except:
            return False

    def copy(self) -> list:
        return list(iter(self))

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
            return dict(self.items()) == dict(o.items())
        except:
            return False

    def copy(self) -> dict:
        return dict(self.items())

class Sparse_:
    __slots__=('_row')
    ##Instantiating classes which inherit from Sparse is moderately expensive due to the ABC checks.
    ##Therefore we keep Sparse around for public API checks but internally we use Sparse_ for inheritance.

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._row, attr)

    def __eq__(self, o: object) -> bool:
        try:
            return dict(self.items()) == dict(o.items())
        except:
            return False

    def copy(self) -> dict:
        return dict(self.items())

class HashableSparse(abc.Mapping):
    __slots__=('_item','_hash')
    def __init__(self,item:Sparse):
        self._item = item

    def __getitem__(self,key):
        return self._item[key]

    def __iter__(self):
        return iter(self._item)

    def __len__(self):
        return len(self._item)

    def __hash__(self) -> int:
        try:
            return self._hash
        except:
            self._hash = hash(frozenset(self._item.items()))
            return self._hash

    def __eq__(self, o: object) -> bool:
        try:
            return frozenset(o.items()) == frozenset(self._item.items())
        except:
            return False

    def __repr__(self) -> str:
        return repr(self._item)

    def __str__(self) -> str:
        return str(self._item)

    def copy(self):
        return self._item.copy()

class HashableDense(abc.Sequence):
    __slots__=('_item','_hash')

    def __init__(self, item: Sequence) -> None:
        self._item = item

    def __getitem__(self,index):
        return self._item[index]

    def __iter__(self):
        return iter(self._item)

    def __len__(self):
        return len(self._item)

    def __eq__(self, o: object) -> bool:
        try:
            return len(self._item) == len(o) and all(map(eq, self._item, o))
        except:
            return False

    def __hash__(self) -> int:
        try:
            return self._hash
        except:
            self._hash = hash(tuple(self._item))
            return self._hash

    def __repr__(self) -> str:
        return repr(self._item)

    def __str__(self) -> str:
        return str(self._item)

Sparse.register(HashableSparse)
Sparse.register(abc.Mapping)
Sparse.register(Sparse_)
Dense.register(HashableDense)
Dense.register(list)
Dense.register(tuple)
Dense.register(Dense_)
