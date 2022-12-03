from abc import abstractmethod, ABC
from collections import abc
from operator import eq
from typing import Sequence, Any, Iterator, Iterable, Mapping

class Categorical(str):
    __slots__ = ('levels','onehot')
    
    def __new__(cls, value:str, levels: Sequence[str]) -> str:
        return str.__new__(Categorical,value)
    
    def __init__(self, value:str, levels: Sequence[str]) -> None:
        self.levels = levels
        self.onehot = [0]*len(levels)
        self.onehot[levels.index(value)] = 1
        self.onehot = tuple(self.onehot)

    def __repr__(self) -> str:
        return f"Categorical('{self}',{self.levels})"

class Batch(list):
    pass

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
            return dict(self.items()) == dict(o.items())
        except:
            return False

class HashableSparse(abc.Mapping):
    __slots__=('_item','_hash')
    def __init__(self,item:Mapping):
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
            self._hash = hash(tuple(self._item.items()))
            return self._hash
    
    def __repr__(self) -> str:
        return repr(self._item)

    def __str__(self) -> str:
        return str(self._item)

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

Dense.register(HashableDense)
Dense.register(list)
Dense.register(tuple)