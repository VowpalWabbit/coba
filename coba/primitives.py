from operator import eq
from collections import abc, Counter
from abc import ABC, abstractmethod
from typing import Union, Mapping, Sequence, Iterator, Iterable, Callable, TypeVar, Generic, Any

def is_batch(item):
    return hasattr(item,'is_batch')

Context = Union[None, str, int, float, Sequence, Mapping]
Action  = Union[str, int, float, Sequence, Mapping]
Actions = Union[Sequence[Action],None]
Reward  = float
Rewards = Callable[[Action],Reward]

class Categorical(str):
    __slots__ = ('levels','as_int','as_onehot')

    def __new__(cls, value:str, levels: Sequence[str]) -> str:
        return str.__new__(Categorical,value)

    def __init__(self, value:str, levels: Sequence[str]) -> None:
        self.levels = levels
        self.as_int = levels.index(value)
        onehot = [0]*len(levels)
        onehot[self.as_int] = 1
        self.as_onehot = tuple(onehot)

    def __repr__(self) -> str:
        return f"Categorical('{self}',{self.levels})"

    def __reduce__(self):
        return Categorical, (str(self),list(map(str,self.levels)))

############################
#       dense/sparse       #
############################

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

############################
#        coba.pipes        #
############################

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

class Line(ABC, Pipe):
    """A pipe that can be run."""

    @abstractmethod
    def run(self):
        """Run the pipe."""
        ...

def resolve_params(pipes:Sequence[Pipe]):

    params = [p.params for p in pipes if hasattr(p,'params')]
    keys   = [ k for p in params for k in p.keys() ]
    counts = Counter(keys)
    index  = {}

    def resolve_key_conflicts(key):
        if counts[key] == 1:
            return key
        else:
            index[key] = index.get(key,0)+1
            return f"{key}{index[key]}"

    return { resolve_key_conflicts(k):v for p in params for k,v in p.items() }
