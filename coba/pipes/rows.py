from abc import ABC, abstractmethod

from bisect import bisect_left
from collections import abc, deque
from typing import Any, Union, Callable, Iterator, overload, List, Dict, Deque, Hashable, Sequence, Mapping, Optional
from coba.backports import Literal

from coba.exceptions import CobaException
from coba.pipes.primitives import Filter, Source

class IDenseRow(ABC):

    @abstractmethod
    def __getitem__(self, key) -> Any:
        ...

    @abstractmethod
    def __setitem__(self, key, item) -> None:
        ...

    @abstractmethod
    def __delitem__(self, key) -> None:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __iter__(self) -> Iterator:
        ...

    @abstractmethod
    def pop(self, key) -> Any:
        ...

class ISparseRow(ABC):

    @abstractmethod
    def __getitem__(self, key) -> Any:
        ...

    @abstractmethod
    def __setitem__(self, key, item) -> None:
        ...

    @abstractmethod
    def __delitem__(self, key) -> None:
        ...
    
    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __iter__(self) -> Iterator:
        ...
    
    @abstractmethod
    def pop(self, key) -> Any:
        ...

class DenseRow(abc.MutableSequence, IDenseRow):

    @overload
    def __init__(self,*,loader: Source[List], missing: bool = False) -> None:
        ...

    @overload
    def __init__(self,*,loaded: List, missing: bool = False) -> None:
        ...

    def __init__(self,**kwargs) -> None:

        self._loader : Source[List] = kwargs.get('loader',None)
        self._loaded : List         = kwargs.get('loaded',None)
        self._indexes: List[int]    = list(range(0 if not self._loaded else len(self._loaded)))

        self._cmds: Deque = deque()
        self._label_enc = None

        self.encoders    : Union[Sequence,Mapping] = None
        self.headers     : Mapping[str,int]        = None
        self.headers_inv : Mapping[int,str]        = None
        self.missing     : bool                    = kwargs.get('missing',False)

        self._label_key = None
        self._label_enc = None
        self._label_alias = None

        if self._loader:
            def load(): 
                self._loaded  = self._loader.read()
                self._indexes = list(range(len(self._loaded)))
            self._cmds.appendleft((load,()))

    def _run_cmds(self) -> None:
        while self._cmds: 
            f,a = self._cmds.pop()
            f(*a)

    def __getitem__(self, key: Union[str,int]):
        if self._cmds: self._run_cmds()
        key  = self._label_key if key == self._label_alias else key
        enc2 = self._label_enc if key == self._label_key else None
        key  = self._indexes[key] if key.__class__ is int else self.headers[key]
        val  = self._loaded[key]
        enc1 = self.encoders[key] if self.encoders else None
        if enc1: val = enc1(val)
        if enc2: val = enc2(val)
        return val

    def __setitem__(self, key: Hashable, item: Any):
        if self._loaded is None:
            self._cmds.appendleft((self.__setitem__,(key,item)))
        else:
            key = self._indexes[key] if key.__class__ is int else self.headers[key]
            self._loaded[key] = item

    def __delitem__(self, key: Hashable):
        if self._loaded is None:
            self._cmds.appendleft((self.__delitem__,(key,)))
        else:
            key = self._label_key if key == self._label_alias else key
            key = self._indexes[key] if key.__class__ is int else self.headers[key]
            del self._indexes[bisect_left(self._indexes,key)]

    def insert(self, key: int, item: Any):
        raise NotImplementedError()

    def __len__(self) -> int:
        if self._cmds: self._run_cmds()
        return len(self._loaded)

    def __eq__(self, o: object) -> bool:
        return isinstance(o,abc.Sequence) and list(self) == list(o)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"DenseRow: {str(self._loaded) if self._loaded is not None else 'Unloaded'}"

    def set_label(self, key: Hashable, alias: Hashable, encoding: Callable[[Any],Any]) -> None:
        self._label_key   = key
        self._label_alias = alias
        self._label_enc   = encoding

    @property
    def feats(self) -> Sequence[Any]:
        if self._cmds: self._run_cmds()
        label_index = self._label_key if not self.headers else self.headers.get(self._label_key,self._label_key)
        return [ self[i] for i in range(len(self._indexes)) if self._indexes[i] != label_index]

    @property
    def label(self) -> Any:
        if self._label_key is not None: return self.__getitem__(self._label_key)
        else: raise CobaException("This row has no defined label.")

class SparseRow(abc.MutableMapping, ISparseRow):
    @overload
    def __init__(self,*,loader: Source[Dict], missing: bool = False) -> None:
        ...

    @overload
    def __init__(self,*,loaded: Dict, missing: bool = False) -> None:
        ...

    def __init__(self,**kwargs) -> None:
        self._loader : Source[Dict] = kwargs.get('loader',None)
        self._loaded : Dict         = kwargs.get('loaded',None)

        self._cmds: Deque[Callable] = deque()

        self.encoders   : Union[Sequence,Mapping] = None
        self.headers    : Mapping[str,str]        = None
        self.headers_inv: Mapping[str,str]        = None
        self.missing    : bool                    = kwargs.get('missing',False)

        self._label_key = None
        self._label_enc = None
        self._label_alias = None

        if self._loader:
            def load(): self._loaded = self._loader.read()
            self._cmds.appendleft((load,()))

    def _run_cmds(self):
        while self._cmds: 
            f,a = self._cmds.pop()
            f(*a)

    def __getitem__(self, key: Hashable):
        if self._cmds: self._run_cmds()
        key  = self._label_key if key == self._label_alias else key
        enc2 = self._label_enc if key == self._label_key else None
        key  = key if not self.headers else self.headers[key]
        val  = self._loaded.get(key,0)
        enc1 = self.encoders[key] if self.encoders else None
        if enc1: val = enc1(val)
        if enc2: val = enc2(val)
        return val

    def __setitem__(self, key: Hashable, item: Any):
        if self._loaded is None:
            self._cmds.appendleft((self.__setitem__,(key,item)))
        else:
            key = key if not self.headers else self.headers[key]
            self._loaded[key] = item

    def __delitem__(self, key: Hashable):
        if self._loaded is None:
            self._cmds.appendleft( (self.__delitem__, (key,)) )
        else:
            key = self._label_key if key == self._label_alias else key
            key = key if not self.headers else self.headers[key]
            if key in self._loaded: del self._loaded[key]

    def __iter__(self) -> Iterator:
        if self._cmds: self._run_cmds()
        
        return iter(self._loaded if not self.headers_inv else map(self.headers_inv.__getitem__,self._loaded))

    def __len__(self) -> int:
        while self._cmds: 
            f,a = self._cmds.pop()
            f(*a)
        return len(self._loaded)

    def __eq__(self, o: object) -> bool:
        return isinstance(o,abc.Mapping) and dict(self) == dict(o)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"SparseRow: {str(self._loaded) if self._loaded else 'Unloaded'}"

    def set_label(self, key: Hashable, alias:Hashable, encoding: Optional[Callable[[Any],Any]]) -> None:
        self._label_key   = key
        self._label_enc   = encoding
        self._label_alias = alias
    
    @property
    def feats(self) -> Dict[str,Any]:
        if self._cmds: self._run_cmds()
        return { k:self[k] for k in self if k != self._label_key }

    @property
    def label(self) -> Any:
        if self._label_key is not None: return self.__getitem__(self._label_key)
        else: raise CobaException("This row has no defined label.")

class DropRow(Filter[Union[IDenseRow,ISparseRow],Union[IDenseRow,ISparseRow]]):

    def __init__(self, cols: Sequence) -> None:
        self._cols = sorted(set(cols),reverse=True) or []

    def filter(self, item: Union[IDenseRow,ISparseRow]) -> Union[IDenseRow,ISparseRow]:
        for c in self._cols: del item[c]
        return item

class EncodeRow(Filter[Union[IDenseRow,ISparseRow],Union[IDenseRow,ISparseRow]]):

    def __init__(self, encoders: Union[Sequence,Mapping]) -> None:
        self._encoders = encoders

    def filter(self, item: Union[IDenseRow,ISparseRow]) -> Union[IDenseRow,ISparseRow]:

        try:
            item.encoders = self._encoders
        except:
            item = SparseRow(loaded=item) if isinstance(item,dict) else DenseRow(loaded=item)
            item.encoders    = self._encoders
        
        return item

class IndexRow(Filter[Union[IDenseRow,ISparseRow],Union[IDenseRow,ISparseRow]]):

    def __init__(self, index: Mapping) -> None:
        self._fwd_index = index
        self._rev_index = { v:k for k,v in index.items()}

    def filter(self, item: Union[IDenseRow,ISparseRow]) -> Union[IDenseRow,ISparseRow]:
        
        try:
            item.headers     = self._fwd_index
            item.headers_inv = self._rev_index
        except:
            item = SparseRow(loaded=item) if isinstance(item,dict) else DenseRow(loaded=item)
            item.headers     = self._fwd_index
            item.headers_inv = self._rev_index

        return item

class LabelRow(Filter[Union[IDenseRow,ISparseRow],Union[IDenseRow,ISparseRow]]):

    def __init__(self, label: Union[int,str], alias:str = None, encoder: Callable[[Any],Any]=None) -> None:
        self._label    = label
        self._alias    = alias
        self._encoding = encoder

    def filter(self, item: Union[IDenseRow,ISparseRow]) -> Union[IDenseRow,ISparseRow]:

        try:
            item.set_label(self._label, self._alias, self._encoding)
        except:
            item = SparseRow(loaded=item) if isinstance(item,dict) else DenseRow(loaded=item)
            item.set_label(self._label, self._alias, self._encoding)

        return item
