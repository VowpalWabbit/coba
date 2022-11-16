from abc import ABC, abstractmethod

from bisect import bisect_left
from collections import abc, deque
from itertools import chain
from typing import Any, Union, Callable, Iterator, List, Dict, Deque
from typing import Hashable, Sequence, Mapping, Optional, Iterable, Tuple

from coba.exceptions import CobaException
from coba.pipes.primitives import Filter

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

class IRow(ABC):

    @abstractmethod
    def __getitem__(self, key) -> Any:
        ...

    @abstractmethod
    def __setitem__(self, key, item) -> None:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __iter__(self) -> Iterator:
        ...

class DenseRow2(IRow):

    def __init__(self, loader: Callable[[],Sequence] = None, loaded: Sequence = None) -> None:
        self._loader = loader
        self._loaded = loaded

    def _load_or_get(self):
        loaded = self._loaded
        if loaded is None: 
            loaded = self._loader()
            self._loaded = loaded
        return loaded

    def __getitem__(self, key: int):
        return self._load_or_get()[key]

    def __setitem__(self, key: Union[str,int], item: Any):
        self._load_or_get()[key] = item

    def __len__(self) -> int:
        return len(self._load_or_get())

    def __eq__(self, o: object) -> bool:
        try:
            return list(self) == list(o)
        except:
            return False

    def __iter__(self) -> Iterator:
        return iter(self._load_or_get())

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"DenseRow: {str(self._loaded) if self._loaded is not None else 'Unloaded'}"
 
class SparseRow2(IRow):

    def __init__(self, loader: Callable[[],Mapping] = None, loaded: Mapping = None) -> None:
        self._loader = loader
        self._loaded = loaded

    def _load_or_get(self):
        loaded = self._loaded
        if loaded is None: 
            loaded = self._loader()
            self._loaded = loaded
        return loaded

    def __getitem__(self, key: str):
        return self._load_or_get()[key]

    def __setitem__(self, key: str, item: Any):
        self._load_or_get()[key] = item

    def __len__(self) -> int:
        return len(self._load_or_get())

    def __eq__(self, o: object) -> bool:
        try:
            return dict(self) == dict(o)
        except:
            return False

    def keys(self):
        return self._load_or_get().keys()

    def __iter__(self) -> Iterator:
        return iter(self._load_or_get())

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"SparseRow: {str(self._loaded) if self._loaded is not None else 'Unloaded'}"

class HeaderRow(IRow):

    def __init__(self, row: IRow, head_map: Mapping[str,int], head_seq: Sequence[str] = None) -> None:
        self._row      = row
        self._head_map = head_map
        self.headers   = head_seq or sorted(head_map.keys(), key=lambda k: head_map[k])

    def _row_index(self, key:Union[int,str]) -> int:
        return key if key.__class__ is int else self._head_map[key]

    def __getitem__(self, key: Union[str,int]):
        return self._row[self._row_index(key)]

    def __setitem__(self, key: Union[str,int], item: Any):
        self._row[self._row_index(key)] = item

    def __len__(self) -> int:
        return len(self._row)

    def __eq__(self, o: object) -> bool:
        return self._row == o

    def __iter__(self) -> Iterator:
        return iter(self._row)

    def __repr__(self) -> str:
        return str(self._row)

    def __getattr__(self, name):
        return getattr(self._row, name)

class EncoderRow(IRow):

    def __init__(self, row: IRow, encoders: Sequence) -> None:
        self._row      = row
        self._encoders = encoders

    def __getitem__(self, key: int):
        return self._encoders[key](self._row[key])

    def __setitem__(self, key: int, item: Any):
        self._encoders[key] = lambda x:x
        self._row[key] = item

    def __len__(self) -> int:
        return len(self._row)

    def __eq__(self, o: object) -> bool:
        return list(self) == list(o)

    def __iter__(self) -> Iterator:
        return iter( e(v) for e,v in zip(self._encoders, self._row))

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self._row)

    def __getattr__(self, name):
        return getattr(self._row, name)

class SelectRow(IRow):
    
    def __init__(self, row: IRow, select: Sequence) -> None:
        self._row        = row
        self._select_seq = select
        self._select_set = set(select)

    def _key_check(self,key):
        if key.__class__ is int:
            return self._select_seq[key]
        elif key in self._select_set:
            return key
        else:
            raise KeyError(key)

    def __getitem__(self, key: Union[int,str]):
        return self._row[self._key_check(key)]

    def __setitem__(self, key: Union[int,str], item: Any):
        self._row[self._key_check(key)] = item

    def __len__(self) -> int:
        return len(self._select_seq)

    def __eq__(self, o: object) -> bool:
        return list(self) == list(o)

    def __iter__(self) -> Iterator:
        return iter(map(self._row.__getitem__, self._select_seq))

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(list(self))

    def __getattr__(self, name):
        return getattr(self._row, name)

class LabelRow2(IRow):

    def __init__(self, row: IRow, label: Union[str,int], label_alias: str) -> None:
        self._row       = row
        self._lbl_key   = label
        self._lbl_alias = label_alias

    def __getitem__(self, key: Union[int,str]):
        return self._row[self._lbl_key if key == self._lbl_alias else key]

    def __setitem__(self, key: Union[int,str], item: Any):
        self._row[self._lbl_key if key == self._lbl_alias else key] = item

    def __len__(self) -> int:
        return len(self._row)

    def __eq__(self, o: object) -> bool:
        return self._row == o

    def __iter__(self) -> Iterator:
        return iter(self._row)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self._row)

    def __getattr__(self, name):
        return getattr(self._row, name)

class DenseRow(abc.MutableSequence, IDenseRow):

    def __init__(self, loader: Callable[[],List] = None, loaded: List = None, missing: bool = False) -> None:

        self._loader : Callable[[],List] = loader
        self._loaded : List              = loaded
        self._indexes: List[int]         = []

        self._cmds: Deque = deque()

        self.encoders    : Union[Sequence,Mapping] = None
        self.headers     : Mapping[str,int]        = None
        self.headers_inv : Mapping[int,str]        = None
        self.missing     : bool                    = missing

        self._label_key = None
        self._label_enc = None
        self._label_ali = None

        if self._loader:
            def load():
                self._loaded  = self._loader()
                self._indexes = list(range(len(self._loaded)))
            self._cmds.appendleft((load,()))
        
        elif self._loaded:
            self._indexes = list(range(len(self._loaded)))

    def _run_cmds(self) -> None:
        while self._cmds: 
            f,a = self._cmds.pop()
            f(*a)

    def __getitem__(self, key: Union[str,int]):
        if self._cmds: self._run_cmds()

        if self._label_key:
            if key == self._label_ali or key == self._label_key:
                key = self._label_key
                key = self._indexes[key] if key.__class__ is int else self.headers[key]
                val = self._loaded[key]
                if self.encoders : val = self.encoders[key](val)
                if self._label_enc: val = self._label_enc(val)
                return val

        key = self._indexes[key] if key.__class__ is int else self.headers[key]
        return self.encoders[key](self._loaded[key]) if self.encoders else self._loaded[key]

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
            if self._label_key and (key == self._label_key) or (key == self._label_ali):
                key = self._label_key
                self._label_key = None
                self._label_ali = None
            
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
        self._label_key = key
        self._label_ali = alias
        self._label_enc = encoding

    def to_builtin(self):
        if self._cmds: self._run_cmds()
        
        V = map(self._loaded.__getitem__, self._indexes)
        if self.encoders:
            E = map(self.encoders.__getitem__, self._indexes)
            L = [ e(v) for e,v in zip(E,V) ]
        else:
            L = list(map(self._loaded.__getitem__, self._indexes))

        if self._label_enc:
            key = self._label_key
            ind = self._indexes[key] if key.__class__ is int else self.headers[key]
            lbl = bisect_left(self._indexes,ind)
            L[lbl] = self._label_enc(L[lbl])

        return tuple(L)

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

    def __init__(self, loader: Callable[[],Dict] = None, loaded: List = None, missing: bool = False) -> None:
        self._loader = loader
        self._loaded = loaded

        self._cmds: Deque[Tuple[Callable,Tuple]] = deque()

        self.encoders   : Union[Sequence,Mapping] = None
        self.headers    : Mapping[str,str]        = None
        self.headers_inv: Mapping[str,str]        = None
        self.missing    : bool                    = missing

        self._label_key   = None
        self._label_enc   = None
        self._label_ali = None

        if self._loader:
            def load(): self._loaded = self._loader()
            self._cmds.appendleft((load,()))

    def _run_cmds(self):
        while self._cmds: 
            f,a = self._cmds.pop()
            f(*a)

    def __getitem__(self, key: Hashable):
        if self._cmds: self._run_cmds()
        if self._label_key:
            if key == self._label_ali or key == self._label_key:
                key = self._label_key
                key  = key if not self.headers else self.headers[key]
                val = self._loaded.get(key,0)
                if self.encoders  : val = self.encoders[key](val)
                if self._label_enc: val = self._label_enc(val)
                return val

        key  = key if not self.headers else self.headers[key]
        val  = self._loaded.get(key,0)
        return self.encoders[key](val) if self.encoders else val

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
            if self._label_key and (key == self._label_key) or (key == self._label_ali):
                key = self._label_key
                self._label_key = None
                self._label_ali = None

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
        self._label_key = key
        self._label_enc = encoding
        self._label_ali = alias
    
    def to_builtin(self):
        from coba.utilities import HashableDict

        if self._cmds: self._run_cmds()
        if self.encoders:
            L = { k:self.encoders[k](v) for k,v in self._loaded.items() }
        else:
            L = dict(self._loaded)

        if self._label_enc:
            key    = self._label_key
            key    = key if not self.headers else self.headers[key]
            L[key] = self._label_enc(L[key])

        return HashableDict(L)

    @property
    def feats(self) -> Dict[str,Any]:
        if self._cmds: self._run_cmds()
        return { k:self[k] for k in self if k != self._label_key }

    @property
    def label(self) -> Any:
        if self._label_key is not None: return self.__getitem__(self._label_key)
        else: raise CobaException("This row has no defined label.")

class EncodeRow(Filter[Iterable[Union[IDenseRow,ISparseRow]],Iterable[Union[IDenseRow,ISparseRow]]]):

    def __init__(self, encoders: Union[Sequence,Mapping]) -> None:
        self._encoders = encoders

    def filter(self, items: Iterable[Union[IDenseRow,ISparseRow]]) -> Iterable[Union[IDenseRow,ISparseRow]]:

        first = next(iter(items))
        items = chain([first],items)

        encoders = self._encoders

        if isinstance(first,(SparseRow,DenseRow)):
            for item in items:
                item.encoders = encoders
                yield item
        else:
            for item in items:
                item = SparseRow(loaded=item) if isinstance(item,dict) else DenseRow(loaded=item)
                item.encoders    = self._encoders
                yield item

class IndexRow(Filter[Iterable[Union[IDenseRow,ISparseRow]],Iterable[Union[IDenseRow,ISparseRow]]]):

    def __init__(self, index: Mapping) -> None:
        self._fwd_index = index
        self._rev_index = { v:k for k,v in index.items()}

    def filter(self, items: Iterable[Union[IDenseRow,ISparseRow]]) -> Iterable[Union[IDenseRow,ISparseRow]]:

        first = next(iter(items))
        items = chain([first],items)

        fwd_index = self._fwd_index
        rev_index = self._rev_index

        if isinstance(first,(SparseRow,DenseRow)):
            for item in items:
                item.headers     = fwd_index
                item.headers_inv = rev_index
                yield item
        else:
            for item in items:
                item = SparseRow(loaded=item) if isinstance(item,dict) else DenseRow(loaded=item)
                item.headers     = self._fwd_index
                item.headers_inv = self._rev_index
                yield item

class LabelRow(Filter[Iterable[Union[IDenseRow,ISparseRow]],Iterable[Union[IDenseRow,ISparseRow]]]):

    def __init__(self, label: Union[int,str], alias:str = None, encoder: Callable[[Any],Any]=None) -> None:
        self._label    = label
        self._alias    = alias
        self._encoding = encoder

    def filter(self, items: Iterable[Union[IDenseRow,ISparseRow]]) -> Iterable[Union[IDenseRow,ISparseRow]]:

        first = next(iter(items))
        items = chain([first],items)

        label    = self._label
        encoding = self._encoding
        alias    = self._alias

        if isinstance(first,(SparseRow,DenseRow)):
            for item in items:
                item._label_key = label
                item._label_enc = encoding
                item._label_ali = alias
                yield item
        else:
            for item in items:
                item = SparseRow(loaded=item) if isinstance(item,dict) else DenseRow(loaded=item)
                item.set_label(label, alias, encoding)
                yield item
