from collections import abc
from itertools import count, compress, chain, filterfalse
from typing import Any, Union, Callable, Iterator, Sequence, Mapping, Iterable, Tuple

from coba.utilities import peek_first
from coba.pipes.primitives import Filter, Sparse, Dense

class LazyDense(Dense):
    __slots__ = ('_row','missing')
    
    def __init__(self, row: Union[Callable[[],Sequence],Sequence], missing: bool = None) -> None:        
        self._row    = row
        self.missing = missing

    def _load_or_get(self) -> Sequence:
        row = self._row
        if callable(row):
            row = self._row()
            self._row = row
        self._load_or_get = lambda : row
        return row

    def __getitem__(self, key: int):
        return self._load_or_get()[key]

    def __iter__(self) -> Iterator:
        return iter(self._load_or_get())

    def __len__(self) -> int:
        return len(self._load_or_get())

class HeadDense(Dense):
    __slots__=('_row','_head_map','headers')

    def __init__(self, row: Dense, head_seq: Sequence[str], head_map: Mapping[str,int] = None) -> None:
        self._row      = row
        self._head_map = head_map or dict(zip(head_seq, count()))
        self.headers   = head_seq

    def __getitem__(self, key: Union[str,int]):
        return self._row[key if key.__class__ is int else self._head_map[key]]

    def __iter__(self) -> Iterator:
        return iter(self._row)

    def __len__(self) -> int:
        return len(self._row)

class EncodeDense(Dense):
    __slots__=('_row','_encoders')
    
    def __init__(self, row: Dense, encoders: Sequence) -> None:
        self._row      = row
        self._encoders = encoders

    def __getitem__(self, key: Union[int,str]):
        enc = self._encoders[key]
        return enc(self._row[key]) if enc else self._row[key]

    def __iter__(self) -> Iterator:
        return iter( e(v) if e else v for e,v in zip(self._encoders, self._row))

    def __len__(self) -> int:
        return len(self._row)

class KeepDense(Dense):
    __slots__=('_row', '_map', '_sel', '_len')
    def __init__(self, row: Dense, mapping: Mapping[Union[str,int],int], selects: Sequence, len: int) -> None:
        self._row = row
        self._map = mapping
        self._sel = selects
        self._len = len

    def __getitem__(self, key: Union[int,str]):
        return self._row[self._map.get(key,10000000)]

    def __iter__(self) -> Iterator:
        return iter(compress(self._row, self._sel))

    def __len__(self) -> int:
        return self._len

class LabelDense(Dense):
    __slots__=('_row', '_lbl_keys', '_lbl_enc')
    def __init__(self, row: Dense, keys: Union[Tuple[int],Tuple[int,str]], encoder: Callable[[Any],Any]) -> None:
        self._row      = row
        self._lbl_keys = keys
        self._lbl_enc  = encoder

    def __getitem__(self, key: Union[int,str]):
        if self._lbl_enc and key in self._lbl_keys:
            return self._lbl_enc(self._row[key])
        return self._row[key]            

    def __iter__(self) -> Iterator:
        enc = self._lbl_enc
        if not enc:
            return iter(self._row)
        else:
            key = self._lbl_keys[0]
            return iter(v if i!= key else enc(v) for i,v in enumerate(self._row))

    def __len__(self) -> int:
        return len(self._row)

    @property
    def labeled(self)-> Tuple[Dense,Any]:
        feats = list(self._row)
        label = feats.pop(self._lbl_keys[0])
        if self._lbl_enc:
            return (feats, self._lbl_enc(label))
        else:
            return (feats, label)

class LazySparse(Sparse):

    def __init__(self, row: Union[Callable[[],Mapping],Mapping], missing:bool = None) -> None:
        self._row  = row
        self.missing = missing 

    def _load_or_get(self) -> Mapping:
        row = self._row
        if callable(row):
            row = self._row()
            self._row = row
        self._load_or_get = lambda : row
        return row

    def __getitem__(self, key: str):
        return self._load_or_get()[key]

    def __iter__(self) -> Iterator:
        return iter(self._load_or_get())

    def __len__(self) -> int:
        return len(self._load_or_get())

    def keys(self) -> abc.KeysView:
        return self._load_or_get().keys()

    def items(self) -> Sequence:
        return tuple(self._load_or_get().items())

class HeadSparse(Sparse):

    def __init__(self, row: Sparse, head_map_fwd: Mapping[str,str], head_map_inv: Mapping[str,str]) -> None:
        self._row          = row
        self._head_map_fwd = head_map_fwd
        self._head_map_inv = head_map_inv

    def __getitem__(self, key: Union[str,int]):
        return self._row[self._head_map_fwd[key]]

    def __iter__(self) -> Iterator:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._row)

    def keys(self) -> abc.KeysView:
        head_map_inv_get = self._head_map_inv.__getitem__ 
        return set(map(head_map_inv_get, self._row.keys()))

    def items(self) -> Sequence:
        head_map_inv_get = self._head_map_inv.__getitem__ 
        return tuple((head_map_inv_get(k),v) for k,v in self._row.items())

class EncodeSparse(Sparse):

    def __init__(self, row: Sparse, encoders: Mapping) -> None:
        self._row      = row
        self._encoders = encoders

    def __getitem__(self, key: Union[int,str]):
        return self._encoders[key](self._row[key])

    def __iter__(self) -> Iterator:
        return iter(self._row)

    def __len__(self) -> int:
        return len(self._row)

    def keys(self) -> abc.KeysView:
        return self._row.keys()

    def items(self) -> Sequence:
        enc = self._encoders
        return tuple((k, enc[k](v) if k in enc else v ) for k,v in self._row.items())

class DropSparse(Sparse):

    def __init__(self, row: Sparse, drop_set: set = None) -> None:
        self._row      = row
        self._drop_set = drop_set

    def _key_check(self,key):
        if key not in self._drop_set:
            return key
        raise KeyError(key)

    def __getitem__(self, key: Union[int,str]):
        return self._row[self._key_check(key)]

    def __iter__(self) -> Iterator:
        return iter(self._row.keys()-self._drop_set)

    def __len__(self) -> int:
        return len(self.keys())

    def keys(self) -> abc.KeysView:
        return self._row.keys() - self._drop_set

    def items(self) -> Sequence:
        drop = self._drop_set
        yield from (item for item in self._row.items() if item[0] not in drop) 

class LabelSparse(Sparse):

    def __init__(self, row: Sparse, label: str, encoder: Callable[[Any],Any] = None) -> None:
        self._row     = row
        self._lbl_key = label
        self._lbl_enc = encoder

        lbl_key_set = {label}
        self._keys = lambda: self._row.keys() | lbl_key_set

    def __getitem__(self, key: str):
        if key != self._lbl_key:
            return self._row[key]
        else:
            enc = self._lbl_enc
            try:
                val = self._row[key]
            except KeyError:
                val = 0
            return val if not enc else enc(val)

    def __iter__(self) -> Iterator:
        return iter(self._keys())

    def __len__(self) -> int:
        return len(self._keys())

    def keys(self) -> abc.KeysView:
        return self._keys()

    def items(self) -> Sequence:
        enc  = self._lbl_enc
        key  = self._lbl_key
        if key in self._row.keys():
            if not enc:
                return tuple(self._row.items())
            else:
                return tuple(kv if kv[0]!=key else (kv[0],enc(kv[1])) for kv in self._row.items())
        else:
            if not enc:
                return tuple(self._row.items()) + ((key,0),)
            else:
                return tuple(self._row.items()) + ((key,enc(0)),)

    @property
    def labeled(self)-> Tuple[Dense,Any]:
        feats = dict(self._row.items())
        label = feats.pop(self._lbl_key,0)
        if self._lbl_enc:
            return (feats, self._lbl_enc(label))
        else:
            return (feats, label)

class EncodeRows(Filter[Iterable[Union[Dense,Sparse]],Iterable[Union[Dense,Sparse]]]):

    def __init__(self, encoders: Union[Sequence,Mapping]) -> None:
        self._encoders = encoders

    def filter(self, rows: Iterable[Union[Dense,Sparse]]) -> Iterable[Union[Dense,Sparse]]:
        encs = self._encoders
        first, rows = peek_first(rows)

        if first and isinstance(first,Dense):
            if isinstance(encs,abc.Mapping):
                if isinstance(list(encs.keys())[0],str):
                    encs = [ encs.get(h) for h in first.headers ]
                else:
                    encs = [ encs.get(h) for h in range(len(first)) ]
            yield from (EncodeDense(row, encs) for row in rows)
        else:
            if isinstance(encs, abc.Sequence): encs = dict(enumerate(encs))
            yield from (EncodeSparse(row, encs) for row in rows)

class HeadRows(Filter[Iterable[Union[Dense,Sparse]],Iterable[Union[Dense,Sparse]]]):

    def __init__(self, headers: Union[Sequence,Mapping]) -> None:
        if isinstance(headers, abc.Mapping):
            self._sequence = None
            self._mapping  = headers
        else:
            self._sequence = headers
            self._mapping  = dict(zip(headers, count()))

    def filter(self, rows: Iterable[Union[Dense,Sparse]]) -> Iterable[Union[Dense,Sparse]]:
        first, rows = peek_first(rows)

        if isinstance(first, Dense):
            sequence = self._sequence or sorted(self._mapping, key=lambda k:self._mapping[k])
            mapping  = self._mapping
            yield from (HeadDense(row, sequence, mapping) for row in rows)
        else:
            mapping     = self._mapping
            mapping_inv = dict((v,k) for k,v in self._mapping.items()) 
            yield from (HeadSparse(row, mapping, mapping_inv) for row in rows)

class LabelRows(Filter[Iterable[Union[Dense,Sparse]],Iterable[Union[Dense,Sparse]]]):

    def __init__(self, label: Union[int,str], encoder: Callable[[Any],Any] = None) -> None:
        self.label  = label
        self.encoder = encoder

    def filter(self, rows: Iterable[Union[Dense,Sparse]]) -> Iterable[Union[Dense,Sparse]]:
        label = self.label
        first, rows = peek_first(rows)

        if isinstance(first,Dense):
            if isinstance(label,str): 
                keys = (first.headers.index(label),label)
            elif hasattr(first,'headers'):
                keys = (label, first.headers[label])
            else:
                keys = (label,)
            yield from (LabelDense(row, keys, self.encoder) for row in rows)
        else:
            yield from (LabelSparse(row, label, self.encoder) for row in rows)

class DropRows(Filter[Iterable[Union[Sequence,Mapping]], Iterable[Union[Sequence,Mapping]]]):
    """A filter which drops rows and columns from in table shaped data."""

    def __init__(self,
        drop_cols: Sequence[Union[str,int]] = [],
        drop_row: Callable[[Union[Sequence,Mapping]], bool] = None) -> None:
        """Instantiate a Drop filter.

        Args:
            drop_cols: Feature names which should be dropped.
            drop_row: A function that accepts a row and returns True if it should be dropped.
        """

        self._drop_cols = set(drop_cols)
        self._drop_row  = drop_row

    def filter(self, rows: Iterable[Union[Sequence,Mapping]]) -> Iterable[Union[Sequence,Mapping]]:

        drop_cols   = self._drop_cols
        first, rows = peek_first(rows)

        rows = rows if not self._drop_row else filterfalse(self._drop_row, rows)

        if not drop_cols:
            yield from rows
        elif isinstance(first,Dense):
            try:
                selects = [ not any(i in drop_cols for i in I) for I in enumerate(first.headers) ]
                headers = list(compress(first.headers, selects))
                indexes = list(compress(range(len(first)), selects))
            except:
                selects = [ i not in drop_cols for i in range(len(first)) ]
                headers = []
                indexes = list(compress(range(len(first)), selects))

            mapping = {k:v for k,v in chain(enumerate(indexes),zip(headers,indexes))}
            length  = len(indexes)
            yield from (KeepDense(row, mapping, selects, length) for row in rows)
        else:
            drop_set = set(drop_cols)
            yield from (DropSparse(row, drop_set) for row in rows)
