from collections import abc
from itertools import count, compress, chain, filterfalse
from typing import Any, Union, Callable, Iterator, Sequence, Mapping, Iterable, Tuple
from coba.backports import Literal

from coba.encodings import OneHotEncoder
from coba.utilities import peek_first
from coba.pipes.primitives import Filter, Sparse, Dense
from coba.pipes.filters import Flatten

class Categorical:
    def __init__(self, value: Any, levels: Sequence[Any]) -> None:
        self.value = value
        self.levels = levels

class LazyDense(Dense):
    __slots__ = ('_row','missing')

    def __init__(self, row: Union[Callable[[],Sequence],Sequence], missing: bool = None) -> None:
        self._row    = row
        self.missing = missing

    def _load_or_get(self) -> Sequence:
        row = self._row
        if not callable(row): return row
        row = row()
        self._row = row
        return row

    def __getitem__(self, key: int):
        return self._load_or_get()[key]

    def __iter__(self) -> Iterator:
        return iter(self._load_or_get())

    def __len__(self) -> int:
        return len(self._load_or_get())

class HeadDense(Dense):
    __slots__=('_row','_map','headers')

    def __init__(self, row: Dense, head_seq: Sequence[str], head_map: Mapping[str,int] = None) -> None:
        self._row    = row
        self._map    = head_map or dict(zip(head_seq, count()))
        self.headers = head_seq

    def __getitem__(self, key: Union[str,int]):
        return self._row[key if key.__class__ is int else self._map[key]]

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
        return self._encoders[key](self._row[key])

    def __iter__(self) -> Iterator:
        return (e(v) for e,v in zip(self._encoders, self._row))

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
    __slots__=('_row', '_labeled')

    def __init__(self, row: Dense, key: Union[int,str], tipe: Literal['c','r','m'], feats: Dense) -> None:
        self._row     = row
        self._labeled = (feats, row[key], tipe)

    def __getitem__(self, key: Union[int,str]):
        return self._row[key]

    def __iter__(self) -> Iterator:
        return iter(self._row)

    def __len__(self) -> int:
        return len(self._row)

    @property
    def labeled(self)-> Tuple[Dense,Any,Literal['c','r','m']]:
        return self._labeled

class LazySparse(Sparse):
    __slots__=('_row', '_row', '_missing')
    
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

    __slots__=('_row', '_head_map_fwd', '_head_map_inv')

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
    __slots__=('_row', '_enc','_nsp')

    def __init__(self, row: Sparse, encoders: Mapping, not_sparse: set) -> None:
        self._row = row
        self._enc = encoders
        self._nsp = not_sparse

    def __getitem__(self, key: Union[int,str]):
        try:
            val = self._row[key]
        except KeyError:
            val = "0"
        return self._enc.get(key, lambda x:x)(val)

    def __iter__(self) -> Iterator:
        return iter(self._row.keys() | self._nsp)

    def __len__(self) -> int:
        return len(self._row.keys() | self._nsp)

    def keys(self) -> abc.KeysView:
        return self._row.keys() | self._nsp

    def items(self) -> Sequence:
        t1 = tuple((k, self._enc.get(k,lambda x:x)(v)) for k,v in self._row.items())
        t2 = tuple((k, self._enc[k]("0")) for k in self._nsp-self._row.keys())
        return t1+t2

class DropSparse(Sparse):
    __slots__=('_row','_drop_set')
    
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
    __slots__=('_row','_lbl_key', '_labeled', '_keys')
    
    def __init__(self, row: Sparse, label: str, tipe: Literal['c','r','m'], feats: Sparse) -> None:

        def labeled():
            try:
                val = row[label]
            except KeyError:
                val = 0

            return (feats, val, tipe)

        self._row     = row
        self._lbl_key = label
        self._labeled = labeled
        self._keys    = lambda: self._row.keys() | {label}

    def __getitem__(self, key: str):
        try:
            return self._row[key]
        except KeyError:
            if key == self._lbl_key: 
                return 0
            raise

    def __iter__(self) -> Iterator:
        return iter(self._keys())

    def __len__(self) -> int:
        return len(self._keys())

    def keys(self) -> abc.KeysView:
        return self._keys()

    def items(self) -> Sequence:
        key = self._lbl_key
        row = self._row
        items = tuple(row.items())
        if key not in row.keys():
            items += ((key,0),)
        return items

    @property
    def labeled(self)-> Tuple[Dense,Any,Literal['c','r','m']]:
        return self._labeled()

class EncodeRows(Filter[Iterable[Union[Dense,Sparse]],Iterable[Union[Dense,Sparse]]]):

    def __init__(self, encoders: Union[Sequence,Mapping]) -> None:
        self._encoders = encoders

    def filter(self, rows: Iterable[Union[Dense,Sparse]]) -> Iterable[Union[Dense,Sparse]]:
        encs = self._encoders
        first, rows = peek_first(rows)

        if first is None:
            return rows

        if isinstance(first,Dense):
            if isinstance(encs,abc.Mapping):
                if hasattr(first, 'headers'):
                    encs = [ encs.get(h, encs.get(i, lambda x:x)) for i,h in enumerate(first.headers) ]
                else:
                    encs = [ encs.get(i, lambda x:x)              for i   in range(len(first))        ]
            return ( EncodeDense(row, encs) for row in rows )

        if isinstance(first,Sparse):
            if isinstance(encs, abc.Sequence): encs = dict(enumerate(encs))
            not_sparse_encoding = set(k for k,v in encs.items() if v('0')!=0)
            return ( EncodeSparse(row, encs, not_sparse_encoding) for row in rows )

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

    def __init__(self, label: Union[int,str], tipe: Literal['c','r','m']) -> None:
        self.label = label
        self.tipe  = tipe

    def filter(self, rows: Iterable[Union[Dense,Sparse]]) -> Iterable[Union[Dense,Sparse]]:
        label = self.label
        tipe  = self.tipe
        first, rows = peek_first(rows)

        if isinstance(first,Dense):
            mapping, selects, length = DropRows.make_drop_row_args(first, [label])            
            for row in rows:
                feats = KeepDense(row, mapping, selects, length)
                yield LabelDense(row, label, tipe, feats)
        else:
            drop_set = DropRows.make_drop_row_args(first, [label])
            for row in rows:
                feats = DropSparse(row, drop_set)
                yield LabelSparse(row, label, tipe, feats)

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

    @staticmethod
    def make_drop_row_args(first, drop_cols) -> Tuple:
        if isinstance(first,Dense):
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
            return mapping, selects, length
        else:
            return set(drop_cols)

    def filter(self, rows: Iterable[Union[Sequence,Mapping]]) -> Iterable[Union[Sequence,Mapping]]:

        drop_cols   = self._drop_cols
        first, rows = peek_first(rows)

        rows = rows if not self._drop_row else filterfalse(self._drop_row, rows)

        if not drop_cols:
            yield from rows
        elif isinstance(first,Dense):
            mapping, selects, length = DropRows.make_drop_row_args(first, drop_cols)
            yield from (KeepDense(row, mapping, selects, length) for row in rows)
        else:
            drop_set = DropRows.make_drop_row_args(first, drop_cols)
            yield from (DropSparse(row, drop_set) for row in rows)

class EncodeCatRows(Filter[Iterable[Union[Sequence,Mapping]], Iterable[Union[Sequence,Mapping]]]):
    def __init__(self, tipe=Literal["onehot","onehot_tuple","string"]) -> None:
        self._tipe = tipe

    def filter(self, rows: Iterable[Union[Any,Sequence,Mapping]]) -> Iterable[Union[Any,Sequence,Mapping]]:

        first, rows = peek_first(rows)
        
        if self._tipe == 'string':
            make_encoder = lambda levels: (lambda c: str(c.value if isinstance(c, Categorical) else c))
        else:
            make_encoder = lambda levels: (lambda c, e=OneHotEncoder(levels):e.encode(c.value if isinstance(c, Categorical) else c))

        if not isinstance(first,(Dense,Sparse)):
            return rows
        elif isinstance(first,Dense):
            enc = { i: make_encoder(v.levels) for i,v in enumerate(first) if isinstance(v,Categorical) }
        elif isinstance(first,Sparse):
            enc = { k: make_encoder(v.levels) for k,v in first.items() if isinstance(v,Categorical) }

        rows = EncodeRows(enc).filter(rows)

        if self._tipe =='onehot':
            rows = Flatten().filter(rows)

        return rows
