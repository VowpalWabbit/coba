from collections import abc
from itertools import count, compress, chain, filterfalse, islice, repeat
from typing import Any, Union, Callable, Iterator, Sequence, Mapping, Iterable, Tuple
from coba.backports import Literal

from coba.primitives import Sparse, Dense, Categorical
from coba.primitives.rows import Dense_, Sparse_
from coba.utilities import peek_first
from coba.pipes.primitives import Filter
from coba.pipes.filters import Flatten

class LazyDense(Dense_):
    __slots__ = ('_row','_enc','headers','missing')

    def __init__(self, row: Union[Callable[[],Sequence],Sequence], enc: Sequence = None, headers: Mapping = None, missing: bool = None) -> None:
        self._row = row
        self._enc = enc

        if headers is not None: self.headers = headers
        if missing is not None: self.missing = missing

    def _load_or_get(self) -> Sequence:
        row = self._row
        if not callable(row): return row
        row = row()
        self._row = row
        return row

    def __getitem__(self, key: int):
        key = key if key.__class__ is int else self.headers[key]
        val = self._load_or_get()[key]
        enc = self._enc

        if not enc:
            return val

        else:
            try:
                return enc[key](val)
            except:
                if val in ['?','']: return None
                raise

    def __iter__(self) -> Iterator:
        if self._enc:
            return self._enc_all()
        else:
            return iter(self._load_or_get())

    def __len__(self) -> int:
        return len(self._load_or_get())

    def _enc_all(self)->Iterator:
        #placing this here gives us a slight perf boost
        #but is a bit of a conflation of functionality.
        #In theory this error checking logic with None
        #should be in an ARFF specific implementation.
        for e,v in zip(self._enc, self._load_or_get()):
            try:
                yield e(v)
            except:
                if v in ['?','']: 
                    yield None
                else:
                    raise

class LazySparse(Sparse_):
    __slots__=('_row','_enc','_nsp','_fwd','_inv','missing')

    def __init__(self, row: Union[Callable[[],Mapping],Mapping], enc: Mapping = {}, nsp: set = set(), fwd: Mapping = {}, inv: Mapping = {}, missing:bool = None) -> None:
        self._row    = row
        self._enc    = enc
        self._nsp    = nsp
        self._fwd    = fwd
        self._inv    = inv
        if missing is not None: self.missing = missing

    def _load_or_get(self) -> Mapping:
        row = self._row
        if not callable(row): return row
        row = row()
        self._row = row
        return row

    def __getitem__(self, key: str):
        key = self._fwd.get(key,key)
        enc = self._enc
        
        try:
            val = self._load_or_get()[key]
        except KeyError:
            if key not in self._nsp: raise
            val = "0"

        if not enc: 
            return val
        else:
            try:
                return enc.get(key,lambda x:x)(val)
            except:
                if val in ['?','']: return None
                raise

    def __iter__(self) -> Iterator:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._load_or_get())

    def keys(self) -> abc.KeysView:
        hdr_inv_get = self._inv.__getitem__ if self._inv else lambda x:x
        return set(map(hdr_inv_get, self._load_or_get().keys() | self._nsp))

    def items(self) -> Sequence:
        inv = self._inv
        enc = self._enc
        row = self._load_or_get()

        if enc:
            return tuple(self._enc_items())
        elif inv:
            return tuple((inv.get(k,k), v) for k,v in row.items())
        else:
            return tuple(row.items())
        
    def _enc_items(self)->Iterator:
        enc   = self._enc
        inv   = self._inv
        raw   = self._load_or_get()
        items = chain(raw.items(), zip(self._nsp-raw.keys(), repeat("0")))

        for k,v in items:
            try:
                v = enc.get(k,lambda x:x)(v)
            except:
                if v in ['?','']: v = None
                else: raise

            yield (inv.get(k,k) if inv else k,v)

class HeadDense(Dense_):
    __slots__=('_row','headers')

    def __init__(self, row: Dense, headers: Mapping[str,int] = None) -> None:
        self._row    = row
        self.headers = headers

    def __getitem__(self, key: Union[str,int]):
        return self._row[key if key.__class__ is int else self.headers[key]]

    def __iter__(self) -> Iterator:
        return iter(self._row)

    def __len__(self) -> int:
        return len(self._row)

class HeadSparse(Sparse_):

    __slots__=('_row', '_fwd', '_inv')

    def __init__(self, row: Sparse, head_fwd: Mapping[str,str], head_inv: Mapping[str,str]) -> None:
        self._row = row
        self._fwd = head_fwd
        self._inv = head_inv

    def __getitem__(self, key: Union[str,int]):
        return self._row[self._fwd[key]]

    def __iter__(self) -> Iterator:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._row)

    def keys(self) -> abc.KeysView:
        head_map_inv_get = self._inv.__getitem__ 
        return set(map(head_map_inv_get, self._row.keys()))

    def items(self) -> Sequence:
        head_map_inv_get = self._inv.__getitem__ 
        return tuple((head_map_inv_get(k),v) for k,v in self._row.items())

class HeadRows(Filter[Iterable[Union[Dense,Sparse]],Iterable[Union[Dense,Sparse]]]):

    def __init__(self, headers: Union[Sequence,Mapping]) -> None:
        if isinstance(headers, abc.Mapping):
            self._mapping = headers
        else:
            self._mapping = dict(zip(headers, count()))

    def filter(self, rows: Iterable[Union[Dense,Sparse]]) -> Iterable[Union[Dense,Sparse]]:
        first, rows = peek_first(rows)

        if isinstance(first, Dense):
            mapping  = self._mapping
            yield from (HeadDense(row, mapping) for row in rows)
        else:
            mapping     = self._mapping
            mapping_inv = {v:k for k,v in self._mapping.items()}
            yield from (HeadSparse(row, mapping, mapping_inv) for row in rows)

class EncodeDense(Dense_):
    __slots__=('_row','_encoders')

    def __init__(self, row: Dense, encoders: Sequence) -> None:
        self._row      = row
        self._encoders = encoders

    def __getitem__(self, key: Union[int,str]):
        return self._encoders[key](self._row[key])

    def __iter__(self) -> Iterator:
        return (e(v) for e,v in zip(self._encoders, self._row))

    def __len__(self) -> int:
        return len(self._encoders)

class EncodeSparse(Sparse_):
    __slots__=('_row','_enc','_nsp')

    def __init__(self, row: Sparse, encoders: Mapping, not_sparse: set) -> None:
        self._row = row
        self._enc = encoders
        self._nsp = not_sparse

    def __getitem__(self, key: Union[int,str]):
        try:
            return self._enc.get(key,lambda x:x)(self._row[key])
        except KeyError:
            if key in self._nsp: return self._enc.get(key, lambda x:x)("0")

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

class EncodeRows(Filter[Iterable[Union[Dense,Sparse]],Iterable[Union[Dense,Sparse]]]):

    def __init__(self, encoders: Union[Callable,Sequence,Mapping]) -> None:
        self._encoders = encoders

    def filter(self, rows: Iterable[Union[Any,Dense,Sparse]]) -> Iterable[Union[Dense,Sparse]]:
        enc = self._encoders
        first, rows = peek_first(rows)

        if not rows: return []

        if isinstance(first,Dense):
            if isinstance(enc,abc.Mapping):
                if hasattr(first, 'headers'):
                    enc = [ enc.get(h, enc.get(i, lambda x:x)) for i,h in enumerate(first.headers) ]
                else:
                    enc = [ enc.get(i, lambda x:x)             for i   in range(len(first))        ]
            return ( EncodeDense(row, enc) for row in rows )

        elif isinstance(first,Sparse):
            if isinstance(enc, abc.Sequence): enc = dict(enumerate(enc))
            nsp = set()
            for k,v in enc.items():
                try:
                    if v('0')!=0: nsp.add(k)
                except: pass #pragma: no cover
            return ( EncodeSparse(row, enc, nsp) for row in rows )

class DropOne(Dense_):
    __slots__=('_row','_ind')

    def __init__(self, row: Dense, ind: int) -> None:
        self._row = row
        self._ind = ind

    def __getitem__(self, key: int):
        if key >= self._ind: key += 1
        return self._row[key]

    def __iter__(self) -> Iterator:
        row = self._row
        ind = self._ind
        return iter(chain(islice(row,ind),islice(row,ind+1,None)))

    def __len__(self) -> int:
        return len(self._row)-1

class KeepDense(Dense_):
    __slots__=('_row', '_map', '_sel', '_len','headers')
    def __init__(self, row: Dense, mapping: Mapping[Union[str,int],int], selects: Sequence, len: int, headers: Mapping,) -> None:
        self._row = row
        self._map = mapping
        self._sel = selects
        self._len = len
        if headers: self.headers = headers

    def __getitem__(self, key: Union[int,str]):
        return self._row[self._map.get(key,10000000)]

    def __iter__(self) -> Iterator:
        return iter(compress(self._row, self._sel))

    def __len__(self) -> int:
        return self._len

class DropSparse(Sparse_):
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

class DropRows(Filter[Iterable[Union[Dense,Sparse]], Iterable[Union[Dense,Sparse]]]):
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
                headers = first.headers.items()
                indexes = list(compress(range(len(first)), selects))
            except:
                selects = [ i not in drop_cols for i in range(len(first)) ]
                headers = []
                indexes = list(compress(range(len(first)), selects))

            mapping = {k:v for k,v in chain(enumerate(indexes),headers)}
            length  = len(indexes)

            if headers: 
                external_indexes = dict(zip(indexes,count()))
                external_headers = { h: external_indexes[i] for h,i in headers if i in external_indexes }
            else:
                external_headers = None

            return mapping, selects, length, external_headers
        else:
            return set(drop_cols)

    def filter(self, rows: Iterable[Union[Dense,Sparse]]) -> Iterable[Union[Dense,Sparse]]:

        drop_cols   = self._drop_cols
        first, rows = peek_first(rows)

        rows = rows if not self._drop_row else filterfalse(self._drop_row, rows)

        if not drop_cols:
            return rows
        elif isinstance(first,Dense):
            mapping, selects, length, headers = DropRows.make_drop_row_args(first, drop_cols)
            return (KeepDense(row, mapping, selects, length, headers) for row in rows)
        else:
            drop_set = DropRows.make_drop_row_args(first, drop_cols)
            return (DropSparse(row, drop_set) for row in rows)

class LabelDense(Dense_):
    __slots__=('_row','_ind','_tipe')

    def __init__(self, row: Dense, ind: int, tipe: Literal['c','r','m']) -> None:
        self._row  = row
        self._ind  = ind
        self._tipe = tipe

    def __getitem__(self, key: Union[int,str]):
        return self._row[key]

    def __iter__(self) -> Iterator:
        return iter(self._row)

    def __len__(self) -> int:
        return len(self._row)

    @property
    def feats(self) -> Dense:
        return DropOne(self._row, self._ind)

    @property
    def label(self) -> Any:
        return self._row[self._ind]

    @property
    def tipe(self) -> Literal['c','r','m']:
        return self._tipe

    @property
    def labeled(self)-> Tuple[Dense,Any,Literal['c','r','m']]:
        return (DropOne(self._row, self._ind), self._row[self._ind], self._tipe)

class LabelSparse(Sparse_):
    __slots__=('_row', '_tipe', '_key')

    def __init__(self, row: Sparse, key: str, tipe: Literal['c','r','m']) -> None:
        self._row  = row
        self._key  = key
        self._tipe = tipe

    def __getitem__(self, key: str):
        try:
            return self._row[key]
        except KeyError:
            if key == self._key:
                return 0
            raise

    def __iter__(self) -> Iterator:
        return iter(self._row.keys() | {self._key})

    def __len__(self) -> int:
        return len(self._row.keys() | {self._key})

    def keys(self) -> abc.KeysView:
        return self._row.keys() | {self._key}

    def items(self) -> Sequence:
        key = self._key
        row = self._row
        items = tuple(row.items())
        if key not in row.keys():
            items += ((key,0),)
        return items

    @property
    def feats(self) -> Dense:
        return DropSparse(self._row, {self._key})

    @property
    def label(self) -> Any:
        return self[self._key]

    @property
    def tipe(self) -> Literal['c','r','m']:
        return self._tipe

    @property
    def labeled(self)-> Tuple[Sparse,Any,Literal['c','r','m']]:
        return (DropSparse(self._row, {self._key}), self[self._key], self._tipe)

class LabelRows(Filter[Iterable[Union[Dense,Sparse]],Iterable[Union[Dense,Sparse]]]):

    def __init__(self, label: Union[int,str], tipe: Literal['c','r','m']) -> None:
        self.label = label
        self.tipe  = tipe

    def filter(self, rows: Iterable[Union[Dense,Sparse]]) -> Iterable[Union[Dense,Sparse]]:
        label = self.label
        tipe  = self.tipe
        first, rows = peek_first(rows)

        if isinstance(first,Dense):
            ind = first.headers[label] if isinstance(label,str) else label
            return map(LabelDense, rows, repeat(ind), repeat(tipe))
        else:
            return map(LabelSparse, rows, repeat(label), repeat(tipe))

class EncodeCatRows(Filter[Iterable[Union[Any,Dense,Sparse]], Iterable[Union[Any,Dense,Sparse]]]):
    def __init__(self, tipe=Literal["onehot","onehot_tuple","string"], value_rows:bool = False) -> None:
        self._tipe = tipe
        self._value_rows = value_rows

    def filter(self, rows: Iterable[Union[Any,Dense,Sparse]]) -> Iterable[Union[Any,Dense,Sparse]]:

        if self._tipe is None: return rows
        first, rows = peek_first(rows)
        if not rows: return []
    
        if self._value_rows:
            rows = self._encode_value_generator(rows)
        elif isinstance(first,Dense):
            rows = self._encode_dense_generator(rows, first)
        elif isinstance(first,Sparse):
            rows = self._encode_sparse_generator(rows, first)
        else:
            rows = rows

        if self._tipe =='onehot':
            rows = Flatten().filter(rows)

        return rows

    def _encode_value_generator(self, rows):
        first,rows = peek_first(rows)
        if not isinstance(first,Categorical):
            yield from rows
        elif self._tipe == "string":
            yield from map(str,rows)
        elif "onehot" in self._tipe:
            for row in rows: yield row.as_onehot

    def _encode_dense_generator(self, rows, first):
        is_mutable = isinstance(first, list)
        cat_cols =  [i for i,v in enumerate(first) if isinstance(v,Categorical) ]
       
        if self._tipe == "string":
            for row in rows:
                row = row.copy() if is_mutable else list(row)
                for i in cat_cols: row[i] = str(row[i])
                yield row
        elif "onehot" in self._tipe:
            for row in rows:
                row = row.copy() if is_mutable else list(row)
                for i in cat_cols: row[i] = row[i].as_onehot
                yield row

    def _encode_sparse_generator(self, rows, first):
        cat_cols = [k for k,v in first.items() if isinstance(v,Categorical) ]
        
        if self._tipe == "string":
            for row in rows:
                row = row.copy()
                for k in cat_cols: row[k] = str(row[k])
                yield row
        elif "onehot" in self._tipe:
            for row in rows:
                row = row.copy()
                for k in cat_cols: row[k] = row[k].as_onehot
                yield row
