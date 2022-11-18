from abc import ABC, abstractmethod

from collections import abc
from itertools import count, chain
from typing import Any, Union, Callable, Iterator, Sequence, Mapping, Iterable

from coba.utilities import peek_first
from coba.pipes.primitives import Filter, Sparse, Dense

class LazyDense(Dense):

    def __init__(self, loader_or_row: Union[Callable[[],Sequence],Sequence]) -> None:
        
        if callable(loader_or_row):
            self._loader = loader_or_row
            self._row    = None
        else:
            self._loader = None
            self._row    = loader_or_row

    def _load_or_get(self):
        row = self._row
        if row is None: 
            row = self._loader()
            self._row = row
        return row

    def __getitem__(self, key: int):
        return self._load_or_get()[key]

    def __iter__(self) -> Iterator:
        return iter(self._load_or_get())

    def __len__(self) -> int:
        return len(self._load_or_get())

class MissingDense(Dense):
    def __init__(self, row: Dense, missing:bool) -> None:
        self._row = row
        self.missing = missing
    
    def __getitem__(self, key: Union[str,int]):
        return self._row[key]

    def __iter__(self) -> Iterator:
        return iter(self._row)

    def __len__(self) -> int:
        return len(self._row)

class HeadDense(Dense):

    def __init__(self, row: Dense, head_seq: Sequence[str], head_map: Mapping[str,int] = None) -> None:
        self._row      = row
        self.headers   = head_seq
        self._head_map = head_map or dict(zip(head_seq, count()))

    def _row_index(self, key:Union[int,str]) -> int:
        return key if key.__class__ is int else self._head_map[key]

    def __getitem__(self, key: Union[str,int]):
        return self._row[self._row_index(key)]

    def __iter__(self) -> Iterator:
        return iter(self._row)

    def __len__(self) -> int:
        return len(self._row)

class EncodeDense(Dense):

    def __init__(self, row: Dense, encoders: Union[Sequence,Mapping]) -> None:
        self._row      = row
        self._encoders = encoders

    def __getitem__(self, key: Union[int,str]):
        val = self._row[key]
        try:
            return self._encoders[key](val)
        except KeyError:
            return val

    def __iter__(self) -> Iterator:
        return iter( e(v) for e,v in zip(self._encoders, self._row))

    def __len__(self) -> int:
        return len(self._row)

class KeepDense(Dense):

    def __init__(self, row: Dense, indexes: Sequence[int], headers: Sequence[str] = None, headers_set: set = None) -> None:

        self._row     = row
        self._indexes = indexes

        if headers is not None:
            self.headers = headers
            self._headers_set = headers_set or set(headers)

    def _key_check(self,key):
        if key.__class__ is int:
            return self._indexes[key]
        if self._headers_set is None or key in self._headers_set:
            return key
        raise IndexError()

    def __getitem__(self, key: Union[int,str]):
        return self._row[self._key_check(key)]

    def __iter__(self) -> Iterator:
        return iter(map(self._row.__getitem__, self._indexes))

    def __len__(self) -> int:
        return len(self._indexes)

class LabelDense(Dense):

    def __init__(self, row: Dense, label: int) -> None:
        self._row = row
        self._lbl_key = label

    def __getitem__(self, key: Union[int,str]):
        return self._row[key]

    def __iter__(self) -> Iterator:
        return iter(self._row)

    def __len__(self) -> int:
        return len(self._row)

    @property
    def label(self) -> Any:
        return self._row[self._lbl_key]

    @property
    def feats(self) -> Dense:
        feat_indexes = chain(range(self._lbl_key), range(self._lbl_key+1, len(self._row)))
        return list(map(self._row.__getitem__, feat_indexes))

class LazySparse(Sparse):

    def __init__(self, loader_or_row: Union[Callable[[],Mapping],Mapping]) -> None:
        if callable(loader_or_row):
            self._loader = loader_or_row
            self._row    = None
        else:
            self._loader = None
            self._row    = loader_or_row

    def _load_or_get(self):
        row = self._row
        if row is None: 
            row = self._loader()
            self._row = row
        return row

    def __getitem__(self, key: str):
        return self._load_or_get()[key]

    def __iter__(self) -> Iterator:
        return iter(self._load_or_get())

    def __len__(self) -> int:
        return len(self._load_or_get())

    def keys(self) -> abc.KeysView:
        return self._load_or_get().keys()

    def items(self) -> Iterable:
        return self._load_or_get().items()

class MissingSparse(Sparse):

    def __init__(self, row: Sparse, missing: bool) -> None:
        self._row = row
        self.missing = missing

    def __getitem__(self, key: str):
        return self._row[key]

    def __iter__(self) -> Iterator:
        return iter(self._row)

    def __len__(self) -> int:
        return len(self._row)

    def keys(self) -> abc.KeysView:
        return self._row.keys()

    def items(self) -> Iterable:
        return self._row.items()

class HeadSparse(Sparse):

    def __init__(self, row: Sparse, head_map: Mapping[str,str]) -> None:
        self._row      = row
        self._head_map = head_map

    def __getitem__(self, key: Union[str,int]):
        return self._row[self._head_map[key]]

    def __iter__(self) -> Iterator:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._row)

    def keys(self) -> abc.KeysView:
        row_keys = self._row.keys()
        return set(k1 for k1,k2 in self._head_map.items() if k2 in row_keys)

    def items(self) -> Iterable:
        row = self._row
        row_keys = self._row.keys()
        yield from ((k1,row[k2]) for k1,k2 in self._head_map.items() if k2 in row_keys)

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

    def items(self) -> Iterable:
        enc = self._encoders
        yield from ((k, enc[k](v) if k in enc else v ) for k,v in self._row.items())

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
        return iter(self._select_set)

    def __len__(self) -> int:
        return len(self.keys())

    def keys(self) -> abc.KeysView:
        return self._row.keys() - self._drop_set

    def items(self) -> Iterable:
        drop = self._drop_set
        yield from (item for item in self._row.items() if item[0] not in drop) 

class LabelSparse(Sparse):

    def __init__(self, row: Sparse, label: str) -> None:
        self._row     = row
        self._lbl_key = label

    def __getitem__(self, key: str):
        return self._row[key]

    def __iter__(self) -> Iterator:
        return iter(self._row)

    def __len__(self) -> int:
        return len(self._row)

    def keys(self) -> abc.KeysView:
        return self._row.keys()

    def items(self) -> Iterable:
        yield self._row.items()

    @property
    def label(self) -> Any:
        return self._row[self._lbl_key]

    @property
    def feats(self) -> Sparse:
        return dict(item for item in self._row.items() if item[0] != self._lbl_key)

class EncodeRows(Filter[Iterable[Union[Dense,Sparse]],Iterable[Union[Dense,Sparse]]]):

    def __init__(self, encoders: Union[Sequence,Mapping]) -> None:
        self._encoders = encoders

    def filter(self, rows: Iterable[Union[Dense,Sparse]]) -> Iterable[Union[Dense,Sparse]]:
        encoders = self._encoders
        first, rows = peek_first(rows)

        if first and isinstance(first,Dense):
            yield from (EncodeDense(row, encoders) for row in rows)
        else:
            yield from (EncodeSparse(row, encoders) for row in rows)

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
            mapping  = self._mapping
            yield from (HeadSparse(row, mapping) for row in rows)

class LabelRows(Filter[Iterable[Union[Dense,Sparse]],Iterable[Union[Dense,Sparse]]]):

    def __init__(self, label: Union[int,str]) -> None:
        self.label = label

    def filter(self, rows: Iterable[Union[Dense,Sparse]]) -> Iterable[Union[Dense,Sparse]]:
        label = self.label

        first, rows = peek_first(rows)

        if isinstance(first,Dense):
            if isinstance(label,str): label = first.headers.index(label)
            yield from (LabelDense(row, label) for row in rows)
        else:
            yield from (LabelSparse(row, label) for row in rows)
