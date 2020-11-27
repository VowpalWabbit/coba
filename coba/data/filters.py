"""The data.filters module contains core classes for filters used in data pipelines.

TODO add docstrings for all filters
TODO add unittests for all filters
"""

import csv
from abc import ABC, abstractmethod
from typing import Generic, Iterable, TypeVar, Any, Sequence, Union, Tuple, Callable

from coba.data.encoders import Encoder
from coba.json import CobaJsonEncoder, CobaJsonDecoder

_T_out = TypeVar("_T_out", bound=Any, covariant=True)
_T_in  = TypeVar("_T_in", bound=Any, contravariant=True)

class Filter(ABC, Generic[_T_in, _T_out]):
    @abstractmethod
    def filter(self, items:_T_in) -> _T_out:
        ...

class JsonEncode(Filter[Iterable[Any], Iterable[str]]):
    def filter(self, items: Iterable[Any]) -> Iterable[str]:
        encoder = CobaJsonEncoder()
        for item in items: yield encoder.encode(item)

class JsonDecode(Filter[Iterable[str], Iterable[Any]]):
    def filter(self, items: Iterable[str]) -> Iterable[Any]:
        decoder = CobaJsonDecoder()
        for item in items: yield decoder.decode(item)

class CsvReader(Filter[Iterable[str], Iterable[Sequence[str]]]):
    def __init__(self, csv_reader  : Callable[[Iterable[str]], Iterable[Sequence[str]]] = csv.reader) -> None: #type: ignore #pylance complains
        self._csv_reader = csv_reader

    def filter(self, items: Iterable[str]) -> Iterable[Sequence[str]]:
        return self._csv_reader(items)

class Transposer(Filter[Iterable[Sequence[_T_in]], Iterable[Sequence[_T_out]]]):
    def filter(self, items: Iterable[Sequence[_T_in]]) -> Iterable[Sequence[_T_out]]:
        return zip(*items)

class ColumnSplitter(Filter[Iterable[Iterable[Any]], Tuple[Iterable[Iterable[Any]], Iterable[Iterable[Any]]]]):
    def __init__(self, split1_columns: Union[Sequence[int],Sequence[str]] = []):
        self._split1_columns    = split1_columns
        self._is_column_headers = len(split1_columns) > 0 and isinstance(split1_columns[0],str)

    def filter(self, columns: Iterable[Sequence[Any]]) -> Tuple[Iterable[Sequence[Any]], Iterable[Sequence[Any]]]:

        split1 = []
        split2 = []

        for index, raw_col in enumerate(columns): 

            is_split1_header =     self._is_column_headers and (raw_col[0] in self._split1_columns) 
            is_split1_index  = not self._is_column_headers and (index      in self._split1_columns)

            if is_split1_header or is_split1_index:
                split1.append(raw_col)
            else:
                split2.append(raw_col)
                
        return (split1, split2)

class ColumnRemover(Filter[Iterable[Sequence[Any]], Iterable[Sequence[Any]]]):
    def __init__(self, removed_columns: Union[Sequence[int],Sequence[str]] = []):
        self._removed_columns = removed_columns
        self._is_column_headers = len(removed_columns) > 0 and isinstance(removed_columns[0],str)

    def filter(self, columns: Iterable[Sequence[Any]]) -> Iterable[Sequence[Any]]:

        for index, raw_col in enumerate(columns):

            is_ignored_header =     self._is_column_headers and (raw_col[0] in self._removed_columns) 
            is_ignored_index  = not self._is_column_headers and (index      in self._removed_columns)

            if not is_ignored_header and not is_ignored_index:
                yield raw_col

class ColumnDecoder(Filter[Iterable[Sequence[str]], Iterable[Sequence[Any]]]):

    def __init__(self, encoders: Sequence[Encoder], headers: Sequence[str] = None) -> None:

        assert headers is None or len(encoders) == len(headers), "The given headers didn't match the encoders"

        self._encoders  = encoders
        self._headers   = headers

    def filter(self, columns: Iterable[Sequence[str]]) -> Iterable[Sequence[Any]]:
    
        for index, raw_col in enumerate(columns):

            encoder_idx = index if self._headers is None else self._headers.index(raw_col[0])
            encoder     = self._encoders[encoder_idx]            
            encoder     = encoder if encoder.is_fit else encoder.fit(raw_col)

            yield [raw_col[0]] + list(encoder.encode(raw_col[1:])) # type:ignore
