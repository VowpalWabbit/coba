"""The data.filters module contains core classes for filters used in data pipelines.

TODO add docstrings for all filters
TODO add unittests for all filters
"""

import csv
import collections
import itertools

from abc import ABC, abstractmethod
from typing import Generic, Iterable, TypeVar, Any, Sequence, Union, Tuple, Callable

from coba.data.encoders import Encoder, OneHotEncoder
from coba.json import CobaJsonEncoder, CobaJsonDecoder
from coba.execution import ExecutionContext

_T_out = TypeVar("_T_out", bound=Any, covariant=True)
_T_in  = TypeVar("_T_in", bound=Any, contravariant=True)

class Filter(ABC, Generic[_T_in, _T_out]):
    @abstractmethod
    def filter(self, items:_T_in) -> _T_out:
        ...

class ForeachFilter(Filter[Iterable[Any], Iterable[Any]]):

    def __init__(self, filter: Filter):
        self._filter = filter

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:

        for item in items:
            yield self._filter.filter(item)

class JsonEncode(Filter[Iterable[Any], Iterable[str]]):
    def filter(self, items: Iterable[Any]) -> Iterable[str]:
        encoder = CobaJsonEncoder()
        for item in items: yield encoder.encode(item)

class JsonDecode(Filter[Iterable[str], Iterable[Any]]):
    def filter(self, items: Iterable[str]) -> Iterable[Any]:
        decoder = CobaJsonDecoder()
        for item in items: yield decoder.decode(item)

class ColSplitter(Filter[Iterable[Iterable[Any]], Tuple[Iterable[Iterable[Any]], Iterable[Iterable[Any]]]]):
    
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

class ColRemover(Filter[Iterable[Sequence[Any]], Iterable[Sequence[Any]]]):
    def __init__(self, remove_columns: Union[Sequence[int],Sequence[str]] = []):
        
        self._removed_columns = remove_columns
        self._is_column_headers = len(remove_columns) > 0 and isinstance(remove_columns[0],str)

    def filter(self, columns: Iterable[Sequence[Any]]) -> Iterable[Sequence[Any]]:

        for index, raw_col in enumerate(columns):

            is_ignored_header =     self._is_column_headers and (raw_col[0] in self._removed_columns) 
            is_ignored_index  = not self._is_column_headers and (index      in self._removed_columns)

            if not is_ignored_header and not is_ignored_index:
                yield raw_col

class ColEncoder(Filter[Iterable[Sequence[str]], Iterable[Sequence[Any]]]):

    def __init__(self, encoders: Sequence[Encoder], headers: Sequence[str] = None) -> None:

        assert headers is None or len(encoders) == len(headers), "The given headers didn't match the encoders"

        self._encoders  = encoders
        self._headers   = headers

    def filter(self, columns: Iterable[Sequence[str]]) -> Iterable[Sequence[Any]]:

        for index, raw_col in enumerate(columns):

            raw_hdr  = raw_col[0]
            raw_vals = raw_col[1:]

            encoder_idx = index if self._headers is None else self._headers.index(raw_hdr)
            encoder     = self._encoders[encoder_idx]
            encoder     = encoder if encoder.is_fit else encoder.fit(raw_vals)

            if isinstance(encoder, OneHotEncoder):
                encoded_values = list(zip(*encoder.encode(raw_vals)))
            else:
                encoded_values = encoder.encode(raw_vals)

            yield [raw_hdr] + list(encoded_values) # type:ignore

class RowRemover(Filter[Iterable[Sequence[Any]], Iterable[Sequence[Any]]]):
    def __init__(self, remove_rows: Sequence[int] = []):
        self._remove_rows = remove_rows

    def filter(self, items: Iterable[Sequence[Any]]) -> Iterable[Sequence[Any]]:
        for i, item in enumerate(items):
            if i not in self._remove_rows:
                yield item

class CsvReader(Filter[Iterable[str], Iterable[Sequence[str]]]):
    def __init__(self, csv_reader  : Callable[[Iterable[str]], Iterable[Sequence[str]]] = csv.reader) -> None: #type: ignore #pylance complains
        self._csv_reader = csv_reader

    def filter(self, items: Iterable[str]) -> Iterable[Sequence[str]]:
        
        return filter(None,self._csv_reader(items))

class CsvTransposer(Filter[Iterable[Sequence[_T_in]], Iterable[Sequence[_T_out]]]):    
    
    def __init__(self, flatten: bool = False):
        self._flatten = flatten

    def filter(self, items: Iterable[Sequence[_T_in]]) -> Iterable[Sequence[_T_out]]:
        
        items = filter(None, items)
        items = items if not self._flatten else self._flatter(items)

        return zip(*list(items))

    def _flatter(self, items: Iterable[Sequence[_T_in]]) -> Iterable[Sequence[_T_in]]:
        flat_items = []

        for item in items:
            if isinstance(item[1], collections.Sequence) and not isinstance(item[1], str):
                for i in item[1:]:
                    yield [item[0]] + list(i)
            else:
                yield item

        return flat_items

class CsvCleaner(Filter[Iterable[str], Iterable[Sequence[Any]]]):

    def __init__(self,
        headers: Sequence[str] = [],
        encoders: Sequence[Encoder] = [],
        ignored: Sequence[bool] = [],
        output_rows: bool = True):

        self._encoders = encoders
        self._headers = headers
        self._ignored = ignored
        self._output_row_major = output_rows

    def filter(self, items: Iterable[str]) -> Iterable[Sequence[Any]]:

        ignored_headers = list(itertools.compress(self._headers, self._ignored))

        cleaning_steps = [
            CsvTransposer(), ColRemover(ignored_headers), ColEncoder(self._encoders, self._headers)
        ]

        output: Any = items

        with ExecutionContext.Logger.log('encoding data... '):
            for cleaning_step in cleaning_steps: output = cleaning_step.filter(output)
            return output if not self._output_row_major else CsvTransposer().filter(output)

class LabeledCsvCleaner(Filter[Iterable[str], Tuple[Iterable[Sequence[Any]],Iterable[Sequence[Any]]]]):
    def __init__(self, 
        label_col : Union[int,str],
        headers   : Sequence[str]     = [],
        encoders  : Sequence[Encoder] = [], 
        ignored   : Sequence[bool]    = [],
        rmv_header: bool              = False):

        self._label_col  = label_col
        self._encoders   = encoders
        self._headers    = headers
        self._ignored    = ignored
        self._rmv_header = rmv_header

    def filter(self, items: Iterable[str]) -> Tuple[Iterable[Sequence[Any]],Iterable[Sequence[Any]]]:

        clean      = CsvCleaner(self._headers, self._encoders, self._ignored, output_rows=False)
        split      = ColSplitter([self._label_col]) #type: ignore
        rows       = ForeachFilter(CsvTransposer(True))
        rmv_header = ForeachFilter(RowRemover([0]))

        output: Any = items

        with ExecutionContext.Logger.log('encoding data... '):
            for step in [clean, split, rows]: output = step.filter(output)
            if self._rmv_header: output = rmv_header.filter(output)

            labels   = next(output)
            features = next(output)

            return features, labels