"""The data.filters module contains core classes for filters used in data pipelines.

TODO add docstrings for all filters
TODO add unittests for all filters
"""

import csv
import collections
import itertools
import json

from abc import ABC, abstractmethod
from typing import Generic, Hashable, Iterable, TypeVar, Any, Sequence, Union, Tuple, Callable, cast

from requests import Response

from coba.data.encoders import Encoder, OneHotEncoder
from coba.json import CobaJsonEncoder, CobaJsonDecoder
from coba.tools import CobaConfig

_T_out = TypeVar("_T_out", bound=Any, covariant=True)
_T_in  = TypeVar("_T_in", bound=Any, contravariant=True)

class Filter(ABC, Generic[_T_in, _T_out]):
    @abstractmethod
    def filter(self, item:_T_in) -> _T_out:
        ...

class Cartesian(Filter[Union[Any,Iterable[Any]], Iterable[Any]]):

    def __init__(self, filter: Union[Filter,Sequence[Filter]]):
        
        self._filters = filter if isinstance(filter, collections.Sequence) else [filter]

    def filter(self, item: Union[Any,Iterable[Any]]) -> Iterable[Any]:

        items = item if isinstance(item, collections.Iterable) else [item]
        
        for item in items:
            for filter in self._filters:
                yield filter.filter(item)

class IdentityFilter(Filter[Any, Any]):
    def filter(self, item:Any) -> Any:
        return item

class StringJoin(Filter[Iterable[str], str]):

    def __init__(self, separator:str = '') -> None:
        self._separator = separator

    def filter(self, item: Iterable[str]) -> str:
        return self._separator.join(item)

class ResponseToText(Filter[Response, str]):
    def filter(self, item: Response) -> str:
        
        if item.status_code != 200:
            message = (
                f"The response from {item.url} reported an error. "
                "The status and reason were {item.status_code}-{item.reason}.")
            
            raise Exception(message) from None

        return item.content.decode('utf-8')

class JsonEncode(Filter[Any, str]):
    def __init__(self, encoder: json.encoder.JSONEncoder = CobaJsonEncoder()) -> None:
        self._encoder = encoder

    def filter(self, item: Any) -> str:
        return self._encoder.encode(item)

class JsonDecode(Filter[str, Any]):
    def __init__(self, decoder: json.decoder.JSONDecoder = CobaJsonDecoder()) -> None:
        self._decoder = decoder

    def filter(self, item: str) -> Any:
        return self._decoder.decode(item)

class ColSplitter(Filter[Iterable[Sequence[Any]], Tuple[Iterable[Sequence[Any]], Iterable[Sequence[Any]]]]):
    
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

    def __init__(self, headers: Sequence[str] = [], encoders: Sequence[Encoder] = [], default: Encoder = None) -> None:

        assert len(headers) == 0 or len(encoders) <= len(headers), "The given encoders didn't match the given headers."
        assert len(encoders) > 0 or default is not None, "A valid encoder was not provided to ColEncoder."

        self._encoders = encoders
        self._headers  = headers
        self._default  = default

    def filter(self, columns: Iterable[Sequence[str]]) -> Iterable[Sequence[Any]]:

        for index, raw_col in enumerate(columns):

            raw_hdr  = raw_col[0]
            raw_vals = raw_col[1:]

            encoder = self._get_encoder(index, raw_hdr)
            encoder = encoder if encoder.is_fit else encoder.fit(raw_vals)

            encoded_values: Sequence[Hashable]

            if isinstance(encoder, OneHotEncoder):
                encoded_values = list(zip(*encoder.encode(raw_vals)))
            else:
                encoded_values = list(encoder.encode(raw_vals))

            yield [cast(Hashable,raw_hdr)] + encoded_values

    def _get_encoder(self, index: int, header: str) -> Encoder:

        encoded_headers = self._headers[0:len(self._encoders)]

        if header in encoded_headers:
            return self._encoders[encoded_headers.index(header)]

        if len(encoded_headers) == 0 and index < len(self._encoders):
            return self._encoders[index]

        if self._default is not None:
            return self._default

        raise Exception("We were unable to find an encoder for the column.")

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

        return zip(*list(items)) #type: ignore

    def _flatter(self, items: Iterable[Sequence[_T_in]]) -> Iterable[Sequence[_T_in]]:
        for item in items:
            if isinstance(item[1], collections.Sequence) and not isinstance(item[1], str):
                for i in item[1:]:
                    yield [item[0]] + list(i)
            else:
                yield item

class CsvCleaner(Filter[Iterable[str], Iterable[Sequence[Any]]]):

    def __init__(self,
        headers: Sequence[str] = [],
        encoders: Sequence[Encoder] = [],
        default: Encoder = None,
        ignored: Sequence[bool] = [],
        output_rows: bool = True):

        self._headers  = headers
        self._encoders = encoders
        self._default  = default
        self._ignored  = ignored
        self._output_rows = output_rows

    def filter(self, items: Iterable[str]) -> Iterable[Sequence[Any]]:

        ignored_headers = list(itertools.compress(self._headers, self._ignored))

        cleaning_steps: Sequence[Filter] = [
            CsvTransposer(), ColRemover(ignored_headers), ColEncoder(self._headers, self._encoders, self._default)
        ]

        output: Any = items
        
        for cleaning_step in cleaning_steps: output = cleaning_step.filter(output)
        return output if not self._output_rows else CsvTransposer().filter(output)

class LabeledCsvCleaner(Filter[Iterable[Sequence[str]], Tuple[Iterable[Sequence[Any]],Iterable[Sequence[Any]]]]):
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

    def filter(self, items: Iterable[Sequence[str]]) -> Tuple[Iterable[Sequence[Any]],Iterable[Sequence[Any]]]:

        split_column = cast(Union[Sequence[str],Sequence[int]], [self._label_col])

        clean      = CsvCleaner(self._headers, self._encoders, None, self._ignored, output_rows=False)
        split      = ColSplitter(split_column)
        rows       = Cartesian(CsvTransposer(True))
        rmv_header = Cartesian(RowRemover([0]))

        output: Any = items

        #with CobaConfig.Logger.time('encoding data... '):
        output = rows.filter(split.filter(clean.filter(output)))

        if self._rmv_header: 
            output = rmv_header.filter(output)

        labels   = next(output)
        features = next(output)

        return features, labels