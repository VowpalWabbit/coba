"""The data.filters module contains core classes for filters used in data pipelines.

TODO add docstrings for all filters
TODO add unittests for all filters
"""

import csv
import collections
import itertools
import json

from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Generic, Iterable, TypeVar, Any, Sequence, Union, Tuple, List

from requests import Response

from coba.data.encoders import Encoder, OneHotEncoder, NumericEncoder, StringEncoder
from coba.json import CobaJsonEncoder, CobaJsonDecoder
import re

# one dict for all rows, one dict for each row
# one dict for all columns, one dict for each column

_T_DenseData  = Iterable[Sequence[Any]]
_T_SparseData = Iterable[Tuple[Tuple[int,...], Tuple[Any,...]]]
_T_Data       = Union[_T_DenseData, _T_SparseData]

_T_out = TypeVar("_T_out", bound=Any, covariant=True)
_T_in  = TypeVar("_T_in", bound=Any, contravariant=True)

def _is_dense(items: _T_Data)-> Tuple[bool, _T_Data]:

    items = iter(items)
    item0 = next(items)

    #a sparse item has the following structure ([ids], [values])
    #this check isn't full proof but I think should be good enough
    is_dense = (len(item0) != 2) or not all([isinstance(i, collections.Sequence) for i in item0])

    return is_dense, itertools.chain([item0], items)

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

class ArffReader(Filter[Iterable[str], Any]):
    """
        https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/
    """

    def __init__(self):

        # Match a comment
        self._r_comment = re.compile(r'^%')

        # Match an empty line
        self.r_empty = re.compile(r'^\s+$')

        #@ lines give metadata describing the file. These always come at the top of the file
        self._r_meta = re.compile(r'^\s*@\S*')

        #The @relation line simply names the data. In practice we don't really care about it.
        self._r_relation = re.compile(r'^@[Rr][Ee][Ll][Aa][Tt][Ii][Oo][Nn]\s*(\S*)')

        #The @attribute lines contain typing information for 'columns'
        self._r_attribute = re.compile(r'^\s*@[Aa][Tt][Tt][Rr][Ii][Bb][Uu][Tt][Ee]\s*(..*$)')

        #The @data line indicates when the data begins. After @data there should be no more @ lines.
        self._r_data = re.compile(r'^@[Dd][Aa][Tt][Aa]')

    def _is_dense(self, data: Sequence[Sequence[str]]) -> bool:
        _starts_with_curly = data[0][ 0].startswith("{")
        _ends_with_curly   = data[0][-1].endswith("}")
        return not _starts_with_curly or not _ends_with_curly

    def _determine_encoder(self, tipe: str) -> Encoder:

        is_numeric = tipe in ['numeric', 'integer', 'real']
        is_one_hot = '{' in tipe

        if is_numeric: return NumericEncoder()
        if is_one_hot: return OneHotEncoder(singular_if_binary=True)

        return StringEncoder()

    def _parse_file(self, lines: Iterable[str]) -> Tuple[_T_Data,Sequence[Encoder]]:
        in_meta_section=True
        in_data_section=False

        headers : List[str]           = []
        encoders: List[Encoder]       = []
        data    : List[Sequence[str]] = []

        for line in lines:

            if in_meta_section:

                if self._r_comment.match(line): continue
                if self._r_relation.match(line): continue

                attribute_match = self._r_attribute.match(line)

                if attribute_match:
                    attribute_text = attribute_match.group(1).lower().strip()
                    attribute_type = re.split('[ ]', attribute_text, 1)[1]
                    attribute_name = re.split('[ ]', attribute_text)[0]

                    headers.append(attribute_name)
                    encoders.append(self._determine_encoder(attribute_type))

                if self._r_data.match(line):
                    in_data_section = True
                    in_meta_section = False
                    continue

            if in_data_section and line != '':
                data.append(re.split('[,]', line))

        return self._parse_data(headers,data), encoders

    def _parse_data(self, headers: Sequence[str], data: Sequence[Sequence[str]]) -> _T_Data:

        if self._is_dense(data): return [headers] + list(data)

        sparse_data_rows: List[Tuple[Tuple[int,...], Tuple[str,...]]] = []
        sparse_data_rows.append( ( tuple(range(len(headers))), tuple(headers) ) )

        for data_row in data:

            index_list: List[int] = []
            value_list: List[str] = []

            for item in data_row:
                split = re.split(' ', item.strip("}{"), 1)

                index_list.append(int(split[0]))
                value_list.append(split[1])

            sparse_data_rows.append( ( tuple(index_list), tuple(value_list)) )

        return sparse_data_rows

    def filter(self, source: Iterable[str]):

        data, encoders = self._parse_file(source)

        return data

        #Do we want to encode here? If we do, then this code won't be quite as seemless with OpenML.
        #I think for now we will leave encoding out from this portion of code. In the 
        #future if ARFF support is desired outside of the OpenML context it can be added in.
        #This particulary causes issues with label encoding when reading OpenML Sources.
        # data       = list(data)
        # header_row = data.pop(0)
        # data_rows  = data
        # data_rows  = Transpose().filter(Encode(encoders).filter(Transpose().filter(data_rows)))
        # return [header_row] + list(data_rows)

class CsvReader(Filter[Iterable[str], _T_DenseData]):
    def filter(self, items: Iterable[str]) -> Iterable[Sequence[str]]:
        return filter(None,csv.reader(items))

class Transpose(Filter[_T_Data, _T_Data]):
    def filter(self, items: _T_Data) -> _T_Data:

        is_dense,items =_is_dense(items)

        if is_dense:
            return zip(*items)
        else:
            sparse_transposed_items = defaultdict( lambda: ([],[]))

            for outer_id, item in enumerate(items):
                for inner_id, value in zip(item[0], item[1]):
                    sparse_transposed_items[inner_id][0].append(outer_id)
                    sparse_transposed_items[inner_id][1].append(value)

            max_key = max(sparse_transposed_items.keys())

            #this loop ensures the column order is maintained and empty columns aren't lost
            return [ sparse_transposed_items[key] for key in range(max_key+1) ]

class Flatten(Filter[_T_Data, _T_Data]):
    def filter(self, items: Iterable[Sequence[Any]]) -> Iterable[Sequence[Any]]:
        is_dense,items =_is_dense(items)
        
        return map(tuple,map(self._flat, items)) if is_dense else items

    def _flat(self, item: Union[Sequence[Any], Any]) -> Sequence[Any]:
        return sum(map(self._flat, item),[]) if isinstance(item, collections.Sequence) else [item]

    def _flatter(self, items: Iterable[Sequence[Any]]) -> Iterable[Sequence[Any]]:
        for item in items:
            if isinstance(item[1], collections.Sequence) and not isinstance(item[1], str):
                for i in item[1:]:
                    yield [item[0]] + list(i)
            else:
                yield item

class Encode(Filter[_T_Data, _T_Data]):

    def __init__(self, encoders: Sequence[Encoder]):
        self._encoders = encoders

    def filter(self, items: _T_Data) -> _T_Data:
        
        is_dense,items =_is_dense(items)

        for encoder, column in zip(self._encoders, items):

            raw_values = column if is_dense else column[1]

            encoder = encoder if encoder.is_fit else encoder.fit(raw_values)

            encoded_values = encoder.encode(raw_values)

            yield encoder.encode(raw_values) if is_dense else (column[0], encoded_values)
