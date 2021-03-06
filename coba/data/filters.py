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

_T_DenseData  = Sequence[Any]
_T_SparseData = Tuple[Sequence[int], Sequence[Any]]
_T_Data       = Union[_T_DenseData, _T_SparseData]

_T_out = TypeVar("_T_out", bound=Any, covariant=True)
_T_in  = TypeVar("_T_in", bound=Any, contravariant=True)

def _is_dense(items: Iterable[_T_Data])-> Tuple[bool, Iterable[_T_Data]]:

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
    # Takes in ARFF bytes and splits it into attributes, encoders, and data while handling sparse data

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

    def _determine_encoder(self, tipe):

        is_numeric = tipe in ['numeric', 'integer', 'real']
        is_one_hot = '{' in tipe

        if is_numeric: return NumericEncoder()
        if is_one_hot: return OneHotEncoder(singular_if_binary=True)

        return StringEncoder()

    def _parse_file(self, lines: Iterable[str]):
        in_meta_section=True
        in_data_section=False

        headers  = []
        encoders = []
        data     = []

        for line in lines:

            if in_meta_section:

                if self._r_comment.match(line): continue
                if self._r_relation.match(line): continue

                attribute_match = self._r_attribute.match(line)

                if attribute_match:
                    attribute_text = attribute_match.group(1).lower().strip()
                    attribute_type  = re.split('[ ]', attribute_text, 1)[1]
                    attribute_name  = re.split('[ ]', attribute_text)[0]

                    headers.append(attribute_name)
                    encoders.append(self._determine_encoder(attribute_type))

                if self._r_data.match(line):
                    in_data_section = True
                    in_meta_section = False
                    continue

            if in_data_section and line != '':
                data.append(re.split('[,]', line))

        return headers, encoders, data

    def _sparse_filler(self, items: List[List[str]], encoders: List[Encoder]) -> List[List[str]]: # Currently quite inefficient
        """Handles Sparse ARFF data

        Args
            items:      Data from openML api call as returned by read_header
            encoders:   Encoders from openML api call as returned by read_header
        Ret
            if sparse --     full:  non-sparse version of data
            if non-sparse -- items: original data
        """

        _starts_with_curly = items[0][0][0] == "{"
        _ends_with_curly = items[0][-1][-1] == "}"
        if(not _starts_with_curly or not _ends_with_curly):
            return items

        full = []
        # Creates non-sparse version of data. 
        for i in range(len(items)):
            r = []
            for encoder in encoders:
                app = ""
                if(isinstance(encoder, NumericEncoder)):
                    app = "0"
                r.append(app)
            full.append(r)

        # Fills in data from items
        for i in range(len(items)):
            items[i][0] = items[i][0].replace('{', '', 1)
            items[i][-1] = items[i][-1].replace('}', '', 1)
            for j in range(len(items[i])):
                split = re.split(' ', items[i][j], 1)
                index = int(split[0])
                val = split[1]
                full[i][index] = val
        return full

    def filter(self, source: Iterable[str]):

        attributes, encoders, items = self._parse_file(source)

        items = self._sparse_filler(items, encoders)

        #do we want to encode here? If we do it won't be quite as seemless with OpenML.
        #I think for now we will leave encoding out from this portion of code. In the 
        #future if ARFF support is desired outside of the OpenML context it can be added in.

        return [attributes] + items

class CsvReader(Filter[Iterable[str], Iterable[_T_DenseData]]):
    def filter(self, items: Iterable[str]) -> Iterable[Sequence[str]]:
        return filter(None,csv.reader(items))

class Transpose(Filter[Iterable[_T_Data], Iterable[_T_Data]]):
    def filter(self, items: Iterable[_T_Data]) -> Iterable[_T_Data]:

        is_dense,items =_is_dense(items)

        if is_dense:
            return zip(*items)
        else:
            sparse_transposed_items = defaultdict( lambda: ([],[]))

            for outer_id, item in enumerate(items):
                for inner_id, value in zip(item[0], item[1]):
                    sparse_transposed_items[inner_id][0].append(outer_id)
                    sparse_transposed_items[inner_id][1].append(value)

            return list(sparse_transposed_items.values())

class Flatten(Filter[Iterable[_T_Data], Iterable[_T_Data]]):
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

class Encode(Filter[Iterable[_T_Data],Iterable[_T_Data]]):

    def __init__(self, encoders: Sequence[Encoder]):
        self._encoders = encoders

    def filter(self, items: Iterable[_T_Data]) -> Iterable[_T_Data]:
        
        is_dense,items =_is_dense(items)

        for encoder, column in zip(self._encoders, items):

            raw_values = column if is_dense else column[1]

            encoder = encoder if encoder.is_fit else encoder.fit(raw_values)

            encoded_values = encoder.encode(raw_values)

            yield encoder.encode(raw_values) if is_dense else (column[0], encoded_values)
