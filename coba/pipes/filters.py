"""The data.filters module contains core classes for filters used in data pipelines.

TODO add docstrings for all filters
"""
import re
import csv
import collections
import itertools
import json

from itertools import islice, count
from collections import defaultdict
from typing import Iterable, Any, Sequence, Union, Tuple, List, Dict, cast

from requests import Response

from coba.encodings import Encoder, OneHotEncoder, NumericEncoder, StringEncoder, CobaJsonEncoder, CobaJsonDecoder
from coba.pipes.core import Filter

_T_DenseRow   = Sequence[Any]
_T_SparseRow  = Tuple[Tuple[int,...], Tuple[Any,...]]
_T_DenseData  = Iterable[_T_DenseRow]
_T_SparseData = Iterable[_T_SparseRow]
_T_Data       = Union[_T_DenseData, _T_SparseData]

def _is_dense(items: _T_Data)-> Tuple[bool, _T_Data]:

    items = iter(items)
    item0 = next(items)

    #a sparse item has the following structure ([ids], [values])
    #this check isn't full proof but I think should be good enough
    is_dense = (len(item0) != 2) or not all([isinstance(i, collections.Sequence) and not isinstance(i, str) for i in item0])

    return is_dense, itertools.chain([item0], items)

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

class ResponseToLines(Filter[Response, Iterable[str]]):
    def filter(self, item: Response) -> Iterable[str]:
        
        if item.status_code != 200:
            
            message = (
                f"The response from {item.url} reported an error. "
                "The status and reason were {item.status_code}-{item.reason}.")
            
            raise Exception(message) from None

        return item.content.decode('utf-8').split('\n')

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

class ArffReader(Filter[Iterable[str], _T_Data]):
    """
        https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/
    """

    def __init__(self, skip_encoding: List[Union[str,int]] = []):

        self._skip_encoding = skip_encoding

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

    def _determine_encoder(self, index:int, name: str, tipe: str) -> Encoder:

        is_numeric = tipe in ['numeric', 'integer', 'real']
        is_one_hot = '{' in tipe

        if index in self._skip_encoding or name in self._skip_encoding:
            return StringEncoder()

        if is_numeric: return NumericEncoder()
        if is_one_hot: return OneHotEncoder(fit_values=[ v.strip() for v in tipe.strip("}{").split(',')], singular_if_binary=True)

        return StringEncoder()

    def _parse_file(self, lines: Iterable[str]) -> Tuple[_T_Data,Sequence[Encoder]]:
        in_meta_section=True
        in_data_section=False

        headers   : List[str]     = []
        encoders  : List[Encoder] = []
        data_lines: List[str]     = []

        for line in lines:

            if in_meta_section:

                if self._r_comment.match(line): continue
                if self._r_relation.match(line): continue

                attribute_match = self._r_attribute.match(line)

                if attribute_match:
                    attribute_text  = attribute_match.group(1).strip()
                    attribute_type  = re.split('[ ]', attribute_text, 1)[1]
                    attribute_name  = re.split('[ ]', attribute_text)[0]
                    attribute_index = len(headers)

                    headers.append(attribute_name)
                    encoders.append(self._determine_encoder(attribute_index,attribute_name,attribute_type))

                if self._r_data.match(line):
                    in_data_section = True
                    in_meta_section = False
                    continue

            if in_data_section and line != '':
                data_lines.append(line)

        parsed_data = CsvReader().filter(itertools.chain([",".join(headers)], data_lines))

        return parsed_data, encoders

    def filter(self, source: Iterable[str]) -> _T_Data:

        data, encoders = self._parse_file(source)

        data_iter = iter(data)
        header    = tuple(next(data_iter))
        encoded   = Transpose().filter(Encode(encoders).filter(Transpose().filter(data_iter)))

        return itertools.chain([header], encoded)

class CsvReader(Filter[Iterable[str], _T_Data]):
    def filter(self, items: Iterable[str]) -> _T_Data:
        
        lines = iter(filter(None, csv.reader( i.strip() for i in items)))

        try:
            row1 = next(lines)
        except StopIteration:
            return []
        try:
            row2 = next(lines)
        except StopIteration:
            row2 = None

        data_row = row2 if row2 is not None else row1

        is_sparse  = data_row[0].startswith("{") and data_row[-1].endswith("}")

        lines = itertools.chain(filter(None, [row1,row2]), lines)

        return self._sparse_parser(lines) if is_sparse else self._dense_parser(lines)

    def _dense_parser(self, lines: Iterable[Sequence[str]]) -> _T_DenseData:        
        return lines
    
    def _sparse_parser(self, lines: Iterable[Sequence[str]]) -> _T_SparseData:

        lines_iter = iter(lines)

        # we know there is at least one row otherwise we wouldn't have gotten here
        line1 = next(lines_iter)
        line1_is_header = not line1[0].startswith("{")

        if line1_is_header:
            yield (tuple(range(len(line1))), tuple(line1))
        else:
            lines_iter = itertools.chain([line1], lines_iter)

        for data_line in lines_iter:

            index_list: List[int] = []
            value_list: List[str] = []

            for item in data_line:
                split = item.strip("}{").split(' ', 1)
                
                index_list.append(int(split[0]))
                value_list.append(split[1])

            yield ( tuple(index_list), tuple(value_list) )        

class LibSvmReader(Filter[Iterable[str], _T_Data]):
    
    """https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/"""
    """https://github.com/cjlin1/libsvm"""

    def filter(self, input_lines: Iterable[str]) -> _T_Data:

        output_lines: List[Tuple[Tuple[int,...], Tuple[Any,...]]] = []
        feature_index = defaultdict(lambda x=count(1): next(x))

        for input_line in filter(None,input_lines):

            items  = input_line.strip().split(' ')
            labels = items[0].split(',')
            label  = labels[0] if len(labels) == 1 else tuple(labels)

            output_line: List[Tuple[int,Any]] = [ (0,label) ]
            
            for item in items[1:]:
                split = item.split(":")
                index = feature_index[split[0]]
                value = float(split[1])
                output_line.append((index,value))

            output_lines.append(tuple(zip(*output_line))) #type: ignore
            
        return output_lines

class ManikReader(Filter[Iterable[str], _T_Data]):
    
    """http://manikvarma.org/downloads/XC/XMLRepository.html"""
    """https://drive.google.com/file/d/1u7YibXAC_Wz1RDehN1KjB5vu21zUnapV/view"""


    def filter(self, input_lines: Iterable[str]) -> _T_Data:

        # we skip first line because it just has metadata
        return LibSvmReader().filter(islice(input_lines,1,None))

class Transpose(Filter[_T_Data, _T_Data]):
    def filter(self, items: _T_Data) -> _T_Data:

        is_dense,items =_is_dense(items)

        if is_dense:
            return zip(*items)
        else:
            sparse_transposed_items: Dict[int, Tuple[List[int],List[Any]]] = defaultdict( lambda: ([],[]))

            for outer_id, item in enumerate(items):
                for inner_id, value in zip(item[0], item[1]):
                    sparse_transposed_items[inner_id][0].append(outer_id)
                    sparse_transposed_items[inner_id][1].append(value)

            max_key = max(sparse_transposed_items.keys())

            #this loop ensures the column order is maintained, empty columns aren't lost and arrays are tuples
            return [ tuple(map(tuple,sparse_transposed_items[key]))  for key in range(max_key+1) ] #type: ignore

class Flatten(Filter[_T_Data, _T_Data]):
    #Assumes column major order

    def filter(self, data: _T_Data) -> _T_Data:
        
        for col in data:
            
            #this will fail on a two row dense representation with a one hot encoded column
            is_sparse = (len(col) == 2) and isinstance(col[0], collections.Sequence) and isinstance(col[0], collections.Sequence)

            if not is_sparse: 
                if isinstance(col[0],collections.Sequence):
                    for flat_col in zip(*col):
                        yield flat_col
                else:
                    yield tuple(col)
            
            else:
                if isinstance(col[1][0],collections.Sequence):
                    for flat_col in zip(*col[1]):
                        yield (tuple(col[0]), flat_col)
                else:
                    yield (tuple(col[0]), tuple(col[1]))

        #return map(tuple,map(self._flat, data)) if is_dense else data

    def _flat(self, item: Union[_T_DenseRow, _T_SparseRow] ) -> Union[_T_DenseRow, _T_SparseRow]:
        return sum(map(self._flat, item),[]) if isinstance(item, collections.Sequence) else [item]

class Encode(Filter[_T_Data, _T_Data]):

    #Assumes column major order

    def __init__(self, encoders: Sequence[Encoder]):
        self._encoders = encoders

    def filter(self, items: _T_Data) -> _T_Data:
        
        is_dense,items =_is_dense(items)

        for encoder, column in zip(self._encoders, items):

            raw_values = column if is_dense else column[1]

            encoder = encoder if encoder.is_fit else encoder.fit(raw_values)

            encoded_values = encoder.encode(raw_values)

            yield encoded_values if is_dense else (tuple(column[0]), tuple(encoded_values))
