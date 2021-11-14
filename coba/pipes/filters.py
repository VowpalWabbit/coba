"""The coba.pipes.filters module contains core classes for filters used in data pipelines."""

import re
import csv
import collections
import itertools
import json
import math

from itertools import islice
from typing_extensions import OrderedDict
from typing import Iterable, Any, Sequence, Union, Tuple, List, Dict, Callable, Optional

from requests import Response

from coba.random import CobaRandom
from coba.encodings import Encoder, OneHotEncoder, NumericEncoder, StringEncoder, CobaJsonEncoder, CobaJsonDecoder
from coba.exceptions import CobaException
from coba.pipes.core import Filter

_T_DenseRow   = Sequence[Any]
_T_SparseRow  = Dict[Any,Any]
_T_DenseData  = Iterable[_T_DenseRow]
_T_SparseData = Iterable[_T_SparseRow]

_T_Row        = Union[_T_DenseRow,  _T_SparseRow ]
_T_Data       = Union[_T_DenseData, _T_SparseData]

class Cartesian(Filter[Union[Any,Iterable[Any]], Iterable[Any]]):

    def __init__(self, filter: Union[Filter,Sequence[Filter]]):
        
        self._filters = filter if isinstance(filter, collections.Sequence) else [filter]

    def filter(self, item: Union[Any,Iterable[Any]]) -> Iterable[Any]:

        items = item if isinstance(item, collections.Iterable) else [item]
        
        for item in items:
            for filter in self._filters:
                yield filter.filter(item)

class Identity(Filter[Any, Any]):
    def filter(self, item:Any) -> Any:
        return item
    
    def __repr__(self) -> str:
        return "{ Identity }"

class Shuffle(Filter[Iterable[Any], Iterable[Any]]):

    def __init__(self, seed:Optional[int]) -> None:

        if seed is not None and (not isinstance(seed,int) or seed < 0):
            raise ValueError(f"Invalid parameter for Shuffle: {seed}. An optional integer value >= 0 was expected.")

        self._seed = seed

    @property
    def params(self) -> Dict[str, Any]:
        return { "shuffle": self._seed }

    def filter(self, items: Iterable[Any]) -> Iterable[Any]: 
        return CobaRandom(self._seed).shuffle(list(items))

    def __repr__(self) -> str:
        return str(self.params)

class Take(Filter[Iterable[Any], Iterable[Any]]):
    """Take a given number of items from an iterable."""

    def __init__(self, count:Optional[int], keep_first:bool = False, seed: int = None,) -> None:
        """Instantiate a Take filter.

        Args:
            count     : The number of items we wish to take from the given iterable.
            keep_first: Indicates if the first row should be kept and take on the rest. Useful for files with headers.
            seed      : An optional random seed to determine which random count items to take.

        Remarks:
            We use Algorithm L as described by Kim-Hung Li. (1994) to ranomdly take count items.

        References:
            Kim-Hung Li. 1994. Reservoir-sampling algorithms of time complexity O(n(1 + log(N/n))). 
            ACM Trans. Math. Softw. 20, 4 (Dec. 1994), 481–493. DOI:https://doi.org/10.1145/198429.198435
        """

        if count is not None and (not isinstance(count,int) or count < 0):
            raise ValueError(f"Invalid parameter for Take: {count}. An optional integer value >= 0 was expected.")

        self._count = count
        self._seed  = seed
        self._keep_first = keep_first

    @property
    def params(self) -> Dict[str, Any]:

        if self._seed is not None:
            return { "take": self._count, "take_seed": self._seed }
        else: 
            return { "take": self._count }

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:

        if self._count is None: 
            return items
        else:

            items    = iter(items)
            first    = [next(items)] if self._keep_first else []
            resevoir = list(islice(items,self._count))

            if self._seed is not None:            
                rng = CobaRandom(self._seed)
                W = 1

                try:
                    while True:
                        [r1,r2,r3] = rng.randoms(3)
                        W = W * math.exp(math.log(r1)/self._count)
                        S = math.floor(math.log(r2)/math.log(1-W))
                        resevoir[int(r3*self._count-.001)] = next(itertools.islice(items,S,S+1))
                except StopIteration:
                    pass

            return itertools.chain( first, resevoir if len(resevoir) == self._count else [])
        
    def __repr__(self) -> str:
        return str(self.params)

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

    def _intify(self,obj):

        if isinstance(obj,float) and obj.is_integer():
            return int(obj)

        if isinstance(obj,tuple):
            obj = list(obj)

        if isinstance(obj,list):
            for i in range(len(obj)):
                obj[i] = self._intify(obj[i])

        if isinstance(obj,dict):
            for key in obj:
                obj[key] = self._intify(obj[key])

        return obj

    def __init__(self, minify=True) -> None:
        self._minify = minify

        if self._minify:
            self._encoder = CobaJsonEncoder(separators=(',', ':'))
        else:
            self._encoder = CobaJsonEncoder()

    def filter(self, item: Any) -> str:
        if self._minify:
            #JsonEncoder writes floats with .0 regardless of if they are integers
            #Therefore we preprocess and turn all float whole numbers into integers
            return self._encoder.encode(self._intify(item))
        else:
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

    def __init__(self, skip_encoding: Union[bool,Sequence[Union[str,int]]] = False):

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

        if self._skip_encoding != False and (self._skip_encoding == True or index in self._skip_encoding or name in self._skip_encoding):
            return StringEncoder()

        if is_numeric: return NumericEncoder()
        if is_one_hot: return OneHotEncoder([v.strip() for v in tipe.strip("}{").split(',')])

        return StringEncoder()

    def _parse_file(self, lines: Iterable[str]) -> Tuple[_T_Data, Dict[str,Encoder]]:
        in_meta_section=True
        in_data_section=False

        headers   : List[str        ] = []
        encoders  : Dict[str,Encoder] = {}
        data_lines: List[str        ] = []

        for line in lines:

            if in_meta_section:

                if self._r_comment.match(line): continue
                if self._r_relation.match(line): continue

                attribute_match = self._r_attribute.match(line)

                if attribute_match:
                    attribute_text  = attribute_match.group(1).strip()
                    attribute_type  = re.split('[ ]', attribute_text, 1)[1]
                    attribute_name  = re.split('[ ]', attribute_text)[0].lower()
                    attribute_index = len(headers)

                    headers.append(attribute_name)
                    encoders[attribute_name] = self._determine_encoder(attribute_index,attribute_name,attribute_type)

                if self._r_data.match(line):
                    in_data_section = True
                    in_meta_section = False
                    continue

            if in_data_section and line != '':
                data_lines.append(line)

        parsed_data = CsvReader(True).filter(itertools.chain([",".join(headers)], data_lines))

        return parsed_data, encoders

    def filter(self, source: Iterable[str]) -> _T_Data:

        data, encoders = self._parse_file(source)

        return data if self._skip_encoding == True else Encode(encoders).filter(data)

class CsvReader(Filter[Iterable[str], _T_Data]):

    def __init__(self, has_header: bool):
        self._has_header = has_header

    def filter(self, items: Iterable[str]) -> _T_Data:

        lines = iter(filter(None, csv.reader(i.strip() for i in items)))

        try:
            header     = [ h.lower() for h in next(lines)] if self._has_header else None
            first_data = next(lines)
        except StopIteration:
            return [header] #[None] because every other filter method assumes there is some kind of a header row.

        is_sparse = first_data[0].startswith("{") and first_data[-1].endswith("}")
        parser    = self._sparse_parser if is_sparse else self._dense_parser

        return parser(header, itertools.chain([first_data], lines))

    def _dense_parser(self, header: Optional[Sequence[str]],  lines: Iterable[Sequence[str]]) -> Iterable[Sequence[str]]:
        return itertools.chain([header], lines)

    def _sparse_parser(self, header: Optional[Sequence[str]],  lines: Iterable[Sequence[str]]) -> Iterable[Dict[int,str]]:
        yield header if header is None else OrderedDict(zip(header, itertools.count()))

        for line in lines:
            yield OrderedDict((int(k),v) for l in line for k,v in [l.strip("}{").strip().split(' ', 1)])

class LibSvmReader(Filter[Iterable[str], _T_SparseData]):

    """https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/"""
    """https://github.com/cjlin1/libsvm"""

    def filter(self, lines: Iterable[str]) -> _T_SparseData:

        yield None # we yield None because there is no header row

        for line in filter(None,lines):

            items  = line.strip().split(' ')
            labels = items[0].split(',')
            row    = { int(k):float(v) for i in items[1:] for k,v in [i.split(":")] }
            row[0] = labels

            yield row

class ManikReader(Filter[Iterable[str], _T_SparseData]):

    """http://manikvarma.org/downloads/XC/XMLRepository.html"""
    """https://drive.google.com/file/d/1u7YibXAC_Wz1RDehN1KjB5vu21zUnapV/view"""


    def filter(self, lines: Iterable[str]) -> _T_SparseData:

        # we skip first line because it just has metadata
        return LibSvmReader().filter(islice(lines,1,None))

class Encode(Filter[_T_Data, _T_Data]):

    def __init__(self, encoders: Dict[str,Encoder], fit_using=None, has_header:bool = True):
        self._encoders   = encoders
        self._fit_using  = fit_using
        self._has_header = has_header

    def filter(self, items: _T_Data) -> _T_Data:

        items  = iter(items) # this makes sure items are pulled out for fitting
        
        if not self._has_header:
            header_synced_encoders = self._encoders

        else:
            header = next(items)

            if header is None:
                yield header
                header_synced_encoders = dict(self._encoders)

            elif isinstance(header, dict):
                yield header
                header_synced_encoders = { header[k]:encoder for k,encoder in self._encoders.items() if k in header } 

            elif isinstance(header, list):
                yield header
                header_synced_encoders = { header.index(k):encoder for k,encoder in self._encoders.items() if k in header }

            else:
                raise CobaException(f"Unrecognized type ({type(header).__name__}) passed to Encodes.")

        fit_using  = 0 if all([e.is_fit for e in header_synced_encoders.values()]) else self._fit_using
        fit_items  = list(islice(items, fit_using))
        fit_values = collections.defaultdict(list)

        for item in fit_items:
            for k,v in (item.items() if isinstance(item,dict) else enumerate(item)):
                fit_values[k].append(v)

        for k,v in fit_values.items():
            if not header_synced_encoders[k].is_fit:
                header_synced_encoders[k] = header_synced_encoders[k].fit(v)

        for item in itertools.chain(fit_items, items):

            for k,v in (item.items() if isinstance(item,dict) else enumerate(item)):
                item[k] = header_synced_encoders[k].encode([v])[0]

            yield item

class Drop(Filter[_T_Data, _T_Data]):

    def __init__(self, drop_cols: Sequence[Any] = [], drop_row: Callable[[_T_Row], bool] = None) -> None:
        self._drop_cols = drop_cols
        self._drop_row  = drop_row

    def filter(self, data: _T_Data) -> _T_Data:
        
        if not self._drop_cols and not self._drop_row: return data
        else:

            data   = iter(data)
            header = next(data)

            if header is None:
                drop_keys = self._drop_cols
                yield header
            
            elif isinstance(header,dict):
                drop_keys = [ header.pop(k) for k in self._drop_cols ]
                yield header
            
            elif isinstance(header,list):
                drop_keys = [ header.index(k) for k in self._drop_cols ]
                for i in sorted(drop_keys,reverse=True): header.pop(i)
                yield header 
            
            else:
                raise CobaException(f"Unrecognized type ({type(header).__name__}) passed to Drops.")

            for row in data:

                if self._drop_row and self._drop_row(row):
                    continue

                for k in sorted(drop_keys,reverse=True):
                    row.pop(k)                

                yield row

class Structure(Filter[_T_Data, Iterable[Any]]):

    def __init__(self, split_cols: Sequence[Any]) -> None:
        self._col_structure = split_cols

    def filter(self, data: _T_Data) -> _T_Data:
        
        data   = iter(data)
        header = next(data)

        key_structure = self._recursive_structure_keys(header, self._col_structure)

        for row in data:
           yield self._recursive_structure_rows(row, key_structure) 

    def _recursive_structure_keys(self, header, cols):

        if header is None:
            return cols

        elif isinstance(cols,(list,tuple)):
            return [ self._recursive_structure_keys(header,s) for s in cols ]

        elif cols is None:
            return cols

        elif isinstance(header,list):
            return header.index(cols)

        elif isinstance(header,dict):
            return header[cols]

        else:
            raise CobaException(f"Unrecognized type ({type(header).__name__}) passed to Structure.")

    def _recursive_structure_rows(self, row, keys):
        if keys is None:
            return row
        elif isinstance(keys,(list,tuple)):
            return [ self._recursive_structure_rows(row,k) for k in keys ]
        else:
            return row.pop(keys)

class Flatten(Filter[_T_Data, _T_Data]):

    def filter(self, data: _T_Data) -> _T_Data:

        for row in data:

            if isinstance(row,dict):
                for k in list(row.keys()):
                    if isinstance(row[k],(list,tuple)):
                        row.update([((k,i), v) for i,v in enumerate(row.pop(k))])

            elif isinstance(row,list):
                for k in range(len(row)):
                    if isinstance(row[k],(list,tuple)):
                        row.extend(row.pop(k))

            else:
                raise CobaException(f"Unrecognized type ({type(row).__name__}) passed to Flattens.")

            yield row

class Default(Filter[_T_Data, _T_Data]):

    def __init__(self, defaults: Dict[str, Any]) -> None:
        self._defaults = defaults

    def filter(self, data: _T_Data) -> _T_Data:

        if not self._defaults: return data
        else:

            data   = iter(data)
            header = next(data)

            if header is None:
                yield header
                header_synced_defaults = dict(self._defaults)

            elif isinstance(header, dict):
                yield header
                header_synced_defaults = { header[k]:encoder for k,encoder in self._defaults.items() } 

            elif isinstance(header, list):
                yield header
                header_synced_defaults = { header.index(k):encoder for k,encoder in self._defaults.items() }

            else:
                raise CobaException(f"Unrecognized type ({type(header).__name__}) passed to Defaults.")

            for row in data:

                if isinstance(row,dict):
                    for k,v in header_synced_defaults.items():
                        if k not in row:
                            row[k] = v

                yield row
