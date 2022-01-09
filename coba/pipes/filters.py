import re
import csv
import json
import math

from collections import defaultdict
from itertools import islice, takewhile, chain, count
from typing import Generic, Iterable, Any, Sequence, TypeVar, List, Dict, Callable, Optional, Union
from coba.backports import Protocol

from coba.random import CobaRandom
from coba.exceptions import CobaException
from coba.utilities import HeaderDict, HeaderList
from coba.encodings import Encoder, OneHotEncoder, NumericEncoder, StringEncoder, CobaJsonEncoder, CobaJsonDecoder

from coba.pipes.primitives import Filter

_TK = TypeVar('_TK', bound=Any)
_TV = TypeVar('_TV', bound=Any)

class MutableMap(Protocol, Generic[_TK,_TV]):
    def __getitem__(self, key: _TK) -> _TV: pass
    def __setitem__(self, key: _TK, val: _TV) -> None: pass
    def pop( key:_TK) -> _TV: pass

class Identity(Filter[Any, Any]):
    """Return whatever is given to the filter."""
    def filter(self, item:Any) -> Any:
        return item

class Shuffle(Filter[Iterable[Any], Iterable[Any]]):
    """Shuffle a sequence of items."""

    def __init__(self, seed:Optional[int]) -> None:
        """Instantiate a Shuffle filter.

        Args:
            seed: A random number seed which determines the new sequence order.
        """

        if seed is not None and (not isinstance(seed,int) or seed < 0):
            raise ValueError(f"Invalid parameter for Shuffle: {seed}. An optional integer value >= 0 was expected.")

        self._seed = seed

    def filter(self, items: Iterable[Any]) -> Iterable[Any]: 
        return CobaRandom(self._seed).shuffle(list(items))

class Take(Filter[Iterable[Any], Iterable[Any]]):
    """Take a fixed number of items from an iterable."""

    def __init__(self, count:Optional[int]) -> None:
        """Instantiate a Take filter.

        Args:
            count: The number of items we wish to take from the given iterable.
        """

        if count is not None and (not isinstance(count,int) or count < 0):
            raise ValueError(f"Invalid parameter for count: {count}. An optional integer value >= 0 was expected.")

        self._count = count

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        items =  list(islice(items,self._count))
        return items if len(items) == self._count else []

class Reservoir(Filter[Iterable[Any], Iterable[Any]]):
    """Take a fixed number of random items from an iterable.
    
    Remarks:
        We use Algorithm L as described by Kim-Hung Li. (1994) to take a random count of items.

    References:
        Kim-Hung Li. 1994. Reservoir-sampling algorithms of time complexity O(n(1 + log(N/n))). 
        ACM Trans. Math. Softw. 20, 4 (Dec. 1994), 481–493. DOI:https://doi.org/10.1145/198429.198435
    """

    def __init__(self, count:Optional[int], seed: int = 1) -> None:
        """Instantiate a Resevoir filter.

        Args:
            count     : The number of items we wish to take from the given iterable.
            seed      : An optional random seed to determine which random count items to take.
            keep_first: Indicate whether the first row should be kept as is (useful for files with headers).
        """

        if count is not None and (not isinstance(count,int) or count < 0):
            raise ValueError(f"Invalid parameter for Take: {count}. An optional integer value >= 0 was expected.")

        self._count      = count
        self._seed       = seed

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:

        items     = iter(items)
        reservoir = list(islice(items,self._count))

        this_count = len(reservoir) if self._count is None else self._count

        if this_count == 0:
            return []

        if self._seed is not None:
            rng = CobaRandom(self._seed)
            W = 1

            try:
                while True:
                    [r1,r2,r3] = rng.randoms(3)
                    W = W * math.exp(math.log(r1)/this_count)
                    S = math.floor(math.log(r2)/math.log(1-W))
                    reservoir[int(r3*this_count-.001)] = next(islice(items,S,S+1))
            except StopIteration:
                pass

        return reservoir if len(reservoir) == self._count or self._count is None else []

class JsonEncode(Filter[Any, str]):
 
    def _min(self,obj):
        #WARNING: This method doesn't handle primitive types such int, float, or str. We handle this shortcoming
        #WARNING: by making sure no primitive type is passed to this method in filter. Accepting the shortcoming
        #WARNING: improves the performance of this method by a few percentage points. 

        #JsonEncoder writes floats with .0 regardless of if they are integers so we convert them to int to save space
        #JsonEncoder also writes floats out 16 digits so we truncate them to 5 digits here to reduce file size

        if isinstance(obj,tuple):
            obj = list(obj)
            kv  = enumerate(obj) 
        elif isinstance(obj,list):
            kv = enumerate(obj)
        elif isinstance(obj,dict):
            kv = obj.items()
        else:
            return obj

        for k,v in kv:
            if isinstance(v, (int,str)):
                obj[k] = v
            elif isinstance(v, float):
                if v.is_integer():
                    obj[k] = int(v) 
                elif math.isnan(v) or math.isinf(v):
                    obj[k] = v                    
                else: 
                    #rounding by any means is considerably slower than this crazy method
                    #we format as a truncated string and then manually remove the string
                    #indicators from the json via string replace methods
                    obj[k] = f"|{v:0.5g}|" 
            else:
                obj[k] = self._min(v)

        return obj

    def __init__(self, minify=True) -> None:
        self._minify = minify

        if self._minify:
            self._encoder = CobaJsonEncoder(separators=(',', ':'))
        else:
            self._encoder = CobaJsonEncoder()

    def filter(self, item: Any) -> str:        
        return self._encoder.encode(self._min([item])[0] if self._minify else item).replace('"|',"").replace('|"',"")

class JsonDecode(Filter[str, Any]):
    def __init__(self, decoder: json.decoder.JSONDecoder = CobaJsonDecoder()) -> None:
        self._decoder = decoder

    def filter(self, item: str) -> Any:
        return self._decoder.decode(item)

class ArffReader(Filter[Iterable[str], Iterable[MutableMap[int,Any]]]):
    """
        https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/
    """

    def __init__(self, skip_encoding: bool = False, **dialect):

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

        self._dialect = dialect

    def filter(self, source: Iterable[str]) -> Iterable[MutableMap[int,Any]]:
        headers   : List[str    ] = []
        encoders  : List[Encoder] = []

        lines_iter = iter(( line.strip() for line in source ))
        meta_lines = takewhile(lambda line: not self._r_data.match(line), lines_iter)
        data_lines = lines_iter

        for line in meta_lines:

            if self._r_comment.match(line): 
                continue

            if self._r_relation.match(line): 
                continue

            attribute_match = self._r_attribute.match(line)

            if attribute_match:
                attribute_text  = attribute_match.group(1).strip()
                attribute_type  = re.split('\s+', attribute_text, 1)[1]
                attribute_name  = re.split('\s+', attribute_text)[0]
                attribute_index = len(headers)

                headers.append(attribute_name.strip().strip('\'"').replace('\\',''))
                encoders.append(self._determine_encoder(attribute_index,attribute_name,attribute_type))

        try:
            first_line = next(data_lines)
        except StopIteration:
            return []
        
        is_dense   = not (first_line.strip().startswith('{') and first_line.strip().endswith('}'))

        data_lines = chain([first_line], data_lines)
        data_lines = filter(None,(d.strip('} {') for d in data_lines if not d.startswith("%")))

        data_lines = self._parse(data_lines, headers, is_dense)
        data_lines = data_lines if self._skip_encoding == True else Encode(dict(zip(count(),encoders))).filter(data_lines)

        return data_lines

    def _determine_encoder(self, index:int, name: str, tipe: str) -> Encoder:
        is_numeric = tipe in ['numeric', 'integer', 'real']
        is_one_hot = '{' in tipe

        if self._skip_encoding != False and (self._skip_encoding == True or index in self._skip_encoding or name in self._skip_encoding):
            return StringEncoder()

        if is_numeric: return NumericEncoder()
        if is_one_hot: return OneHotEncoder([v.strip() for v in tipe.strip("}{").split(',')])

        return StringEncoder()

    def _parse(self, lines: Iterable[str], headers: Sequence[str], is_dense: bool) -> Iterable[Dict[int,str]]:
        
        lines = CsvReader(has_header=False,**self._dialect).filter(lines)

        if is_dense:
            for line in lines:
                yield HeaderList(line,headers)
        else:
            for line in lines:
                to_pair = lambda k,v: (int(k),v)
                pairing = [ to_pair(*l.split(' ', 1)) for l in line ]
                yield HeaderDict(pairing,headers)

class CsvReader(Filter[Iterable[str], Iterable[List[str]]]):

    def __init__(self, has_header: bool=False, **dialect):
        self._dialect    = dialect
        self._has_header = has_header 

    def filter(self, items: Iterable[str]) -> Iterable[Sequence[str]]:

        lines = iter(csv.reader(iter(filter(None,(i.strip() for i in items))), **self._dialect))

        if self._has_header:
            headers = next(lines)

        for line in lines:
            yield line if not self._has_header else HeaderList(line,headers)

class LibSvmReader(Filter[Iterable[str], Iterable[Dict[int,float]]]):

    """https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/"""
    """https://github.com/cjlin1/libsvm"""

    def filter(self, lines: Iterable[str]) -> Iterable[Dict[int,float]]:

        for line in filter(None,lines):

            items  = line.strip().split(' ')
            labels = items[0].split(',')
            row    = { int(k):float(v) for i in items[1:] for k,v in [i.split(":")] }
            row[0] = labels

            yield row

class ManikReader(Filter[Iterable[str], Iterable[Dict[int,float]]]):

    """http://manikvarma.org/downloads/XC/XMLRepository.html"""
    """https://drive.google.com/file/d/1u7YibXAC_Wz1RDehN1KjB5vu21zUnapV/view"""

    def filter(self, lines: Iterable[str]) -> Iterable[Dict[int,float]]:

        # we skip first line because it just has metadata
        return LibSvmReader().filter(islice(lines,1,None))

class Encode(Filter[Iterable[MutableMap[int,Any]], Iterable[MutableMap[int,Any]]]):

    def __init__(self, encoders:Dict[Union[str,int],Encoder], fit_using:int = None):
        self._encoders   = encoders
        self._fit_using  = fit_using

    def filter(self, items: Iterable[MutableMap[int,Any]]) -> Iterable[MutableMap[int,Any]]:

        items = iter(items) # this makes sure items are pulled out for fitting

        try:
            first_item = next(items)
            items      = chain([first_item], items)
        except StopIteration:
            return []

        is_dense = not isinstance(first_item,dict) 

        if not is_dense:
            encoders = self._encoders
        else:
            encoders = {}
            for k,v in self._encoders.items():
                try:
                    first_item[k]
                    encoders[k] = v
                except:
                    pass

        unfit_enc = { k:v for k,v in encoders.items() if not v.is_fit }
        fit_using = self._fit_using if unfit_enc else 0

        items_for_fitting  = list(islice(items, fit_using))
        values_for_fitting = defaultdict(list)

        for item in items_for_fitting:
            for k in unfit_enc:
                if is_dense:
                    values_for_fitting[k].append(item[k])
                else:
                    values_for_fitting[k].append(item.get(k,0))

        for k,v in values_for_fitting.items():
            encoders[k] = encoders[k].fit(v)

        for item in chain(items_for_fitting, items):
            for k,v in encoders.items():
                if is_dense or k in item:
                    item[k] = encoders[k].encode(item[k])
            yield item

class Drop(Filter[Iterable[MutableMap[int,Any]], Iterable[MutableMap[int,Any]]]):

    def __init__(self, drop_cols: Sequence[int] = [], drop_row: Callable[[MutableMap[int,Any]], bool] = lambda r: False) -> None:
        self._drop_cols = sorted(drop_cols, reverse=True)
        self._drop_row  = drop_row

    def filter(self, data: Iterable[MutableMap[int,Any]]) -> Iterable[MutableMap[int,Any]]:
        
        keep_row = lambda r: not self._drop_row(r) if self._drop_row else True
        for row in filter(keep_row, data):            
            if row is not None:
                for col in self._drop_cols:
                    row.pop(col)            
            yield row

class Structure(Filter[Iterable[MutableMap[int,Any]], Iterable[Any]]):

    def __init__(self, split_cols: Sequence[Any]) -> None:
        self._col_structure = split_cols

    def filter(self, data: Iterable[MutableMap[int,Any]]) -> Iterable[Any]:
        for row in data:
           yield self._structure_row(row, self._col_structure) 

    def _structure_row(self, row: MutableMap[int,Any], col_structure: Sequence[Any]):
        if col_structure is None:
            return row

        elif isinstance(col_structure,(list,tuple)):
            return [ self._structure_row(row, k) for k in col_structure ]

        else:
            return row.pop(col_structure)

class Flatten(Filter[Iterable[Any], Iterable[Any]]):

    def filter(self, data: Iterable[Any]) -> Iterable[Any]:

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

class Default(Filter[Iterable[MutableMap[int,Any]], Iterable[MutableMap[int,Any]]]):

    def __init__(self, defaults: Dict[Any, Any]) -> None:
        self._defaults = defaults

    def filter(self, data: Iterable[MutableMap[Any,Any]]) -> Iterable[MutableMap[Any,Any]]:

        for row in data:

            if isinstance(row,dict):
                for k,v in self._defaults.items():
                    if k not in row: row[k] = v

            yield row
