import re
import csv
import collections.abc

from itertools import islice, takewhile, chain, count
from tokenize import String
from typing import Iterable, Sequence, List, Dict, Union, Any, Iterator

from coba.exceptions import CobaException
from coba.encodings import Encoder, IdentityEncoder, OneHotEncoder, NumericEncoder, StringEncoder, MissingEncoder

from coba.pipes.primitives import MutableMap, Filter

class LazyDense(collections.abc.MutableSequence):
    def __init__(self, items: Sequence[Any], headers: Sequence[str] = [], encoders: Sequence[Encoder] = []) -> None:
        self._sentinel = object()
        self._headers  = dict(zip(headers,count()))

        if not encoders:
            self._enc_vals = items
            self._encoders = None
            self._raw_vals = None 
        else:
            self._enc_vals = [self._sentinel]*len(items)
            self._encoders = list(encoders)
            self._raw_vals = list(items)
    
    def __getitem__(self, index: Union[str,int]) -> Any:
        index: int = self._headers.get(index,index)

        if self._enc_vals[index] == self._sentinel and self._encoders:
            self._enc_vals[index] = self._encoders[index].encode(self._raw_vals[index])

        return self._enc_vals[index]

    def __setitem__(self, index: Union[str,int], value: Any) -> None:
        self._enc_vals[self._headers.get(index,index)] = value

    def __delitem__(self, index: Union[str,int]):
        index = self._headers.pop(index,index)
        
        for k,v in list(self._headers.items()):
            if v > index:
                self._headers[k] = v-1
        
        self._enc_vals.pop(index)
        
        if self._raw_vals:
            self._raw_vals.pop(index)
        
        if self._encoders:
            self._encoders.pop(index)

    def __len__(self) -> int:
        return len(self._enc_vals)

    def insert(self, index: int, value:Any):
        raise NotImplementedError()

    def __eq__(self, __o: object) -> bool:
        return list(self).__eq__(__o)

    def __repr__(self) -> str:
        return str(list(self))

    def __str__(self) -> str:
        return str(list(self))

class LazySparse(collections.abc.MutableMapping):
    
    def __init__(self, items: Dict[Any,str], headers: Dict[str,Any] = {}, encoders: Dict[Any,Encoder] = {}, modifiers: Sequence[str] = []) -> None:
        self._headers  = dict(headers)
        
        if not encoders:
            self._enc_vals = dict(items)
            self._raw_vals = {}
            self._encoders = {}
        else:
            self._encoders = dict(encoders)
            self._enc_vals = {}
            self._raw_vals = dict(items)

        if modifiers:
            for k in modifiers:
                if k not in self._raw_vals: 
                    self._raw_vals[k] = 0 #we add it so that it will be encoded if queried
        else:
            for k,e in self._encoders.items():
                base_encoder = e if not isinstance(e,MissingEncoder) else e._encoder
                base_is_modifying_encoder = not isinstance(base_encoder,(IdentityEncoder,NumericEncoder))

                if base_is_modifying_encoder and k not in self._raw_vals:
                    self._raw_vals[k] = 0 #we add it so that it will be encoded if queried
    
    def __getitem__(self, index: Union[str,int]) -> Any:
        index: int = self._headers.get(index,index)

        if index not in self._enc_vals:
            self._enc_vals[index] = self._encoders[index].encode(self._raw_vals[index])

        return self._enc_vals[index]

    def __setitem__(self, index: Union[str,int], value: Any) -> None:
        self._enc_vals[self._headers.get(index,index)] = value

    def __delitem__(self, index: Union[str,int]):
        index = self._headers.pop(index,index)

        self._enc_vals.pop(index,None)
        self._raw_vals.pop(index,None)
        self._encoders.pop(index,None)

    def __len__(self) -> int:
        return len(self._enc_vals)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._enc_vals.keys() | self._raw_vals.keys())

    def __eq__(self, __o: object) -> bool:
        return dict(self.items()).__eq__(__o)

    def __repr__(self) -> str:
        return str(dict(self))

    def __str__(self) -> str:
        return str(dict(self))

class Reader(Filter[Iterable[str], Iterable[MutableMap]]):
    pass

class ArffReader(Reader):
    """
        https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/
    """

    def __init__(self, cat_as_str=False, skip_encoding: bool = False):

        dialect = dict(quotechar="'", escapechar="\\", doublequote=False, skipinitialspace=True)
        
        self._str_enc_cat   = cat_as_str 
        self._csv_reader    = CsvReader(has_header=False,**dialect)
        self._skip_encoding = skip_encoding

        self._r_attribute = re.compile(r'^@attribute', re.IGNORECASE)
        self._r_data = re.compile(r'^@data', re.IGNORECASE)
        self._r_unescaped_single_quote = re.compile("(?<!\\\)'")
        self._r_unescaped_double_quote = re.compile('(?<!\\\)"')

    def filter(self, source: Iterable[str]) -> Iterable[MutableMap]:
        headers   : List[str           ] = []
        encoders  : List[MissingEncoder] = []

        #strip all lines to remove leading/trailing spaces
        source = (line.strip() for line in source)

        #remove all comment lines, empty lines, and empty sparse lines
        source = (line for line in source if not line.startswith("%") and line != "")

        lines_iter = iter(source)
        meta_lines = takewhile(lambda line: not self._r_data.match(line), lines_iter)
        data_lines = lines_iter

        for line in meta_lines:

            if self._r_attribute.match(line):
                
                attribute_split = self._split_line(line[11:], ' ', 1)
                attribute_name  = attribute_split[0]
                attribute_type  = attribute_split[1]

                headers.append(attribute_name)
                encoders.append(self._determine_encoder(attribute_type))

        try:
            first_line = next(data_lines)
        except StopIteration:
            return []

        is_dense = not (first_line.strip().startswith('{') and first_line.strip().endswith('}'))
        encoders = self._encoder_prep(is_dense, encoders)

        data_lines = chain([first_line], data_lines)
        data_lines = filter(None,(d.strip('} {') for d in data_lines))
        data_lines = self._parse_data(is_dense, data_lines, headers, encoders)

        return data_lines

    def _determine_encoder(self, tipe: str) -> Encoder:
        is_numeric = tipe in ['numeric', 'integer', 'real']
        is_one_hot = tipe.startswith('{') and tipe.endswith('}')
        is_string  = tipe == "string"
        is_date    = tipe.startswith('date')
        is_relate  = tipe.startswith('relational')

        if is_numeric: 
            return NumericEncoder()
        
        if is_one_hot: 
            if self._str_enc_cat: 
                return StringEncoder()
            else:
                return OneHotEncoder(self._split_line(tipe[1:-1], ','), err_if_unknown=True)

        if is_string or is_date or is_relate:
            return StringEncoder()

        raise CobaException(f"An unrecognized type was found in the arff attributes: {tipe}.")

    def _encoder_prep(self, is_dense: bool, encoders: List[Encoder]) -> Sequence[Encoder]:
        
        if self._skip_encoding:
            return []

        #there is a bug in ARFF where the first class value in an ARFF class can will dropped from the 
        #actual data because it is encoded as 0. Therefore our ARFF reader automatically adds a 0 value 
        #to all sparse categorical one-hot encoders to protect against this.
        if not is_dense:
            for i in range(len(encoders)):
                if isinstance(encoders[i],OneHotEncoder):
                    encoders[i] = OneHotEncoder([0]+list(encoders[i]._onehots.keys()))

        #ARFF allows for missing values so we wrap all our encoders with a missing encoder to handle these
        for i in range(len(encoders)):
            encoders[i] = MissingEncoder(encoders[i])
        
        return encoders

    def _parse_data(self, is_dense: bool, lines: Iterable[str], headers: Sequence[str], encoders: Sequence[Encoder]) -> Iterable[MutableMap]:

        if is_dense:
            for i, line in enumerate(self._csv_reader.filter(lines)):

                if len(line) < len(headers):
                    raise CobaException(f"There are not enough elements on line {i} in the ARFF file.")
                
                if len(line) > len(headers):
                    raise CobaException(f"There are too many elements on line {i} in the ARFF file.")

                yield LazyDense(line, headers, encoders) 
        else:
            dict_encoders = dict(enumerate(encoders))
            dict_headers  = dict(zip(headers,count()))
            modifiers     = [k for k,e in enumerate(encoders) if not isinstance(e._encoder,(IdentityEncoder,NumericEncoder))]

            for i,line in enumerate(lines):
                
                keys_and_vals = re.split('\s*,\s*|\s+', line)

                keys = list(map(int,keys_and_vals[0::2]))
                vals = keys_and_vals[1::2]

                if max(keys) > len(headers):
                    raise CobaException(f"There are elements we can't associate with a header on line {i} in the ARFF file.")

                yield LazySparse(dict(zip(keys,vals)), dict_headers, dict_encoders, modifiers)

    def _split_line(self, line: str, delimiter:str, count=float('inf')) -> Sequence[str]:

        line = line.strip()
        items: List[str] = []

        while line != "":

            if line.startswith("'"):
                line = line[1:]
                end  = self._r_unescaped_single_quote.search(line).start(0)
            elif line.startswith('"'):
                line = line[1:]
                end  = self._r_unescaped_double_quote.search(line).start(0)
            else:
                end = line.find(delimiter)

            if end == -1:
                items.append(line)
                line = ""
            else:
                items.append(line[:end])
                line = line[end+1:].strip()

            count -= 1

            if count == 0:
                items.append(line)
                line = ""

        return [ i.strip().strip('\'"').replace('\\','') for i in items]

class CsvReader(Reader):

    def __init__(self, has_header: bool=False, **dialect):
        self._dialect    = dialect
        self._has_header = has_header 

    def filter(self, items: Iterable[str]) -> Iterable[MutableMap]:

        lines = iter(csv.reader(iter(filter(None,(i.strip() for i in items))), **self._dialect))

        if self._has_header:
            headers = next(lines)

        for line in lines:
            yield line if not self._has_header else LazyDense(line,headers)

class LibSvmReader(Reader):

    """https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/"""
    """https://github.com/cjlin1/libsvm"""

    def filter(self, lines: Iterable[str]) -> Iterable[MutableMap]:

        for line in filter(None,lines):

            items  = line.strip().split(' ')
            labels = items[0].split(',')
            row    = { int(k):float(v) for i in items[1:] for k,v in [i.split(":")] }
            row[0] = labels

            yield row

class ManikReader(Reader):

    """http://manikvarma.org/downloads/XC/XMLRepository.html"""
    """https://drive.google.com/file/d/1u7YibXAC_Wz1RDehN1KjB5vu21zUnapV/view"""

    def filter(self, lines: Iterable[str]) -> Iterable[MutableMap]:

        # we skip first line because it just has metadata
        return LibSvmReader().filter(islice(lines,1,None))
