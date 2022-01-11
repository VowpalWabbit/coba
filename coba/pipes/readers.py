import re
import csv
import collections.abc

from itertools import islice, takewhile, chain, count, repeat
from typing import Iterable, Sequence, List, Dict, Union, Any, Iterator

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

    def filter(self, source: Iterable[str]) -> Iterable[MutableMap]:
        headers   : List[str           ] = []
        encoders  : List[MissingEncoder] = []

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
                if "'" in attribute_text:  # some attributes are quoted i.e. 'total weight' in openml(720)
                    attribute_text = attribute_text.replace("'","")
                    attribute_type = re.split('\s+', attribute_text)[-1] # the last word is the type
                    attribute_name = " ".join(re.split('\s+', attribute_text)[:-1]) # the rest is the name
                else:
                    attribute_type = re.split('\s+', attribute_text, 1)[1]
                    attribute_name = re.split('\s+', attribute_text)[0]
                attribute_index = len(headers)

                headers.append(attribute_name.strip().strip('\'"').replace('\\',''))
                encoders.append(MissingEncoder(self._determine_encoder(attribute_index,attribute_name,attribute_type)))

        #Remove empty lines and comments
        data_lines = filter(None,(d.strip() for d in data_lines if not d.startswith("%")))

        try:
            first_line = next(data_lines)
        except StopIteration:
            return []

        is_dense   = not (first_line.strip().startswith('{') and first_line.strip().endswith('}'))
        data_lines = chain([first_line], data_lines)
        
        if not is_dense:
            #this is a bug in ARFF such that it is not uncommon for the first class value in an ARFF class 
            #list to be dropped from the actual data because it is encoded as 0. Therefore our ARFF reader
            #automatically adds a 0 value to all categorical one-hot encoders to protect against this.
            new_encoders = []
            for e in encoders:
                if isinstance(e._encoder,OneHotEncoder):
                    new_encoders.append(MissingEncoder(OneHotEncoder([0]+list(e._encoder._onehots.keys()))))
                else:
                    new_encoders.append(e)
            encoders = new_encoders

        #remove sparse brackets before parsing
        data_lines = filter(None,(d.strip('} {') for d in data_lines))
        data_lines = self._parse(data_lines, headers, is_dense, encoders)

        return data_lines

    def _determine_encoder(self, index:int, name: str, tipe: str) -> Encoder:
        is_numeric = tipe in ['numeric', 'integer', 'real']
        is_one_hot = '{' in tipe

        if self._skip_encoding != False and (self._skip_encoding == True or index in self._skip_encoding or name in self._skip_encoding):
            return StringEncoder()
        
        if is_numeric: 
            return NumericEncoder()
        
        if is_one_hot and not self._str_enc_cat: 
            return OneHotEncoder(list(self._csv_reader.filter([tipe.strip("}{")]))[0], err_if_unknown=True)

        return StringEncoder()

    def _parse(self, lines: Iterable[str], headers: Sequence[str], is_dense: bool, encoders: Sequence[MissingEncoder]) -> Iterable[MutableMap]:

        if is_dense:
            for line in self._csv_reader.filter(lines):
                yield LazyDense(line, headers, encoders) 
        else:
            dict_encoders = dict(enumerate(encoders))
            dict_headers  = dict(zip(headers,count()))
            modifiers     = [k for k,e in enumerate(encoders) if not isinstance(e._encoder,(IdentityEncoder,NumericEncoder))]

            for line in lines:
                
                keys_and_vals = re.split('\s*,\s*|\s+', line)

                keys = list(map(int,keys_and_vals[0::2]))
                vals = keys_and_vals[1::2]

                yield LazySparse(dict(zip(keys,vals)), dict_headers, dict_encoders, modifiers)

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
