import re
import csv

from itertools import islice, takewhile, chain, count
from typing import Iterable, Sequence, List, Dict

from coba.utilities import HeaderDict, HeaderList
from coba.encodings import Encoder, OneHotEncoder, NumericEncoder, StringEncoder

from coba.pipes.filters import Encode
from coba.pipes.primitives import MutableMap, Filter

class Reader(Filter[Iterable[str], Iterable[MutableMap]]):
    pass

class ArffReader(Reader):
    """
        https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/
    """

    def __init__(self, str_encode_cat=False, skip_encoding: bool = False, **dialect):

        dialect = dict(quotechar="'", escapechar="\\", doublequote=False, skipinitialspace=True)
        
        self._str_enc_cat   = str_encode_cat 
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

        #Remove empty lines and comments
        data_lines = filter(None,(d.strip() for d in data_lines if not d.startswith("%")))

        try:
            first_line = next(data_lines)
        except StopIteration:
            return []

        is_dense   = not (first_line.strip().startswith('{') and first_line.strip().endswith('}'))

        data_lines = chain([first_line], data_lines)
        
        #remove sparse brackets before parsing
        data_lines = filter(None,(d.strip('} {') for d in data_lines))
        data_lines = self._parse(data_lines, headers, is_dense)
        data_lines = data_lines if self._skip_encoding == True else Encode(dict(zip(count(),encoders)), missing_val='?').filter(data_lines)

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

    def _parse(self, lines: Iterable[str], headers: Sequence[str], is_dense: bool) -> Iterable[Dict[int,str]]:
        
        lines = self._csv_reader.filter(lines)

        if is_dense:
            for line in lines:
                yield HeaderList(line,headers)
        else:
            for line in lines:
                to_pair = lambda k,v: (int(k),v)
                pairing = [ to_pair(*l.strip().split(' ', 1)) for l in line ]
                yield HeaderDict(pairing,headers)

class CsvReader(Reader):

    def __init__(self, has_header: bool=False, **dialect):
        self._dialect    = dialect
        self._has_header = has_header 

    def filter(self, items: Iterable[str]) -> Iterable[MutableMap]:

        lines = iter(csv.reader(iter(filter(None,(i.strip() for i in items))), **self._dialect))

        if self._has_header:
            headers = next(lines)

        for line in lines:
            yield line if not self._has_header else HeaderList(line,headers)

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
