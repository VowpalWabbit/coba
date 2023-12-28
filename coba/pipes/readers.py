import re
import csv

from operator import methodcaller
from collections import deque
from itertools import islice, chain, count, takewhile
from typing import Iterable, Sequence, Union, Any, Pattern, Tuple, Mapping
from typing import MutableSequence, MutableMapping, Callable

from coba.exceptions import CobaException
from coba.encodings import Encoder, CategoricalEncoder
from coba.utilities import peek_first
from coba.primitives import Dense, Sparse

from coba.pipes.rows import HeadRows, LazyDense, LazySparse
from coba.pipes.primitives import Filter

class CsvReader(Filter[Iterable[str], Iterable[MutableSequence]]):
    """A filter capable of parsing CSV formatted data."""

    def __init__(self, has_header: bool=False, **dialect):
        """Instantiate a CsvReader.

        Args:
            has_header: Indicates if the CSV data has a header row.
            **dialect: This has the same values as Python's csv.reader dialect.
        """
        self._dialect    = dialect
        self._has_header = has_header

    def filter(self, items: Iterable[str]) -> Iterable[Dense]:

        lines = iter(csv.reader(iter(filter(None,(i.strip() for i in items))), **self._dialect))
        first = next(lines)

        if self._has_header:
            return HeadRows(first).filter(lines)
        else:
            return chain([first],lines)

class ArffAttrReader(Filter[Iterable[str], Iterable[Tuple[str,Callable]]]):

    class CategoricalDict(dict):
        def __missing__(self, key):
            raise CobaException(f"We were unable to find '{key}' in {sorted(self.keys())}.")

    def __init__(self, is_dense:bool) -> None:
        self._is_dense = is_dense
        self._r_space  = re.compile("(\s+)")
        self._r_comma  = re.compile("(,)")

    def filter(self, lines: Iterable[str]) -> Iterable[Tuple[str,Callable]]:

        headers = set()

        for line in lines:
            if line[0:10].lower() == "@attribute":
                header, encoding = tuple(self._split(line[11:], self._r_space, n=2))

                if header in headers:
                    raise CobaException("Two columns in the ARFF file had identical header values.")
                else:
                    headers.add(header)

                yield header, self._encoder(encoding)

    def _split(self, line: str, pattern: Pattern[str], n=None):

        items  = iter(pattern.split(line))
        count  = 0
        quotes = '"'+"'"

        try:
            while True:

                item = next(items).lstrip()
                if not item or pattern.match(item): continue

                count += 1
                if count == n:
                    items = chain([item],items)
                    break

                if item[0] in quotes:
                    q  = item[0]
                    while item.rstrip()[-1] != q or item.rstrip()[-2]=="\\":
                        item += next(items)

                    item = item.strip().rstrip()[1:-1].replace("\\",'')
                else:
                    item = item.strip()

                yield item

            yield "".join(items).strip()
        except StopIteration:
            pass

    def _encoder(self, encoding: str) -> Encoder:
        numeric_types = ('numeric', 'integer', 'real')
        string_types  = ("string", "date", "relational")

        if encoding.lower() in numeric_types:
            return float

        elif encoding.lower().startswith(string_types):
            return lambda x: None if x == "?" else x

        elif encoding.startswith('{'):
            categories = list(self._split(encoding[1:-1], self._r_comma))

            if not self._is_dense:
                #there is a bug in ARFF where the first class value in an ARFF class can will dropped from the
                #actual data because it is encoded as 0. Therefore, our ARFF reader automatically adds a 0 value
                #to all sparse categorical one-hot encoders to protect against this.
                categories = ["0"] + categories

            return ArffAttrReader.CategoricalDict(CategoricalEncoder(categories)._categoricals).__getitem__

        else:
            raise CobaException(f"An unrecognized encoding was found in the arff attributes: {encoding}.")

class ArffDataReader(Filter[Iterable[str], Iterable[Union[Dense,Sparse]]]):

    _trans = str.maketrans('','',' \t\n\r\v\f')

    def __init__(self, is_dense:bool) -> None:
        self._is_dense = is_dense

    def filter(self, lines: Iterable[str]) -> Iterable[Tuple[str,bool]]:
        return (self._dense if self._is_dense else self._sparse)(lines)

    def _dense(self, lines: Iterable[str]) -> Iterable[Tuple[str,bool]]:
        for line in lines:
            if line[0] == "%": continue
            if "?" not in line:
                missing = False
            elif line[:2] == "?,":
                missing = True
            elif line[-2:] == ",?":
                missing = True
            else:
                compact = line.translate(self._trans)
                missing = compact[:2] == '?,' or ',?,' in compact or compact[-2:] == ',?'

            yield line,missing

    def _sparse(self, lines: Iterable[str]) -> Iterable[Tuple[str,bool]]:

        for line in lines:
            if line[0] == "%": continue
            missing = " ?," in line or line[-3:] == " ?}"
            yield line,missing

class ArffLineReader(Filter[str, Sequence[str]]):
    def __init__(self, is_dense:bool, n_columns:int):
        self._is_dense = is_dense
        self._n_columns = n_columns

        #only used when parsing dense lines
        self._fallback_delim = None
        self._quotes         = '"'+"'"
        self._dialect        = dict(skipinitialspace=True,escapechar="\\",doublequote=False)
        self._quotechar      = None

        if self._is_dense:
            self._set_filter(self._dense)
        else:
            self._set_filter(self._sparse)

    def _set_filter(self,method)->None:
        self.filter = method

    def filter(self,line:str) -> Union[Sequence[str],Mapping[str,str]]:
        #this is defined in __init__ for performance purposes
        pass #pragma: no cover

    def _dense(self, line:str) -> Sequence[str]:

        self._set_filter(self._dense_simple)

        dialect = self._dialect

        double_quote_in_line = '"' in line
        single_quote_in_line = "'" in line

        if single_quote_in_line and double_quote_in_line:
            self._set_filter(self._dense_advanced)
        elif double_quote_in_line:
            dialect['quotechar'] = '"'
            self._quotechar      = '"'
        elif single_quote_in_line:
            dialect['quotechar'] = "'"
            self._quotechar      = "'"

        if len(next(csv.reader([line], **dialect, delimiter=","))) == self._n_columns:
            dialect['delimiter'] = ','
        elif len(next(csv.reader([line], **dialect, delimiter='\t'))) == self._n_columns:
            dialect['delimiter'] = '\t'
        else:
            self._set_filter(self._dense_advanced)

        return self.filter(line)

    def _sparse(self, line:str) -> Mapping[int,str]:
        keys_and_vals = re.split('\s*,\s*|\s+', line.strip("} {"))

        if keys_and_vals != ['']:
            keys = list(map(int,keys_and_vals[0::2]))
            vals = keys_and_vals[1::2]
        else:
            keys = []
            vals = []

        parsed = dict(zip(keys,vals))
        if parsed and (min(parsed.keys()) < 0 or self._n_columns <= max(parsed.keys())):
                raise CobaException(f"We were unable to parse a line in a way that matched the expected attributes.")
        return parsed

    def _dense_simple(self,line:str)-> Sequence[str]:
        dialect = self._dialect
        quotechar = self._quotechar

        if '"' in line:
            if quotechar == '"':
                pass
            elif quotechar is None:
                self._quotechar = '"'
                dialect['quotechar'] = '"'
            else:
                self._set_filter(self._dense_advanced)
                return self.filter(line)

        if "'" in line:
            if quotechar == "'":
                pass
            elif quotechar is None:
                self._quotechar = "'"
                dialect['quotechar'] = "'"
            else:
                self._set_filter(self._dense_advanced)
                return self.filter(line)

        parsed = next(csv.reader([line], **dialect))
        if len(parsed) != self._n_columns:
            raise CobaException(f"We were unable to parse a line in a way that matched the expected attributes.")
        return parsed

    def _dense_advanced(self,line:str) -> Sequence[str]:
        #the file does not appear to follow a readable csv.reader dialect
        #we fall back now to a slightly slower, but more flexible, parser

        if self._fallback_delim is None:
            #this isn't airtight but we can only infer so much.
            self._fallback_delim = ',' if len(line.split(',')) > len(line.split('\t')) else "\t"

        d_line = deque(line.split(self._fallback_delim))
        parsed = []

        while d_line:
            item = d_line.popleft().lstrip()

            if item[0] in self._quotes:
                possible_quotechar = item[0]
                while item.rstrip()[-1] != possible_quotechar or item.rstrip()[-2] == "\\":
                    item += "," + d_line.popleft()
                item = item.strip()[1:-1]

            parsed.append(item.replace('\\',''))

        if len(parsed) != self._n_columns:
            raise CobaException(f"We were unable to parse a line in a way that matched the expected attributes.")
        return parsed

class ArffReader(Filter[Iterable[str], Iterable[Union[Dense,Sparse]]]):

    _strip = methodcaller("strip")

    def __init__(self):
        """Instantiate an ArffReader."""

    def filter(self, lines: Iterable[str]) -> Iterable[Union[Dense,Sparse]]:

        lines = iter(filter(None,map(self._strip, lines)))
        attrs = [ l for l in takewhile(lambda l: l.lower()!="@data", lines) if l[:5].lower() == "@attr" ]
        data  = lines

        first_data,data = peek_first(data)
        while data and first_data[0]=="%":
            first_data,data = peek_first(islice(data,1,None))

        if not data: return []

        is_dense = not first_data.startswith("{") or not first_data.endswith("}")

        attr_reader = ArffAttrReader(is_dense)
        data_reader = ArffDataReader(is_dense)
        line_reader = ArffLineReader(is_dense,len(attrs))

        headers,encoders = zip(*attr_reader.filter(attrs))

        if is_dense:
            hdr_map = dict(zip(headers, count()))
            for line,missing in data_reader.filter(data):
                yield LazyDense(lambda line=line:line_reader.filter(line),encoders,hdr_map,missing)

        else:
            encs = dict(enumerate(encoders))
            fwd  = dict(zip(headers,count()))
            inv  = dict(zip(count(),headers))
            nsp  = set()
            for k,v in encs.items():
                try:
                    if v('0')!=0: nsp.add(k)
                except: pass #pragma: no cover

            for line,missing in data_reader.filter(data):
                yield LazySparse(lambda line=line:line_reader.filter(line), encs, nsp, fwd, inv, missing)

class LibsvmReader(Filter[Iterable[str], Iterable[Tuple[MutableMapping,Any]]]):
    """A filter capable of parsing Libsvm formatted data.

    For a complete description of the libsvm format see `here`__ and `here`__.

    __ https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
    __ https://github.com/cjlin1/libsvm
    """

    def filter(self, lines: Iterable[str]) -> Iterable[Tuple[MutableMapping, Any]]:

        for line in filter(None,lines):

            items  = line.strip().split(' ')

            no_label_line = items[0] == '' or ":" in items[0]

            if not no_label_line:
                labels = items[0].split(',')
                row    = { int(k):float(v) for i in items[1:] for k,v in [i.split(":")] }
                yield (row, labels)

class ManikReader(Filter[Iterable[str], Iterable[Tuple[MutableMapping,Any]]]):
    """A filter capable of parsing Manik formatted data.

    For a complete description of the manik format see `here`__ and `here`__.

    __ http://manikvarma.org/downloads/XC/XMLRepository.html
    __ https://drive.google.com/file/d/1u7YibXAC_Wz1RDehN1KjB5vu21zUnapV/view
    """

    def filter(self, lines: Iterable[str]) -> Iterable[Tuple[MutableMapping, Any]]:
        # we skip first line because it just has metadata
        return LibsvmReader().filter(islice(lines,1,None))
