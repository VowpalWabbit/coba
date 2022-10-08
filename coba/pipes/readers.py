import re
import csv

from collections import deque
from itertools import islice, chain, count
from typing import Iterable, Sequence, List, Union, Any, Pattern, Tuple, Dict
from typing import MutableSequence, MutableMapping

from coba.exceptions import CobaException
from coba.encodings import Encoder, OneHotEncoder

from coba.pipes.core import Pipes
from coba.pipes.rows import EncodeRow, IndexRow, DenseRow, SparseRow
from coba.pipes.sources import LambdaSource
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

    def filter(self, items: Iterable[str]) -> Iterable[MutableSequence]:

        lines = iter(csv.reader(iter(filter(None,(i.strip() for i in items))), **self._dialect))

        if self._has_header:
            headers = dict(zip(next(lines), count()))
            indexer = IndexRow(headers)

        for line in lines:
            row = DenseRow(loaded=line,missing=False)
            yield row if not self._has_header else indexer.filter(row)

class ArffReader(Filter[Iterable[str], Iterable[Union[MutableSequence,MutableMapping]]]):
    """A filter capable of parsing ARFF formatted data.

    For a complete description of the ARFF format see `here`__.

    __ https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/

    Remarks:
        This class has been highly highly optimized. Before modifying anything run
        Performance_Tests.test_arffreader_performance to get a performance baseline.
    """

    class SparseRowParser:
        def __init__(self, max_key:int, sparse_onehots:Dict[int,str]) -> None:
            self._max_key = max_key
            self._sparse_onehots = sparse_onehots

        def parse(self, i:int, line:str) -> Dict[str,str]:
            keys_and_vals = re.split('\s*,\s*|\s+', line.strip("} {"))

            if keys_and_vals != ['']:
                keys = list(map(int,keys_and_vals[0::2]))
                vals = keys_and_vals[1::2]
            else:
                keys = []
                vals = []

            if keys and (max(keys) >= self._max_key or min(keys) < 0):
                raise CobaException(f"We were unable to parse line {i} in a way that matched the expected attributes.")

            final = { **self._sparse_onehots, ** dict(zip(keys,vals)) }

            return final

    class DenseRowParser:
        def __init__(self, column_count:int) -> None:
            self._column_count = column_count
            self._fallback_delimieter = None

            self._quotechars   = {'"',"'"}
            self._quotes       = '"'+"'"
            self._quotechar    = None
            self._delimiter    = None
            self._use_advanced = False
            self._base_dialect = dict(skipinitialspace=True,escapechar="\\",doublequote=False)

        def parse(self, i:int, line:str) -> List[str]:
            if not self._use_advanced:
                for quote in self._quotechars-{self._quotechar}:
                    if quote in line:
                        self._quotechar = self._quotechar or quote
                        self._use_advanced = self._quotechar != quote

            if not self._use_advanced and not self._delimiter:
                if len(next(csv.reader([line], ))) == self._column_count:
                    self._delimiter = ','
                elif len(next(csv.reader([line], quotechar=(self._quotechar or '"'), delimiter='\t', **self._base_dialect))) == self._column_count:
                    self._delimiter = '\t'
                else:
                    self._use_advanced = True

            if not self._use_advanced:
                final = next(csv.reader([line], quotechar=(self._quotechar or '"'), delimiter=self._delimiter, **self._base_dialect))

            else:
                #the file does not appear to follow a readable csv.reader dialect
                #we fall back now to a slightly slower, but more flexible, parser
                if self._fallback_delimieter is None:
                    #this isn't airtight but we can only infer so much.
                    self._fallback_delimieter = ',' if len(line.split(',')) > len(line.split('\t')) else "\t"

                d_line = deque(line.split(self._fallback_delimieter))
                final = []

                while d_line:
                    item = d_line.popleft().lstrip()

                    if item[0] in self._quotes:
                        possible_quotechar = item[0]
                        while item.rstrip()[-1] != possible_quotechar or item.rstrip()[-2] == "\\":
                            item += "," + d_line.popleft()
                        item = item.strip()[1:-1]

                    final.append(item.replace('\\',''))

            if len(final) != self._column_count:
                raise CobaException(f"We were unable to parse line {i} in a way that matched the expected attributes.")

            return final

    def __init__(self, cat_as_str: bool =False, missing_value: Any = float('nan')):
        """Instantiate an ArffReader.

        Args:
            cat_as_str: Indicates that categorical features should be encoded as a string rather than one hot encoded.
            missing_value: The value to replace missing values with
        """

        self._quotes        = '"'+"'"
        self._cat_as_str    = cat_as_str
        self._missing_value = missing_value

    def filter(self, source: Iterable[str]) -> Iterable[Union[MutableSequence,MutableMapping]]:
        headers  : List[str    ] = []
        encodings: List[Encoder] = []

        source = (line.strip() for line in source)
        source = (line for line in source if line and not line.startswith("%"))

        lines = iter(source)

        r_space = re.compile("(\s+)")
        for line in lines:
            if line[0:10].lower() == "@attribute":
                header, encoding = tuple(self._pattern_split(line[11:], r_space, n=2))
                headers.append(header)
                encodings.append(encoding)
            elif line[0:5].lower() == "@data":
                break

        try:
            first_data_line = next(lines)
        except StopIteration:
            return []

        is_dense = not (first_data_line.startswith('{') and first_data_line.endswith('}'))
        encoders = list(self._encoders(encodings,is_dense))

        if len(headers) != len(set(headers)):
            raise CobaException("Two columns in the ARFF file had identical header values.")

        if is_dense:
            return self._parse_dense_data(chain([first_data_line], lines), headers, encoders)
        else:
            return self._parse_sparse_data(chain([first_data_line], lines), headers, encoders)

    def _encoders(self, encodings: Sequence[str], is_dense:bool) -> Encoder:
        numeric_types = ('numeric', 'integer', 'real')
        string_types  = ("string", "date", "relational")
        r_comma       = None

        for encoding in encodings:

            if encoding.lower() in numeric_types:
                yield lambda x: self._missing_value if x=="?" or x=="" else float(x)
            elif encoding.lower().startswith(string_types):
                yield lambda x: self._missing_value if x=="?" else x.strip()
            elif encoding.startswith('{'):
                r_comma = r_comma or re.compile("(,)")
                categories = list(self._pattern_split(encoding[1:-1], r_comma))

                if not is_dense:
                    #there is a bug in ARFF where the first class value in an ARFF class can will dropped from the
                    #actual data because it is encoded as 0. Therefore, our ARFF reader automatically adds a 0 value
                    #to all sparse categorical one-hot encoders to protect against this.
                    categories = ["0"] + categories

                def encoder(x:str,cats=categories,get=OneHotEncoder(categories)._onehots.__getitem__):

                    if x =="?":
                        return self._missing_value

                    #if x not in cats and x[0] in self._quotes and x[0]==x[-1] and len(x) > 1:
                    #    x = x[1:-1]

                    if x not in cats:
                        raise CobaException("We were unable to find one of the categorical values in the arff data.")

                    return x if self._cat_as_str else get(x)

                yield encoder
            else:
                raise CobaException(f"An unrecognized encoding was found in the arff attributes: {encoding}.")

    def _parse_dense_data(self, lines: Iterable[str], headers: Sequence[str], encoders: Sequence[Encoder]) -> Iterable[MutableSequence]:

        parser       = ArffReader.DenseRowParser(len(headers))
        row_indexer  = IndexRow(dict(zip(headers,count())))
        row_encoder  = EncodeRow(encoders)

        for i,line in enumerate(lines):
            compact = line.translate(str.maketrans('','', ' \t\n\r\v\f'))
            missing = compact[0] in '?,' or compact.find(',?,') != -1 or compact.find(',,') != -1 or compact[-1] in '?,'            
            loader  = LambdaSource(lambda line=line, i=i:  parser.parse(i,line))
            
            yield Pipes.join(row_encoder, row_indexer).filter(DenseRow(loader=loader,missing=missing))

    def _parse_sparse_data(self, lines: Iterable[str], headers: Sequence[str], encoders: Sequence[Encoder]) -> Iterable[MutableMapping]:

        #if there is a onehot column whose value is 0 it will be missing. This fills it in.
        onehots     = { k:"0" for k in range(len(encoders)) if encoders[k]("0") != 0 }
        parser      = ArffReader.SparseRowParser(len(headers), onehots)
        row_indexer = IndexRow(dict(zip(headers,count())))
        row_encoder = EncodeRow(encoders)

        for i,line in enumerate(lines):
            missing = " ?," in line or line[-2:] == " ?"
            loader  = LambdaSource(lambda line=line, i=i:  parser.parse(i,line))
            yield Pipes.join(row_encoder, row_indexer).filter(SparseRow(loader=loader,missing=missing))

    def _pattern_split(self, line: str, pattern: Pattern[str], n=None):

        items  = iter(pattern.split(line))
        count  = 0
        quotes = self._quotes

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
                    item.strip()

                yield item

            yield "".join(items).strip()
        except StopIteration:
            pass

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
