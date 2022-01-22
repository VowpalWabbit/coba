import re
import csv
import collections.abc

from collections import deque, defaultdict
from itertools import islice, chain, count, product
from typing import Iterable, Sequence, List, Dict, Union, Any, Iterator, Pattern, Callable, Set

from coba.exceptions import CobaException
from coba.encodings import Encoder, OneHotEncoder

from coba.pipes.primitives import MutableMap, Filter

class LazyDualDense(collections.abc.MutableSequence):
    def __init__(self, values: Sequence[Any], headers: Dict[str,int] = {}, encoders: Sequence[Callable[[Any],Any]] = []) -> None:
        self._headers  = headers
        self._encoders = encoders
        self._values   = values
        
        self._removed  = 0
        self._offsets  = [0]*len(values)
        self._encoded  = [not encoders]*len(values)

    def __getitem__(self, index: Union[str,int]) -> Any:
        index = self._headers[index] if index in self._headers else self._offsets[index] + index

        if not self._encoded[index] and self._encoders:
            self._values[index] = self._encoders[index](self._values[index])
            self._encoded[index] = True

        return self._values[index]

    def __setitem__(self, index: Union[str,int], value: Any) -> None:
        index = self._headers[index] if index in self._headers else self._offsets[index] + index
        self._values[index] = value
        self._encoded[index] = True

    def __delitem__(self, index: Union[str,int]):
        index = self._headers[index] if index in self._headers else self._offsets[index] + index

        for i in range(index, len(self._values)):
            self._offsets[i] += 1

        self._removed += 1

    def __len__(self) -> int:
        return len(self._values) - self._removed

    def insert(self, index: int, value:Any):
        raise NotImplementedError()

    def __eq__(self, __o: object) -> bool:
        return list(self).__eq__(__o)

    def __repr__(self) -> str:
        return str(list(self._values))

    def __str__(self) -> str:
        return str(list(self._values))

class LazyDualSparse(collections.abc.MutableMapping):

    def __init__(self, values: Dict[Any,str], headers: Dict[str,Any] = {}, encoders: Dict[Any,Callable[[Any],Any]] = {}) -> None:
        self._headers  = headers
        self._encoders = encoders
        self._values   = values

        self._removed: Set[Any] = set()
        self._encoded: Dict[Any,bool] = defaultdict(bool)
    
    def __getitem__(self, index: Union[str,int]) -> Any:
        index = self._headers[index] if index in self._headers else index
        if index in self._removed: raise KeyError(index)

        if not self._encoded[index] and self._encoders:
            self._values[index] = self._encoders[index](self._values[index])
            self._encoded[index] = True

        return self._values[index]

    def __setitem__(self, index: Union[str,int], value: Any) -> None:
        index = self._headers[index] if index in self._headers else index
        if index in self._removed: raise KeyError(index)
        self._values[index] = value

    def __delitem__(self, index: Union[str,int]):
        index = self._headers[index] if index in self._headers else index
        if index in self._removed: raise KeyError(index)
        self._removed.add(index)

    def __len__(self) -> int:
        return len(self._values) - len(self._removed)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values.keys()-self._removed)

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

    #this class has been highly highly optimized. Before modifying anything one should 
    #run Performance_Tests.test_arffreader_performance to get a performance baseline.

    def __init__(self, cat_as_str=False, skip_encoding: bool = False, lazy_encoding: bool = True, header_indexing: bool = True):

        self._quotes = '"'+"'"

        self._cat_as_str      = cat_as_str 
        self._skip_encoding   = skip_encoding
        self._lazy_encoding   = lazy_encoding
        self._header_indexing = header_indexing 

    def filter(self, source: Iterable[str]) -> Iterable[MutableMap]:
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

        if is_dense:
            return self._parse_dense_data(chain([first_data_line], lines), headers, encoders)
        else:
            return self._parse_sparse_data(chain([first_data_line], lines), headers, encoders)

    def _encoders(self, encodings: Sequence[str], is_dense:bool) -> Encoder:
        numeric_types = ('numeric', 'integer', 'real')
        string_types  = ("string", "date", "relational")
        cat_as_str    = self._cat_as_str
        r_comma       = None
        identity      = lambda x: None if x=="?" else x.strip()

        for encoding in encodings:
            
            if self._skip_encoding:
                yield identity
            elif encoding in numeric_types: 
                yield lambda x: None if x=="?" else float(x)
            elif encoding.startswith(string_types):
                yield identity
            elif encoding.startswith('{'):
                if cat_as_str:
                    yield identity
                else:
                    #there is a bug in ARFF where the first class value in an ARFF class can will dropped from the 
                    #actual data because it is encoded as 0. Therefore our ARFF reader automatically adds a 0 value 
                    #to all sparse categorical one-hot encoders to protect against this.
                    r_comma = r_comma or re.compile("(,)")
                    categories = list(self._pattern_split(encoding[1:-1], r_comma))
                    
                    if not is_dense: 
                        categories = ["0"] + categories

                    def encoder(x,cats=categories,get=OneHotEncoder(categories, err_if_unknown=True)._onehots.__getitem__):
                        x=x.strip()
                        if x =="?":
                            return None
                        if x not in cats:
                            raise CobaException("We were unable to find one of the categorical values in the arff data.")
                        return get(x)
                    
                    yield encoder
            else:
                raise CobaException(f"An unrecognized encoding was found in the arff attributes: {encoding}.")

    def _parse_dense_data(self, lines: Iterable[str], headers: Sequence[str], encoders: Sequence[Encoder]) -> Iterable[MutableMap]:
        headers_dict        = dict(zip(headers,count()))
        possible_dialects   = self._possible_dialects()

        dialect             = possible_dialects.pop()
        fallback_delimieter = None

        for i,line in enumerate(lines):

            if dialect is not None:
                final = next(csv.reader([line], dialect=dialect))

                while len(final) != len(headers) and possible_dialects:
                    dialect = possible_dialects.pop()
                    final   = next(csv.reader([line], dialect=dialect))

                if len(final) != len(headers): dialect = None

            if dialect is None:
                #None of the csv dialects we tried were successful at parsing
                #we fall back now to a slightly slower, but more flexible, parser

                if fallback_delimieter is None:
                    #this isn't airtight but we can only infer so much.
                    fallback_delimieter = ',' if len(line.split(',')) > len(line.split('\t')) else "\t"

                line = deque(line.split(fallback_delimieter))
                final = []

                while line:
                    item = line.popleft()

                    if "'" in item or '"' in item:
                        quotechar = item[min(i for i in [item.find("'"), item.find('"')] if i >= 0)]
                        while item.rstrip()[-1] != quotechar and item.rstrip()[-2] != "\\":
                            item += "," + line.popleft()
                        item = item.strip()[1:-1]

                    final.append(item)

            if len(final) != len(headers):
                raise CobaException(f"We were unable to parse line {i} in a way that matched the expected attributes.")

            final_headers  = headers_dict if self._header_indexing else {}
            final_encoders = encoders if self._lazy_encoding else []
            final_items    = final if self._lazy_encoding else [ e(f) for e,f in zip(encoders,final)]

            if not self._lazy_encoding and not self._header_indexing:
                yield final_items
            else:
                yield LazyDualDense(final_items, final_headers, final_encoders)

    def _parse_sparse_data(self, lines: Iterable[str], headers: Sequence[str], encoders: Sequence[Encoder]) -> Iterable[MutableMap]:

        headers_dict  = dict(zip(headers,count()))
        defaults_dict = { k:"0" for k in range(len(encoders)) if encoders[k]("0") != 0 }
        encoders_dict = dict(zip(count(),encoders))

        for i,line in enumerate(lines):

            keys_and_vals = re.split('\s*,\s*|\s+', line.strip("} {"))

            keys = list(map(int,keys_and_vals[0::2]))
            vals = keys_and_vals[1::2]

            if max(keys) >= len(headers) or min(keys) < 0:
                raise CobaException(f"We were unable to parse line {i} in a way that matched the expected attributes.")

            final = { **defaults_dict, ** dict(zip(keys,vals)) }

            final_headers  = headers_dict if self._header_indexing else {}
            final_encoders = encoders_dict if self._lazy_encoding else {}
            final_items    = final if self._lazy_encoding else { k:encoders[k](v) for k,v in final.items() }

            if not self._lazy_encoding and not self._header_indexing:
                yield final_items
            else:
                yield LazyDualSparse(final_items, final_headers, final_encoders)

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
                    while item.strip()[-1] not in quotes:
                        item += next(items)
                    item = item.rstrip()[1:-1]

                yield item.strip()

            yield "".join(items).strip()
        except StopIteration:
            pass

    def _possible_dialects(self):
        legal_quotechars = ['"', "'"]
        legal_delimeters = [",", "\t"]

        return [
            {"delimeter":d, "quotechar": q, "skipinitialspace":True} for d,q in product(legal_delimeters, legal_quotechars) 
        ]

class CsvReader(Reader):

    def __init__(self, has_header: bool=False, **dialect):
        self._dialect    = dialect
        self._has_header = has_header 

    def filter(self, items: Iterable[str]) -> Iterable[MutableMap]:

        lines = iter(csv.reader(iter(filter(None,(i.strip() for i in items))), **self._dialect))

        if self._has_header:
            headers = dict(zip(next(lines), count()))

        for line in lines:
            yield line if not self._has_header else LazyDualDense(line,headers)

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
