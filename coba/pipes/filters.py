import json
import math
import collections.abc
import copy

from collections import defaultdict
from itertools import islice, chain
from typing import Iterable, Any, Sequence, Dict, Callable, Optional, Union

from coba.random import CobaRandom
from coba.exceptions import CobaException
from coba.encodings import Encoder, CobaJsonEncoder, CobaJsonDecoder

from coba.pipes.primitives import Filter, MutableMap

class Identity(Filter[Any, Any]):
    """Return whatever is given to the filter."""
    def filter(self, item:Any) -> Any:
        return item

class Shuffle(Filter[Iterable[Any], Sequence[Any]]):
    """Shuffle a sequence of items."""

    def __init__(self, seed:Optional[int]) -> None:
        """Instantiate a Shuffle filter.

        Args:
            seed: A random number seed which determines the new sequence order.
        """

        if seed is not None and (not isinstance(seed,int) or seed < 0):
            raise ValueError(f"Invalid parameter for Shuffle: {seed}. An optional integer value >= 0 was expected.")

        self._seed = seed

    def filter(self, items: Iterable[Any]) -> Sequence[Any]: 
        return CobaRandom(self._seed).shuffle(list(items))

class Take(Filter[Iterable[Any], Sequence[Any]]):
    """Take a fixed number of items from an iterable."""

    def __init__(self, count:Optional[int]) -> None:
        """Instantiate a Take filter.

        Args:
            count: The number of items we wish to take from the given iterable.
        """

        if count is not None and (not isinstance(count,int) or count < 0):
            raise ValueError(f"Invalid parameter for count: {count}. An optional integer value >= 0 was expected.")

        self._count = count

    def filter(self, items: Iterable[Any]) -> Sequence[Any]:
        items =  list(islice(items,self._count))
        return items if len(items) == self._count else []

class Reservoir(Filter[Iterable[Any], Sequence[Any]]):
    """Take a fixed number of random items from an iterable.
    
    Remarks:
        We use Algorithm L as described by Kim-Hung Li. (1994) to take a random count of items.

    References:
        Kim-Hung Li. 1994. Reservoir-sampling algorithms of time complexity O(n(1 + log(N/n))). 
        ACM Trans. Math. Softw. 20, 4 (Dec. 1994), 481–493. DOI:https://doi.org/10.1145/198429.198435
    """

    def __init__(self, count: Optional[int], seed: int = 1) -> None:
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

    def filter(self, items: Iterable[Any]) -> Sequence[Any]:

        if self._count is None:
            return items

        if self._count == 0:
            return []

        items     = iter(items)
        reservoir = list(islice(items,self._count))

        if len(reservoir) < self._count:
            return []

        rng = CobaRandom(self._seed)
        W = 1

        try:
            while True:
                [r1,r2,r3] = rng.randoms(3)
                W = W * math.exp(math.log(r1)/self._count)
                S = math.floor(math.log(r2)/math.log(1-W))
                reservoir[int(r3*self._count-.001)] = next(islice(items,S,S+1))
        except StopIteration:
            pass

        return reservoir

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

class Flatten(Filter[Iterable[Any], Iterable[Any]]):

    def filter(self, data: Iterable[Any]) -> Iterable[Any]:

        for row in data:

            if isinstance(row,dict):
                for k in list(row.keys()):
                    if isinstance(row[k],(list,tuple)):
                        row.update([((k,i), v) for i,v in enumerate(row.pop(k))])

            elif isinstance(row,list):
                for k in reversed(range(len(row))):
                    if isinstance(row[k],(list,tuple)):
                        for v in reversed(row.pop(k)):
                            row.insert(k,v) 

            else:
                raise CobaException(f"Unrecognized type ({type(row).__name__}) passed to Flattens.")

            yield row

class Encode(Filter[Iterable[MutableMap], Iterable[MutableMap]]):

    def __init__(self, encoders:Dict[Union[str,int],Encoder], fit_using:int = None, missing_val: str = None):
        self._encoders    = encoders
        self._fit_using   = fit_using
        self._missing_val = missing_val

    def filter(self, items: Iterable[MutableMap]) -> Iterable[MutableMap]:

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
                val = item[k] if is_dense else item.get(k,0)
                if val not in  ['',self._missing_val]: values_for_fitting[k].append(val)

        for k,v in values_for_fitting.items():
            encoders[k] = encoders[k].fit(v)

        for item in chain(items_for_fitting, items):
            for k,v in encoders.items():
                if is_dense or k in item:
                    val = item[k]
                    item[k] = encoders[k].encode(val) if val not in ['',self._missing_val] else val
            yield item

class Drop(Filter[Iterable[MutableMap], Iterable[MutableMap]]):

    def __init__(self, drop_cols: Sequence[int] = [], drop_row: Callable[[MutableMap], bool] = lambda r: False) -> None:
        self._drop_cols = sorted(drop_cols, reverse=True)
        self._drop_row  = drop_row

    def filter(self, data: Iterable[MutableMap]) -> Iterable[MutableMap]:
        
        keep_row = lambda r: not self._drop_row(r) if self._drop_row else True
        for row in filter(keep_row, data):            
            if row is not None:
                for col in self._drop_cols:
                    row.pop(col)            
            yield row

class Structure(Filter[Iterable[MutableMap], Iterable[Any]]):

    def __init__(self, structure: Sequence[Any]) -> None:
        self._structure = []
        stack = [structure]
        while stack:
            item = stack.pop()
            if isinstance(item,list):
                stack.append("LO")
                stack.extend(item)
                stack.append("LC")
            elif isinstance(item,tuple):
                stack.append("TO")
                stack.extend(item)
                stack.append("TC")
            else:
                self._structure.insert(0,item)

    def filter(self, data: Iterable[MutableMap]) -> Iterable[Any]:
        for row in data:
            stack   = []
            working = []
            for item in self._structure:
                if item in ["LO","TO"]:
                    stack.append(working)
                    working = []
                elif item == "LC":
                    new_working = stack.pop()
                    new_working.append(working)
                    working = new_working
                elif item == "TC":
                    new_working = stack.pop()
                    new_working.append(tuple(working))
                    working = new_working
                elif item == None:
                    working.append(row)
                else:
                    working.append(row.pop(item))
            
            yield working[0]

class Default(Filter[Iterable[MutableMap], Iterable[MutableMap]]):

    def __init__(self, defaults: Dict[Any, Any]) -> None:
        self._defaults = defaults

    def filter(self, data: Iterable[MutableMap]) -> Iterable[MutableMap]:

        for row in data:

            if isinstance(row,dict):
                for k,v in self._defaults.items():
                    if k not in row: row[k] = v

            yield row
