import json
import math
import copy

from collections import defaultdict, abc
from itertools import islice, chain
from typing import Iterable, Any, Sequence, Mapping, Optional, Union

from coba.random import CobaRandom
from coba.encodings import Encoder, CobaJsonEncoder, CobaJsonDecoder
from coba.utilities import peek_first

from coba.pipes.primitives import Filter, Dense, Sparse

class Identity(Filter[Any, Any]):
    """A filter which returns what is given."""
    def filter(self, item:Any) -> Any:
        return item

    @property
    def params(self) -> Mapping[str, Any]:
        return {}

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

    @property
    def params(self) -> Mapping[str, Any]:
        return { "shuffle": self._seed }

class Take(Filter[Iterable[Any], Sequence[Any]]):
    """Take a fixed number of items from an iterable."""

    def __init__(self, count: Optional[int] ) -> None:
        """Instantiate a Take filter.

        Args:
            count: The number of items we wish to take from an iterable.
        """

        is_valid_count = (count is None) or (isinstance(count,int) and count >= 0)

        if not is_valid_count:
            raise ValueError(f"Invalid value for Take: {count}. A positive integer or None was expected.")

        self._count = count

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        return islice(items, self._count)

    @property
    def params(self) -> Mapping[str, Any]:
        return { "take": self._count }

class Reservoir(Filter[Iterable[Any], Sequence[Any]]):
    """Take a fixed number of random items from an iterable.

    Remarks:
        We use Algorithm L as described by Kim-Hung Li. (1994) to take a random count of items.

    References:
        Kim-Hung Li. 1994. Reservoir-sampling algorithms of time complexity O(n(1 + log(N/n))).
        ACM Trans. Math. Softw. 20, 4 (Dec. 1994), 481–493. DOI:https://doi.org/10.1145/198429.198435
    """

    def __init__(self, count: Optional[int], seed: float = 1) -> None:
        """Instantiate a Reservoir filter.

        Args:
            count: The number of items we wish to sample from an iterable (expressed as either an exact value or range).
            seed : The seed which determines which random items to take.
        """

        is_valid_count = (count is None) or (isinstance(count,int) and count >= 0)

        if not is_valid_count:
            raise ValueError(f"Invalid value for Reservoir: {count}. A positive integer or None was expected.")

        self._count = count
        self._seed  = seed

    @property
    def params(self) -> Mapping[str, Any]:
        return { "reservoir_count": self._count, "reservoir_seed": self._seed }

    def filter(self, items: Iterable[Any]) -> Sequence[Any]:

        rng = CobaRandom(self._seed)

        if self._count == 0:
            return []

        if self._count is None:
            return rng.shuffle(items)

        W         = 1
        items     = iter(items)
        reservoir = rng.shuffle(list(islice(items,self._count)))

        try:
            while True:
                [r1,r2,r3] = rng.randoms(3)
                W = W * math.exp(math.log(r1)/ (self._count or 1) )
                S = math.floor(math.log(r2)/math.log(1-W))
                reservoir[int(r3*self._count-.001)] = next(islice(items,S,S+1))
        except StopIteration:
            pass

        return Take(self._count).filter(reservoir)

class JsonEncode(Filter[Any, str]):
    """A filter which turn a Python object into JSON strings."""

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
                    #where we format as a truncated string and then manually remove the 
                    #string indicators from the json via string replace methods
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
        return self._encoder.encode(self._min(copy.deepcopy([item]))[0] if self._minify else item).replace('"|',"").replace('|"',"")

class JsonDecode(Filter[str, Any]):
    """A filter which turns a JSON string into a Python object."""

    def __init__(self, decoder: json.decoder.JSONDecoder = CobaJsonDecoder()) -> None:
        self._decoder = decoder

    def filter(self, item: str) -> Any:
        return self._decoder.decode(item)

class Flatten(Filter[Iterable[Any], Iterable[Any]]):
    """A filter which flattens rows in table shaped data."""

    def filter(self, data: Iterable[Any]) -> Iterable[Any]:

        try:
            data  = iter(data)
            first = next(data)
        except StopIteration:
            return []

        if isinstance(first,abc.MutableSequence):
            first_type = "list" 
        elif isinstance(first,dict):
            first_type = "dict"
        elif isinstance(first,abc.Sequence):
            first_type = "tuple"
        else:
            first_type = None

        is_flattable = lambda v: isinstance(v,abc.Iterable) and not isinstance(v,str)

        flattable = []
        if first_type in ["list","tuple"]:
            flattable = [ is_flattable(v) for v in first ]
            any_flattable = any(flattable)
        elif first_type == "dict":
            #we assume that flattable entries will be populated in every row even though the dataset is sparse
            flattable = {k:[f"{k}_{i}" for i in range(len(v))] for k,v in first.items() if is_flattable(v) }
            any_flattable = bool(flattable)
        else:
            any_flattable = False

        def flatter_list(row):
            for f,r in zip(flattable,row):
                if f: yield from r 
                else: yield r

        for row in chain([first], data):

            if not any_flattable:
                yield row

            else:
                if first_type == "list":
                    row = list(flatter_list(row))
                elif isinstance(row,tuple):
                    row = tuple(flatter_list(row))
                elif first_type == "dict":
                    #in-line dict comprehension was faster than a generator like we did with list
                    row = { k:v for k,v in row.items() for k,v in ( zip(flattable[k],v) if k in flattable else ((k,v),)) }
                yield row

class Encode(Filter[Iterable[Union[Sequence,Mapping]], Iterable[Union[Sequence,Mapping]]]):
    """A filter which encodes features in table shaped data."""

    def __init__(self, encoders:Mapping[Any,Encoder], fit_using:int = None, missing_val: str = None):
        """Instantiate an Encode filter.

        Args:
            encoders: A mapping from feature names to the Encoder to be used on the data.
            fit_using: The number of rows that should be used to fit the encoder.
            missing_val: The value that should be used when a feature is missing from a row.
        """
        self._encoders    = encoders
        self._fit_using   = fit_using
        self._missing_val = missing_val

    def filter(self, items: Iterable[Union[Sequence,Mapping]]) -> Iterable[Union[Sequence,Mapping]]:

        items = iter(items) # this makes sure items are pulled out for fitting

        try:
            first_item = next(items)
            items      = chain([first_item], items)
        except StopIteration:
            return []

        is_dense = isinstance(first_item,Dense)

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
            item = list(item) if is_dense else dict(item)
            for k,v in encoders.items():
                if is_dense or k in item:
                    val = item[k]
                    item[k] = encoders[k].encode(val) if val not in ['',self._missing_val] else val
            yield item

class Structure(Filter[Iterable[Union[Sequence,Mapping]], Iterable[Any]]):
    """A filter which restructures rows in table shaped data."""

    def __init__(self, structure: Sequence[Any]) -> None:
        """Instantiate Structure filter.

        Args:
            structure: The structure that each row should be reformed into. Items in the structure should be
                feature names. A None value indicates the location that all left-over features will be placed.
        """

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

    def filter(self, data: Iterable[Union[Sequence,Mapping]]) -> Iterable[Any]:
        first, data = peek_first(data)
        is_dense = isinstance(first, Dense)

        for row in data:

            row = list(row) if is_dense else dict(row.items())

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

class Default(Filter[Iterable[Union[Sequence,Mapping]], Iterable[Union[Sequence,Mapping]]]):
    """A filter which sets default values for row features in table shaped data."""

    def __init__(self, defaults: Mapping[Any, Any]) -> None:
        """Instantiate a Default filter.

        Args:
            defaults: A mapping from feature names to their default values.
        """
        self._defaults = defaults

    def filter(self, data: Iterable[Union[Sequence,Mapping]]) -> Iterable[Union[Sequence,Mapping]]:

        for row in data:

            if isinstance(row, Sparse):
                row = dict(row)
                for k,v in self._defaults.items():
                    if k not in row: row[k] = v

            yield row

class Cache(Filter[Iterable[Any], Iterable[Any]]):

    def __init__(self) -> None:
        self._cache = []

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        if self._cache: 
            yield from self._cache
        else:
            for item in items:
                self._cache.append(item)
                yield item
