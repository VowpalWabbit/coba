"""This module contains utility classes for transforming data between encodings."""

import json
import time
import collections.abc

from collections import Counter, OrderedDict, defaultdict
from itertools import count, accumulate, chain
from abc import ABC, abstractmethod

from typing import Iterator, Sequence, Generic, TypeVar, Any, Tuple, Union, Dict

from coba.exceptions import CobaException

_T_out = TypeVar('_T_out', bound=Any, covariant=True) 

class Encoder(Generic[_T_out], ABC):
    """The Encoder interface.

    Remarks:
        While it can't be enforced by the interface, the assumption is that all Encoder
        implementations are immutable. This means that the `fit` method should always
        return a new Encoder.
    """

    @property
    @abstractmethod
    def is_fit(self) -> bool:
        """Indicates if the encoder has been fit.

        Returns:
            True if the Encoder is ready to encode otherwise False.
        """
        ...

    @abstractmethod
    def fit(self, values: Sequence[Any]) -> 'Encoder':
        """Fit the encoder given training data.

        Args:
            values: The encoder training values.

        Returns:
            A fit Encoder.
        """
        ...

    @abstractmethod
    def encode(self, value: Any) -> Sequence[_T_out]:
        """Encode the given value.

        Args:
            value: The value to encode.

        Returns:
            The encoded value.
        """
        ...

    @abstractmethod
    def encodes(self, values: Sequence[Any]) -> Sequence[_T_out]:
        """Encode the given values.

        Args:
            values: The values toencode.

        Returns:
            The encoded values.
        """
        ...

    def fit_encodes(self, values: Sequence[Any]) -> Sequence[_T_out]:
        """Fit and then encode the given values.

        Args:
            values: The values toencode.

        Returns:
            The encoded values.
        """
        if self.is_fit:
            return self.encodes(values)
        else:
            return self.fit(values).encodes(values)

class IdentityEncoder(Encoder[Any]):

    @property
    def is_fit(self) -> bool:
        return True

    def fit(self, values: Sequence[Any]) -> 'Encoder':
        return self

    def encode(self, value: Any) -> Any:
        return value

    def encodes(self, values: Sequence[Any]) -> Sequence[Any]:
        return values

class MissingEncoder(Encoder[Any]):

    def __init__(self, encoder:Encoder = IdentityEncoder(), missing_vals: Sequence[Any] = ["?",""], missing_rep: Any = None) -> None:
        self._encoder      = encoder
        self._missing_vals = missing_vals
        self._missing_rep  = missing_rep

    @property
    def is_fit(self) -> bool:
        return True

    def fit(self, values: Sequence[Any]) -> 'Encoder':
        return self

    def encode(self, value: Any) -> Any:
        return self._missing_rep if value in self._missing_vals else self._encoder.encode(value)

    def encodes(self, values: Sequence[Any]) -> Sequence[Any]:
        return [ self.encode(v) for v in values ]

class StringEncoder(Encoder[str]):
    """An Encoder implementation that turns incoming values into string values."""

    @property
    def is_fit(self) -> bool:
        return True

    def fit(self, values: Sequence[Any]) -> 'StringEncoder':
        return StringEncoder()

    def encode(self, value: Any) -> str:
        return str(value)

    def encodes(self, values: Sequence[Any]) -> Sequence[str]:
        return list(map(str,values))

class NumericEncoder(Encoder[float]):
    """An Encoder implementation that turns incoming values into float values."""

    @property
    def is_fit(self) -> bool:
        return True

    def fit(self, values: Sequence[Any]) -> 'NumericEncoder':
        return NumericEncoder()

    def encode(self, value: Any) -> float:
        try:
            return float(value)
        except:
            return float('nan')

    def encodes(self, values: Sequence[Any]) -> Sequence[float]:
        return list(self._float_generator(values))

    def _float_generator(self,values) -> Iterator[float]:
        for value in values:
            try:
                yield float(value)
            except:
                yield float('nan')

class OneHotEncoder(Encoder[Tuple[int,...]]):
    """An Encoder implementation that turns incoming values into a one hot representation."""

    def __init__(self, values: Sequence[Any] = [], err_if_unknown = False) -> None:
        """Instantiate a OneHotEncoder.

        Args:
            values: Provide the universe of values for encoding and set `is_fit==True`.
            err_if_unknown: When an unknown value is passed to `encode` throw an exception (otherwise encode as all 0's).
        """

        self._err_if_unknown = err_if_unknown
        self._onehots        = None
        self._default        = None

        if values:

            values = sorted(set(values), key=lambda v: values.index(v))

            self._default = tuple([0] * len(values))
            known_onehots = [ [0] * len(values) for _ in range(len(values)) ]

            for i,k in enumerate(known_onehots):
                k[i] = 1

            keys_and_values = zip(values, map(tuple,known_onehots))
    
            if self._err_if_unknown:
                self._onehots = dict(keys_and_values)
            else:
                self._onehots = defaultdict(lambda:self._default, keys_and_values)

    @property
    def is_fit(self) -> bool:
        return self._onehots is not None

    def fit(self, values: Sequence[Any]) -> 'OneHotEncoder':
        return OneHotEncoder(values = values, err_if_unknown = self._err_if_unknown)

    def encode(self, value: Any) -> Tuple[int,...]:

        if self._onehots is None:
            raise CobaException("This encoder must be fit before it can be used.")

        try:
            return self._onehots[value]
        except KeyError as e:
            raise CobaException(f"We were unable to find {e} in {list(self._onehots.keys())}")

    def encodes(self, values: Sequence[Any]) -> Sequence[Tuple[int,...]]:
        
        if self._onehots is None:
            raise CobaException("This encoder must be fit before it can be used.")

        try:
            return list(map(self._onehots.__getitem__,values))
        except KeyError as e:
            raise CobaException(f"We were unable to find {e} in {list(self._onehots.keys())}")

class FactorEncoder(Encoder[int]):
    """An Encoder implementation that turns incoming values into factor representation."""

    def __init__(self, values: Sequence[Any] = [], err_if_unknown = False) -> None:
        """Instantiate a FactorEncoder.

        Args:
            values: Provide the universe of values for encoding and set `is_fit==True`.
            err_if_unknown: When an unknown value is passed to `encode` throw an exception (otherwise encode as all 0's).
        """

        self._err_if_unknown = err_if_unknown
        self._levels         = None

        if values:

            values = sorted(set(values), key=lambda v: values.index(v))
            levels  = [ i + 1 for i in range(len(values)) ]

            pairs = zip(values, levels)
            self._levels = dict(pairs) if self._err_if_unknown else defaultdict(lambda: float('nan'), pairs)

    @property
    def is_fit(self) -> bool:
        return self._levels is not None

    def fit(self, values: Sequence[Any]) -> 'FactorEncoder':
        return FactorEncoder(values = values, err_if_unknown = self._err_if_unknown)

    def encode(self, value: Any) -> int:

        if self._levels is None:
            raise CobaException("This encoder must be fit before it can be used.")

        try:
            return self._levels[value]
        except KeyError as e:
            raise CobaException(f"We were unable to find {e} in {self._levels.keys()}") from None

    def encodes(self, values: Sequence[Any]) -> Sequence[int]:
        
        if self._levels is None:
            raise CobaException("This encoder must be fit before it can be used.")

        try:
            return list(map(self._levels.__getitem__,values))
        except KeyError as e:
            raise CobaException(f"We were unable to find {e} in {self._levels.keys()}") from None

class CobaJsonEncoder(json.JSONEncoder):
    """A json encoder that allows for potential COBA extensions in the future."""

class CobaJsonDecoder(json.JSONDecoder):
    """A json decoder that allows for potential COBA extensions in the future."""

class InteractionsEncoder:
    
    def __init__(self, interactions: Sequence[str]) -> None:

        self.times       = [0,0,0,0]
        self.n           = 0
        self._cross_pows = OrderedDict(zip(interactions,map(OrderedDict,map(Counter,interactions))))
        self._ns_max_pow = { n:max(p.get(n,0) for p in self._cross_pows.values()) for n in set(''.join(interactions)) }

    def encode(self, **ns_raw_values: Union[str, float, Sequence[Union[str,float]], Dict[Union[str,int],Union[str,float]]]) -> Union[Sequence[float], Dict[str,float]]:

        self.n+= 1

        is_sparse_type = lambda f: isinstance(f,dict) or isinstance(f,str)
        is_sparse_sequ = lambda f: isinstance(f, collections.abc.Sequence) and any(map(is_sparse_type,f))

        is_sparse = any(is_sparse_type(v) or is_sparse_sequ(v) for v in ns_raw_values.values())

        def make_all_dict_values(v) -> Dict[str,Union[str,float]]:
            return v if isinstance(v,dict) else dict(zip(map(str,count()),v)) if isinstance(v,(list,tuple)) else { "0":v }

        def make_all_list_values(v) -> Sequence[Union[str,float]]:
            return v if isinstance(v, (list,tuple)) else [v]

        def handle_str_values(v: Dict[str,Union[str,float]]) -> Dict[str,float]:
            return { (f"{x}{y}" if isinstance(y,str) else x):(1 if isinstance(y,str) else y) for x,y in v.items() }

        start = time.time()
        if is_sparse:
            ns_values = {ns:handle_str_values(make_all_dict_values(v)) for ns,v in ns_raw_values.items()}
            ns_values = {ns:{f"{ns}{k}":v for k,v in values.items()}   for ns,values in ns_values.items() }
        else:
            ns_values = {ns:make_all_list_values(v) for ns,v in ns_raw_values.items()}
        self.times[0] += time.time()-start

        if is_sparse:
            start = time.time()
            key_pows = { ns: self._pows(list(ns_values[ns].keys()  ), max_pow) for ns, max_pow in self._ns_max_pow.items() }
            val_pows = { ns: self._pows(list(ns_values[ns].values()), max_pow) for ns, max_pow in self._ns_max_pow.items() }
            self.times[1] += time.time()-start

            start = time.time()
            key_crosses = [ self._cross(key_pows, cross_pow) for cross_pow in self._cross_pows.values() ]
            val_crosses = [ self._cross(val_pows, cross_pow) for cross_pow in self._cross_pows.values() ]
            self.times[2] += time.time()-start

            start = time.time()
            encoded = dict(zip(chain.from_iterable(key_crosses), chain.from_iterable(val_crosses)))
            self.times[3] += time.time()-start

            return encoded
        else:
            start = time.time()
            val_pows = { ns: self._pows(ns_values[ns], max_pow) for ns, max_pow in self._ns_max_pow.items() }
            self.times[1] += time.time()-start

            start = time.time()
            val_crosses = [ self._cross(val_pows, cross_pow) for cross_pow in self._cross_pows.values() ]
            self.times[2] += time.time()-start

            start = time.time()
            encoded = sum(val_crosses,[])
            self.times[3] += time.time()-start

            return encoded

    def _pows(self, values, degree):
        #WARNING: This function has been extremely optimized. Test speed before and after making any changes.
        #WARNING: Look in test_performance for three existing performance tests.
        starts = [1]*len(values)
        terms  = [['']] if isinstance(values[0],str) else [[1]]

        for d in range(degree):
            if isinstance(values[0],str):
                terms.append([v+t for v,s in zip(values,starts) for t in terms[d][(s-1):]])
            else:
                terms.append([v*t for v,s in zip(values,starts) for t in terms[d][(s-1):]])

            starts = list(accumulate(starts[:1]+starts[-1:]+starts[1:-1]))

        return terms

    def _cross(self, ns_pows, cross_pow):
        #WARNING: This function has been extremely optimized. Test speed before and after making any changes.
        #WARNING: Look in test_performance for three existing performance tests.

        values = [ ns_pows[ns][p] for ns,p in cross_pow.items() ]
        cross  = values[0]

        if isinstance(cross[0],str):
            for vs in values[1:]: cross = [ o+v for o in cross for v in vs ]
        else:
            for vs in values[1:]: cross = [ o*v for o in cross for v in vs ]

        return cross
