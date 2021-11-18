"""The encodings module contains utility classes for transforming data between encodings."""

import json
import collections
import time

from numbers import Number
from itertools import product
from abc import ABC, abstractmethod
from typing import Iterator, Sequence, Generic, TypeVar, Any, Tuple, Union

_T_out = TypeVar('_T_out', bound=Any, covariant=True) 

class Encoder(Generic[_T_out], ABC):
    """The interface for encoder implementations.

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
            A boolean indicating if this encoder has been fit (i.e., ready to "encode").
        """
        ...

    @abstractmethod
    def fit(self, values: Sequence[Any]) -> 'Encoder':
        """Determine how to encode from given training data.

        Args:
            values: A collection of values to use for determining the encoding.

        Returns:
            An Encoder that has been fit.

        Remarks:
            This method should return a new Encoder that is fit without altering
            the original Encoder. If the original Encoder is already fit this 
            should raise an Exception.
        """
        ...

    @abstractmethod
    def encode(self, values: Sequence[Any]) -> Sequence[_T_out]:
        """Encode the given value into the implementation's generic type.

        Args:
            values: The values that need to be encoded as the given generic type.

        Returns:
            The encoded value as a sequence of generic types.

        Remarks:
            This method should raise an Exception if `is_fit == False`.
        """
        ...

    def fit_encode(self, values: Sequence[Any]) -> Sequence[_T_out]:
        if self.is_fit:
            return self.encode(values)
        else:
            return self.fit(values).encode(values)

class IdentityEncoder(Encoder[Any]):
    
    @property
    def is_fit(self) -> bool:
        return True

    def fit(self, values: Sequence[Any]) -> 'Encoder':
        return self

    def encode(self, values: Sequence[Any]) -> Sequence[Any]:
        return values

class StringEncoder(Encoder[str]):
    """An Encoder implementation that turns incoming values into string values."""

    @property
    def is_fit(self) -> bool:
        """Indicates if the encoder has been fit.

        Remarks:
            See the base class for more information.
        """

        return True

    def fit(self, values: Sequence[Any]) -> 'StringEncoder':
        """Determine how to encode from given training data.

        Args:
            values: A collection of values to use for determining the encoding.

        Returns:
            An Encoder that has been fit.

        Remarks:
            See the base class for more information.
        """

        return StringEncoder()

    def encode(self, values: Sequence[Any]) -> Sequence[str]:
        """Encode the given values as a sequence of strings.

        Args:
            values: The values that needs to be encoded as a sequence of strings.

        Remarks:
            See the base class for more information.
        """

        return [str(value) for value in values]

class NumericEncoder(Encoder[float]):
    """An Encoder implementation that turns incoming values into float values."""

    @property
    def is_fit(self) -> bool:
        """Indicates if the encoder has been fit.

        Remarks:
            See the base class for more information.
        """

        return True

    def fit(self, values: Sequence[Any]) -> 'NumericEncoder':
        """Determine how to encode from given training data.

        Args:
            values: A collection of values to use for determining the encoding.

        Returns:
            An Encoder that has been fit.

        Remarks:
            See the base class for more information.
        """

        return NumericEncoder()

    def encode(self, values: Sequence[Any]) -> Sequence[float]:
        """Encode the given values as a sequence of floats.

        Args:
            value: The value that needs to be encoded as a sequence of floats.

        Remarks:
            See the base class for more information.
        """

        def float_generator() -> Iterator[float]:
            for value in values:
                try:
                    yield float(value)
                except:
                    yield float('nan')

        return list(float_generator())

class OneHotEncoder(Encoder[Tuple[int,...]]):
    """An Encoder implementation that turns incoming values into a one hot representation."""

    #this in theory could be made more efficient via bit array rather than a byte array
    class MemoryEffecientStorage(bytes):
        def __repr__(self) -> str:
            return str(tuple(self))

        def __eq__(self, x: object) -> bool:
            return x == tuple(self)

        def __hash__(self) -> int:
            return hash(tuple(self))

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
            self._onehots   = dict(keys_and_values)

    @property
    def is_fit(self) -> bool:
        """Indicates if the encoder has been fit.

        Remarks:
            See the base class for more information.
        """

        return self._onehots is not None

    def fit(self, values: Sequence[Any]) -> 'OneHotEncoder':
        """Determine how to encode from given training data.

        Args:
            values: A collection of values to use for determining the encoding.

        Returns:
            An Encoder that has been fit.

        Remarks:
            See the base class for more information.
        """

        return OneHotEncoder(values = values, err_if_unknown = self._err_if_unknown)

    def encode(self, values: Sequence[Any]) -> Sequence[Tuple[int,...]]:
        """Encode the given value as a sequence of 0's and 1's.

        Args:
            value: The value that needs to be encoded as a sequence of 0's and 1's.

        Returns:
            The encoded value as a sequence of 0's and 1's.

        Remarks:
            See the base class for more information.
        """

        if self._onehots is None:
            raise Exception("This encoder must be fit before it can be used.")

        try:
            return [ self._onehots[value] if self._err_if_unknown else self._onehots.get(value, self._default) for value in values ]
        except KeyError as e:
            raise Exception(f"We were unable to find {e} in {self._onehots.keys()}")

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
        self._default        = None

        if values:

            values = sorted(set(values), key=lambda v: values.index(v))

            self._default = float('nan')
            known_levels  = [ i + 1 for i in range(len(values)) ]                

            keys_and_values = zip(values, known_levels)
            self._levels    = dict(keys_and_values)

    @property
    def is_fit(self) -> bool:
        """Indicates if the encoder has been fit.

        Remarks:
            See the base class for more information.
        """

        return self._levels is not None

    def fit(self, values: Sequence[Any]) -> 'FactorEncoder':
        """Determine how to encode from given training data.

        Args:
            values: A collection of values to use for determining the encoding.

        Returns:
            An Encoder that has been fit.

        Remarks:
            See the base class for more information.
        """

        return FactorEncoder(values = values, err_if_unknown = self._err_if_unknown)

    def encode(self, values: Sequence[Any]) -> Sequence[int]:
        """Encode the given values as a sequence factor levels.

        Args:
            values: The values that needs to be encoded as factor levels.

        Remarks:
            See the base class for more information.
        """

        if self._levels is None:
            raise Exception("This encoder must be fit before it can be used.")

        try:
            return [ self._levels[value] if self._err_if_unknown else self._levels.get(value, self._default) for value in values ]
        except KeyError as e:
            raise Exception(f"We were unable to find {e} in {self._levels.keys()}") from None

class CobaJsonEncoder(json.JSONEncoder):
    """A json encoder that allows for potential COBA extensions in the future."""

    def default(self, o: Any) -> Any:
        return super().default(o)

class CobaJsonDecoder(json.JSONDecoder):
    """A json decoder that allows for potential COBA extensions in the future."""

class InteractionTermsEncoder:

    def __init__(self, interactions: Sequence[str]) -> None:

        self._terms = []

        self.times = [0,0]
        self.n = 0

        for term in interactions:
            term  = term.lower()
            x_num = term.count('x')
            a_num = term.count('a')

            if x_num + a_num != len(term):
                raise Exception("Letters other than x and a were passed for parameter interactions. Please remove other letters/characters.")

            self._terms.append((x_num, a_num))

    def encode(self,*, x: Union[list,dict], a: Union[list,dict]):

        is_sparse_type = lambda f: isinstance(f,dict) or isinstance(f,str)
        is_sparse_sequ = lambda f: isinstance(f, collections.Sequence) and any(map(is_sparse_type,f))

        is_sparse = is_sparse_type(x) or is_sparse_sequ(x) or is_sparse_type(a) or is_sparse_sequ(a)
        
        self.n+= 1

        def get_name_values(namespace,features):
            if isinstance(features, dict):
                values = list(features.values())
                names  = [ f"{namespace}{k}" for k in features.keys() ]
            elif isinstance(features,str):
                values = [1]
                names  = [f"{namespace}{features}"]
            elif isinstance(features,Number):
                values = [features]
                names  = [f"{namespace}0"]
            elif isinstance(features, collections.Sequence):
                values = list(features or [1])
                names  = [ f"{namespace}{i}" for i in range(len(values)) ]
            else:
                raise Exception("The features provided to InteractionTermsEncoder are not supported.")
            
            return (names,values) if is_sparse else ([],values)

        def handle_string_values(names, values):
            for i in range(len(values)):
                if isinstance(values[i],str):
                    names[i]  = f"{names[i]}{values[i]}"
                    values[i] = 1
            
            return names,values

        context_names,context_values = get_name_values("x",x)
        action_names,action_values   = get_name_values("a",a)

        context_names,context_values = handle_string_values(context_names,context_values)
        action_names,action_values   = handle_string_values(action_names,action_values)

        max_x_term = max([t[0] for t in self._terms])
        max_a_term = max([t[1] for t in self._terms])

        #.16
        start = time.time()
        x_f_n_by_degree = self._degree_terms(context_values, context_names, max_x_term, is_sparse)
        a_f_n_by_degree = self._degree_terms(action_values , action_names , max_a_term, is_sparse)
        self.times[0] += time.time()-start
        
        #.22
        start = time.time()
        features,names = self._interaction_terms(x_f_n_by_degree,a_f_n_by_degree)
        self.times[1] += time.time()-start

        return features if not is_sparse else list(zip(names,features))

    def _degree_terms(self,values,names,maxd,sparse):

        s_by_degree = dict()
        f_by_degree = dict()
        n_by_degree = dict()

        for degree in range(1,maxd+1):

            if degree == 1:
                n_by_degree[degree] = names
                f_by_degree[degree] = values
                s_by_degree[degree] = [1]*len(values)
            else:
                n_by_degree[degree] = []
                f_by_degree[degree] = []
                s_by_degree[degree] = []

                j  = 0
                for i in range(len(values)):
                    
                    f_by_degree[degree].extend([f*values[i] for f in f_by_degree[degree-1][j:]])
                    n_by_degree[degree].extend([n+ names[i] for n in n_by_degree[degree-1][j:]])
                        
                    s_by_degree[degree].append(len(f_by_degree[degree-1])-j)
                    j = j + s_by_degree[degree-1][i]

        return f_by_degree, n_by_degree

    def _interaction_terms(self, x_f_n_by_degree, a_f_n_by_degree):

        f_interactions = []
        n_interactions = []

        for term in self._terms:

            f_x = x_f_n_by_degree[0].get(term[0], [1])
            f_a = a_f_n_by_degree[0].get(term[1], [1])

            n_x = x_f_n_by_degree[1].get(term[0], [''])
            n_a = a_f_n_by_degree[1].get(term[1], [''])

            f_interactions.extend([p[0]*p[1] for p in product(f_x,f_a)])
            n_interactions.extend([p[0]+p[1] for p in product(n_x,n_a)])

        return f_interactions, n_interactions
