"""The preprocessing module contains generic classes for data preparation.

Remarks:
    This module is used primarily for the creation of simulations from data sets.
"""

import json

from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Iterator, Sequence, Generic, TypeVar, Any, Optional, Hashable, Union, Dict, Tuple

T_out = TypeVar('T_out', bound=Hashable, covariant=True) 

class Encoder(ABC, Generic[T_out]):
    """The interface for encoder implementations.

    Remarks:
        While it can't be enforced by the interface, the assumption is that all Encoder
        implementations are immutable. This means that the `fit` method should always
        return a new Encoder.
    """

    @staticmethod
    def from_json(json_val:str) -> 'Encoder':
        """Construct an Encoder object from JSON.
        
        Args:
            json_val: Either a json string or the decoded json object.
        
        Returns:
            The Encoder representation of the given JSON string or object.
        """

        if json_val == "numeric" : return NumericEncoder()
        if json_val == "onehot"  : return OneHotEncoder()
        if json_val == "string"  : return StringEncoder()
        if json_val == "inferred": return InferredEncoder()

        raise Exception('We were unable to determine the appropriate encoder from json')

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
    def encode(self, values: Sequence[Any]) -> Sequence[T_out]:
        """Encode the given value into the implementation's generic type.

        Args:
            values: The values that need to be encoded as the given generic type.

        Returns:
            The encoded value as a sequence of generic types.

        Remarks:
            This method should raise an Exception if `is_fit == False`.
        """
        ...

    def fit_encode(self, values: Sequence[Any]) -> Sequence[T_out]:
        if self.is_fit:
            return self.encode(values)
        else: 
            return self.fit(values).encode(values)

class StringEncoder(Encoder[str]):
    """An Encoder implementation that turns incoming values into string values."""

    def __init__(self, is_fit = True) -> None:
        """Instantiate a StringEncoder.

        Args:
            is_fit: Indicates if the encoder is instantiated as already being fit.
        """
        self._is_fit = is_fit

    @property
    def is_fit(self) -> bool:
        """Indicates if the encoder has been fit.

        Remarks:
            See the base class for more information.
        """

        return self._is_fit

    def fit(self, values: Sequence[Any]) -> 'StringEncoder':
        """Determine how to encode from given training data.

        Args:
            values: A collection of values to use for determining the encoding.

        Returns:
            An Encoder that has been fit.

        Remarks:
            See the base class for more information.
        """

        if self.is_fit:
            raise Exception("This encoder has already been fit.")

        return StringEncoder(is_fit=True)

    def encode(self, values: Sequence[Any]) -> Sequence[str]:
        """Encode the given values as a sequence of strings.

        Args:
            values: The values that needs to be encoded as a sequence of strings.

        Remarks:
            See the base class for more information.
        """

        if not self.is_fit:
            raise Exception("This encoder must be fit before it can be used.")

        return [str(value) for value in values]

class NumericEncoder(Encoder[float]):
    """An Encoder implementation that turns incoming values into float values."""

    def __init__(self, is_fit = True) -> None:
        """Instantiate a NumericEncoder.

        Args:
            is_fit: Indicates if the encoder is instantiated as already being fit.
        """

        self._is_fit = is_fit

    @property
    def is_fit(self) -> bool:
        """Indicates if the encoder has been fit.
        
        Remarks:
            See the base class for more information.
        """

        return self._is_fit

    def fit(self, values: Sequence[Any]) -> 'NumericEncoder':
        """Determine how to encode from given training data.

        Args:
            values: A collection of values to use for determining the encoding.

        Returns:
            An Encoder that has been fit.

        Remarks:
            See the base class for more information.
        """

        if self.is_fit:
            raise Exception("This encoder has already been fit.")

        return NumericEncoder(is_fit=True)

    def encode(self, values: Sequence[Any]) -> Sequence[float]:
        """Encode the given values as a sequence of floats.

        Args:
            value: The value that needs to be encoded as a sequence of floats.

        Remarks:
            See the base class for more information.
        """

        if not self.is_fit:
            raise Exception("This encoder must be fit before it can be used.")

        #The fastnumbers package seems like it could potentially provide around a 20% speed increase.        
        #if isinstance(values[0],str):
        #    return [float(value) if cast(str,value).isnumeric() else float('nan') for value in values]
        #else:

        def float_generator() -> Iterator[float]:
            for value in values:
                try:
                    yield float(value)
                except:
                    yield float('nan')

        return list(float_generator())

class OneHotEncoder(Encoder[Tuple[int,...]]):
    """An Encoder implementation that turns incoming values into a one hot representation."""

    def __init__(self, fit_values: Sequence[Any] = [], singular_if_binary: bool = True, error_if_unknown = False) -> None:
        """Instantiate a OneHotEncoder.

        Args:
            fit_values: Provide the universe of values for encoding and set `is_fit==True`.
            singular_if_binary: Indicate if a universe with two values should be encoded as [1] or [0]
                rather than the more standard [1 0] and [0 1].
            error_if_unknown: Indicates if an error is thrown when an unknown value is passed to `encode`
                or if a sequence of all 0's with a length of the universe is returned.
        """
        self._fit_values         = fit_values
        self._singular_if_binary = singular_if_binary
        self._error_if_unknown   = error_if_unknown
        self._is_fit             = len(fit_values) > 0

        if fit_values:

            if len(fit_values) == 2 and singular_if_binary:
                unknown_onehot = tuple([0])
                known_onehots = [[1],[0]]
            else:
                unknown_onehot = tuple([0] * len(fit_values))
                known_onehots  = [ [0] * len(fit_values) for _ in range(len(fit_values)) ]
                
                for i,k in enumerate(known_onehots):
                    k[i] = 1

            keys_and_values = zip(fit_values, map(lambda k_oh: tuple(k_oh), known_onehots))
            default_factory = lambda:unknown_onehot

            self._onehots: Dict[Any,Tuple[int,...]]

            if self._error_if_unknown:
                self._onehots = dict(keys_and_values)
            else:
                self._onehots = defaultdict(default_factory, keys_and_values)

    @property
    def is_fit(self) -> bool:
        """Indicates if the encoder has been fit.

        Remarks:
            See the base class for more information.
        """

        return self._is_fit

    def fit(self, values: Sequence[Any]) -> 'OneHotEncoder':
        """Determine how to encode from given training data.

        Args:
            values: A collection of values to use for determining the encoding.

        Returns:
            An Encoder that has been fit.

        Remarks:
            See the base class for more information.
        """

        if self.is_fit:
            raise Exception("This encoder has already been fit.")

        fit_values = sorted(set(values))

        return OneHotEncoder(
            fit_values         = fit_values, 
            singular_if_binary = self._singular_if_binary, 
            error_if_unknown   = self._error_if_unknown)

    def encode(self, values: Sequence[Any]) -> Sequence[Tuple[int,...]]:
        """Encode the given value as a sequence of 0's and 1's.

        Args:
            value: The value that needs to be encoded as a sequence of 0's and 1's.

        Returns:
            The encoded value as a sequence of 0's and 1's.

        Remarks:
            See the base class for more information.
        """

        if not self.is_fit:
            raise Exception("This encoder must be fit before it can be used.")

        return [ self._onehots[value] for value in values ]

class FactorEncoder(Encoder[int]):
    """An Encoder implementation that turns incoming values into factor representation."""

    def __init__(self, fit_values: Sequence[Any] = [], error_if_unknown = False) -> None:
        """Instantiate a OneHotEncoder.

        Args:
            fit_values: Provide the universe of values for encoding and set `is_fit==True`.
            error_if_unknown: Indicates if an error is thrown when an unknown value is passed to `encode`
                or if a sequence of all 0's with a length of the universe is returned.
        """
        self._fit_values       = fit_values
        self._error_if_unknown = error_if_unknown
        self._is_fit           = len(fit_values) > 0

        if fit_values:
            unknown_level = 0
            known_levels  = [ i + 1 for i in range(len(fit_values)) ]                

            keys_and_values = zip(fit_values, known_levels)
            default_factory = lambda: unknown_level

            self._levels: Dict[Any,int]

            if self._error_if_unknown:
                self._levels = dict(keys_and_values)
            else:
                self._levels = defaultdict(default_factory, keys_and_values)

    @property
    def is_fit(self) -> bool:
        """Indicates if the encoder has been fit.

        Remarks:
            See the base class for more information.
        """

        return self._is_fit

    def fit(self, values: Sequence[Any]) -> 'FactorEncoder':
        """Determine how to encode from given training data.

        Args:
            values: A collection of values to use for determining the encoding.

        Returns:
            An Encoder that has been fit.

        Remarks:
            See the base class for more information.
        """

        if self.is_fit:
            raise Exception("This encoder has already been fit.")

        fit_values = sorted(set(values))

        return FactorEncoder(
            fit_values         = fit_values, 
            error_if_unknown   = self._error_if_unknown)

    def encode(self, values: Sequence[Any]) -> Sequence[int]:
        """Encode the given values as a sequence factor levels.

        Args:
            values: The values that needs to be encoded as factor levels.

        Remarks:
            See the base class for more information.
        """

        if not self.is_fit:
            raise Exception("This encoder must be fit before it can be used.")

        return [ self._levels[value] for value in values ]

class InferredEncoder(Encoder[Hashable]):
    """An Encoder implementation that looks at its given `fit` values and infers the best Encoder."""

    @property
    def is_fit(self) -> bool:
        """Indicates if the encoder has been fit.

        Remarks:
            See the base class for more information.
        """

        return False

    def fit(self, values: Sequence[Any]) -> Encoder:
        """Determine how to encode from given training data.

        Args:
            values: A collection of values to use for determining the encoding.

        Returns:
            An Encoder that has been fit.

        Remarks:
            See the base class for more information.
        """

        if all(v.isnumeric() for v in values) and len(set(values)) > len(values)/2:
            return NumericEncoder(is_fit=False).fit(values)

        if len(set(values)) < 200:
            return OneHotEncoder().fit(values)

        return StringEncoder(is_fit=False).fit(values)

    def encode(self, value: Any) -> Sequence[Hashable]:
        """This implementation never encodes since `fit` returns specific implementations."""

        raise Exception("This encoder must be fit before it can be used.")

T_ignore  = TypeVar('T_ignore' , bound = Optional[bool]   , covariant=True) 
T_label   = TypeVar('T_label'  , bound = Optional[bool]   , covariant=True) 
T_encoder = TypeVar('T_encoder', bound = Optional[Encoder], covariant=True)

class Metadata(Generic[T_ignore, T_label, T_encoder]):
    """A storage class for Optional meta information describing features."""

    @staticmethod
    def from_json(json_val:Union[str, Dict[str,Any]]) -> 'Metadata':
        """Construct a Metadata object from JSON.

        Args:
            json_val: Either a json string or the decoded json object.

        Returns:
            The Metadata representation of the given JSON string or object.
        """

        config = json.loads(json_val) if isinstance(json_val,str) else json_val

        ignore  = None if "ignore"   not in config else config["ignore"]
        label   = None if "label"    not in config else config["label" ]
        encoder = None if "encoding" not in config else Encoder.from_json(config["encoding"])

        return Metadata(ignore,label,encoder)

    @staticmethod
    def default() -> 'Metadata[bool,bool,Encoder]':
        return Metadata(False,False,InferredEncoder())

    @property
    def ignore(self) -> T_ignore:
        return self._ignore

    @property
    def label(self) -> T_label:
        return self._label

    @property
    def encoder(self) -> T_encoder:
        return self._encoder

    def __init__(self, ignore: T_ignore, label: T_label, encoder: T_encoder) -> None:
        ...
        """Instantiate PartialMeta.

        Args:
            ignore: Indicates if the feature should be ignored.
            label: Indicates if the feature should be regarded as a supervised label
            encoder: The Encoder that should be used when ingesting features.
        """
        self._ignore  = ignore
        self._label   = label
        self._encoder = encoder

    def clone(self) -> 'Metadata[T_ignore, T_label, T_encoder]':
        """Clone the current DefiniteMeta. 

        Returns:
            Returns a new DefiniteMeta with identical properties.
        """

        return Metadata(self.ignore, self.label, self.encoder)

    def override(self, override: 'Metadata') -> 'Metadata[T_ignore, T_label, T_encoder]':
        """Apply by overriding DefiniteMeta properties with not none PartialMeta properties.

        Args:
            override: A Metadata that should be used to override properties in the DefiniteMeta.

        Returns:
            Returns a new DefiniteMeta with properties overriden by not None PartialMeta properties.
        """

        ignore  = override.ignore  if override.ignore  is not None else self.ignore
        label   = override.label   if override.label   is not None else self.label
        encoder = override.encoder if override.encoder is not None else self.encoder

        return Metadata[T_ignore, T_label, T_encoder](ignore, label, encoder)