"""The encodings module contains utility classes for transforming data between encodings.

Remarks:
    This module is used primarily for the creation of simulations from data sets.
"""

import json
import array

from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Iterator, Sequence, Generic, TypeVar, Any, Dict, Tuple

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

    class MemoryEffecientStorage(bytes):
        def __repr__(self) -> str:
            return str(tuple(self))

        def __eq__(self, x: object) -> bool:
            return x == tuple(self)

        def __hash__(self) -> int:
            return hash(tuple(self))

    def __init__(self, fit_values: Sequence[Any] = [], singular_if_binary: bool = False, error_if_unknown = True) -> None:
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
                unknown_onehot = OneHotEncoder.MemoryEffecientStorage([0])
                known_onehots = [OneHotEncoder.MemoryEffecientStorage([1]),OneHotEncoder.MemoryEffecientStorage([0])]
            else:
                unknown_onehot = OneHotEncoder.MemoryEffecientStorage([0] * len(fit_values))
                known_onehots  = [ [0] * len(fit_values) for _ in range(len(fit_values)) ]
                
                for i,k in enumerate(known_onehots):
                    k[i] = 1

            keys_and_values = zip(fit_values, map(OneHotEncoder.MemoryEffecientStorage, known_onehots))
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

        fit_values = sorted(set(values), key=lambda v: values.index(v))

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

        try:
            return [ self._onehots[value] for value in values ]
        except KeyError as e:
            raise Exception(f"We were unable to find {e} in {self._onehots.keys()}")

class FactorEncoder(Encoder[int]):
    """An Encoder implementation that turns incoming values into factor representation."""

    def __init__(self, fit_values: Sequence[Any] = [], error_if_unknown = False) -> None:
        """Instantiate a FactorEncoder.

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

        try:
            return [ self._levels[value] for value in values ]
        except KeyError as e:
            raise Exception(f"We were unable to find {e} in {self._levels.keys()}") from None

class CobaJsonEncoder(json.JSONEncoder):
    """A json encoder that allows for potential COBA extensions in the future."""

    def default(self, obj):
        return super().default(self._intify(obj))

class CobaJsonDecoder(json.JSONDecoder):
    """A json decoder that allows for potential COBA extensions in the future."""

    def __init__(self, *args, **kwargs):
        """Instantiate a CobaJsonDecoder."""
        super().__init__(object_hook=self._object_hook, *args, **kwargs)

    def _object_hook(self, json_obj: Dict[str,Any]) -> Any:
        return json_obj
