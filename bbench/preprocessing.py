"""The preprocessing module contains generic classes for data preparation.

Remarks:
    This module is used primarily for the creation of simulations from data sets.
"""

from abc import ABC, abstractmethod
from typing import Sequence, List, Generic, TypeVar, Any, Optional, Hashable

T_out = TypeVar('T_out', bound=Hashable, covariant=True) 

class Encoder(ABC, Generic[T_out]):
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
    def encode(self, value: Any) -> Sequence[T_out]:
        """Encode the given value into the implementation's generic type.

        Args:
            value: The value that needs to be encoded as the given generic type.

        Returns:
            The encoded value as a sequence of generic types.

        Remarks:
            This method should raise an Exception if `is_fit == False`.
        """
        ...

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

    def encode(self, value: Any) -> Sequence[str]:
        """Encode the given value as a sequence of strings.

        Args:
            value: The value that needs to be encoded as a sequence of strings.

        Returns:
            The encoded value as a sequence of strings.

        Remarks:
            See the base class for more information.
        """

        if not self.is_fit:
            raise Exception("This encoder must be fit before it can be used.")

        return [str(value)]

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

    def encode(self, value: Any) -> Sequence[float]:
        """Encode the given value as a sequence of floats.

        Args:
            value: The value that needs to be encoded as a sequence of floats.

        Returns:
            The encoded value as a sequence of floats.

        Remarks:
            See the base class for more information.
        """

        if not self.is_fit:
            raise Exception("This encoder must be fit before it can be used.")

        return [float(value)]

class OneHotEncoder(Encoder[int]):
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

    @property
    def is_fit(self) -> bool:
        """Indicates if the encoder has been fit.

        Remarks:
            See the base class for more information.
        """

        return len(self._fit_values) > 0

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

        return OneHotEncoder(fit_values = sorted(set(values)), singular_if_binary=self._singular_if_binary, error_if_unknown = self._error_if_unknown)

    def encode(self, value: Any) -> Sequence[int]:
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

        if self._error_if_unknown and value not in self._fit_values:
            raise Exception(f"An unkown value ('{value}') given to the encoder.")

        encoding: Sequence[int] = [ 1 if value == fit_value else 0 for fit_value in self._fit_values]

        if self._singular_if_binary and len(encoding) == 2:
            encoding = [ encoding[0] ]

        return encoding

class InferredEncoder(Encoder[Any]):
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

    def encode(self, value: Any) -> Sequence[Any]:
        """This implementation never encodes since `fit` returns specific implementations."""

        raise Exception("This encoder must be fit before it can be used.")

class PartialMeta:
    """A storage class for Optional meta information describing features."""

    def __init__(self,
        ignore  : Optional[bool] = None,
        label   : Optional[bool] = None,
        encoder : Optional[Encoder] = None) -> None:
        """Instantiate PartialMeta.

        Args:
            ignore: Indicates if the feature should be ignored.
            label: Indicates if the feature should be regarded as a supervised label
            encoder: The Encoder that should be used when ingesting features.
        """

        self.ignore  = ignore
        self.label   = label
        self.encoder = encoder

class DefiniteMeta:
    """A storage class for required meta information describing features."""

    def __init__(self, ignore: bool = False, label: bool = False, encoder: Encoder = InferredEncoder()) -> None:
        """Instantiate DefiniteMeta.

        Args:
            ignore: Indicates if the feature should be ignored.
            label: Indicates if the feature should be regarded as a supervised label
            encoder: The Encoder that should be used when ingesting features.
        """

        self.ignore  = ignore
        self.label   = label
        self.encoder = encoder

    def clone(self) -> 'DefiniteMeta':
        """Clone the current DefiniteMeta. 

        Returns:
            Returns a new DefiniteMeta with identical properties.
        """

        return DefiniteMeta(self.ignore, self.label, self.encoder)

    def apply(self, overrides: PartialMeta) -> 'DefiniteMeta':
        """Apply by overriding DefiniteMeta properties with not none PartialMeta properties.

        Args:
            overrides: A PartialMeta that should be used to override properties in the DefiniteMeta.

        Returns:
            Returns a new DefiniteMeta with properties overriden by not None PartialMeta properties.
        """

        new = self.clone()

        new.ignore  = overrides.ignore  if overrides.ignore  is not None else self.ignore
        new.label   = overrides.label   if overrides.label   is not None else self.label
        new.encoder = overrides.encoder if overrides.encoder is not None else self.encoder

        return new