"""The preprocessing module contains generic classes for data preparation.

This module is used primarily for the creation of simulations from data sets.

Todo:
    *Finish adding docstrings.
"""

from abc import ABC, abstractmethod
from typing import Sequence, List, Generic, TypeVar, Any, Optional, Hashable

T_out = TypeVar('T_out', bound=Hashable, covariant=True) 

class Encoder(ABC, Generic[T_out]):
    """The interface for encoder implementations."""

    @property
    @abstractmethod
    def is_fit(self) -> bool:
        """Indicates if the encoder has been fit."""
        ...

    @abstractmethod
    def fit(self, values:Sequence[str]) -> 'Encoder':
        """Determine how to encode from given training data.
        
        Args:
            values: A collection of values to use for determining the encoding. 
        """
        ...

    @abstractmethod
    def encode(self, value:str) -> Sequence[T_out]:
        """Encode the given string as a numeric sequence.

        Args:
            value: The string value that needs to be encoded as a numeric sequence.
        
        Returns:
            The encoded value as a sequence of floats.
        """
        ...

class StringEncoder(Encoder[str]):
    def __init__(self, is_fit = True) -> None:
        self._is_fit = is_fit

    @property
    def is_fit(self) -> bool:
        return self._is_fit

    def fit(self, values:Sequence[str]) -> 'StringEncoder':
        if self.is_fit:
            raise Exception("This encoder has already been fit.")

        return StringEncoder(is_fit=True)

    def encode(self, value: str) -> Sequence[str]:
        if not self.is_fit:
            raise Exception("This encoder must be fit before it can be used.")

        return [value]

class NumericEncoder(Encoder[float]):
    def __init__(self, is_fit = True) -> None:
        self._is_fit = is_fit

    @property
    def is_fit(self) -> bool:
        return self._is_fit

    def fit(self, values:Sequence[str]) -> 'NumericEncoder':
        if self.is_fit:
            raise Exception("This encoder has already been fit.")

        return NumericEncoder(is_fit=True)

    def encode(self, value: str) -> Sequence[float]:
        if not self.is_fit:
            raise Exception("This encoder must be fit before it can be used.")

        return [float(value)]

class OneHotEncoder(Encoder[int]):
    def __init__(self, fit_values: List[str] = [], singular_if_binary: bool = True, error_if_unknown = False) -> None:
        
        self._fit_values         = fit_values
        self._singular_if_binary = singular_if_binary
        self._error_if_unknown   = error_if_unknown
    
    @property
    def is_fit(self) -> bool:
        return len(self._fit_values) > 0

    def fit(self, values: Sequence[str]) -> 'OneHotEncoder':

        if self.is_fit:
            raise Exception("This encoder has already been fit.")

        return OneHotEncoder(fit_values = sorted(set(values)), singular_if_binary=self._singular_if_binary, error_if_unknown = self._error_if_unknown)

    def encode(self, value: str) -> Sequence[int]:
        
        if not self.is_fit:
            raise Exception("This encoder must be fit before it can be used.")

        if self._error_if_unknown and value not in self._fit_values:
            raise Exception(f"An unkown value ('{value}') given to the encoder.")

        encoding: Sequence[int] = [ 1 if value == fit_value else 0 for fit_value in self._fit_values]

        if self._singular_if_binary and len(encoding) == 2:
            encoding = [ encoding[0] ]

        return encoding

class InferredEncoder(Encoder[Any]):

    @property
    def is_fit(self) -> bool:
        return False

    def fit(self, values: Sequence[str]) -> Encoder:
        if all(v.isnumeric() for v in values) and len(set(values)) > len(values)/2:
            return NumericEncoder(is_fit=False).fit(values)

        if not all(v.isnumeric() for v in values) and len(set(values)) < 30:
            return OneHotEncoder().fit(values)

        return StringEncoder(is_fit=False).fit(values)

    def encode(self, value: str) -> Sequence[Any]:
        raise Exception("This encoder must be fit before it can be used.")

class PartialMeta:

    def __init__(self,
        ignore  : Optional[bool] = None,
        label   : Optional[bool] = None,
        encoder : Optional[Encoder] = None) -> None:

        self.ignore  = ignore
        self.label   = label
        self.encoder = encoder

class DefiniteMeta:

    def __init__(self, ignore: bool = False, label: bool = False, encoder: Encoder = InferredEncoder()) -> None:
        
        self.ignore : bool    = ignore
        self.label  : bool    = label
        self.encoder: Encoder = encoder

    def clone(self) -> 'DefiniteMeta':
        return DefiniteMeta(self.ignore, self.label, self.encoder)

    def apply(self, overrides: PartialMeta) -> None:
        
        self.ignore  = overrides.ignore  if overrides.ignore  is not None else self.ignore
        self.label   = overrides.label   if overrides.label   is not None else self.label
        self.encoder = overrides.encoder if overrides.encoder is not None else self.encoder