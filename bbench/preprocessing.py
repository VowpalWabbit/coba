"""The preprocessing module contains generic classes for data preparation.

This module is used primarily for the creation of simulations from data sets.
"""

from abc import ABC, abstractmethod
from typing import Sequence, List, Optional

class Encoder(ABC):
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
    def encode(self, value:str) -> Sequence[float]:
        """Encode the given string as a numeric sequence.

        Args:
            value: The string value that needs to be encoded as a numeric sequence.
        
        Returns:
            The encoded value as a sequence of floats.
        """
        ...

class NumericEncoder(Encoder):
    def __init__(self, auto_fit = True) -> None:
        self._is_fit = auto_fit

    @property
    def is_fit(self) -> bool:
        return self._is_fit

    def fit(self, values:Sequence[str]) -> Encoder:
        if self.is_fit:
            raise Exception("This encoder has already been fit.")

        self._is_fit = True

        return self

    def encode(self, value: str) -> Sequence[float]:
        if not self.is_fit:
            raise Exception("This encoder must be fit before it can be used.")

        return [float(value)]

class OneHotEncoder(Encoder):
    def __init__(self, fit_values: List[str] = [], singular_if_binary: bool = True, error_if_unknown = False) -> None:
        
        self._fit_values = fit_values
        self._singular_if_binary = singular_if_binary
        self._error_if_unknown = error_if_unknown
    
    @property
    def is_fit(self) -> bool:
        return len(self._fit_values) > 0

    def fit(self, values: Sequence[str]) -> Encoder:

        if self.is_fit:
            raise Exception("This encoder has already been fit.")

        #we sort to ensure a fixed order for testing
        self._fit_values = sorted(list(set(values)))

        return self

    def encode(self, value: str) -> Sequence[float]:
        
        if not self.is_fit:
            raise Exception("This encoder must be fit before it can be used.")

        encoding = [ int(value == fit_value) for fit_value in self._fit_values ]

        if all(e == 0 for e in encoding) and self._error_if_unknown:
            raise Exception(f"An unkown value ('{value}') given to the encoder.")

        if len(encoding) == 2 and self._singular_if_binary:
            encoding = [ encoding[0] ]

        return encoding

class InferredEncoder(Encoder):

    def __init__(self) -> None:
        self._fit_encoder: Optional[Encoder] = None

    @property
    def is_fit(self) -> bool:
        return self._fit_encoder is not None

    def fit(self, values: Sequence[str]) -> Encoder:
        if self.is_fit:
            raise Exception("This encoder has already been fit.")
        
        if all(v.isnumeric() for v in values) and len(set(values)) > len(values)/2:
            self._fit_encoder = NumericEncoder(auto_fit=False)

        if not all(v.isnumeric() for v in values) and len(set(values)) < 30:
            self._fit_encoder = OneHotEncoder()

        if(self._fit_encoder is None):
            raise Exception("An appropriate encoder couldn't be inferred.")

        self._fit_encoder.fit(values)

        return self

    def encode(self, value: str) -> Sequence[float]:
        if not self.is_fit:
            raise Exception("This encoder must be fit before it can be used.")

        return self._fit_encoder.encode(value) #type: ignore