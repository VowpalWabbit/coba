"""The preprocessing module contains generic classes for data preparation.

Remarks:
    This module is used primarily for the creation of simulations from data sets.

TODO InferredEncoder needs a lot of love. Currently its inferences are quite bad.
"""

import json

from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Iterator, Optional, Sequence, Generic, TypeVar, Any, Hashable, Union, Dict, Tuple

from coba.data.encoders import Encoder, StringEncoder

T_ignore  = TypeVar('T_ignore' ) 
T_label   = TypeVar('T_label'  ) 
T_encoder = TypeVar('T_encoder')

class Metadata(Generic[T_ignore, T_label, T_encoder]):
    """A storage class for Optional meta information describing features."""

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

        return Metadata(ignore, label, encoder)

class PartMeta(Metadata[Optional[bool], Optional[bool], Optional[Encoder]]):
    """A storage class for partial meta information describing features."""

    @staticmethod
    def from_json(json_val:Union[str, Dict[str,Any]]) -> 'PartMeta':
        """Construct a PartMeta object from JSON.

        Args:
            json_val: Either a json string or the decoded json object.

        Returns:
            The PartMeta representation of the given JSON string or object.
        """

        config = json.loads(json_val) if isinstance(json_val,str) else json_val

        ignore  = None if "ignore"   not in config else config["ignore"]
        label   = None if "label"    not in config else config["label" ]
        encoder = None if "encoding" not in config else Encoder.from_json(config["encoding"])

        return PartMeta(ignore,label,encoder)

    def __init__(self, ignore: bool = None, label: bool = None, encoder: Encoder = None):
        """Instantiate FullMeta.

        Args:
            ignore: Indicates if the feature should be ignored.
            label: Indicates if the feature should be regarded as a supervised label
            encoder: The Encoder that should be used when ingesting features.
        """

        super().__init__(ignore, label, encoder)

class FullMeta(Metadata[bool, bool, Encoder]):
    """A storage class for full meta information describing features."""

    @staticmethod
    def from_json(json_val:Union[str, Dict[str,Any]]) -> 'FullMeta':
        """Construct a FullMeta object from JSON.

        Args:
            json_val: Either a json string or the decoded json object.

        Returns:
            The FullMeta representation of the given JSON string or object.
        """

        part_meta = PartMeta.from_json(json_val)

        if part_meta.ignore is not None and part_meta.label is not None and part_meta.encoder is not None:
            return FullMeta(part_meta.ignore, part_meta.label, part_meta.encoder)
        
        raise Exception("FullMeta JSON must define 'ignore', 'label' and 'encoder'")

    def __init__(self, ignore: bool = False, label: bool = False, encoder: Encoder = StringEncoder()):
        """Instantiate FullMeta.

        Args:
            ignore: Indicates if the feature should be ignored.
            label: Indicates if the feature should be regarded as a supervised label
            encoder: The Encoder that should be used when ingesting features.
        """

        super().__init__(ignore, label, encoder)