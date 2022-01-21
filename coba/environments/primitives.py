import collections.abc

from numbers import Number
from abc import abstractmethod, ABC
from typing import Any, Union, Iterable, Dict

from coba.utilities import HashableDict
from coba.pipes import Source

Action  = Union[str, Number, tuple, HashableDict]
Context = Union[None, str, Number, tuple, HashableDict]

class Interaction:
    """An individual interaction that occurs in an Environment."""

    def __init__(self, context: Context) -> None:
        """Instantiate an Interaction.

        Args:
            context: The context in which the interaction occured.
        """

        self._raw_context = context
        self._hash_context = None

    @property
    def context(self) -> Context:
        """The context in which the interaction occured."""
        if self._hash_context is None:
            self._hash_context = self._hashable(self._raw_context)

        return self._hash_context

    def _hashable(self, feats):

        if isinstance(feats, collections.abc.Mapping):
            return HashableDict(feats)

        if isinstance(feats,collections.abc.Sequence) and not isinstance(feats,str):
            return tuple(feats)

        return feats

class Environment(Source[Iterable[Interaction]], ABC):
    """An Environment that produces Contextual Bandit data"""

    @property
    @abstractmethod
    def params(self) -> Dict[str,Any]:
        """Paramaters describing the simulation.

        Remarks:
            These will become columns in the environment table of an experiment result.
        """
        ...

    @abstractmethod
    def read(self) -> Iterable[Interaction]:
        """The sequence of interactions in the simulation.

        Remarks:
            This function should always be "re-iterable".
        """
        ...

    def __str__(self) -> str:
        return str(self.params) if self.params else self.__class__.__name__
