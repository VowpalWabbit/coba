from numbers import Number
from abc import abstractmethod, ABC
from typing import Any, Union, Iterable, Dict

from coba.utilities import HashableDict
from coba.pipes import Source

Action  = Union[str, Number, tuple, HashableDict]
Context = Union[None, str, Number, tuple, HashableDict]

class Interaction:
    def __init__(self, context: Context) -> None:
        self._context = context

    @property
    def context(self) -> Context:
        """The context in which an action was taken."""
        return self._context

    def _hashable(self, feats):

        if isinstance(feats, dict):
            return HashableDict(feats)

        if isinstance(feats,list):
            return tuple(feats)

        return feats

class Environment(Source[Iterable[Interaction]], ABC):

    @property
    @abstractmethod
    def params(self) -> Dict[str,Any]:
        """Paramaters describing the simulation.

        Remarks:
            These will become columns in the environments data of an experiment result.
        """
        ...
    
    @abstractmethod
    def read(self) -> Iterable[Interaction]:
        """The sequence of interactions in a simulation.

        Remarks:
            This function should always be "re-iterable".
        """
        ...

    def __str__(self) -> str:
        return str(self.params) if self.params else self.__class__.__name__
