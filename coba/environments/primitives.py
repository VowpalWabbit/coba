import collections.abc

from numbers import Number
from abc import abstractmethod, ABC
from typing import Any, Union, Iterable, Dict, Callable

from coba.utilities import HashableDict
from coba.pipes import Source, Filter, DiskIO, ListIO

Action  = Union[str, Number, tuple, HashableDict]
Context = Union[None, str, Number, tuple, HashableDict]

class Interaction:
    """An individual interaction that occurs in an Environment."""
    
    def __init__(self, context: Context) -> None:
        """Instantiate an Interaction.
        
        Args:
            context: The context in which the interaction occured.
        """
        self._context = context

    @property
    def context(self) -> Context:
        """The context in which the interaction occured."""
        return self._hashable(self._context)

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

class ReaderEnvironment(Environment):

    def __init__(self,
        source: Union[str,Source[Iterable[str]]],
        reader: Filter[Iterable[str], Iterable[Any]],
        inters: Filter[Iterable[Any], Iterable[Interaction]]) -> None:

        self._source = DiskIO(source) if isinstance(source,str) else source
        self._reader = reader
        self._inters = inters

    @property
    def params(self) -> Dict[str, Any]:
        if isinstance(self._source,DiskIO):
            return {"source": str(self._source._filename) }
        elif isinstance(self._source, ListIO):
            return {"source": 'memory' }
        else:
            return {"source": self._source.__class__.__name__}

    def read(self) -> Iterable[Interaction]:
        return self._inters.filter(self._reader.filter(self._source.read()))
