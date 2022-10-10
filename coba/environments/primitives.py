import collections.abc

from numbers import Number
from abc import abstractmethod, ABC
from typing import Any, Union, Iterable, Dict

from coba.utilities import HashableDict
from coba.pipes import Source, SourceFilters

Action  = Union[str, Number, tuple, HashableDict]
Context = Union[None, str, Number, tuple, HashableDict]

class Interaction:
    """An individual interaction that occurs in an Environment."""

    def __init__(self, context: Context, **kwargs) -> None:
        """Instantiate an Interaction.

        Args:
            context: The context in which the interaction occured.
        """
        self._context = context
        self._kwargs  = kwargs

    @property
    def context(self) -> Context:
        """The context in which the interaction occured."""
        if not isinstance(self._context, collections.abc.Hashable):
            self._context = self._make_hashable(self._context)
        return self._context

    @property
    def kwargs(self) -> Dict[str,Any]:
        """Additional information associatd with the Interaction."""
        return self._kwargs

    def _make_hashable(self, feats):

        try:
            return feats.to_builtin()
        except Exception:
            if isinstance(feats, collections.abc.Sequence):
                return tuple(feats)

            if isinstance(feats, collections.abc.Mapping):
                return HashableDict(feats)

            return feats

class Environment(Source[Iterable[Interaction]], ABC):
    """An Environment that produces Contextual Bandit data"""

    @property
    def params(self) -> Dict[str,Any]: # pragma: no cover
        """Paramaters describing the simulation.

        Remarks:
            These will become columns in the environments table of experiment results.
        """
        return {}

    @abstractmethod
    def read(self) -> Iterable[Interaction]:
        """The sequence of interactions in the simulation.

        Remarks:
            This function should always be "re-iterable".
        """
        ...

    def __str__(self) -> str:
        return str(self.params) if self.params else self.__class__.__name__

class SafeEnvironment(Environment):
    """A wrapper for environment-likes that guarantees interface consistency."""

    def __init__(self, environment: Environment) -> None:
        """Instantiate a SafeEnvironment.

        Args:
            environment: The environment we wish to make sure has the expected interface
        """

        self._environment = environment if not isinstance(environment, SafeEnvironment) else environment._environment

    @property
    def params(self) -> Dict[str, Any]:
        try:
            params = self._environment.params
        except AttributeError:
            params = {}

        if "type" not in params:

            if isinstance(self._environment, SourceFilters):
                params["type"] = self._environment._source.__class__.__name__
            else:
                params["type"] = self._environment.__class__.__name__

        return params

    def read(self) -> Iterable[Interaction]:
        return self._environment.read()

    def __str__(self) -> str:
        params = dict(self.params)
        tipe   = params.pop("type")

        if len(params) > 0:
            return f"{tipe}({','.join(f'{k}={v}' for k,v in params.items())})"
        else:
            return tipe
