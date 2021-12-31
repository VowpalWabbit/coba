
from abc import abstractmethod, ABC
from typing import Iterable, Dict, Any

from coba import pipes

from coba.environments.primitives import Environment, Interaction

class EnvironmentFilter(pipes.Filter[Iterable[Interaction],Iterable[Interaction]], ABC):
    """A filter that can be applied to an Environment."""

    @property
    @abstractmethod
    def params(self) -> Dict[str, Any]:
        """Parameters that describe the filter."""
        ...

    @abstractmethod
    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        """Apply a filter to an Environment's interactions."""
        ...

    def __str__(self) -> str:
        return str(self.params) if self.params else self.__class__.__name__

class Identity(pipes.Identity, EnvironmentFilter):
    """Return whatever interactions are given to the filter."""

    @property
    def params(self) -> Dict[str, Any]:
        return {}

    def __str__(self) -> str:
        return "{ Identity }"

class FilteredEnvironment(Environment):
    """An Environment with a sequence of filters to apply."""

    def __init__(self, environment: Environment, *filters: EnvironmentFilter):
        """Instantiate a FilteredEnvironment.

        Args:
            environment: The environment to apply filters to.
            *filters: The sequence of filters to apply to the environment. 
        """

        if isinstance(environment, FilteredEnvironment):
            self._source = environment._source
            self._filter = pipes.Pipe.join([environment._filter] + list(filters))
        elif len(filters) > 0:
            self._source  = environment
            self._filter = pipes.Pipe.join(list(filters))
        else:
            self._source = environment
            self._filter = Identity()

    @property
    def params(self) -> Dict[str, Any]:
        params = self._safe_params(self._source)
        params.update(self._safe_params(self._filter))
        return params

    def read(self) -> Iterable[Interaction]:
        return self._filter.filter(self._source.read())

    def _safe_params(self, obj) -> Dict[str, Any]:
        try:
            return obj.params
        except AttributeError:
            pass

        try:
            params = {}
            for filter in obj._filters:
                params.update(self._safe_params(filter))
            return params
        except AttributeError:
            pass

        return {}

    def __str__(self) -> str:

        str_source = str(self._source)
        str_filter = str(self._filter)

        return ','.join(filter(None,[str_source,str_filter])).replace('{ Identity }','').replace(',,',',').strip(',')
