
from abc import abstractmethod, ABC
from typing import Iterable, Dict, Any

from coba import pipes

from coba.environments.primitives import Environment, Interaction

class Identity(pipes.Identity):
    
    @property
    def params(self) -> Dict[str, Any]:
        return {}

    def __str__(self) -> str:
        return "{ Identity }"

class EnvironmentFilter(pipes.Filter[Iterable[Interaction],Iterable[Interaction]], ABC):

    @property
    @abstractmethod
    def params(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        """Apply a filter to a Simulation's interactions."""
        ...

    def __str__(self) -> str:
        return str(self.params) if self.params else self.__class__.__name__

class FilteredEnvironment(Environment):

    def __init__(self, source: Environment, *filters: EnvironmentFilter):
        
        if isinstance(source, FilteredEnvironment):
            self._source = source._source
            self._filter = pipes.Pipe.join([source._filter] + list(filters))
        elif len(filters) > 0:
            self._source  = source
            self._filter = pipes.Pipe.join(list(filters))
        else:
            self._source = source
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
