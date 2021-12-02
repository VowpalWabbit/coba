from typing import Iterable, Dict, Any

from coba.pipes import Pipe

from coba.environments.primitives import SimulatedEnvironment, SimulatedInteraction
from coba.environments.filters import EnvironmentFilter

class EnvironmentPipe(SimulatedEnvironment):

    def __init__(self, source: SimulatedEnvironment, *filters: EnvironmentFilter):
        
        if isinstance(source, EnvironmentPipe):
            self._source = source._source
            self._filter = Pipe.join([source._filter] + list(filters))
        else:
            self._source  = source
            self._filter = Pipe.join(list(filters))

    @property
    def params(self) -> Dict[str, Any]:
        params = self._safe_params(self._source)

        for filter in self._filter._filters:
            params.update(self._safe_params(filter))

        return params

    @property
    def source_repr(self) -> str:

        source_params = self._safe_params(self._source)

        return str(source_params) if source_params else self._source.__class__.__name__

    def read(self) -> Iterable[SimulatedInteraction]:
        interactions = self._source.read()

        for filter in self._filter:
            interactions = filter.filter(interactions)

        return interactions

    def _safe_params(self, obj) -> Dict[str, Any]:
        try:
            return obj.params
        except AttributeError:
            return {}

    def __repr__(self) -> str:
        return str([self._source, self._filter])