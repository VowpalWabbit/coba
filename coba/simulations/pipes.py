from typing import Sequence, Iterable, Dict, Any

from coba.simulations.core import Simulation, Interaction
from coba.simulations.filters import SimulationFilter

class SimSourceFilters(Simulation):

    def __init__(self, source: Simulation, filters: Sequence[SimulationFilter]):
        
        if isinstance(source, SimSourceFilters):
            self._source  = source._source
            self._filters = list(source._filters) + list(filters)
        else:
            self._source  = source
            self._filters = list(filters)

    @property
    def params(self) -> Dict[str, Any]:
        params = self._safe_params(self._source)

        for filter in self._filters:
            params.update(self._safe_params(filter))

        return params

    @property
    def source_repr(self) -> str:

        source_params = self._safe_params(self._source)

        return str(source_params) if source_params else self._source.__class__.__name__

    def read(self) -> Iterable[Interaction]:
        interactions = self._source.read()

        for filter in self._filters:
            interactions = filter.filter(interactions)

        return interactions

    def _safe_params(self, obj) -> Dict[str, Any]:
        try:
            return obj.params
        except AttributeError:
            return {}
