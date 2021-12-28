from typing import Iterable, Dict, Any

from coba.pipes import Pipe

from coba.environments.primitives import SimulatedEnvironment, SimulatedInteraction
from coba.environments.filters import EnvironmentFilter, Identity

class EnvironmentPipe(SimulatedEnvironment):

    def __init__(self, source: SimulatedEnvironment, *filters: EnvironmentFilter):
        
        if isinstance(source, EnvironmentPipe):
            self._source = source._source
            self._filter = Pipe.join([source._filter] + list(filters))
        elif len(filters) > 0:
            self._source  = source
            self._filter = Pipe.join(list(filters))
        else:
            self._source = source
            self._filter = Identity()

    @property
    def params(self) -> Dict[str, Any]:
        params = self._safe_params(self._source)
        params.update(self._safe_params(self._filter))
        return params

    def read(self) -> Iterable[SimulatedInteraction]:
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
