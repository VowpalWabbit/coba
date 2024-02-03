from typing import Iterable, Mapping, Any

from coba.exceptions import CobaException
from coba.primitives import Source, Environment, Interaction

class ResultEnvironment(Environment):

    def __init__(self, int_source:Source[str], env_params:dict, lrn_params:dict, val_params:dict) -> None:
        self._int_source = int_source
        self._env_params = env_params
        self._lrn_params = lrn_params
        self._val_params = val_params

    @property
    def params(self) -> Mapping[str,Any]:
        params = {}
        if self._env_params: params.update(self._env_params)
        if self._lrn_params: params.update(self._lrn_params)
        if self._val_params: params.update(self._val_params)
        return params

    def read(self) -> Iterable[Interaction]:
        interactions = self._int_source.read()

        is_all_data = {'actions','rewards'}.issubset(interactions.keys())
        is_log_data = {'action' ,'reward' }.issubset(interactions.keys())

        if not (is_all_data or is_log_data):
            raise CobaException(
                "It is not possible to create a ResultEnvironment if the Result does "
                "not contain at least (`actions`,`rewards`) or (`action`,`reward`).")

        for i in range(len(next(iter(interactions.values())))):
            yield {k:interactions[k][i] for k in interactions.keys()}
