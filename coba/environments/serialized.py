import pickle

from itertools import islice
from typing import Mapping, Iterable, Any

from coba.pipes import Source
from coba.environments.primitives import Environment, Interaction

class EnvironmentToBytes(Source[Iterable[bytes]]):
    def __init__(self, environment: Environment) -> None:
        self._env = environment

    def read(self) -> Iterable[bytes]:
        yield pickle.dumps({"version":1})
        yield pickle.dumps(self._env.params)
        yield from map(pickle.dumps,self._env.read())

class EnvironmentFromBytes(Environment):
    def __init__(self, source: Source[Iterable[bytes]]) -> None:
        self._bytes = source

    @property
    def params(self) -> Mapping[str,Any]:
        return pickle.loads(next(islice(self._bytes.read(),1,None)))

    def read(self) -> Iterable[Interaction]:
        yield from map(pickle.loads,islice(self._bytes.read(),2,None))
