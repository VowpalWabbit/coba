import pickle

from zipfile import ZipFile
from itertools import islice, repeat
from typing import Sequence, Mapping, Iterable, Any

from coba.backports import version
from coba.pipes import Source, Sink, Filter
from coba.environments.primitives import Environment, Interaction

class ObjectsToZipMember(Sink[Iterable[Sequence[object]]]):

    def __init__(self, zip:str):
        self._zip    = zip

    def write(self, envs: Iterable[Sequence[object]]) -> None:
        with ZipFile(self._zip,mode='a') as zip:
            for i,env in enumerate(envs):
                with zip.open(str(i),mode='w') as f:
                   f.writelines(map(pickle.dumps,env))

class ZipMemberToObjects(Source[Iterable[object]]):

    def __init__(self, zip:str, member:str):
        self._zip    = zip
        self._member = member

    def read(self) -> Iterable[object]:
        try:
            with ZipFile(self._zip) as zip:
                with zip.open(self._member) as f:
                    yield from map(pickle.load, repeat(f))
        except EOFError:
            pass

class EnvironmentToObjects(Filter[Environment, Iterable]):
    def filter(self, env: Environment) -> Sequence[object]:
        return [{"version":1,"coba_version":version("coba")},env.params] + list(env.read())

class EnvironmentFromObjects(Environment):
    def __init__(self, source: Source[Iterable[object]]) -> None:
        self._source = source

    @property
    def params(self) -> Mapping[str,Any]:
        return next(islice(self._source.read(),1,None))

    def read(self) -> Iterable[Interaction]:
        yield from islice(self._source.read(),2,None)
