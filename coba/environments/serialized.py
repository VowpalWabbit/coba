import pickle

from zipfile import ZipFile
from itertools import islice, repeat
from typing import Mapping, Iterable, Any

from coba.pipes import Source
from coba.environments.primitives import Environment, Interaction

class ObjectsToZipMember:

    def __init__(self, zip:str, member:str):
        self._zip    = zip
        self._member = member

    def write(self, objs: Iterable[object]) -> None:
        with ZipFile(self._zip,mode='a') as zip:
            with zip.open(self._member,mode='w') as f:
                f.writelines(map(pickle.dumps,objs))

class ZipMemberToObjects:

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

class EnvironmentToObjects(Source[Iterable[object]]):
    def __init__(self, environment: Environment) -> None:
        self._env = environment

    def read(self) -> Iterable[object]:
        yield {"version":1}
        yield self._env.params
        yield from self._env.read()

class EnvironmentFromObjects(Environment):
    def __init__(self, source: Source[Iterable[object]]) -> None:
        self._source = source

    @property
    def params(self) -> Mapping[str,Any]:
        return next(islice(self._source.read(),1,None))

    def read(self) -> Iterable[Interaction]:
        yield from islice(self._source.read(),2,None)
