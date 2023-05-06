import pickle

from pathlib import Path
from collections import abc
from zipfile import ZipFile, ZIP_DEFLATED
from itertools import islice, repeat, chain
from typing import Union, Sequence, Mapping, Iterable, Any

from coba.contexts import CobaContext
from coba.backports import version
from coba.pipes import Source, Sink, Filter
from coba.environments.primitives import Environment, Interaction

class ObjectsToZipMember(Sink[Iterable[Sequence[object]]]):
    def __init__(self, zip:str):
        self._zip = zip
        self._start = 0
        if Path(self._zip).exists():
            for n in ZipFile(self._zip).namelist():
                try:
                    self._start = max(self._start,int(n)+1)
                except ValueError: #pragma: no cover
                    pass

    def write(self, envs: Iterable[Sequence[object]]) -> None:
        for i,env in enumerate(envs,self._start):
            with CobaContext.logger.time(f"Writing environment {i}..."):
                # Dump before opening the file so we
                # can keep it open for as short a time 
                # as possible.
                dumps = list(map(pickle.dumps,env))
                with ZipFile(self._zip,mode='a',compression=ZIP_DEFLATED) as zip:
                    with zip.open(str(i),mode='w') as f:
                        f.writelines(dumps)

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

class EnvironmentsToObjects(Filter[Environment, Iterable]):
    def filter(self, envs: Union[Environment,Sequence[Environment]]) -> Iterable[Iterable[object]]:
        if not isinstance(envs,(abc.Iterable)): envs = [envs]
        for env in envs:
            with CobaContext.logger.time("Materializing environment..."):
                yield list(self._env_to_objects(env))

    def _env_to_objects(self,env):
        yield {"version":2,"coba_version":version("coba")}
        yield env.params
        I = iter(env.read())
        batch = list(islice(I,1000))
        while batch: 
            yield batch
            batch = list(islice(I,1000))

class EnvironmentFromObjects(Environment):
    def __init__(self, source: Source[Iterable[object]]) -> None:
        self._source = source

    @property
    def params(self) -> Mapping[str,Any]:
        return next(islice(self._source.read(),1,2))

    def read(self) -> Iterable[Interaction]:
        version_data = next(iter(self._source.read()))
        if version_data['version'] == 1: #pragma: no cover
            yield from islice(self._source.read(),2,None)
        else:
            yield from chain(*islice(self._source.read(),2,None))
