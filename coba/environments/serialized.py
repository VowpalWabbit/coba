import pickle

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from itertools import islice, repeat, chain
from typing import Union, Sequence, Mapping, Iterable, Any

from coba.contexts import CobaContext
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
            with ZipFile(self._zip) as z:
                with z.open(self._member) as m:
                    yield from map(pickle.load, repeat(m))
        except EOFError:
            pass

class EnvironmentsToObjects(Filter[Environment, Iterable]):
    def filter(self, env: Union[Environment,Sequence[Environment]]) -> Iterable[Iterable[object]]:        
        with CobaContext.logger.time("Materializing environment..."):
            yield list(self._env_to_objects(env))

    def _env_to_objects(self,env):
        from coba import __version__ #imported here to avoid circular dependency
        yield {"version":2,"coba_version":__version__}
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
        return list(islice(self._source.read(),1,2))[0]

    def read(self) -> Iterable[Interaction]:

        I = iter(self._source.read())
        version_data = list(islice(I,2))[0]
        
        if version_data['version'] == 1: #pragma: no cover
            yield from I
        else:
            yield from chain.from_iterable(I)
