
from typing import Sequence, overload, Union, Iterable

from coba.pipes import Source, ResponseToLines, HttpIO, DiskIO, JsonDecode

from coba.environments.pipes import EnvironmentPipe
from coba.environments.core import Environment
from coba.environments.filters import Shuffle, Take, SimulationFilter
from coba.environments.formats import EnvironmentFileFmtV1

class Environments:

    @overload
    @staticmethod
    def from_file(filesource:Union[Source[str], Source[Iterable[str]]]) -> 'Environments': ...

    @overload
    @staticmethod
    def from_file(filename:str) -> 'Environments': ...
    
    @staticmethod #type: ignore #(this apppears to be a mypy bug https://github.com/python/mypy/issues/7781)
    def from_file(arg) -> 'Environments': #type: ignore
        """Instantiate a Benchmark from a config file."""

        if isinstance(arg,str) and arg.startswith('http'):
            content = '\n'.join(ResponseToLines().filter(HttpIO(arg).read()))
        
        elif isinstance(arg,str) and not arg.startswith('http'):
            content = '\n'.join(DiskIO(arg).read())

        else:
            content = arg.read() #type: ignore

        return Environments(EnvironmentFileFmtV1().filter(JsonDecode().filter(content)))


    def __init__(self, environments: Sequence[Environment]):
        self._environments = environments

    def shuffle(self, seeds: Sequence[int]) -> 'Environments':
        shuffle_filters = [ Shuffle(seed) for seed in seeds ]
        self._environments = [ EnvironmentPipe(e,s) for e in self._environments for s in shuffle_filters ]
        return self

    def take(self, n_interactions: int) -> 'Environments':
        take_filter = Take(n_interactions)
        self._environments = [ EnvironmentPipe(e,take_filter) for e in self._environments ]
        return self

    def filter(self, *filters: SimulationFilter) -> 'Environments':
        self._environments = [ EnvironmentPipe(e,f) for e in self._environments for f in filters ]
        return self