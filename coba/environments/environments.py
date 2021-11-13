
from typing import Sequence, overload, Union, Iterable, Iterator

from coba.pipes import Source, ResponseToLines, HttpIO, DiskIO, JsonDecode, Shuffle, Take

from coba.environments.pipes import EnvironmentPipe
from coba.environments.core import Environment
from coba.environments.filters import SimulationFilter, Binary
from coba.environments.formats import EnvironmentFileFmtV1
from coba.environments.simulations import ValidationSimulation

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

    @staticmethod
    def from_test_sim(n_interactions:int, n_actions:int, n_context_features:int, n_action_features:int, seed:int=1) -> 'Environments':
        return Environments(
            ValidationSimulation(n_interactions, n_actions=n_actions, n_context_feats=n_context_features, n_action_feats=n_action_features, seed=seed)
        )

    #@staticmethod
    #def from_csv_sim(file:str, actions:Sequence[Any], label_column:Union[str,int], label_type:Literal[] reward_type: Literal["binary","continuous"])

    def __init__(self, *environments: Environment):
        self._environments = list(environments)

    def binary(self) -> 'Environments':
        self._environments = [ EnvironmentPipe(e,Binary()) for e in self._environments ]
        return self

    def shuffle(self, seeds: Sequence[int]) -> 'Environments':
        shuffle_filters    = [ Shuffle(seed) for seed in seeds ]
        self._environments = [ EnvironmentPipe(e,s) for e in self._environments for s in shuffle_filters ]
        return self

    def take(self, n_interactions: int, seed: int = None) -> 'Environments':
        take_filter = Take(n_interactions, seed)
        self._environments = [ EnvironmentPipe(e,take_filter) for e in self._environments ]
        return self

    def filter(self, *filters: SimulationFilter) -> 'Environments':
        self._environments = [ EnvironmentPipe(e,f) for e in self._environments for f in filters ]
        return self

    def __getitem__(self, index:int) -> Environment:
        return self._environments[index]

    def __iter__(self) -> Iterator[Environment]:
        return self._environments.__iter__()

    def __len__(self) -> int:
        return len(self._environments)