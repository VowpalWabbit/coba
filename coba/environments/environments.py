from typing_extensions import Literal
from typing import Sequence, overload, Union, Iterable, Iterator
from coba.environments.core import LoggedEnvironment, SimulatedEnvironment, WarmStartEnvironment

from coba.pipes import Source, ResponseToLines, HttpIO, DiskIO, JsonDecode, Shuffle, Take

from coba.environments.pipes import EnvironmentPipe
from coba.environments.filters import SimulationFilter, Binary
from coba.environments.formats import EnvironmentFileFmtV1
from coba.environments.simulations import DebugSimulation
from coba.environments.openml import OpenmlSimulation

class Environments:

    @overload
    @staticmethod
    def from_file(filesource:Union[Source[str], Source[Iterable[str]]]) -> 'Environments': ...

    @overload
    @staticmethod
    def from_file(filename:str) -> 'Environments': ...
    
    @staticmethod #type: ignore #(this apppears to be a mypy bug https://github.com/python/mypy/issues/7781)
    def from_file(arg) -> 'Environments': #type: ignore
        """Instantiate Environments from an environments definition file."""

        if isinstance(arg,str) and arg.startswith('http'):
            content = '\n'.join(ResponseToLines().filter(HttpIO(arg).read()))
        
        elif isinstance(arg,str) and not arg.startswith('http'):
            content = '\n'.join(DiskIO(arg).read())

        else:
            content = arg.read() #type: ignore

        return Environments(*EnvironmentFileFmtV1().filter(JsonDecode().filter(content)))

    @staticmethod
    def from_debug(
        n_interactions:int = 5000, 
        n_actions:int = 3, 
        n_context_features:int = 2, 
        n_action_features:int = 2, 
        r_noise_var:float = 1/1000,
        interactions:Sequence[str] = ["a","xa"], 
        seed:int=1) -> 'Environments':
        """A simple simulation useful for debugging learning algorithms. It's rewards are linear with respect to the given 
           interactions of context (x) and action (a) features. In the case that no context or action features are requested the 
           interaction terms are calculted by assuming all actions or contexts have a constant  feature of 1."""

        return Environments(
            DebugSimulation(n_interactions, n_actions, n_context_features, n_action_features, r_noise_var, interactions,  seed)
        )

    @staticmethod
    def from_openml(
        openml_ids: Union[int, Sequence[int]], 
        take: int = None,
        cat_as_str:bool = False,
        type: Literal["classification", "regression"] = "classification") -> 'Environments':
        
        if isinstance(openml_ids, int): openml_ids = [openml_ids]
        return Environments(*[OpenmlSimulation(id, take, type, cat_as_str) for id in openml_ids])

    def __init__(self, *environments: Union[SimulatedEnvironment, LoggedEnvironment, WarmStartEnvironment]):
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

    def __getitem__(self, index:int) -> Union[SimulatedEnvironment, LoggedEnvironment, WarmStartEnvironment]:
        return self._environments[index]

    def __iter__(self) -> Iterator[Union[SimulatedEnvironment, LoggedEnvironment, WarmStartEnvironment]]:
        return self._environments.__iter__()

    def __len__(self) -> int:
        return len(self._environments)

    def __repr__(self) -> str:
        return "\n".join([f"{i+1}. {e}" for i,e in enumerate(self._environments)])