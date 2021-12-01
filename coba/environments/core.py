import collections

from typing_extensions import Literal
from typing import Sequence, overload, Union, Iterable, Iterator

from coba.pipes import Source, DiskIO, JsonDecode, Shuffle, Take

from coba.environments.pipes       import EnvironmentPipe
from coba.environments.filters     import SimulationFilter, Binary
from coba.environments.definitions import EnvironmentDefinitionFileV1
from coba.environments.simulations import DebugSimulation
from coba.environments.primitives  import Environment, LoggedEnvironment, SimulatedEnvironment, WarmStartEnvironment
from coba.environments.openml      import OpenmlSimulation

class Environments:

    @overload
    @staticmethod
    def from_file(filesource:Source[Iterable[str]]) -> 'Environments': ...

    @overload
    @staticmethod
    def from_file(filename:str) -> 'Environments': ...
    
    @staticmethod #type: ignore #(this apppears to be a mypy bug https://github.com/python/mypy/issues/7781)
    def from_file(arg) -> 'Environments': #type: ignore
        """Instantiate Environments from an environments definition file."""
        
        if isinstance(arg,str):
            content = '\n'.join(DiskIO(arg).read())
        else:
            content = '\n'.join(arg.read())

        return Environments(*EnvironmentDefinitionFileV1().filter(JsonDecode().filter(content)))

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
           interaction terms are calculted by assuming all actions or contexts have a constant feature of 1."""

        return Environments(
            DebugSimulation(n_interactions, n_actions, n_context_features, n_action_features, r_noise_var, interactions, seed)
        )

    @staticmethod
    def from_openml(
        openml_ids: Union[int, Sequence[int]], 
        take: int = None,
        cat_as_str:bool = False,
        type: Literal["classification", "regression"] = "classification") -> 'Environments':

        if isinstance(openml_ids, int): openml_ids = [openml_ids]
        return Environments(*[OpenmlSimulation(id, take, type, cat_as_str) for id in openml_ids])

    def __init__(self, *environments: Union[Environment, Sequence[Environment]]):

        self._environments = []

        for env in environments:
            if isinstance(env, collections.Sequence):
                self._environments.extend(env)
            else:
                self._environments.append(env)

    def binary(self) -> 'Environments':
        return self.filter(Binary())

    def shuffle(self, seeds: Sequence[int]) -> 'Environments':
        return self.filter([Shuffle(seed) for seed in seeds])

    def take(self, n_interactions: int, seed: int = None) -> 'Environments':
        return self.filter(Take(n_interactions, seed))

    def filter(self, filter: Union[SimulationFilter,Sequence[SimulationFilter]]) -> 'Environments':
        filters = filter if isinstance(filter, collections.Sequence) else [filter]
        self._environments = [ EnvironmentPipe(e,f) for e in self._environments for f in filters ]
        return self

    def __getitem__(self, index:int) -> Union[SimulatedEnvironment, LoggedEnvironment, WarmStartEnvironment]:
        return self._environments[index]

    def __iter__(self) -> Iterator[Union[SimulatedEnvironment, LoggedEnvironment, WarmStartEnvironment]]:
        return self._environments.__iter__()

    def __len__(self) -> int:
        return len(self._environments)

    def __add__(self, other: 'Environments') -> 'Environments':
        return Environments(self._environments,other._environments)

    def __repr__(self) -> str:
        return "\n".join([f"{i+1}. {e}" for i,e in enumerate(self._environments)])
