import collections.abc

from typing import Sequence, overload, Union, Iterable, Iterator
from coba.backports import Literal

from coba.pipes import Source, DiskIO, JsonDecode

from coba.environments.filters     import FilteredEnvironment, EnvironmentFilter
from coba.environments.filters     import Binary, Shuffle, Take, Sparse, Reservoir
from coba.environments.definitions import EnvironmentDefinitionFileV1

from coba.environments          .primitives import Environment
from coba.environments.logged   .primitives import LoggedEnvironment
from coba.environments.simulated.primitives import SimulatedEnvironment
from coba.environments.warmstart.primitives import WarmStartEnvironment

from coba.environments.simulated.synthetics import LinearSyntheticSimulation, LocalSyntheticSimulation
from coba.environments.simulated.openml     import OpenmlSimulation

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
    def from_linear_synthetic(
        n_interactions: int = 500,
        n_actions: int = 3,
        n_context_features: int = 2,
        n_action_features: int = 2,
        r_noise_var: float = 1/1000,
        interactions: Sequence[str] = ["a","xa"],
        seed: int = 1) -> 'Environments':
        """A simple simulation useful for debugging learning algorithms. It's rewards are linear with respect to the given 
           interactions of context (x) and action (a) features. In the case that no context or action features are requested the 
           interaction terms are calculted by assuming all actions or contexts have a constant feature of 1."""

        return Environments([
            LinearSyntheticSimulation(n_interactions, n_actions, n_context_features, n_action_features, r_noise_var, interactions, seed)
        ])

    @staticmethod
    def from_local_synthetic(
        n_interactions: int = 500, 
        n_actions: int = 3, 
        n_context_features: int = 2, 
        n_contexts: int = 200, 
        seed: int = 1) -> 'Environments':
        """A simple simulation useful for debugging learning algorithms. It's rewards are determined by the location of given 
            context and action pairs with respect to a small set of pre-generated exemplar context,action pairs. Location
            is currently determined by equality, though it could potentially be extended to support any number of metric
            based similarity kernels. The "local" in the name is due to its close relationship to 'local regression'."""

        return Environments([
            LocalSyntheticSimulation(n_interactions, n_contexts, n_context_features, n_actions, seed)
        ])

    @staticmethod
    def from_openml(
        openml_ids: Union[int, Sequence[int]], 
        take: int = None,
        cat_as_str:bool = False,
        type: Literal["classification", "regression"] = "classification") -> 'Environments':
        """Create a SimulatedEnvironment from datasets available on openml."""

        if isinstance(openml_ids, int): openml_ids = [openml_ids]
        return Environments(*[OpenmlSimulation(id, take, type, cat_as_str) for id in openml_ids])

    def __init__(self, *environments: Union[Environment, Sequence[Environment]]):

        self._environments = []

        for env in environments:
            if isinstance(env, collections.abc.Sequence):
                self._environments.extend(env)
            else:
                self._environments.append(env)

    def binary(self) -> 'Environments':
        """Convert rewards in an environment to 1 for max reward else 0."""
        return self.filter(Binary())

    def sparse(self, context:bool = True, action:bool = False) -> 'Environments':
        """Convert an environment from a dense representation to sparse. This has little utility beyond debugging."""
        return self.filter(Sparse(context,action))

    def shuffle(self, seeds: Sequence[int]) -> 'Environments':
        """Shuffle the order of the interactions in the Environments."""
        return self.filter([Shuffle(seed) for seed in seeds])

    def take(self, n_interactions: int) -> 'Environments':
        """Take a fixed number of interactions from the Environments."""
        return self.filter(Take(n_interactions))

    def reservoir(self, n_interactions: int, seed: int = None) -> 'Environments':
        """Take a random fixed number of interactions from the Environments."""
        return self.filter(Reservoir(n_interactions, seed))

    def filter(self, filter: Union[EnvironmentFilter,Sequence[EnvironmentFilter]]) -> 'Environments':
        """Apply filters to each environment currently in Environments."""
        filters = filter if isinstance(filter, collections.abc.Sequence) else [filter]
        self._environments = [ FilteredEnvironment(e,f) for e in self._environments for f in filters ]
        return self

    def __getitem__(self, index:int) -> Union[SimulatedEnvironment, LoggedEnvironment, WarmStartEnvironment]:
        return self._environments[index]

    def __iter__(self) -> Iterator[Union[SimulatedEnvironment, LoggedEnvironment, WarmStartEnvironment]]:
        return self._environments.__iter__()

    def __len__(self) -> int:
        return len(self._environments)

    def __add__(self, other: 'Environments') -> 'Environments':
        return Environments(self._environments,other._environments)

    def __str__(self) -> str:
        return "\n".join([f"{i+1}. {e}" for i,e in enumerate(self._environments)])

    def _ipython_display_(self):
        #pretty print in jupyter notebook (https://ipython.readthedocs.io/en/stable/config/integrating.html)
        print(str(self))
