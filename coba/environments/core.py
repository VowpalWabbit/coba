import collections.abc

from typing import Sequence, overload, Union, Iterable, Iterator, Any, Optional
from coba.backports import Literal

from coba.pipes import Source, DiskIO, JsonDecode, Reader, CsvReader

from coba.environments.filters     import FilteredEnvironment, EnvironmentFilter
from coba.environments.filters     import Binary, Shuffle, Take, Sparse, Reservoir, Cycle, Scale, Impute
from coba.environments.definitions import EnvironmentDefinitionFileV1

from coba.environments          .primitives import Environment
from coba.environments.logged   .primitives import LoggedEnvironment
from coba.environments.simulated.primitives import SimulatedEnvironment
from coba.environments.warmstart.primitives import WarmStartEnvironment

from coba.environments.simulated.synthetics import LinearSyntheticSimulation, LocalSyntheticSimulation
from coba.environments.simulated.openml     import OpenmlSimulation
from coba.environments.simulated.supervised import SupervisedSimulation

class Environments:
    """A friendly wrapper around commonly used environment functionality."""

    @overload
    @staticmethod
    def from_file(filesource:Source[Iterable[str]]) -> 'Environments': ...

    @overload
    @staticmethod
    def from_file(filename:str) -> 'Environments': ...
    
    @staticmethod
    def from_file(arg) -> 'Environments':
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
        cross_terms: Sequence[str] = ["a","xa"],
        seed: int = 1) -> 'Environments':
        """A simple simulation useful for debugging learning algorithms. 
        
        The simulation's rewards are linear with respect to the given features and their cross terms. In the case 
        that no context or action features are requested interaction terms are calculted by assuming actions or 
        contexts have a constant feature of 1.
        """

        return Environments([
            LinearSyntheticSimulation(n_interactions, n_actions, n_context_features, n_action_features, r_noise_var, cross_terms, seed)
        ])

    @staticmethod
    def from_local_synthetic(
        n_interactions: int = 500, 
        n_actions: int = 3, 
        n_context_features: int = 2, 
        n_contexts: int = 200, 
        seed: int = 1) -> 'Environments':
        """A simple synthetic simulation useful for debugging learning algorithms. 
                
        The simulation's rewards are linear with respect to the given features and their cross terms. In the case 
        that no context or action features are requested interaction terms are calculted by assuming actions or 
        contexts have a constant feature of 1.
        """

        return Environments([
            LocalSyntheticSimulation(n_interactions, n_contexts, n_context_features, n_actions, seed)
        ])

    @staticmethod
    def from_openml(
        openml_ids: Union[int, Sequence[int]], 
        take: int = None,
        type: Literal["C", "R"] = "C",
        cat_as_str:bool = False) -> 'Environments':
        """Create a SimulatedEnvironment from datasets available on openml."""

        if isinstance(openml_ids, int): openml_ids = [openml_ids]
        return Environments(*[OpenmlSimulation(id, take, type, cat_as_str) for id in openml_ids])

    @overload
    @staticmethod
    def from_supervised(
        source: Union[str, Source[Any]],
        reader: Reader = CsvReader(), 
        label_col: Union[int,str] = 0,
        label_type: Literal["C","R"] = "C",
        take: int = None) -> 'Environments':
        """Create a SimulatedEnvironment from a supervised dataset"""
        ...

    @overload
    @staticmethod
    def from_supervised(
        X = Sequence[Any], 
        Y = Sequence[Any],
        label_type: Literal["C","R"] = "C", 
        take:int = None) -> 'Environments':
        """Create a SimulatedEnvironment from a supervised dataset"""
        ...

    @staticmethod
    def from_supervised(*args, **kwargs) -> 'Environments':
        """Create a SimulatedEnvironment from a supervised dataset"""
        return Environments(SupervisedSimulation(*args, **kwargs))

    def __init__(self, *environments: Union[Environment, Sequence[Environment]]):
        """Instantiate an Environments class.
        
        Args:
            *environments: The base environments to initialize the class.        
        """
        self._environments = []

        for env in environments:
            if isinstance(env, collections.abc.Sequence):
                self._environments.extend(env)
            else:
                self._environments.append(env)

    def binary(self) -> 'Environments':
        """Binarize all rewards to either 1 (max rewards) or 0 (all others)."""
        return self.filter(Binary())

    def sparse(self, context:bool = True, action:bool = False) -> 'Environments':
        """Convert an environment from a dense representation to sparse. This has little utility beyond debugging."""
        return self.filter(Sparse(context,action))

    def shuffle(self, seeds: Union[Sequence[int],int] = 1) -> 'Environments':
        """Shuffle the order of the interactions in the Environments."""
        if isinstance(seeds,int): seeds = [seeds]
        return self.filter([Shuffle(seed) for seed in seeds])

    def cycle(self, after: int) -> 'Environments':
        """Cycle all rewards associated with actions by one place."""
        return self.filter(Cycle(after))

    def take(self, n_interactions: int) -> 'Environments':
        """Take a fixed number of interactions from the Environments."""
        return self.filter(Take(n_interactions))

    def reservoir(self, n_interactions: int, seeds: Sequence[int]) -> 'Environments':
        """Take a random fixed number of interactions from the Environments."""
        return self.filter([Reservoir(n_interactions,seed) for seed in seeds])

    def scale(self,
        shift: Union[float,Literal["min","mean","med"]] ="min", 
        scale: Union[float,Literal["minmax","std","iqr"]]="minmax", 
        using: Optional[int] = None) -> 'Environments':
        """Apply an affine shift and scaling factor to precondition environments."""
        return self.filter(Scale(shift, scale, using))
        
    def impute(self,
        stat : Literal["mean","median","mode"] = "mean",
        using: Optional[int] = None) -> 'Environments':
        """Impute missing values with a feature statistic using a given number of interactions."""
        return self.filter(Impute(stat, using))

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
