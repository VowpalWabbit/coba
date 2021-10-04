
import json
import collections

from abc import abstractmethod, ABC
from itertools import islice
from typing import Optional, Sequence, cast, Union, Iterable, Dict, Any

from coba.random import CobaRandom
from coba.pipes import Filter

from coba.simulations.core import Interaction

class SimulationFilter(Filter[Iterable[Interaction],Iterable[Interaction]], ABC):

    @property
    @abstractmethod
    def params(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        """Apply a filter to a Simulation's interactions."""
        ...

class Shuffle(SimulationFilter):
    
    def __init__(self, seed:Optional[int]) -> None:
        
        if seed is not None and (not isinstance(seed,int) or seed < 0):
            raise ValueError(f"Invalid parameter for Shuffle: {seed}. An optional integer value >= 0 was expected.")

        self._seed = seed

    @property
    def params(self) -> Dict[str, Any]:
        return { "shuffle": self._seed }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:  
        return CobaRandom(self._seed).shuffle(list(interactions))

    def __repr__(self) -> str:
        return str(self.params)

class Take(SimulationFilter):
    
    def __init__(self, count:Optional[int]) -> None:
        
        if count is not None and (not isinstance(count,int) or count < 0):
            raise ValueError(f"Invalid parameter for Take: {count}. An optional integer value >= 0 was expected.")

        self._count = count

    @property
    def params(self) -> Dict[str, Any]:
        return { "take": self._count }    

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        if self._count is None: return interactions

        materialized = list(islice(interactions,self._count))

        return materialized if len(materialized) == self._count else []

    def __repr__(self) -> str:
        return str(self.params)

class Sort(SimulationFilter):

    def __init__(self, *indexes: Union[int,Sequence[int]]) -> None:
        
        flat_indexes = cast(Sequence[int], indexes[0] if isinstance(indexes[0], collections.Sequence) else indexes)

        if not isinstance(flat_indexes, collections.Sequence) or not isinstance(flat_indexes[0],int):
            raise ValueError(f"Invalid parameter for Sort: {flat_indexes}. A sequence of integers was expected.")

        self._indexes = flat_indexes

    @property
    def params(self) -> Dict[str, Any]:
        return { "sort": self._indexes }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        
        return sorted(interactions, key=lambda interaction: tuple(interaction.context[i] for i in self._indexes))

    def __repr__(self) -> str:
        return str(self.params)
