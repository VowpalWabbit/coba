from abc import abstractmethod
import json
import collections

from itertools import islice
from typing import Optional, Sequence, Tuple, cast, Union, Iterable

from coba.utilities import PackageChecker
from coba.random import CobaRandom
from coba.pipes import Filter

from coba.simulations.core import Interaction

class SimulationFilter(Filter[Iterable[Interaction],Iterable[Interaction]]):

    @abstractmethod
    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        """Apply a filter to a Simulation's interactions."""
        ...

class Shuffle(SimulationFilter):
    def __init__(self, seed:Optional[int]) -> None:
        
        if seed is not None and (not isinstance(seed,int) or seed < 0):
            raise ValueError(f"Invalid parameter for Shuffle: {seed}. An optional integer value >= 0 was expected.")

        self._seed = seed

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:  
        return CobaRandom(self._seed).shuffle(list(interactions))

    def __repr__(self) -> str:
        return f'{{"Shuffle":{self._seed}}}'

class Take(SimulationFilter):
    def __init__(self, count:Optional[int]) -> None:
        
        if count is not None and (not isinstance(count,int) or count < 0):
            raise ValueError(f"Invalid parameter for Take: {count}. An optional integer value >= 0 was expected.")

        self._count = count

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        if self._count is None: return interactions

        materialized = list(islice(interactions,self._count))

        return materialized if len(materialized) == self._count else []

    def __repr__(self) -> str:
        return f'{{"Take":{json.dumps(self._count)}}}'

class PCA(SimulationFilter):

    def __init__(self) -> None:
        PackageChecker.numpy("PCA.__init__")

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        
        PackageChecker.numpy("PcaSimulation.__init__")

        import numpy as np #type: ignore

        interactions = list(interactions)

        contexts = [ list(cast(Tuple[float,...],i.context)) for i in interactions]

        feat_matrix          = np.array(contexts)
        comp_vals, comp_vecs = np.linalg.eig(np.cov(feat_matrix.T))

        comp_vecs = comp_vecs[:,comp_vals > 0]
        comp_vals = comp_vals[comp_vals > 0]

        pca_contexts = (feat_matrix @ comp_vecs ) / np.sqrt(comp_vals) #type:ignore
        pca_contexts = pca_contexts[:,np.argsort(-comp_vals)]

        return [ Interaction(tuple(c),i.actions,i.feedbacks) for c, i in zip(pca_contexts,interactions) ]

    def __repr__(self) -> str:
        return '"PCA"'

class Sort(SimulationFilter):

    def __init__(self, *indexes: Union[int,Sequence[int]]) -> None:
        
        flat_indexes = cast(Sequence[int], indexes[0] if isinstance(indexes[0], collections.Sequence) else indexes)

        if not isinstance(flat_indexes, collections.Sequence) or not isinstance(flat_indexes[0],int):
            raise ValueError(f"Invalid parameter for Sort: {flat_indexes}. A sequence of integers was expected.")

        self.indexes = flat_indexes

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        
        return sorted(interactions, key=lambda interaction: tuple(interaction.context[i] for i in self.indexes))

    def __repr__(self) -> str:
        return f'{{"Sort":{json.dumps(self.indexes, separators=(",",":"))}}}'