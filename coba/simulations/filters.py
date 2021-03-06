import json
import collections

from typing import Optional, Sequence, Tuple, overload, cast, Union

from coba.tools import PackageChecker
from coba.random import CobaRandom
from coba.data.filters import Filter
from coba.simulations.core import Simulation, MemorySimulation, BatchedSimulation, Interaction

class Shuffle(Filter[Simulation,Simulation]):
    def __init__(self, seed:Optional[int]) -> None:
        
        if seed is not None and (not isinstance(seed,int) or seed < 0):
            raise ValueError(f"Invalid parameter for Shuffle: {seed}. An optional integer value >= 0 was expected.")

        self._seed = seed

    def filter(self, item: Simulation) -> Simulation:  
        shuffled_interactions = CobaRandom(self._seed).shuffle(item.interactions)
        return MemorySimulation(shuffled_interactions, item.reward)

    def __repr__(self) -> str:
        return f'{{"Shuffle":{self._seed}}}'

class Take(Filter[Simulation,Simulation]):
    def __init__(self, count:Optional[int]) -> None:
        
        if count is not None and (not isinstance(count,int) or count < 0):
            raise ValueError(f"Invalid parameter for Take: {count}. An optional integer value >= 0 was expected.")

        self._count = count

    def filter(self, item: Simulation) -> Simulation:

        if self._count is None:
            return item

        if self._count > len(item.interactions):
            return MemorySimulation([], item.reward)

        return MemorySimulation(item.interactions[0:self._count], item.reward)

    def __repr__(self) -> str:
        return f'{{"Take":{json.dumps(self._count)}}}'

class Batch(Filter[Simulation,BatchedSimulation]):
    
    @overload
    def __init__(self,*,count: int): ...

    @overload
    def __init__(self,*,size: int): ...

    @overload
    def __init__(self,*,sizes: Sequence[int]): ...

    def __init__(self, **kwargs) -> None:
        
        if not any(key in kwargs for key in ['count', 'size', 'sizes']):
            raise ValueError(f"Invalid parameters for Batch: {kwargs}. A count, size or sizes kwarg was expected.")

        self._kwargs = kwargs

        self._count  = cast(Optional[int], kwargs.get("count", None))
        self._size   = cast(Optional[int], kwargs.get("size", None))
        self._sizes  = cast(Optional[Sequence[int]], kwargs.get("sizes", None))

    def filter(self, item: Simulation) -> BatchedSimulation:
        
        sizes: Optional[Sequence[int]] = None

        if self._count is not None:
            n         = len(item.interactions)
            sizes     = [int(float(n)/(self._count))] * self._count
            remainder = n - sum(sizes)
            for i in range(remainder): sizes[int(i*len(sizes)/remainder)] += 1
        
        if self._size is not None:
            n     = len(item.interactions)
            sizes = [self._size] * int(n/self._size)

        if self._sizes is not None:
            sizes = self._sizes

        if sizes is None:
            raise Exception("We were unable to determine an approriate batch sizes")
        else:
            return BatchedSimulation(item, sizes)

    def __repr__(self) -> str:
        return f'{{"Batch":{json.dumps(self._kwargs, separators=(",",":"))}}}'

class PCA(Filter[Simulation,Simulation]):

    def __init__(self) -> None:
        PackageChecker.numpy("PCA.__init__")

    def filter(self, simulation: Simulation) -> Simulation:
        
        PackageChecker.numpy("PcaSimulation.__init__")

        import numpy as np #type: ignore

        contexts = [ list(cast(Tuple[float,...],i.context)) for i in simulation.interactions]

        feat_matrix          = np.array(contexts)
        comp_vals, comp_vecs = np.linalg.eig(np.cov(feat_matrix.T))

        comp_vecs = comp_vecs[:,comp_vals > 0]
        comp_vals = comp_vals[comp_vals > 0]

        new_contexts = (feat_matrix @ comp_vecs ) / np.sqrt(comp_vals) #type:ignore
        new_contexts = new_contexts[:,np.argsort(-comp_vals)]

        interactions = [ Interaction(tuple(c), i.actions, i.key) for c, i in zip(new_contexts,simulation.interactions) ]

        return MemorySimulation(interactions, simulation.reward)

    def __repr__(self) -> str:
        return '"PCA"'

class Sort(Filter[Simulation,Simulation]):

    def __init__(self, *indexes: Union[int,Sequence[int]]) -> None:
        
        flat_indexes = cast(Sequence[int], indexes[0] if isinstance(indexes[0], collections.Sequence) else indexes)

        if not isinstance(flat_indexes, collections.Sequence) or not isinstance(flat_indexes[0],int):
            raise ValueError(f"Invalid parameter for Sort: {flat_indexes}. A sequence of integers was expected.")

        self.indexes = flat_indexes

    def filter(self, simulation: Simulation) -> Simulation:
        
        sort_key            = lambda interaction: tuple([interaction.context[i] for i in self.indexes ])
        sorted_interactions = list(sorted(simulation.interactions, key=sort_key))

        return MemorySimulation(sorted_interactions, simulation.reward)

    def __repr__(self) -> str:
        return f'{{"Sort":{json.dumps(self.indexes, separators=(",",":"))}}}'