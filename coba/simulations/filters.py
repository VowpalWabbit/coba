import collections

from statistics import mean, median, stdev
from numbers import Number
from abc import abstractmethod, ABC
from itertools import islice, chain
from typing import Hashable, Optional, Sequence, cast, Union, Iterable, Dict, Any, List, Tuple

from coba.random import CobaRandom
from coba.pipes import Filter
from coba.statistics import iqr

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

class Identity(SimulationFilter):

    @property
    def params(self) -> Dict[str, Any]:
        return { }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        return interactions

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

class Scale(SimulationFilter):

    def __init__(self, shift:Union[Number,str] ="min", scale:Union[Number,str]="minmax", using:Optional[int]=None):
        assert isinstance(shift,Number) or shift in ["min","mean","med"]
        assert isinstance(scale,Number) or scale in ["minmax","std","iqr"]

        self._shift = shift
        self._scale = scale
        self._using = using

    @property
    def params(self) -> Dict[str, Any]:
        return { "scale_shift": self._shift, "scale_scale":self._scale, "scale_using":self._using }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        underlying_iterable = iter(interactions)
        using_interactions  = list(islice(underlying_iterable,self._using))

        shifts  : Dict[Hashable,float]     = collections.defaultdict(lambda:0)
        scales  : Dict[Hashable,float]     = collections.defaultdict(lambda:1)
        features: Dict[Hashable,List[Any]] = collections.defaultdict(list)

        for interaction in using_interactions:            
            for key,value in self._context_to_key_values(interaction.context):
                features[key].append(value)

        for feature_key,feature_values in features.items():

            if isinstance(feature_values[0],str):
                continue 

            if isinstance(self._shift, Number):
                shifts[feature_key] = self._shift

            if self._shift == "min":
                shifts[feature_key] = min(feature_values)

            if self._shift == "mean":
                shifts[feature_key] = mean(feature_values)

            if self._shift == "med":
                shifts[feature_key] = median(feature_values)

            if isinstance(self._scale, Number):
                num = self._scale
                den = 1
   
            if self._scale == "std":
                num = 1
                den = stdev(feature_values)

            if self._scale == "minmax":
                num = 1
                den = max(feature_values)-min(feature_values)

            if self._scale == "iqr":
                num = 1
                den = iqr(feature_values)

            scales[feature_key] = num/den if den != 0 else 1

        for interaction in chain.from_iterable([using_interactions, underlying_iterable]):

            kv_scaled_context = {}

            for key,value in self._context_to_key_values(interaction.context):
                if isinstance(value,Number):
                    kv_scaled_context[key] = (value-shifts[key])*scales[key]
                else:
                    kv_scaled_context[key] = value

            if interaction.context is None:
                final_context = None
            elif isinstance(interaction.context,dict):
                final_context = kv_scaled_context
            elif isinstance(interaction.context,tuple):
                final_context = tuple( kv_scaled_context[k] for k,_ in self._context_to_key_values(interaction.context))
            else:
                final_context = kv_scaled_context[1]

            yield Interaction(final_context, interaction.actions, reveals=interaction.reveals, **interaction.results)

    def _context_to_key_values(self,context) -> Sequence[Tuple[Hashable,Any]]:
        
        if isinstance(context,dict ): return context.items()
        if isinstance(context,tuple): return enumerate(context)
        if context is not None      : return [(1,context)]

        return []

    def __repr__(self) -> str:
        return str(self.params)

class CycleNoise(SimulationFilter):

    def __init__(self, after:int=0):
        self._after = after

    @property
    def params(self) -> Dict[str, Any]:
        return { "cycle_after": self._after }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        underlying_iterable     = iter(interactions)
        sans_noise_interactions = islice(underlying_iterable, self._after)
        with_noise_interactions = underlying_iterable

        for interaction in sans_noise_interactions:
            yield interaction

        for interaction in with_noise_interactions:
            
            direct_reveals = list(interaction.reveals)
            cycled_reveals  = direct_reveals[1:]+direct_reveals[:1]

            context = interaction.context
            actions = interaction.actions
            results = {**interaction.results, "direct_reveals":direct_reveals}

            yield Interaction(context, actions, reveals=cycled_reveals, **results)

    def __repr__(self) -> str:
        return str(self.params)