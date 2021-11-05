import collections

from math import isnan
from statistics import mean, median, stdev, mode
from numbers import Number
from abc import abstractmethod, ABC
from itertools import islice, chain
from typing_extensions import Literal
from typing import Hashable, Optional, Sequence, cast, Union, Iterable, Dict, Any, List, Tuple

from coba.random import CobaRandom
from coba.pipes import Filter
from coba.statistics import iqr

from coba.environments.core import SimulatedInteraction

class SimulationFilter(Filter[Iterable[SimulatedInteraction],Iterable[SimulatedInteraction]], ABC):

    @property
    @abstractmethod
    def params(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def filter(self, interactions: Iterable[SimulatedInteraction]) -> Iterable[SimulatedInteraction]:
        """Apply a filter to a Simulation's interactions."""
        ...

class Identity(SimulationFilter):

    @property
    def params(self) -> Dict[str, Any]:
        return { }

    def filter(self, interactions: Iterable[SimulatedInteraction]) -> Iterable[SimulatedInteraction]:
        return interactions

class Shuffle(SimulationFilter):
    
    def __init__(self, seed:Optional[int]) -> None:
        
        if seed is not None and (not isinstance(seed,int) or seed < 0):
            raise ValueError(f"Invalid parameter for Shuffle: {seed}. An optional integer value >= 0 was expected.")

        self._seed = seed

    @property
    def params(self) -> Dict[str, Any]:
        return { "shuffle": self._seed }

    def filter(self, interactions: Iterable[SimulatedInteraction]) -> Iterable[SimulatedInteraction]: 
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

    def filter(self, interactions: Iterable[SimulatedInteraction]) -> Iterable[SimulatedInteraction]:

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

    def filter(self, interactions: Iterable[SimulatedInteraction]) -> Iterable[SimulatedInteraction]:
        
        return sorted(interactions, key=lambda interaction: tuple(interaction.context[i] for i in self._indexes))

    def __repr__(self) -> str:
        return str(self.params)

class Scale(SimulationFilter):

    def __init__(self, 
        shift: Union[Number,Literal["min","mean","med"]] ="min", 
        scale: Union[Number,Literal["minmax","std","iqr"]]="minmax", 
        using: Optional[int] = None):

        assert isinstance(shift,Number) or shift in ["min","mean","med"]
        assert isinstance(scale,Number) or scale in ["minmax","std","iqr"]

        self._shift = shift
        self._scale = scale
        self._using = using

    @property
    def params(self) -> Dict[str, Any]:
        return { "scale_shift": self._shift, "scale_scale":self._scale, "scale_using":self._using }

    def filter(self, interactions: Iterable[SimulatedInteraction]) -> Iterable[SimulatedInteraction]:

        iter_interactions  = iter(interactions)
        train_interactions = list(islice(iter_interactions,self._using))
        test_interactions  = chain.from_iterable([train_interactions, iter_interactions])
        
        shifts  : Dict[Hashable,float]     = collections.defaultdict(lambda:0)
        scales  : Dict[Hashable,float]     = collections.defaultdict(lambda:1)
        features: Dict[Hashable,List[Any]] = collections.defaultdict(list)

        for interaction in train_interactions:
            for name,value in self._context_as_name_values(interaction.context):
                if isinstance(value,Number) and not isnan(value):
                    features[name].append(value)

        for feat_name, feat_numeric_values in features.items():

            if isinstance(feat_numeric_values[0],str):
                continue 

            if isinstance(self._shift, Number):
                shifts[feat_name] = self._shift

            if self._shift == "min":
                shifts[feat_name] = min(feat_numeric_values)

            if self._shift == "mean":
                shifts[feat_name] = mean(feat_numeric_values)

            if self._shift == "med":
                shifts[feat_name] = median(feat_numeric_values)

            if isinstance(self._scale, Number):
                num = self._scale
                den = 1
   
            if self._scale == "std":
                num = 1
                den = stdev(feat_numeric_values)

            if self._scale == "minmax":
                num = 1
                den = max(feat_numeric_values)-min(feat_numeric_values)

            if self._scale == "iqr":
                num = 1
                den = iqr(feat_numeric_values)

            scales[feat_name] = num/den if round(den,50) != 0 else 1

        for interaction in test_interactions:

            kv_scaled_context = {}

            for name,value in self._context_as_name_values(interaction.context):
                if isinstance(value,Number):
                    kv_scaled_context[name] = (value-shifts[name])*scales[name]
                else:
                    kv_scaled_context[name] = value

            if interaction.context is None:
                final_context = None
            elif isinstance(interaction.context,dict):
                final_context = kv_scaled_context
            elif isinstance(interaction.context,tuple):
                final_context = tuple( kv_scaled_context[k] for k,_ in self._context_as_name_values(interaction.context))
            else:
                final_context = kv_scaled_context[1]

            yield SimulatedInteraction(final_context, interaction.actions, **interaction.results)

    def _context_as_name_values(self,context) -> Sequence[Tuple[Hashable,Any]]:
        
        if isinstance(context,dict ): return context.items()
        if isinstance(context,tuple): return enumerate(context)
        if context is not None      : return [(1,context)]

        return []

    def __repr__(self) -> str:
        return str(self.params)

class Cycle(SimulationFilter):

    def __init__(self, after:int = 0):
        self._after = after

    @property
    def params(self) -> Dict[str, Any]:
        return { "cycle_after": self._after }

    def filter(self, interactions: Iterable[SimulatedInteraction]) -> Iterable[SimulatedInteraction]:

        underlying_iterable     = iter(interactions)
        sans_cycle_interactions = islice(underlying_iterable, self._after)
        with_cycle_interactions = underlying_iterable

        for interaction in sans_cycle_interactions:
            yield interaction

        for interaction in with_cycle_interactions:
            kwargs = {k:v[1:]+v[:1] for k,v in interaction.results.items()}
            yield SimulatedInteraction(interaction.context, interaction.actions, **kwargs)

    def __repr__(self) -> str:
        return str(self.params)

class Impute(SimulationFilter):

    def __init__(self, 
        stat : Literal["mean","median","mode"] = "mean",
        using: Optional[int] = None):

        assert stat in ["mean","median","mode"]

        self._stat  = stat
        self._using = using

    @property
    def params(self) -> Dict[str, Any]:
        return { "impute_stat": self._stat, "impute_using": self._using }

    def filter(self, interactions: Iterable[SimulatedInteraction]) -> Iterable[SimulatedInteraction]:

        iter_interactions  = iter(interactions)
        train_interactions = list(islice(iter_interactions,self._using))
        test_interactions  = chain.from_iterable([train_interactions, iter_interactions])
        
        stats   : Dict[Hashable,float]     = collections.defaultdict(lambda:0)
        features: Dict[Hashable,List[Any]] = collections.defaultdict(list)

        for interaction in train_interactions:
            for name,value in self._context_as_name_values(interaction.context):
                if isinstance(value,Number) and not isnan(value):
                    features[name].append(value)

        for feat_name, feat_numeric_values in features.items():

            if isinstance(feat_numeric_values[0],str):
                continue 

            if self._stat == "mean":
                stats[feat_name] = mean(feat_numeric_values)

            if self._stat == "median":
                stats[feat_name] = median(feat_numeric_values)

            if self._stat == "mode":
                stats[feat_name] = mode(feat_numeric_values)

        for interaction in test_interactions:

            kv_imputed_context = {}

            for name,value in self._context_as_name_values(interaction.context):
                kv_imputed_context[name] = stats[name] if isinstance(value,Number) and isnan(value) else value

            if interaction.context is None:
                final_context = None
            elif isinstance(interaction.context,dict):
                final_context = kv_imputed_context
            elif isinstance(interaction.context,tuple):
                final_context = tuple( kv_imputed_context[k] for k,_ in self._context_as_name_values(interaction.context))
            else:
                final_context = kv_imputed_context[1]

            yield SimulatedInteraction(final_context, interaction.actions, **interaction.results)

    def _context_as_name_values(self,context) -> Sequence[Tuple[Hashable,Any]]:
        
        if isinstance(context,dict ): return context.items()
        if isinstance(context,tuple): return enumerate(context)
        if context is not None      : return [(1,context)]

        return []

    def __repr__(self) -> str:
        return str(self.params)
