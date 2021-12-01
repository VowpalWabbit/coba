import collections.abc

from math import isnan
from statistics import mean, median, stdev, mode
from numbers import Number
from collections import defaultdict
from abc import abstractmethod, ABC
from itertools import islice, chain
from typing_extensions import Literal
from typing import Hashable, Optional, Sequence, cast, Union, Iterable, Dict, Any, List, Tuple

from coba.random import CobaRandom
from coba.pipes import Filter
from coba.statistics import iqr

from coba.environments.primitives import SimulatedInteraction, LoggedInteraction, Interaction

class SimulationFilter(Filter[Iterable[SimulatedInteraction],Iterable[SimulatedInteraction]], ABC):

    @property
    @abstractmethod
    def params(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def filter(self, interactions: Iterable[SimulatedInteraction]) -> Iterable[SimulatedInteraction]:
        """Apply a filter to a Simulation's interactions."""
        ...

    def __repr__(self) -> str:
        return str(self.params)

class Sort(SimulationFilter):

    def __init__(self, *indexes: Union[int,Sequence[int]]) -> None:
        
        flat_indexes = cast(Sequence[int], indexes[0] if isinstance(indexes[0], collections.abc.Sequence) else indexes)

        if not isinstance(flat_indexes, collections.abc.Sequence) or not isinstance(flat_indexes[0],int):
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
        
        shifts  : Dict[Hashable,float]     = defaultdict(lambda:0)
        scales  : Dict[Hashable,float]     = defaultdict(lambda:1)
        features: Dict[Hashable,List[Any]] = defaultdict(list)

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

            yield SimulatedInteraction(final_context, interaction.actions, **interaction.kwargs)

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
            kwargs = {k:v[1:]+v[:1] for k,v in interaction.kwargs.items()}
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
        
        stats   : Dict[Hashable,float]     = defaultdict(lambda:0)
        features: Dict[Hashable,List[Any]] = defaultdict(list)

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

            yield SimulatedInteraction(final_context, interaction.actions, **interaction.kwargs)

    def _context_as_name_values(self,context) -> Sequence[Tuple[Hashable,Any]]:
        
        if isinstance(context,dict ): return context.items()
        if isinstance(context,tuple): return enumerate(context)
        if context is not None      : return [(1,context)]

        return []

    def __repr__(self) -> str:
        return str(self.params)

class Binary(SimulationFilter):
    @property
    def params(self) -> Dict[str, Any]:
        return { "binary": True }

    def filter(self, interactions: Iterable[SimulatedInteraction]) -> Iterable[SimulatedInteraction]:

        for interaction in interactions:
            kwargs = interaction.kwargs
            max_rwd = max(kwargs["rewards"])
            kwargs["rewards"] = [int(r==max_rwd) for r in kwargs["rewards"]]

            yield SimulatedInteraction(interaction.context, interaction.actions, **kwargs)

class ToWarmStart(SimulationFilter):
    def __init__(self, n_warmstart:int, seed:int = 1):
        
        self._n_warmstart = n_warmstart
        self._seed = seed

    @property
    def params(self) -> Dict[str, Any]:
        return { "n_warmstart": self._n_warmstart }

    def filter(self, interactions: Iterable[SimulatedInteraction]) -> Iterable[Interaction]:

        self._rng = CobaRandom(self._seed)

        underlying_iterable    = iter(interactions)
        logged_interactions    = map(self._to_logged_interaction, islice(underlying_iterable, self._n_warmstart))
        simulated_interactions = underlying_iterable

        return chain(logged_interactions, simulated_interactions)

    def _to_logged_interaction(self, interaction: SimulatedInteraction) -> LoggedInteraction:        
        num_actions   = len(interaction.actions)
        probabilities = [1/num_actions] * num_actions 
        
        selected_index       = self._rng.choice(list(range(num_actions)), probabilities)
        selected_action      = interaction.actions[selected_index]
        selected_probability = probabilities[selected_index]
        selected_reward      = interaction.kwargs.get("reveals", interaction.kwargs.get("rewards", None))[selected_index]

        return LoggedInteraction(interaction.context, selected_action, selected_reward, selected_probability)

    def __repr__(self) -> str:
        return str(self.params)
