import collections.abc

from math import isnan
from statistics import mean, median, stdev, mode
from numbers import Number
from collections import defaultdict
from itertools import islice, chain

from coba.typing import Hashable, Optional, Sequence, Union, Iterable, Dict, Any, List, Tuple, Literal
from coba import pipes
from coba.random import CobaRandom
from coba.statistics import iqr

from coba.environments.primitives import SimulatedInteraction, LoggedInteraction, Interaction, EnvironmentFilter

class Take(pipes.Take):
    
    @property
    def params(self) -> Dict[str, Any]:
        return { "take": self._count }

    def __str__(self) -> str:
        return str(self.params)

class Reservoir(pipes.Reservoir):

    def __init__(self, count: Optional[int], seed:int=1)-> None:
        super().__init__(count, seed)

    @property
    def params(self) -> Dict[str, Any]:
        return { "reservoir_count": self._count, "reservoir_seed": self._seed }

    def __str__(self) -> str:
        return str(self.params)

class Shuffle(pipes.Shuffle):
    
    @property
    def params(self) -> Dict[str, Any]:
        return { "shuffle": self._seed }

    def __str__(self) -> str:
        return str(self.params)

class Identity(pipes.Identity):
    
    @property
    def params(self) -> Dict[str, Any]:
        return {}

    def __str__(self) -> str:
        return "{ Identity }"

class Sort(EnvironmentFilter):

    def __init__(self, *keys: Union[str,int,Sequence[Union[str,int]]]) -> None:
        
        self._keys = []
        
        for key in keys:
            if not isinstance(key, collections.abc.Sequence) or isinstance(key,str):
                self._keys.append(key)
            else:
                self._keys.extend(key)

    @property
    def params(self) -> Dict[str, Any]:
        return { "sort": self._keys }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        return sorted(interactions, key=lambda interaction: tuple(interaction.context[key] for key in self._keys))

class Scale(EnvironmentFilter):

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

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

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
                final_context = tuple(kv_scaled_context[k] for k,_ in self._context_as_name_values(interaction.context))
            else:
                final_context = kv_scaled_context[1]

            try:
                yield SimulatedInteraction(final_context, interaction.actions, **interaction.kwargs)
            except:
                yield LoggedInteraction(final_context, interaction.action, **interaction.kwargs)

    def _context_as_name_values(self,context) -> Sequence[Tuple[Hashable,Any]]:
        
        if isinstance(context,dict ): return context.items()
        if isinstance(context,tuple): return enumerate(context)
        if context is not None      : return [(1,context)]

        return []

class Impute(EnvironmentFilter):

    def __init__(self, 
        stat : Literal["mean","median","mode"] = "mean",
        using: Optional[int] = None):

        assert stat in ["mean","median","mode"]

        self._stat  = stat
        self._using = using

    @property
    def params(self) -> Dict[str, Any]:
        return { "impute_stat": self._stat, "impute_using": self._using }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        iter_interactions  = iter(interactions)
        train_interactions = list(islice(iter_interactions,self._using))
        test_interactions  = chain.from_iterable([train_interactions, iter_interactions])
        
        stats   : Dict[Hashable,float]        = defaultdict(int)
        features: Dict[Hashable,List[Number]] = defaultdict(list)

        for interaction in train_interactions:
            for name,value in self._context_as_name_values(interaction.context):
                if isinstance(value,Number) and not isnan(value):
                    features[name].append(value)

        for feat_name, feat_numeric_values in features.items():

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
                final_context = tuple(kv_imputed_context[k] for k,_ in self._context_as_name_values(interaction.context))
            else:
                final_context = kv_imputed_context[1]

            try:
                yield SimulatedInteraction(final_context, interaction.actions, **interaction.kwargs)
            except:
                yield LoggedInteraction(final_context, interaction.action, **interaction.kwargs)

    def _context_as_name_values(self,context) -> Sequence[Tuple[Hashable,Any]]:
        
        if isinstance(context,dict ): return context.items()
        if isinstance(context,tuple): return enumerate(context)
        if context is not None      : return [(1,context)]

        return []

class Cycle(EnvironmentFilter):

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

class Binary(EnvironmentFilter):
    @property
    def params(self) -> Dict[str, Any]:
        return { "binary": True }

    def filter(self, interactions: Iterable[SimulatedInteraction]) -> Iterable[SimulatedInteraction]:

        for interaction in interactions:
            kwargs  = interaction.kwargs.copy()
            max_rwd = max(kwargs["rewards"])
            
            kwargs["rewards"] = [int(r==max_rwd) for r in kwargs["rewards"]]

            yield SimulatedInteraction(interaction.context, interaction.actions, **kwargs)

class ToWarmStart(EnvironmentFilter):

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
        
        kwargs = {"probability":selected_probability, "actions":interaction.actions}

        if "reveals" in interaction.kwargs:
            kwargs["reveal"] = interaction.kwargs["reveals"][selected_index]

        if "rewards" in interaction.kwargs:
            kwargs["reward"] = interaction.kwargs["rewards"][selected_index]

        return LoggedInteraction(interaction.context, selected_action, **kwargs)

class Sparse(EnvironmentFilter):
    
    def __init__(self, context:bool = True, action:bool = False):

        self._context = context
        self._action  = action

    @property
    def params(self) -> Dict[str, Any]:
        return { "sparse_C": self._context, "sparse_A": self._action }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        for interaction in interactions:

            sparse_context = self._make_sparse(interaction.context) if self._context else interaction.context

            if hasattr(interaction, 'actions'):
                sparse_actions = list(map(self._make_sparse,interaction.actions)) if self._action else interaction.actions
                yield SimulatedInteraction(sparse_context, sparse_actions, **interaction.kwargs)
            else:
                sparse_action = self._make_sparse(interaction.action) if self._action else interaction.action
                yield LoggedInteraction(sparse_context, sparse_action, **interaction.kwargs)

    def _make_sparse(self, value) -> Optional[dict]:
        if isinstance(value,dict) or value is None:
            return value
        if isinstance(value,(list,tuple)):
            return dict(enumerate(value))
        return {0:value}
