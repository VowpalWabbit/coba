from math import isnan
from statistics import mean, median, stdev, mode
from numbers import Number
from collections import defaultdict
from itertools import islice, chain
from typing import Hashable, Optional, Sequence, Union, Iterable, Dict, Any, List, Tuple
from coba.backports import Literal

from coba.exceptions import CobaException
from coba.statistics import iqr

from coba.environments.primitives import Interaction
from coba.environments.logged.primitives import LoggedInteraction
from coba.environments.simulated.primitives import SimulatedInteraction
from coba.environments.filters.primitives import EnvironmentFilter

class Scale(EnvironmentFilter):
    """Shift and scale features to precondition them before learning."""

    def __init__(self, 
        shift: Union[Number,Literal["min","mean","med"]] ="min", 
        scale: Union[Number,Literal["minmax","std","iqr"]]="minmax", 
        using: Optional[int] = None):
        """Instantiate a Scale filter.
        
        Args:
            shift: The statistic to use to shift each context feature.
            scale: The statistic to use to scale each context feature.
            using: The number of interactions to use when calculating the necessary statistics.
        """

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
    """Impute missing values (nan) in Interaction contexts."""

    def __init__(self, 
        stat : Literal["mean","median","mode"] = "mean",
        using: Optional[int] = None):
        """Instantiate an Impute filter.
        
        Args:
            stat: The statistic to use for impuatation.
            using: The number of interactions to use to calculate the imputation statistics.
        """

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

            if isinstance(interaction, SimulatedInteraction):
                yield SimulatedInteraction(final_context, interaction.actions, **interaction.kwargs)
            elif isinstance(interaction, LoggedInteraction):
                yield LoggedInteraction(final_context, interaction.action, **interaction.kwargs)
            else:#pragma: no cover
                raise CobaException("Unknown interactions were given to the Impute filter.") 

    def _context_as_name_values(self,context) -> Sequence[Tuple[Hashable,Any]]:
        
        if isinstance(context,dict ): return context.items()
        if isinstance(context,tuple): return enumerate(context)
        if context is not None      : return [(1,context)]

        return []

class Sparse(EnvironmentFilter):
    """Sparsify an environment's feature representation. 
    
    This has little utility beyond debugging.
    """

    def __init__(self, context:bool = True, action:bool = False):
        """Instantiate a Sparse filter.

        Args:
            context: If True then contexts should be made sparse otherwise leave them alone.
            action: If True then actions should be made sparse otherwise leave them alone.
        """
        
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
