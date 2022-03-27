import collections.abc
import warnings

from math import isnan
from statistics import mean, median, stdev, mode
from abc import abstractmethod, ABC
from numbers import Number
from collections import defaultdict
from itertools import islice, chain
from typing import Hashable, Optional, Sequence, Union, Iterable, Dict, Any, List, Tuple, Callable, Mapping
from coba.backports import Literal

from coba            import pipes
from coba.random     import CobaRandom
from coba.exceptions import CobaException
from coba.statistics import iqr

from coba.environments.primitives import Interaction, Context, Action
from coba.environments.logged.primitives import LoggedInteraction
from coba.environments.simulated.primitives import SimulatedInteraction

class EnvironmentFilter(pipes.Filter[Iterable[Interaction],Iterable[Interaction]], ABC):
    """A filter that can be applied to an Environment."""

    @abstractmethod
    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        """Apply a filter to an Environment's interactions."""
        ...

class Identity(pipes.Identity, EnvironmentFilter):
    """Return whatever interactions are given to the filter."""
    pass

class Take(pipes.Take, EnvironmentFilter):
    """Take a fixed number of interactions from an Environment."""
    pass

class Shuffle(pipes.Shuffle, EnvironmentFilter):
    """Shuffle a sequence of Interactions in an Environment."""
    pass

class Reservoir(pipes.Reservoir, EnvironmentFilter):
    """Take a fixed number of random Interactions from an Environment."""
    pass

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

        has_sparse_zero = set()

        for interaction in train_interactions:
            for name,value in self._context_as_name_values(interaction.context):
                if isinstance(value,Number) and not isnan(value):
                    features[name].append(value)

        for interaction in train_interactions:
            if isinstance(interaction.context,dict):
                has_sparse_zero |= features.keys() - interaction.context.keys()

        for key in has_sparse_zero:
            features[key].append(0)

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

class Cycle(EnvironmentFilter):
    """Cycle all rewards associated with actions by one place.

    This filter is useful for testing an algorithms response to a non-stationary shock.
    """

    def __init__(self, after:int = 0):
        """Instantiate a Cycle filter.

        Args:
            after: How many interactions should be seen before applying the cycle filter.
        """
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

        try:
            first_interaction = next(with_cycle_interactions)

            action_set              = set(first_interaction.actions)
            n_actions               = len(action_set)
            featureless_actions     = [tuple([0]*n+[1]+[0]*(n_actions-n-1)) for n in range(n_actions)]
            with_cycle_interactions = chain([first_interaction], with_cycle_interactions)

            if len(set(action_set) & set(featureless_actions)) != len(action_set):
                warnings.warn("Cycle only works for environments without action features. It will be ignored in this case.")
                for interaction in with_cycle_interactions:
                    yield interaction
            else:
                for interaction in with_cycle_interactions:
                    kwargs = interaction.kwargs.copy()
                    kwargs['rewards'] = kwargs['rewards'][-1:] + kwargs['rewards'][:-1]
                    yield SimulatedInteraction(interaction.context, interaction.actions, **kwargs)
        except StopIteration:
            pass

class Binary(EnvironmentFilter):
    """Binarize all rewards to either 1 (max rewards) or 0 (all others)."""

    @property
    def params(self) -> Dict[str, Any]:
        return { "binary": True }

    def filter(self, interactions: Iterable[SimulatedInteraction]) -> Iterable[SimulatedInteraction]:

        for interaction in interactions:
            kwargs  = interaction.kwargs.copy()
            max_rwd = max(kwargs["rewards"])
            
            kwargs["rewards"] = [int(r==max_rwd) for r in kwargs["rewards"]]

            yield SimulatedInteraction(interaction.context, interaction.actions, **kwargs)

class Sort(EnvironmentFilter):
    """Sort a sequence of Interactions in an Environment."""

    def __init__(self, *keys: Union[str,int,Sequence[Union[str,int]]]) -> None:
        """Instantiate a Sort filter.

        Args:
            *keys: The context items that should be sorted on.
        """
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

class Where(EnvironmentFilter):
    """Define Environment selection criteria for an Environments pipe."""

    def __init__(self, *, n_interactions: Union[int,Tuple[Optional[int],Optional[int]]] = None) -> None:
        """Instantiate a Where filter.

        Args:
            n_interactions: The minimum, maximum or exact number of interactions Environments must have. 
        """

        self._n_interactions = n_interactions

    @property
    def params(self) -> Dict[str, Any]:
        params = {}

        if self._n_interactions is not None:
            params["where_n_interactions"] = self._n_interactions

        return params

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        
        interactions = iter(interactions)

        if self._n_interactions is None or self._n_interactions == (None,None):
            min_interactions  = None
            max_interactions  = None
            take_interactions = 0
        elif isinstance(self._n_interactions, int):
            min_interactions  = self._n_interactions
            max_interactions  = self._n_interactions
            take_interactions = self._n_interactions+1
        else:
            min_interactions  = self._n_interactions[0]
            max_interactions  = self._n_interactions[1]
            take_interactions = max(filter(lambda x: x is not None, list(self._n_interactions)))+1

        taken_interactions = list(islice(interactions, take_interactions))

        if max_interactions is not None and len(taken_interactions) > max_interactions:
            return []

        if min_interactions is not None and len(taken_interactions) < min_interactions:
            return []

        return chain(taken_interactions, interactions)

class WarmStart(EnvironmentFilter):
    """Turn a SimulatedEnvironment into a WarmStartEnvironment."""

    def __init__(self, n_warmstart:int, seed:int = 1):
        """Instantiate a WarmStart filter.

        Args:
            n_warmstart: The number of interactions that should be turned into LoggedInteractions.
            seed: The random number seed that determines the random logging policy for LoggedInteractions.
        """
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

class Noise(EnvironmentFilter):
    """Introduce noise to an environment."""

    def __init__(self,
        context: Callable[[CobaRandom, float], float] = None, 
        action : Callable[[CobaRandom, float], float] = None,
        reward : Callable[[CobaRandom, float], float] = None,
        seed   : int = 1) -> None:
        """Instantiate a Noise EnvironmentFilter.

        Args:
            context: A noise generator for context features.
            action: A noise generator for action features.
            reward: A noise generator for rewards.
            seed: The seed initializing the random state of the noise generators.
        """

        self._no_noise = lambda _, x: x

        if context is None and action is None and reward is None:
            context = lambda rng, x: x+rng.gauss(0,1)

        self._context_noise = context or self._no_noise
        self._action_noise  = action  or self._no_noise
        self._reward_noise  = reward  or self._no_noise
        self._seed          = seed

    @property
    def params(self) -> Dict[str, Any]:

        params = {}

        if self._context_noise != self._no_noise: params['context_noise'] = True 
        if self._action_noise != self._no_noise : params['action_noise' ] = True
        if self._reward_noise != self._no_noise : params['reward_noise' ] = True

        params['noise_seed'] = self._seed

        return params

    def filter(self, interactions: Iterable[SimulatedInteraction]) -> Iterable[SimulatedInteraction]:

        rng = CobaRandom(self._seed)

        for interaction in interactions:

            if isinstance(interaction, LoggedInteraction):
                raise CobaException("We do not currently support adding noise to a LoggedInteraction.")

            noisy_context = self._noises(interaction.context, rng, self._context_noise)
            noisy_actions = [ self._noises(a, rng, self._action_noise) for a in interaction.actions ]

            noisy_kwargs  = {}

            if 'rewards' in interaction.kwargs and self._reward_noise:
                noisy_kwargs['rewards'] = self._noises(interaction.kwargs['rewards'], rng, self._reward_noise)

            yield SimulatedInteraction(noisy_context, noisy_actions, **noisy_kwargs)

    def _noises(self, value:Union[None,float,str,Mapping,Sequence], rng: CobaRandom, noiser: Callable[[CobaRandom], float]):

        if isinstance(value, collections.abc.Mapping):
            #we sort so that noise generation is deterministic with respect to seed
            return { k:self._noise(v, rng, noiser) for k,v in sorted(value.items()) }

        if isinstance(value, collections.abc.Sequence):
            return [ self._noise(v, rng, noiser) for v in value ]

        return self._noise(value, rng, noiser)

    def _noise(self, value:Union[None,float,str], rng: CobaRandom, noiser: Callable[[CobaRandom], float]) -> float:

        return value if not isinstance(value,(int,float)) else noiser(rng,value)
