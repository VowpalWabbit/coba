import time
import pickle
import warnings
import copy

from math import isnan
from statistics import median, stdev, mode
from numbers import Number
from operator import eq, getitem, methodcaller, itemgetter
from collections import defaultdict, deque
from functools import lru_cache
from itertools import islice, chain, tee, compress, repeat
from typing import Optional, Sequence, Union, Iterable, Any, Tuple, Callable, Mapping
from coba.backports import Literal

from coba            import pipes, primitives
from coba.random     import CobaRandom
from coba.exceptions import CobaException
from coba.statistics import iqr
from coba.utilities  import peek_first
from coba.primitives import ScaleReward, BinaryReward, SequenceReward, BatchReward
from coba.primitives import Feedback, BatchFeedback, Categorical
from coba.learners   import Learner, SafeLearner

from coba.environments.primitives import EnvironmentFilter, Interaction

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

class Cache(pipes.Cache, EnvironmentFilter):
    """Cache all interactions that come before this filter and use the cache in the future."""

    def filter(self, items: Iterable[Interaction]) -> Iterable[Interaction]:
        yield from map(methodcaller('copy'), super().filter(items))

class Scale(EnvironmentFilter):
    """Shift and scale features to precondition them before learning."""

    def __init__(self,
        shift: Union[Number,Literal["min","mean","med"]] = 0,
        scale: Union[Number,Literal["minmax","std","iqr","maxabs"]] = "minmax",
        target: Literal["context","rewards","argmax"] = "context",
        using: Optional[int] = None):
        """Instantiate a Scale filter.

        Args:
            shift: The statistic to use to shift each context feature.
            scale: The statistic to use to scale each context feature.
            target: The target data we wish to scale in the environment.
            using: The number of interactions to use when calculating the necessary statistics.
        """

        assert isinstance(shift,Number) or shift in ["min","mean","med"]
        assert isinstance(scale,Number) or scale in ["minmax","std","iqr","maxabs"]

        self._times = [0,0,0,0]

        self._shift  = shift
        self._scale  = scale
        self._using  = using
        self._target = target

    @property
    def params(self) -> Mapping[str, Any]:
        return {
            "scale_shift": self._shift,
            "scale_scale": self._scale,
            "scale_using": self._using,
            "scale_target": self._target
        }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        first, interactions = peek_first(Mutable().filter(interactions))

        if not interactions: return []

        remaining_interactions = iter(interactions)
        fitting_interactions   = list(islice(remaining_interactions,self._using))

        first_context = first['context']
        first_actions = first.get('actions')

        if isinstance(first_context, primitives.Sparse) and self._target=="context" and self._shift != 0:
            raise CobaException("Shift is required to be 0 for sparse environments. Otherwise the environment will become dense.")

        is_discrete       = first_actions and len(first_actions) > 0
        is_dense_context  = isinstance(first_context, primitives.Dense)
        is_sparse_context = isinstance(first_context, primitives.Sparse)
        is_value_context  = not (is_dense_context or is_sparse_context)

        #get the values we wish to scale
        start = time.time()
        if self._target == "context" and is_dense_context:
            potential_cols = [i for i,v in enumerate(first['context']) if isinstance(v,(int,float))]
            if len(potential_cols) == 0:
                unscaled = []
            elif len(potential_cols) == 1:
                unscaled = [ tuple(map(itemgetter(*potential_cols),map(getitem,fitting_interactions,repeat("context")))) ]
            else:
                unscaled = list(zip(*map(itemgetter(*potential_cols),map(getitem,fitting_interactions,repeat("context")))))

        elif self._target == "context" and is_sparse_context:
            unscalable_cols = {k for k,v in first['context'].items() if not isinstance(v,(int,float))}
            unscaled = defaultdict(list)
            for interaction in fitting_interactions:
                context = interaction['context']
                for k in context.keys()-unscalable_cols:
                    unscaled[k].append(context[k])

        elif self._target == "context" and is_value_context:
            unscaled = [interaction['context'] for interaction in fitting_interactions]

        elif self._target == "rewards" and is_discrete:
            rwd_vals = lambda i: list(map(i['rewards'].eval,range(len(i['actions']))))
            unscaled = sum(list(map(rwd_vals,fitting_interactions)),[])

        elif self._target == "rewards" and not is_discrete:
            unscaled = []

        elif self._target == "argmax":
            unscaled = [interaction['rewards'].argmax() for interaction in fitting_interactions]
        self._times[1] += time.time()-start

        start = time.time()
        #determine the scale and shift values
        if self._target == "context" and is_dense_context:
            shifts_scales = []
            for i,col in zip(potential_cols,unscaled):
                shift_scale = self._get_shift_and_scale(col)
                if shift_scale is not None:
                    shifts_scales.append((i,)+shift_scale)

        elif self._target == "context" and is_sparse_context:
            shifts_scales = {}
            for k,col in unscaled.items():
                vals = col + [0]*(len(fitting_interactions)-len(col))
                shift_scale = self._get_shift_and_scale(vals)
                if shift_scale is not None:
                    shifts_scales[k] = shift_scale

        elif self._target in ['context','rewards','argmax']:
            shifts_scales = self._get_shift_and_scale(unscaled)
        self._times[2] += time.time()-start

        start = time.time()
        if not shifts_scales:
            yield from chain(fitting_interactions, remaining_interactions)

        #now scale return
        elif self._target == "context" and is_dense_context:
            for interaction in chain(fitting_interactions, remaining_interactions):
                context = interaction['context']
                for i,shift,scale in shifts_scales:
                     context[i] = (context[i]+shift)*scale
                yield interaction

        elif self._target == "context" and is_sparse_context:
            for interaction in chain(fitting_interactions, remaining_interactions):
                new = interaction # Mutable copies
                context = new['context']
                for k in shifts_scales.keys() & context.keys():
                    (shift,scale) = shifts_scales[k]
                    context[k] = (context[k]+shift)*scale
                yield new

        elif self._target == "context" and is_value_context:
            (shift,scale) = shifts_scales
            for interaction in chain(fitting_interactions, remaining_interactions):
                new = interaction.copy()
                new['context'] = (new['context']+shift)*scale
                yield new

        elif self._target == "rewards":
            (shift,scale) = shifts_scales
            for interaction in chain(fitting_interactions, remaining_interactions):
                new = interaction.copy()
                new['rewards'] = ScaleReward(new['rewards'], shift, scale, 'value')
                yield new

        elif self._target == "argmax":
            (shift,scale) = shifts_scales
            for interaction in chain(fitting_interactions, remaining_interactions):
                new = interaction.copy()
                new['rewards'] = ScaleReward(new['rewards'], shift, scale, 'argmax')
                yield new
        self._times[3] += time.time()-start

    def _get_shift_and_scale(self,values) -> Tuple[float,float]:
        try:
            shift = self._shift_value(values)
            scale = self._scale_value(values,shift)

            if isnan(shift):
                #this is a trick, nan != nan so equality will tell
                #if a value is not equal with itself then it is nan.
                #Using this trick is about 2x faster than using isnan.
                not_nan_vals = list(compress(values,map(eq,values,values)))
                shift = self._shift_value(not_nan_vals)
                scale = self._scale_value(not_nan_vals,shift)

            return shift,scale
        except TypeError:
            return None

    def _shift_value(self, values) -> float:
        shift = self._shift
        if shift == "min":
            return -min(values)
        elif shift == "mean":
            return -sum(values)/len(values) #mean() is very slow due to calculations for precision
        elif shift == "med":
            return -median(values)
        return shift

    def _scale_value(self, values, shift) -> float:
        scale     = self._scale
        scale_num = scale
        scale_den = 1

        if scale == "minmax":
            scale_num = 1
            scale_den = max(values)-min(values)
        elif scale == "std":
            scale_num = 1
            scale_den = stdev(values)
        elif scale == "iqr":
            scale_num = 1
            scale_den = iqr(values)
        elif scale == "maxabs":
            scale_num = 1
            scale_den = max([abs(v+shift) for v in values])

        return scale_num if scale_den < .000001 else scale_num/scale_den

class Impute(EnvironmentFilter):
    """Impute missing values (nan) in Interaction contexts."""

    def __init__(self,
        stat : Literal["mean","median","mode"] = "mean",
        indicator: bool = True,
        using: Optional[int] = None):
        """Instantiate an Impute filter.

        Args:
            stat: The statistic to use for impuatation.
            indicator: Indicates whether a new binary feature should be added for missingness.
            using: The number of interactions to use to calculate the imputation statistics.
        """

        assert stat in ["mean","median","mode"]

        self._stat  = stat
        self._miss  = indicator
        self._using = using
        self._times = [0,0,0,0]

    @property
    def params(self) -> Mapping[str, Any]:
        return { "impute_stat": self._stat, "impute_using": self._using, "impute_indicator":self._miss }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        first, interactions = peek_first(Mutable().filter(interactions))

        if not interactions: return None

        is_dense  = isinstance(first['context'], primitives.Dense)
        is_sparse = isinstance(first['context'], primitives.Sparse)
        is_value  = not is_dense and not is_sparse

        interactions       = iter(interactions)
        using_interactions = list(islice(interactions,self._using))
        other_interactions = interactions

        start = time.time()
        if is_dense:
            if self._stat in ["mean","median"]:
                imputable_cols = [i for i,v in enumerate(first['context']) if isinstance(v,(int,float) or v is None)]
            else:
                imputable_cols = list(range(len(first['context'])))

        elif is_sparse:
            if self._stat in ['mean','median']:
                unimputable_cols = {k for k,v in first['context'].items() if not isinstance(v,(int,float) or v is None)}
            else:
                unimputable_cols = {}

        elif is_value:
            if self._stat in ["mean","median"]:
                imputable_cols = isinstance(first['context'],(int,float)) or (first['context'] is None)
            else:
                imputable_cols = True
        self._times[0] += time.time()-start

        start = time.time()
        #get unimputed values
        if is_dense:
            if len(imputable_cols) == 0:
                unimputed = []
            elif len(imputable_cols) == 1:
                unimputed = [ tuple(map(itemgetter(*imputable_cols),map(getitem,using_interactions,repeat("context")))) ]
            else:
                unimputed = list(zip(*map(itemgetter(*imputable_cols),map(getitem,using_interactions,repeat("context")))))

        elif is_sparse:
            unimputed = defaultdict(list)
            for interaction in using_interactions:
                context = interaction['context']
                for k in context.keys()-unimputable_cols:
                    unimputed[k].append(context[k])

        elif is_value:
            unimputed = [ interaction['context'] for interaction in using_interactions ]
        self._times[1] += time.time()-start

        start = time.time()
        #calculate imputation statistics
        if is_dense:
            imputations = {}
            impute_binary = {}
            for i, col in zip(imputable_cols,unimputed):
                imputation = self._get_imputation(col)
                if imputation is not None:
                    imputations[i] = imputation
                    if self._miss and any([c is None for c in col]): 
                        impute_binary[i] = len(impute_binary)

        elif is_sparse:
            imputations = {}
            impute_binary = {}
            binary_template = {}
            for k,col in unimputed.items():
                imputation = self._get_imputation(col + [0]*(len(using_interactions)-len(col)))
                if imputation is not None:
                    imputations[k] = imputation
                    if self._miss and any([c is None for c in col]):
                        impute_binary[k] = f"{k}_is_missing"
                        binary_template[f"{k}_is_missing"] = 0

        elif is_value:
            imputations = self._get_imputation(unimputed)
            impute_binary = self._miss and any([c is None for c in unimputed])
        self._times[2] += time.time()-start

        start = time.time()
        for interaction in chain(using_interactions,other_interactions):

            context = interaction['context']

            if is_dense:
                is_missing = [0]*len(impute_binary)
                for k,v in enumerate(context):
                    if v is None and k in imputations: 
                        context[k] = imputations[k]
                        if k in impute_binary: 
                            is_missing[impute_binary[k]] = 1
                context += is_missing

            elif is_sparse:

                is_missing = binary_template.copy()
                for k,v in context.items():
                    if v is None: 
                        context[k] = imputations[k]
                        if k in impute_binary:
                            is_missing[impute_binary[k]] = 1
                context.update(is_missing)

            elif is_value:
                if impute_binary:
                    if context is None: 
                        interaction["context"] = [imputations,1]
                    else:
                        interaction["context"] = [context,0]
                else:
                    if context is None:
                        interaction["context"] = imputations

            yield interaction
        self._times[3] += time.time()-start

    def _get_imputation(self,values):
        try:
            values = [v for v in values if v is not None]
            if self._stat == "mean":
                return sum(values)/len(values)
            if self._stat == "median":
                return median(values)
            if self._stat == "mode":
                return mode(values)
        except:
            return None

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
    def params(self) -> Mapping[str, Any]:
        return { "sparse_C": self._context, "sparse_A": self._action }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        for interaction in interactions:

            new = interaction.copy()            

            if self._context:
                new['context'] = self._make_sparse(new['context'])

            if self._action and 'actions' in new:
                new['actions'] = list(map(self._make_sparse,new['actions']))

            if self._action and 'action' in new:
                new['action'] = self._make_sparse(new['action'])

            yield new

    def _make_sparse(self, value) -> Optional[dict]:
        if value is None:
            return value
        if isinstance(value,primitives.Dense):
            return dict(enumerate(value))
        if isinstance(value,primitives.Sparse):
            return value
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
    def params(self) -> Mapping[str, Any]:
        return { "cycle_after": self._after }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        underlying_iterable     = iter(interactions)
        sans_cycle_interactions = islice(underlying_iterable, self._after)
        with_cycle_interactions = underlying_iterable

        yield from sans_cycle_interactions

        first, with_cycle_interactions = peek_first(with_cycle_interactions)

        if with_cycle_interactions:

            action_set = set(first['actions'])
            n_actions  = len(action_set)

            onehot_actions  = {tuple([0]*n+[1]+[0]*(n_actions-n-1)) for n in range(n_actions)}
            is_discrete     = 0 < n_actions and n_actions < float('inf')
            is_onehots      = action_set == onehot_actions
            is_categoricals = all([isinstance(a,str) for a in action_set])

            if not is_discrete or (not is_onehots and not is_categoricals):
                warnings.warn("Cycle only works for discrete environments without action features. It will be ignored in this case.")
                yield from with_cycle_interactions
            else:
                for interaction in with_cycle_interactions:
                    rewards = deque(interaction['rewards'])
                    rewards.rotate(1)

                    new = interaction.copy()
                    new['rewards'] = SequenceReward(list(rewards))

                    yield new

class Flatten(EnvironmentFilter):
    """Flatten the context and action features for interactions."""

    def __init__(self):
        self._flattener = pipes.Flatten()

    @property
    def params(self) -> Mapping[str, Any]:
        return { "flat": True }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        I1,I2,I3 = tee(interactions,3)

        interactions      = I1
        flat_context_iter = self._flattener.filter(i['context'] for i in I2                   )
        flat_actions_iter = self._flattener.filter(a            for i in I3 for a in i['actions'])

        for interaction in interactions:
            new = interaction.copy()
            new['context'] = next(flat_context_iter)
            new['actions'] = list(islice(flat_actions_iter,len(interaction['actions'])))
            yield new

class Binary(EnvironmentFilter):
    """Binarize all rewards to either 1 (max rewards) or 0 (all others)."""

    @property
    def params(self) -> Mapping[str, Any]:
        return { "binary": True }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        for interaction in interactions:
            new = interaction.copy()
            new['rewards'] = BinaryReward(interaction['rewards'].argmax())
            yield new

class Sort(EnvironmentFilter):
    """Sort a sequence of Interactions in an Environment."""

    def __init__(self, *keys: Union[str,int,Sequence[Union[str,int]]]) -> None:
        """Instantiate a Sort filter.

        Args:
            *keys: The context items that should be sorted on.
        """

        self._keys = list(pipes.Flatten().filter([list(keys)]))[0]

    @property
    def params(self) -> Mapping[str, Any]:
        return { "sort": self._keys or '*' }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        full_sorter = lambda interaction: tuple(interaction['context']                                 )
        list_sorter = lambda interaction: tuple(interaction['context'][key]       for key in self._keys)
        dict_sorter = lambda interaction: tuple(interaction['context'].get(key,0) for key in self._keys)

        first, interactions = peek_first(interactions)
        is_sparse           = isinstance(first['context'],primitives.Sparse)

        sorter = full_sorter if not self._keys else dict_sorter if is_sparse else list_sorter

        return sorted(interactions, key=sorter)

class Where(EnvironmentFilter):
    """Define Environment selection criteria for an Environments pipe."""

    def __init__(self, *, n_interactions: Union[int,Tuple[Optional[int],Optional[int]]] = None) -> None:
        """Instantiate a Where filter.

        Args:
            n_interactions: The minimum, maximum or exact number of interactions Environments must have.
        """

        self._n_interactions = n_interactions

    @property
    def params(self) -> Mapping[str, Any]:
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

class Riffle(EnvironmentFilter):
    """Riffle shuffle Interactions by taking actions from the end and evenly distributing into the beginning."""

    def __init__(self, spacing: int = 3, seed=1) -> None:
        """Instantiate a Riffle filter.

        Args:
            spacing: The number of interactions from the beginning between each interaction shuffled in from the end.
            seed: The seed used to determine the location of each ending interaction when placed within its beginning space.
        """
        self._spacing = spacing
        self._seed = seed

    @property
    def params(self) -> Mapping[str, Any]:
        return {"riffle_spacing": self._spacing, "riffle_seed": self._seed}

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        rng          = CobaRandom(self._seed)
        interactions = list(interactions)

        for i in range(int(len(interactions)/(self._spacing+1))):
            interactions.insert(i*self._spacing+rng.randint(0,self._spacing), interactions.pop())

        return interactions

class Noise(EnvironmentFilter):
    """Introduce noise to an environment."""

    def __init__(self,
        context: Callable[[float,CobaRandom], float] = None,
        action : Callable[[float,CobaRandom], float] = None,
        reward : Callable[[float,CobaRandom], float] = None,
        seed   : int = 1) -> None:
        """Instantiate a Noise EnvironmentFilter.

        Args:
            context: A noise generator for context features.
            action : A noise generator for action features.
            reward : A noise generator for rewards.
            seed   : The seed initializing the random state of the noise generators.
        """

        self._args = (context,action,reward,seed)
        self._no_noise = lambda x, _: x

        if context is None and action is None and reward is None:
            context = lambda x, rng: x+rng.gauss(0,1)

        self._context_noise = context or self._no_noise
        self._action_noise  = action  or self._no_noise
        self._reward_noise  = reward  or self._no_noise
        self._seed          = seed

    def __reduce__(self) -> tuple:
        try:
            pickle.dumps(self._args)
        except Exception:
            message = (
                "We were unable to pickle the Noise filter. This is likely due to using lambda functions for noise generation. "
                "To work around this we recommend you first define your lambda functions as a named function and then pass the "
                "named function to Noise."
            )
            raise CobaException(message)
        else:
            return (Noise, self._args)

    @property
    def params(self) -> Mapping[str, Any]:

        params = {}

        if self._context_noise != self._no_noise: params['context_noise'] = True
        if self._action_noise != self._no_noise : params['action_noise' ] = True
        if self._reward_noise != self._no_noise : params['reward_noise' ] = True

        params['noise_seed'] = self._seed

        return params

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        rng = CobaRandom(self._seed)

        for interaction in interactions:
            new = interaction.copy()

            noisy_context = self._noises(interaction['context'], rng, self._context_noise)
            noisy_actions = [self._noises(a, rng, self._action_noise) for a in interaction['actions'] ]
            noisy_rewards = [ self._noises(r, rng, self._reward_noise) for r in interaction['rewards'] ]
            noisy_rewards = SequenceReward(noisy_rewards)

            new['context'] = noisy_context
            new['actions'] = noisy_actions
            new['rewards'] = noisy_rewards

            yield new

    def _noises(self, value:Union[None,float,str,Mapping,Sequence], rng: CobaRandom, noiser: Callable[[float,CobaRandom], float]):

        if isinstance(value, primitives.Sparse):
            #we sort so that noise generation is deterministic with respect to seed
            return { k:self._noise(v, rng, noiser) for k,v in sorted(value.items()) }

        if isinstance(value, primitives.Dense):
            return [ self._noise(v, rng, noiser) for v in value ]

        return self._noise(value, rng, noiser)

    def _noise(self, value:Union[None,float,str], rng: CobaRandom, noiser: Callable[[float,CobaRandom], float]) -> float:

        return value if not isinstance(value,(int,float)) else noiser(value, rng)

class Params(EnvironmentFilter):
    """Add parameters to an environment."""
    def __init__(self, params: Mapping[str, Any]) -> None:
        self._params = params

    @property
    def params(self) -> Mapping[str, Any]:
        return self._params

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        return interactions

class Grounded(EnvironmentFilter):

    class GroundedFeedback(Feedback):
        def __init__(self, goods, bads, argmax, seed):
            self._rng    = None
            self._goods  = goods
            self._bads   = bads
            self._seed   = seed
            self._argmax = argmax

        @lru_cache(maxsize=None)
        def eval(self, arg):
            if not self._rng: self._rng = CobaRandom(self._seed)
            if arg == self._argmax:
                return self._rng.choice(self._goods) 
            else:
                return self._rng.choice(self._bads) 

    def __init__(self, n_users: int, n_normal:int, n_words:int, n_good:int, seed:int) -> None:
        self._n_users  = self._cast_int(n_users , "n_users" )
        self._n_normal = self._cast_int(n_normal, "n_normal")
        self._n_words  = self._cast_int(n_words , "n_words" )
        self._n_good   = self._cast_int(n_good  , "n_good"  )
        self._seed     = seed

        if n_normal > n_users:
            raise CobaException("Igl conversion can't have more normal users (n_normal) than total users (n_users).")

        if n_good > n_words:
            raise CobaException("Igl conversion can't have more good words (n_good) than total words (n_words).")

        self.userids   = list(range(self._n_users))
        self.normalids = self.userids[:self._n_normal]
        self.wordids   = list(range(self._n_words))
        self.goodwords = self.wordids[:self._n_good]
        self.badwords  = self.wordids[self._n_good:]

    @property
    def params(self) -> Mapping[str, Any]:
        return {
            "n_users"  : self._n_users,
            "n_normal" : self._n_normal,
            "n_good"   : self._n_good,
            "n_words"  : self._n_words,
            "igl_seed" : self._seed
        }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        rng = CobaRandom(self._seed)

        #we make it a set for faster contains checks
        normalids       = set(self.normalids) 
        normal          = [u in normalids for u in self.userids]
        userid_isnormal = list(zip(self.userids,normal))

        first,interactions = peek_first(interactions)

        if not interactions: return []

        is_binary_rwd = {0,1} == set(first['rewards'])
        first_context = first['context']

        is_sparse = isinstance(first_context,primitives.Sparse)
        is_dense  = isinstance(first_context,primitives.Dense)

        goods = [(g,) for g in self.goodwords]
        bads  = [(b,) for b in self.badwords ]

        for seed, interaction in enumerate(interactions, self._seed):

            userid,normal = rng.choice(userid_isnormal)
            argmax        = interaction['rewards'].argmax()

            new = interaction.copy()

            if not is_binary_rwd:
                new['rewards'] = BinaryReward(argmax)

            if normal:
                new['feedbacks'] = Grounded.GroundedFeedback(goods,bads,argmax,seed)
            else:
                new['feedbacks'] = Grounded.GroundedFeedback(bads,goods,argmax,seed)

            if is_sparse:
                new['context'] = dict(userid=userid,**new['context'])
            elif is_dense:
                new['context'] = (userid,)+tuple(new['context'])
            else:
                new['context'] = (userid, new['context'])

            new['userid'  ] = userid
            new['isnormal'] = normal

            yield new

    def _cast_int(self, value:Union[float,int], value_name:str) -> int:
        if isinstance(value, int): return value
        if isinstance(value, float) and value.is_integer(): return int(value)
        raise CobaException(f"{value_name} must be a whole number and not {value}.")

class Repr(EnvironmentFilter):
    def __init__(self, 
        cat_context:Literal["onehot","onehot_tuple","string"] = None,
        cat_actions:Literal["onehot","onehot_tuple","string"] = None) -> None:
        self._cat_context = cat_context
        self._cat_actions = cat_actions

    @property
    def params(self) -> Mapping[str, Any]:
        return {"cat_context": self._cat_context, "cat_actions": self._cat_actions}

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        first, interactions = peek_first(interactions)

        if not interactions: return []

        has_actions = 'actions' in first

        I = tee(interactions, 3 if has_actions else 2)

        interactions     = I[0]
        cat_context_iter = pipes.EncodeCatRows(self._cat_context).filter(i['context'] for i in I[1])

        if has_actions:
            cat_actions_iter = pipes.EncodeCatRows(self._cat_actions, True).filter(action for i in I[2] for action in i['actions'])

        for interaction in interactions:
            new = interaction.copy()
            new['context'] = next(cat_context_iter)

            if has_actions:
                new['actions'] = list(islice(cat_actions_iter,len(interaction['actions'])))

            yield new

class Batch(EnvironmentFilter):

    def __init__(self, batch_size: int) -> None:
        self._batch_size = batch_size

    @property
    def params(self) -> Mapping[str,Any]:
        return {'batched': self._batch_size}

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        for batch in self._batched(interactions, self._batch_size):
            new = { k: primitives.Batch(i[k] for i in batch) for k in batch[0] }
            if 'rewards' in new: new['rewards'] = BatchReward(new['rewards'])
            if 'feedbacks' in new: new['feedbacks'] = BatchFeedback(new['feedbacks'])
            yield new

    def _batched(self, iterable, n):
        "Batch data into lists of length n. The last batch may be shorter."
        #taken from python itertools recipes
        it = iter(iterable)
        batch = list(islice(it, n))
        while (batch):
            yield batch
            batch = list(islice(it, n))

class Unbatch(EnvironmentFilter):

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        first, interactions = peek_first(interactions)
        if not interactions: return []
        is_batched = isinstance(first[list(first.keys())[0]], primitives.Batch)

        if not is_batched:
            yield from interactions
        else:
            yield from self._unbatch(interactions)

    def _unbatch(self, interactions: Iterable[Interaction]):
        for interaction in interactions:
            batch_size = len(interaction[list(interaction.keys())[0]])
            for i in range(batch_size):
                yield { k: interaction[k][i] for k in interaction }

class BatchSafe(EnvironmentFilter):

    def __init__(self, filter: EnvironmentFilter) -> None:
        self._filter = filter

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        first, interactions = peek_first(interactions)

        if not interactions: return []

        is_batched = isinstance(first[list(first.keys())[0]], primitives.Batch)

        if not is_batched:
            yield from self._filter.filter(interactions)
        else:
            batch_size = len(first[list(first.keys())[0]])
            unbatched = Unbatch().filter(interactions)
            filtered  = self._filter.filter(unbatched)
            rebatched = Batch(batch_size).filter(filtered)
            yield from rebatched

class Finalize(EnvironmentFilter):

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        first, interactions = peek_first(interactions)

        if not interactions: return []

        first_has_context = 'context' in first
        first_has_actions = 'actions' in first and first['actions']
        first_has_action  = 'action'  in first

        context_materialized = not first_has_context or isinstance(first['context']   ,(list,tuple,dict))
        actions_materialized = not first_has_actions or isinstance(first['actions'][0],(list,tuple,dict))
        action_materialized  = not first_has_action  or isinstance(first['action' ]   ,(list,tuple,dict))

        if first_has_context:
            is_dense_context  = isinstance(first['context'],primitives.Dense)
            is_sparse_context = isinstance(first['context'],primitives.Sparse)
        if first_has_actions:
            is_dense_action  = isinstance(first['actions'][0],primitives.Dense)
            is_sparse_action = isinstance(first['actions'][0],primitives.Sparse)
        elif first_has_action:
            is_dense_action  = isinstance(first['action'],primitives.Dense)
            is_sparse_action = isinstance(first['action'],primitives.Sparse)

        for interaction in Repr("onehot","onehot").filter(interactions):

            new = interaction.copy()

            if not context_materialized and is_dense_context:
                new['context'] = list(new['context'])
            elif not context_materialized and is_sparse_context :
                    new['context'] = new['context'].copy()

            if not actions_materialized and is_dense_action:
                new['actions'] = list(map(list,new['actions']))
            elif not actions_materialized and is_sparse_action:
                new['actions'] = list(map(methodcaller('copy'),new['actions']))

            if not action_materialized and is_dense_action:
                new['action'] = list(new['action'])
            elif not action_materialized and is_sparse_action:
                new['action'] = new['action'].copy()

            yield new

class Chunk(EnvironmentFilter):
    """A placeholder filter that exists only to semantically indicate how an environment pipe should be chunked for processing."""
    def filter(self, items: Iterable[Interaction]) -> Iterable[Interaction]:
        return items

class Logged(EnvironmentFilter):

    def __init__(self, learner: Learner) -> None:
        self._learner = learner

    @property
    def params(self) -> Mapping[str, Any]:
        return {"learner": SafeLearner(self._learner).params, "logged":True}

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        learner = SafeLearner(copy.deepcopy(self._learner))

        first, interactions = peek_first(interactions)

        predict = learner.predict
        learn   = learner.learn

        if 'context' not in first or 'actions' not in first or 'rewards' not in first:
            raise CobaException("We were unable to create a logged representation of the interaction.")

        for interaction in interactions:

            context = interaction['context']
            actions = interaction['actions']
            rewards = interaction['rewards']

            action,prob,info = predict(context, actions)

            reward = rewards.eval(action)

            learn(context, actions, action, reward, prob, **info)

            new = interaction.copy()

            new['action']      = action
            new['probability'] = prob
            new['reward']      = reward

            yield new

class Mutable(EnvironmentFilter):

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        
        first, interactions = peek_first(interactions)
        
        if not interactions: return []

        first_is_dense   = isinstance(first['context'], primitives.Dense)
        first_is_sparse  = isinstance(first['context'], primitives.Sparse)
        first_is_value   = not (first_is_dense or first_is_sparse)
        first_is_mutable = isinstance(first['context'], (list,dict))

        if first_is_mutable:
            for interaction in interactions:
                new = interaction.copy()
                new['context'] = new['context'].copy()
                yield new
                            
        elif first_is_dense:
            for interaction in interactions:
                new = interaction.copy()
                new['context'] = list(new['context'])
                yield new
        
        elif first_is_sparse:
            for interaction in interactions:
                new = interaction.copy()
                new['context'] = dict(new['context'].items())
                yield new

        elif first_is_value:
            for interaction in interactions:
                yield interaction.copy()
