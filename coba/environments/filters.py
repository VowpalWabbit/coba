import time
import pickle
import warnings

from math import isnan
from statistics import mean, median, stdev, mode
from numbers import Number
from collections import defaultdict, deque
from functools import lru_cache
from itertools import islice, chain, tee
from typing import Hashable, Optional, Sequence, Union, Iterable, Dict, Any, List, Tuple, Callable, Mapping
from coba.backports import Literal

from coba            import pipes
from coba.random     import CobaRandom
from coba.exceptions import CobaException
from coba.statistics import iqr
from coba.utilities  import peek_first, Categorical

from coba.environments.primitives import EnvironmentFilter, Interaction, HashableMap, HashableSeq
from coba.environments.primitives import ScaleReward, BinaryReward, Feedback, SequenceReward, MappedReward

def _copy_interaction(interaction: dict, **kwargs):
    new = interaction.copy()
    new.update(**kwargs)
    return new

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
    pass

class Scale(EnvironmentFilter):
    """Shift and scale features to precondition them before learning."""

    def __init__(self,
        shift: Union[Number,Literal["min","mean","med"]] = 0,
        scale: Union[Number,Literal["minmax","std","iqr","maxabs"]] = "minmax",
        target: Literal["features","rewards","argmax"] = "features",
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

        first, interactions = peek_first(interactions)

        if not first: return []

        iter_interactions    = iter(interactions)
        fitting_interactions = list(islice(iter_interactions,self._using))

        shifts  : Dict[Hashable,float]     = defaultdict(lambda:0)
        scales  : Dict[Hashable,float]     = defaultdict(lambda:1)
        unscaled: Dict[Hashable,List[Any]] = defaultdict(list)

        first_context = first['context']
        first_actions = first.get('actions')

        if isinstance(first_context, pipes.Dense):
            context_type = 0
            is_materialized = isinstance(first_context,(list,tuple))
        elif isinstance(first_context, pipes.Sparse):
            context_type = 1
            is_materialized = isinstance(first_context,dict)
        else:
            context_type = 2
            is_materialized = True

        if context_type == 1 and self._shift != 0:
            raise CobaException("Shift is required to be 0 for sparse environments. Otherwise the environment will become dense.")

        is_discrete = first_actions and len(first_actions) > 0

        not_numeric = set()
        mixed_types = set()

        start = time.time()
        for interaction in fitting_interactions:
            if self._target == "features":
                if context_type == 0 and not is_materialized:
                    interaction['context'] = tuple(interaction['context'])
                if context_type == 1 and not is_materialized:
                    interaction['context'] = dict(interaction['context'].items())

                for key,value in self._kv_pairs(context_type,interaction['context']):
                    numeric = isinstance(value,(float,int))

                    if key in unscaled:
                        if numeric and not isnan(value):
                            unscaled[key].append(value)
                        elif not numeric:
                            del unscaled[key]
                            mixed_types.add(key)
                            not_numeric.add(key)
                    else:
                        if numeric:
                            if key in not_numeric:
                                mixed_types.add(key)
                            elif isnan(value):
                                unscaled[key] = []
                            else:
                                unscaled[key].append(value)
                        else:
                            not_numeric.add(key)

            elif self._target == "rewards" and is_discrete:
                unscaled["rewards"].extend(map(interaction['rewards'].eval,interaction['actions']))

            elif self._target == "argmax":
                unscaled["argmax"].append(interaction['rewards'].argmax())
        self._times[0] += time.time()-start

        if mixed_types: 
            warnings.warn(f"Some features were not scaled due to having mixed types: {mixed_types}. ")

        start = time.time()
        if context_type == 1:
            has_sparse_zero = set()

            for interaction in fitting_interactions:
                has_sparse_zero |= unscaled.keys() - interaction['context'].keys() - {"rewards"}

            for key in has_sparse_zero:
                unscaled[key].append(0)
        self._times[1] += time.time()-start

        start = time.time()
        for key,values in unscaled.items():
            if isinstance(self._shift, (int,float)):
                shift = self._shift
            elif self._shift == "min":
                shift = min(values)
            elif self._shift == "mean":
                shift = mean(values)
            elif self._shift == "med":
                shift = median(values)

            if isinstance(self._scale, (int,float)):
                scale_num = self._scale
                scale_den = 1
            elif self._scale == "minmax":
                scale_num = 1
                scale_den = max(values)-min(values)
            elif self._scale == "std":
                scale_num = 1
                scale_den = stdev(values)
            elif self._scale == "iqr":
                scale_num = 1
                scale_den = iqr(values)
            elif self._scale == "maxabs":
                scale_num = 1
                scale_den = max([abs(v-shift) for v in values])

            shifts[key] = shift
            scales[key] = scale_num/(round(scale_den,8) or 1)
        self._times[2] += time.time()-start

        start = time.time()
        for interaction in chain(fitting_interactions, iter_interactions):

            new = interaction.copy()

            if self._target == "features" and shifts:

                if context_type == 0:
                    new_context = list(interaction['context'])
                    for key in shifts:
                        new_context[key] = (new_context[key]-shifts[key])*scales[key]
                    new['context'] = new_context

                elif context_type == 1:
                    new_context = dict(interaction['context'])
                    for key in (new_context.keys() & shifts.keys()):
                        new_context[key] = (new_context[key]-shifts[key])*scales[key]
                    new['context'] = new_context

                elif 0 in shifts:
                    new['context'] = (interaction['context']-shifts[0])*scales[0]

            elif self._target == "rewards":
                new['rewards'] = ScaleReward(interaction['rewards'], -shifts['rewards'], scales['rewards'], 'value')

            elif self._target == "argmax":
                new['rewards'] = ScaleReward(interaction['rewards'], -shifts["argmax"], scales["argmax"], "argmax")

            yield new
        self._times[3] += time.time()-start

        #print(self._times)

    def _kv_pairs(self, context_type, context) -> Sequence[Tuple[Hashable,Any]]:
        if context_type == 0  : return enumerate(context)
        if context_type == 1  : return context.items()
        if context is not None: return [(0,context)]
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
    def params(self) -> Mapping[str, Any]:
        return { "impute_stat": self._stat, "impute_using": self._using }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        iter_interactions  = iter(interactions)
        train_interactions = list(islice(iter_interactions,self._using))
        test_interactions  = chain.from_iterable([train_interactions, iter_interactions])

        stats   : Dict[Hashable,float]        = defaultdict(int)
        features: Dict[Hashable,List[Number]] = defaultdict(list)

        for interaction in train_interactions:
            for name,value in self._context_as_name_values(interaction['context']):
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

            new     = interaction.copy()
            context = interaction['context']
            imputed = {k: stats[k] if isinstance(v,float) and isnan(v) else v for k,v in self._context_as_name_values(context)}

            if isinstance(context, pipes.Sparse):
                new['context'] = imputed
            elif isinstance(context, pipes.Dense):
                new['context'] = tuple(imputed[k] for k,_ in self._context_as_name_values(context))
            elif imputed:
                new['context'] = imputed[0]

            yield new

    def _context_as_name_values(self,context) -> Sequence[Tuple[Hashable,Any]]:

        if isinstance(context,dict ): return context.items()
        if isinstance(context,tuple): return enumerate(context)
        if context is not None      : return [(0,context)]

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
        if isinstance(value,pipes.Dense):
            return dict(enumerate(value))
        if isinstance(value,pipes.Sparse):
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

        if first:
            action_set          = set(first['actions'])
            n_actions           = len(action_set)
            featureless_actions = [tuple([0]*n+[1]+[0]*(n_actions-n-1)) for n in range(n_actions)]

            is_discrete = first['actions'] and 0 < len(first['actions']) and len(first['actions']) < float('inf')

            if not is_discrete or len(set(action_set) & set(featureless_actions)) != len(action_set):
                warnings.warn("Cycle only works for discrete environments without action features. It will be ignored in this case.")
                yield from with_cycle_interactions
            else:
                for interaction in with_cycle_interactions:
                    rewards = deque(interaction['rewards'])
                    rewards.rotate(1)
                    
                    new = interaction.copy()
                    new['rewards'] = SequenceReward(interaction['actions'],list(rewards))
                    
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
        flat_actions_iter = self._flattener.filter(a         for i in I3 for a in i['actions'])

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
        is_sparse           = isinstance(first['context'],pipes.Sparse)

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

class Warm(EnvironmentFilter):
    """Turn a SimulatedEnvironment into a WarmStartEnvironment."""

    def __init__(self, n_warm:int, seed:int = 1):
        """Instantiate a Warm filter.

        Args:
            n_warm: The number of interactions that should be turned into LoggedInteractions.
            seed: The random number seed that determines the random logging policy for LoggedInteractions.
        """
        self._n_warm = n_warm
        self._seed = seed

    @property
    def params(self) -> Mapping[str, Any]:
        return { "n_warm": self._n_warm }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        self._rng = CobaRandom(self._seed)

        underlying_iterable    = iter(interactions)
        logged_interactions    = map(self._to_logged_interaction, islice(underlying_iterable, self._n_warm))
        simulated_interactions = underlying_iterable

        return chain(logged_interactions, simulated_interactions)

    def _to_logged_interaction(self, interaction: Interaction) -> Interaction:
        new = interaction.copy()

        actions       = interaction['actions']
        rewards       = interaction['rewards']
        num_actions   = len(actions)
        probabilities = [1/num_actions] * num_actions

        idx    = self._rng.choice(list(range(num_actions)), probabilities)
        action = actions[idx]
        prob   = probabilities[idx]
        reward = rewards[idx]

        new['type'] = 'logged'
        new['action'] = action
        new['probability'] = prob
        new['reward'] = reward

        return new

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

            noisy_context = self._noises(interaction['context'], rng, self._context_noise)
            noisy_actions = [self._noises(a, rng, self._action_noise) for a in interaction['actions'] ]
            noisy_rewards = [ self._noises(r, rng, self._reward_noise) for r in interaction['rewards'] ]
            noisy_rewards = SequenceReward(noisy_actions,noisy_rewards)

            yield _copy_interaction(interaction, context=noisy_context, actions=noisy_actions, rewards=noisy_rewards)

    def _noises(self, value:Union[None,float,str,Mapping,Sequence], rng: CobaRandom, noiser: Callable[[float,CobaRandom], float]):

        if isinstance(value, pipes.Sparse):
            #we sort so that noise generation is deterministic with respect to seed
            return { k:self._noise(v, rng, noiser) for k,v in sorted(value.items()) }

        if isinstance(value, pipes.Dense):
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

        if not first: return []

        is_binary_rwd = {0,1} == set(first['rewards'])
        first_context = first['context']

        if isinstance(first_context,pipes.Sparse):
            context_type = 0
        elif isinstance(first_context,pipes.Dense):
            context_type = 1
        else:
            context_type = 2

        goods = [(g,) for g in self.goodwords]
        bads  = [(b,) for b in self.badwords ]

        for seed, interaction in enumerate(interactions, self._seed):

            userid,normal = rng.choice(userid_isnormal)
            argmax        = interaction['rewards'].argmax()

            new = interaction.copy()
            new['type'] = 'grounded'

            if not is_binary_rwd:
                new['rewards'] = BinaryReward(argmax)

            if normal:
                new['feedbacks'] = Grounded.GroundedFeedback(goods,bads,argmax,seed)
            else:
                new['feedbacks'] = Grounded.GroundedFeedback(bads,goods,argmax,seed)

            if context_type == 0:
                new['context'] = dict(userid=userid,**new['context'])
            elif context_type == 1:
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

        reward_transformer = lambda rwd: rwd
        try:
            action = first['actions'][0]
            if isinstance(action, Categorical):
                old = [ Categorical(l,action.levels) for l in action.levels ]
                new = list(pipes.EncodeCatRows(self._cat_actions).filter(old))
                fwd = dict(zip(new,old))
                inv = dict(zip(old,new))
                reward_transformer = lambda rwd: MappedReward(rwd,fwd,inv)
        except:
            pass

        I1,I2,I3 = tee(interactions,3)

        interactions      = I1
        cat_context_iter = pipes.EncodeCatRows(self._cat_context).filter( i['context'] for i in I2                       )
        cat_actions_iter = pipes.EncodeCatRows(self._cat_actions).filter( a            for i in I3 for a in i['actions'] )

        for interaction in interactions:
            
            cat_context = next(cat_context_iter)
            cat_actions = list(islice(cat_actions_iter,len(interaction['actions'])))
            rewards     = reward_transformer(interaction['rewards'])
            
            yield _copy_interaction(interaction,context=cat_context,actions=cat_actions,rewards=rewards)

class Finalize(EnvironmentFilter):

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        for interaction in Repr("onehot","onehot").filter(interactions):

            new = interaction.copy()

            if 'context' in new: new['context'] = self._make_hashable(new['context'])
            if 'actions' in new: new['actions'] = [self._make_hashable(a) for a in new['actions']]
            if 'action'  in new: new['action' ] = self._make_hashable(new['action'])

            yield new

    def _make_hashable(self, feats):
        if isinstance(feats, pipes.Dense):
            return HashableSeq(feats)
        if isinstance(feats, pipes.Sparse):
            return HashableMap(feats)
        return feats
