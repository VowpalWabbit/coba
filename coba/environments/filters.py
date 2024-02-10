import time
import pickle
import warnings
import copy

from math import isnan
from statistics import median, stdev, mode
from numbers import Number
from zlib import crc32
from operator import eq, methodcaller, itemgetter
from collections import defaultdict
from functools import lru_cache
from itertools import islice, chain, tee, compress, repeat
from typing import Optional, Sequence, Union, Iterable, Any, Tuple, Callable, Mapping, Literal

from coba              import pipes, primitives
from coba.context      import CobaContext
from coba.random       import CobaRandom
from coba.exceptions   import CobaException
from coba.statistics   import iqr
from coba.utilities    import peek_first, PackageChecker, try_else, minimize
from coba.primitives   import is_batch, Learner, Interaction, EnvironmentFilter, BinaryReward, DiscreteReward
from coba.pipes        import Pipes, SparseDense
from coba.safety       import SafeLearner

class Identity(pipes.Identity, EnvironmentFilter):
    """Return interactions unchanged."""
    pass

class Take(pipes.Take, EnvironmentFilter):
    """Take the first n interactions."""
    pass

class Slice(pipes.Slice, EnvironmentFilter):
    """Take a slice of interactions."""
    pass

class Shuffle(pipes.Shuffle, EnvironmentFilter):
    """Shuffle interaction order."""

    def filter(self, interactions: Iterable[Interaction]) -> Sequence[Interaction]:
        first, interactions = peek_first(interactions)

        if not interactions:
            yield from []

        elif 'action' in first and 'reward' in first:
            #this is here because if it is not offpolicy evaluation can give a
            #very biased estimate when seeds are the same. To see this run.
                # import numpy as np
                # from coba import CobaRandom

                # N    = 1_000_000
                # seed = 9

                # R1 = CobaRandom(seed).randoms(N)
                # R2 = CobaRandom(seed).shuffle(R1)

                # np.corrcoef(R1,R2)

            old_seed = self._seed
            new_seed = self._seed * 3.21 if self._seed is not None else self._seed

            self._seed = new_seed
            yield from super().filter(interactions)
            self._seed = old_seed

        else:
            yield from super().filter(interactions)

class Reservoir(pipes.Reservoir, EnvironmentFilter):
    """Take n random interactions."""
    pass

class Cache(pipes.Cache, EnvironmentFilter):
    """Cache given interactions."""

    def filter(self, items: Iterable[Interaction]) -> Iterable[Interaction]:
        yield from map(methodcaller('copy'), super().filter(items))

class Scale(EnvironmentFilter):
    """Scale and shift features."""

    def __init__(self,
        shift: Union[Number,Literal["min","mean","median"]] = 0,
        scale: Union[Number,Literal["minmax","std","iqr","maxabs"]] = "minmax",
        target: Literal["context"] = "context",
        using: Optional[int] = None):
        """Instantiate a Scale filter.

        Args:
            shift: The statistic to use to shift each context feature.
            scale: The statistic to use to scale each context feature.
            target: The target data we wish to scale in the environment.
            using: The number of interactions to use when calculating the necessary statistics.
        """

        assert isinstance(shift,Number) or shift in ["min","mean","med","median"]
        assert isinstance(scale,Number) or scale in ["minmax","std","iqr","maxabs"]

        self._times = [0,0,0,0]

        self._shift  = shift
        self._scale  = scale
        self._using  = using
        self._target = target

    @property
    def params(self) -> Mapping[str, Any]:
        return {
            "shift": self._shift,
            "scale": self._scale,
            "scale_using": self._using,
        }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        first, interactions = peek_first(Mutable().filter(interactions))

        if not interactions:
            return

        if 'context' not in first:
            yield from interactions
            return

        remaining_interactions = iter(interactions)
        fitting_interactions   = list(islice(remaining_interactions,self._using))

        first_context = first['context']

        if isinstance(first_context, primitives.Sparse) and self._target=="context" and self._shift != 0:
            raise CobaException("Shift is required to be 0 for sparse environments. Otherwise the environment will become dense.")

        is_dense_context  = isinstance(first_context, primitives.Dense)
        is_sparse_context = isinstance(first_context, primitives.Sparse)
        is_value_context  = not (is_dense_context or is_sparse_context)

        #get the values we wish to scale
        if self._target == "context" :
            if is_dense_context:
                potential_cols = [i for i,v in enumerate(first['context']) if isinstance(v,(int,float))]
                if len(potential_cols) == 0:
                    unscaled = []
                elif len(potential_cols) == 1:
                    unscaled = [ tuple(map(itemgetter(*potential_cols),map(itemgetter("context"),fitting_interactions))) ]
                else:
                    unscaled = list(zip(*map(itemgetter(*potential_cols),map(itemgetter("context"),fitting_interactions))))
            elif is_sparse_context:
                unscalable_cols = {k for k,v in first['context'].items() if not isinstance(v,(int,float))}
                unscaled = defaultdict(list)
                for interaction in fitting_interactions:
                    context = interaction['context']
                    for k in context.keys()-unscalable_cols:
                        unscaled[k].append(context[k])
            elif is_value_context:
                unscaled = [interaction['context'] for interaction in fitting_interactions]

        #determine the scale and shift values
        if self._target == "context":
            if is_dense_context:
                shifts_scales = []
                for i,col in zip(potential_cols,unscaled):
                    shift_scale = self._get_shift_and_scale(col)
                    if shift_scale is not None:
                        shifts_scales.append((i,)+shift_scale)
            elif is_sparse_context:
                shifts_scales = {}
                for k,col in unscaled.items():
                    vals = col + [0]*(len(fitting_interactions)-len(col))
                    shift_scale = self._get_shift_and_scale(vals)
                    if shift_scale is not None:
                        shifts_scales[k] = shift_scale
            elif is_value_context:
                shifts_scales = self._get_shift_and_scale(unscaled)

        #now scale
        if not shifts_scales:
            yield from chain(fitting_interactions, remaining_interactions)
        elif self._target == "context":
            if is_dense_context:
                for interaction in chain(fitting_interactions, remaining_interactions):
                    context = interaction['context']
                    for i,shift,scale in shifts_scales:
                        context[i] = (context[i]+shift)*scale
                    yield interaction
            elif is_sparse_context:
                for interaction in chain(fitting_interactions, remaining_interactions):
                    new = interaction # Mutable copies
                    context = new['context']
                    for k in shifts_scales.keys() & context.keys():
                        (shift,scale) = shifts_scales[k]
                        context[k] = (context[k]+shift)*scale
                    yield new
            elif is_value_context:
                (shift,scale) = shifts_scales
                for interaction in chain(fitting_interactions, remaining_interactions):
                    new = interaction.copy()
                    if new['context'] is not None:
                        new['context'] = (new['context']+shift)*scale
                    yield new

    def _get_shift_and_scale(self,values) -> Tuple[float,float]:
        try:
            values = [v for v in values if v is not None]
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
        except (TypeError,ValueError):
            return None

    def _shift_value(self, values) -> float:
        shift = self._shift
        if shift == "min":
            return -min(values)
        elif shift == "mean":
            return -sum(values)/len(values) #mean() is very slow due to calculations for precision
        elif shift == "med" or shift == "median":
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
        """Impute missing data.

        Args:
            stats: The statistic to use for imputation.
            indicator: Indicates whether a binary feature should be added for missingness.
            using: The number of interactions to use to calculate imputation statistics.
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

        if not interactions:
            return

        if 'context' not in first:
            yield from interactions
            return

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
                unimputed = [ tuple(map(itemgetter(*imputable_cols),map(itemgetter("context"),using_interactions))) ]
            else:
                unimputed = list(zip(*map(itemgetter(*imputable_cols),map(itemgetter("context"),using_interactions))))

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

class Sparsify(EnvironmentFilter):
    """Ensure that features are sparse."""

    def __init__(self, context:bool = True, action:bool = False):
        """Instantiate a Sparsify filter.

        Args:
            context: If True then contexts should be made sparse otherwise leave them alone.
            action: If True then actions should be made sparse otherwise leave them alone.
        """

        self._context = context
        self._action  = action

    @property
    def params(self) -> Mapping[str, Any]:
        return { "sparse_c": self._context, "sparse_a": self._action }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        first,interactions = peek_first(interactions)

        context_has_headers = 'context' in first and hasattr(first['context']   ,'headers')
        actions_has_headers = 'actions' in first and hasattr(first['actions'][0],'headers')
        action_has_headers  = 'action'  in first and hasattr(first['action' ]   ,'headers')

        for interaction in interactions:

            new = interaction.copy()

            if self._context and 'context' in new:
                new['context'] = self._make_sparse(new['context'], context_has_headers, 'context')

            if self._action and 'actions' in new:
                new['actions'] = list(map(self._make_sparse,new['actions'],repeat(actions_has_headers),repeat('action')))

            if self._action and 'action' in new:
                new['action'] = self._make_sparse(new['action'],action_has_headers,'action')

            yield new

    def _make_sparse(self, value, has_headers:bool, default_header:str) -> Optional[dict]:
        if value is None:
            return value

        if isinstance(value,primitives.Dense):
            if has_headers:
                value_list = list(value)
                return {k:value_list[i] for k,i in value.headers.items() if i < len(value_list) and value_list[i] != 0}
            else:
                return {str(k):v for k,v in enumerate(value) if v != 0 }

        if isinstance(value,primitives.Sparse):
            return value

        return {default_header:value}

class Densify(EnvironmentFilter):
    """Ensure that features are dense."""

    def __init__(self,
        n_feats: int = 400,
        method : Literal['lookup','hashing'] = 'lookup',
        context: bool = True,
        action : bool = False):
        """Instantiate a Densify filter.

        Args:
            n_feats: The number of features densified environment should have.
            method: How sparse features are turned into dense features. The hashing
                trick is more memory efficient but may have collisions. The lookup
                method is less memory efficient but guaranteed to have no collisions.
            context: If True then contexts should be made sparse otherwise leave them alone.
            action: If True then actions should be made sparse otherwise leave them alone.

            Remarks:
                The "lookup" method produces a more efficient feature representation but its memory
                usage expand to the cardinality of the sparse features. The "hashing" method utilizes
                a less efficient feature representation but does not use any memory beyond the features.
                When using the 'hashing' method `n_feats` should be set to a large number such as 2**16.
        """

        self._n_feats = n_feats
        self._context = context
        self._action  = action
        self._method  = method

        def generator():
            rng = CobaRandom(seed=1)
            while True:
                for i in rng.shuffle(range(n_feats)):
                    yield i

        def factory(g=generator()):
            return next(g)

        self._lookup = defaultdict(factory)

    @property
    def params(self) -> Mapping[str, Any]:
        return { "dense_m": self._method, "dense_n": self._n_feats, "dense_c": self._context, "dense_a": self._action }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        for interaction in interactions:

            new = interaction.copy()

            if self._context and 'context' in new:
                new['context'] = self._make_dense(new['context'])

            if self._action and 'actions' in new:
                new['actions'] = list(map(self._make_dense,new['actions']))

            if self._action and 'action' in new:
                new['action'] = self._make_dense(new['action'])

            yield new

    def _make_dense(self, value) -> Optional[dict]:

        #For this conversion we use the hashing trick. Based on many tests we go with crc32:
        #   > It is fast
        #   > It has few collisions
        #   > It is deterministic between runs

        # Hashing Algorithms We Tested:
        #   from hashlib import md5, sha1, sha256, sha3_128, sha3_256, shake_128, shake_256, blake2b
        #   from zlib import adler32, crc32
        #   from mmh3 import hash as mmh3_hash #(aka, MurmurHash3, requires pip install mmh3)
        #   from builtins import hash

        # Performance Findings:
            # hashlib algorithms took between 0.5 to 0.75 seconds to produce 1 million indexes
            # zlib and mmh3 algorithms took about .16 seconds to produce 1 million indexes
            # builtin hash algorithm took about .09 seconds to produce 1 million indexes

        # Collision Findings:
            # we used the code below to count collisions for 1,000,000 randomly generated strings:
                #import secrets
                #n=10**6
                #1-len(set([<hash-algo>(secrets.token_bytes(7)) for _ in range(n)]))/n

            # hashlib and the builtin hash had 0 collisions
            # mmh3 and crc32 had ~130 collisions with 1 million items (i.e., 0.013%)
            # adler32 had ~250k collisions (i.e., 24.00%)

        #Final decisions:
            # crc32:
            #   > It is considerably faster than the hashlib algorithms
            #   > It has equal runtime and collisions to MurmurHash3
            #   > It is deterministic across runs (unlike the built-in hash)

        if not isinstance(value,primitives.Sparse):
            return value

        else:
            if self._method == 'lookup':
                values = { self._lookup[k]:v for k,v in value.items() }
            else:
                values = { crc32(k.encode('ascii')) % self._n_feats: v for k,v in value.items() }

            return SparseDense(values, self._n_feats)

class Cycle(EnvironmentFilter):
    """Cycle reward values.

    Useful for testing a learner's response to a non-stationary shock.
    """

    def __init__(self, after:int = 0):
        """Instantiate a Cycle filter.

        Args:
            after: The number of interactions to wait before cycling reward values.
        """
        self._after = after

    @property
    def params(self) -> Mapping[str, Any]:
        return { "cycle_after": self._after }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        first, interactions = peek_first(interactions)

        has_actions   = first and 'actions'   in first
        has_rewards   = first and 'rewards'   in first
        has_feedbacks = first and 'feedbacks' in first

        targets = []
        if has_rewards  : targets.append(['rewards'  ,callable(first['rewards'  ])])
        if has_feedbacks: targets.append(['feedbacks',callable(first['feedbacks'])])

        one_hot = lambda n,N: tuple([0]*n+[1]+[0]*(N-n-1))
        rotate  = lambda l: l[-1%n_actions:] + l[:-1%n_actions]

        n_actions      = len(first['actions']) if has_actions else 0
        is_discrete    = 0 < n_actions and n_actions < float('inf')
        onehot_actions = set(map(one_hot,range(n_actions),repeat(n_actions)))
        is_onehot      = has_actions and set(first['actions']) == onehot_actions
        is_categorical = has_actions and all(map(isinstance,first['actions'],repeat(str)))

        is_cyclable = has_actions and has_rewards and is_discrete and (is_onehot or is_categorical)

        if not is_cyclable:
            warnings.warn("Cycle only works for discrete environments without action features. It will be ignored in this case.")
            yield from interactions

        else:
            for i,old in enumerate(interactions):
                new = old.copy()
                if i >= self._after:
                    for target,is_callable in targets:
                        if not is_callable:
                            new[target] = rotate(new[target])
                        else:
                            new[target] = DiscreteReward(old['actions'],rotate(list(map(old[target],old['actions']))))
                yield new

class Flatten(EnvironmentFilter):
    """Flatten contexts and actions."""

    def __init__(self):
        self._flattener = pipes.Flatten()

    @property
    def params(self) -> Mapping[str, Any]:
        return { "flat": True }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        first,interactions = peek_first(interactions)

        has_context = first and 'context' in first
        has_actions = first and 'actions' in first

        if not interactions:
            return

        if not has_context and not has_actions:
            yield from interactions
            return

        I1,I2,I3 = tee(interactions,3)

        interactions = I1

        if has_context: flat_context_iter = self._flattener.filter(i['context'] for i in I2                      )
        if has_actions: flat_actions_iter = self._flattener.filter(a            for i in I3 for a in i['actions'])

        targets = []
        if callable(first.get('rewards'))  : targets.append('rewards')
        if callable(first.get('feedbacks')): targets.append('feedbacks')

        for old in interactions:
            new = old.copy()

            if has_context:
                new['context'] = next(flat_context_iter)

            if has_actions:
                new['actions'] = list(islice(flat_actions_iter,len(old['actions'])))

                if targets and new['actions'] != old['actions']:
                    for target in targets:
                        new[target] = DiscreteReward(new['actions'],list(map(old[target],old['actions'])))

            yield new

class Binary(EnvironmentFilter):
    """Transform reward values to 1 or 0."""

    @property
    def params(self) -> Mapping[str, Any]:
        return { "binary": True }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        first, interactions = peek_first(interactions)

        is_discrete = 'actions' in first and 0 < len(first['actions']) and len(first['actions']) < float('inf')
        is_callable = callable(first['rewards'])

        if not is_discrete:
            warnings.warn("Only discrete action rewards can be made binary. ")
            yield from interactions

        else:
            for interaction in interactions:
                new     = interaction.copy()
                actions = interaction['actions']
                rewards = interaction['rewards']

                if is_callable:
                    argmax = max(actions, key=rewards)
                    new['rewards'] = BinaryReward(argmax)
                else:
                    argmax = max(range(len(actions)),key=rewards.__getitem__)
                    new['rewards'] = [0] * len(new['actions'])
                    new['rewards'][argmax] = 1

                yield new

class Sort(EnvironmentFilter):
    """Sort interactions by features."""

    def __init__(self, *keys: Union[str,int,Sequence[Union[str,int]]]) -> None:
        """Instantiate a Sort filter.

        Args:
            *keys: The context items that should be sorted on.
        """

        self._keys = list(pipes.Flatten().filter([list(keys)]))[0]

    @property
    def params(self) -> Mapping[str, Any]:
        return { "sort_keys": self._keys or '*' }

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        first, interactions = peek_first(interactions)

        if not interactions:
            return

        if first and 'context' not in first:
            yield from interactions
            return

        full_sorter = lambda interaction: tuple(interaction['context']                                 )
        list_sorter = lambda interaction: tuple(interaction['context'][key]       for key in self._keys)
        dict_sorter = lambda interaction: tuple(interaction['context'].get(key,0) for key in self._keys)

        first, interactions = peek_first(interactions)
        is_sparse           = isinstance(first['context'],primitives.Sparse)

        sorter = full_sorter if not self._keys else dict_sorter if is_sparse else list_sorter
        yield from sorted(interactions, key=sorter)

class Where(EnvironmentFilter):
    """Select for characteristics"""

    def __init__(self, *,
        n_interactions: Union[int,Tuple[Optional[int],Optional[int]]] = None,
        n_actions: Union[int,Tuple[Optional[int],Optional[int]]] = None,
        n_features: Union[int,Tuple[Optional[int],Optional[int]]] = None) -> None:
        """Instantiate a Where filter.

        Args:
            n_interactions: The minimum, maximum or exact number of interactions Environments must have.
            n_actions: The minimum, maximum or exact number of actions interactions must have.
            n_features: The minimum, maximum or exact number of features interactions must have.
        """

        self._n_interactions = n_interactions
        self._n_actions = n_actions
        self._n_features = n_features

    @property
    def params(self) -> Mapping[str, Any]:
        params = {}

        if self._n_interactions is not None:
            params["where_n_interactions"] = self._n_interactions

        if self._n_actions is not None:
            params["where_n_actions"] = self._n_actions

        if self._n_features is not None:
            params["where_n_features"] = self._n_features

        return params

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        n_int = self._n_interactions
        n_int = list(n_int) if isinstance(n_int,(list,tuple)) else [n_int]*2

        n_act = self._n_actions
        n_act = list(n_act) if isinstance(n_act,(list,tuple)) else [n_act]*2

        n_fet = self._n_features
        n_fet = list(n_fet) if isinstance(n_fet,(list,tuple)) else [n_fet]*2

        firstn = 1+next((v for v in [*n_int,0] if v is not None))
        first,interactions = peek_first(interactions, n=firstn, reduce=False)

        if not interactions:
            return

        if not self._in_min_max(len(first), *n_int):
            return

        if not self._in_min_max(self._context_len(first), *n_fet):
            return

        for interaction in interactions:
            if n_act == [None,None] or self._in_min_max(len(interaction['actions']),*n_act):
                yield interaction

    def _in_min_max(self,v:int, minv:Optional[float], maxv:Optional[float]):
        return (minv is None or minv <= v) and (maxv is None or v <= maxv)

    def _context_len(self,firstn) -> int:
        try:
            context = firstn[0].get('context')
            return try_else(lambda: len(context), 1) if context or context == 0 else 0
        except:
            return 0

class Riffle(EnvironmentFilter):
    """Riffle shuffle interactions."""

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
    """Add noise to values."""

    def __init__(self,
        context: Union[Tuple[str,float,float],Callable[[float,CobaRandom], float]] = None,
        action : Union[Tuple[str,float,float],Callable[[float,CobaRandom], float]] = None,
        reward : Union[Tuple[str,float,float],Callable[[float,CobaRandom], float]] = None,
        seed   : int = 1) -> None:
        """Instantiate a Noise EnvironmentFilter.

        Args:
            context: A noise generator for context features.
            action : A noise generator for action features.
            reward : A noise generator for rewards.
            seed   : The seed initializing the random state of the noise generators.
        """

        self._args = (context,action,reward,seed)

        def make_noiser(p):
            if not p:
                return None
            elif callable(p):
                return p
            elif len(p) == 2:
                return (lambda x, rng, p=p: x+rng.gauss(*p))
            elif p[0] == 'g':
                return (lambda x, rng, p=p[1:]: x+rng.gauss(*p))
            elif p[0] == 'i':
                return (lambda x, rng, p=p[1:]: x+rng.randint(*p))

        if context is None and action is None and reward is None: context = ('g',0,1)

        self._context_noise = make_noiser(context)
        self._action_noise  = make_noiser(action)
        self._reward_noise  = make_noiser(reward)
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

        if self._context_noise: params['context_noise'] = True
        if self._action_noise : params['action_noise' ] = True
        if self._reward_noise : params['reward_noise' ] = True

        params['noise_seed'] = self._seed

        return params

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        rng = CobaRandom(self._seed)
        first,interactions = peek_first(interactions)

        if not interactions:
            return

        is_callable = callable(first.get('rewards'))

        for interaction in interactions:

            new = interaction.copy()

            if 'context' in new:
                context = new['context']
                noisy_context = self._noises(context, rng, self._context_noise)
                new['context'] = noisy_context

            if 'actions' in new:
                actions = new['actions']
                noisy_actions = [ self._noises(a, rng, self._action_noise) for a in actions ]
                new['actions'] = noisy_actions

            if 'rewards' in new:
                rewards = new['rewards']
                if is_callable: rewards = map(rewards,actions)
                noisy_rewards = [ self._noises(r, rng, self._reward_noise) for r in rewards ]
                if is_callable: noisy_rewards = DiscreteReward(noisy_actions, noisy_rewards)
                new['rewards'] = noisy_rewards

            yield new

    def _noises(self, value:Union[None,float,str,Mapping,Sequence], rng: CobaRandom, noiser: Callable[[float,CobaRandom], float]):
        if noiser is None:
            return value
        if isinstance(value, primitives.Sparse):
            #we sort so that noise generation is deterministic with respect to seed
            return { k:self._noise(v, rng, noiser) for k,v in sorted(value.items()) }
        if isinstance(value, primitives.Dense):
            return [ self._noise(v, rng, noiser) for v in value ]
        return self._noise(value, rng, noiser)

    def _noise(self, value:Union[None,float,str], rng: CobaRandom, noiser: Callable[[float,CobaRandom], float]) -> float:
        return value if not isinstance(value,(int,float)) else noiser(value, rng)

class Params(EnvironmentFilter):
    """Add params to an environment."""
    def __init__(self, params: Mapping[str, Any]) -> None:
        self._params = params

    @property
    def params(self) -> Mapping[str, Any]:
        return self._params

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        return interactions

class Grounded(EnvironmentFilter):
    """Transform simulated interactions to IGL interactions."""

    class GroundedFeedback:
        __slots__ = ('_rng','_goods','_bads','_seed','_argmax')
        def __init__(self, goods, bads, argmax, seed):
            self._rng    = None
            self._goods  = goods
            self._bads   = bads
            self._seed   = seed
            self._argmax = argmax

        @lru_cache(maxsize=None)
        def __call__(self, arg):
            if not self._rng:
                self._rng = CobaRandom(self._seed)
            if arg == self._argmax:
                return self._rng.choice(self._goods)
            else:
                return self._rng.choice(self._bads)

        def __repr__(self) -> str:
            am = self._argmax
            return f"GroundedFeedback({try_else(lambda:minimize(am),str(am))})"

    def __init__(self, n_users: int, n_normal:int, n_words:int, n_good:int, seed:int) -> None:
        """Instantiate a Grounded filter.

        Args:
            n_users: The number of users in the grounded environment.
            n_normal: The number of users with normal grounded behavior.
            n_words: The number of potential feedback words for users.
            n_good: The number of words that mean good out of the n_words.
            seed: Seed for all random values.

        Remarks:
            See `here`__ for more on interaction grounded learning.

        __ https://proceedings.mlr.press/v139/xie21e.html
        """
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

        if not interactions:
            return

        is_callable = callable(first['rewards'])

        if is_callable:
            is_binary_rwd = set(map(first['rewards'], first['actions'])) == {0,1}
        else:
            is_binary_rwd = set(first['rewards']) == {0,1}

        goods = [(g,) for g in self.goodwords]
        bads  = [(b,) for b in self.badwords ]

        for seed, interaction in enumerate(interactions, self._seed):

            userid,normal   = rng.choice(userid_isnormal)
            actions,rewards = interaction['actions'],interaction['rewards']

            new = interaction.copy()

            if is_callable:
                argmax = max(actions,key=rewards)
            else:
                argmax = actions[max(range(len(actions)),key=rewards.__getitem__)]

            if not is_binary_rwd:
                new['rewards'] = BinaryReward(argmax)

            if normal:
                new['feedbacks'] = Grounded.GroundedFeedback(goods,bads,argmax,seed)
            else:
                new['feedbacks'] = Grounded.GroundedFeedback(bads,goods,argmax,seed)

            new['userid'  ] = userid
            new['isnormal'] = normal

            yield new

    def _cast_int(self, value:Union[float,int], value_name:str) -> int:
        if isinstance(value, int): return value
        if isinstance(value, float) and value.is_integer(): return int(value)
        raise CobaException(f"{value_name} must be a whole number and not {value}.")

class Repr(EnvironmentFilter):
    """Change representation of categorical data."""

    def __init__(self,
        categorical_context:Literal["onehot","onehot_tuple","string"] = None,
        categorical_actions:Literal["onehot","onehot_tuple","string"] = None) -> None:
        """Instantiate a Repr filter.

        Args:
            cat_context: How to represent categorical data in contexts.
            cat_actions: How to represent categorical data in actions.
        """
        self._cat_context = categorical_context
        self._cat_actions = categorical_actions

    @property
    def params(self) -> Mapping[str, Any]:
        return {"categoricals_in_context": self._cat_context, "categoricals_in_actions": self._cat_actions}

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        firsts, interactions = peek_first(interactions,n=2)

        if not interactions: return []

        first = firsts[0]

        has_context   = 'context'   in first
        has_actions   = 'actions'   in first and bool(first['actions'])
        has_action    = 'action'    in first
        has_rewards   = 'rewards'   in first
        has_feedbacks = 'feedbacks' in first

        n_tee = 1 + bool(self._cat_context and has_context) + \
                    bool(self._cat_actions and has_actions) + \
                    bool(self._cat_actions and has_action)

        tees = iter([interactions]) if n_tee == 1 else iter(tee(interactions, n_tee))

        if not (self._cat_context and has_context):
            cat_context_iter = None
        else:
            cat_context_iter = pipes.EncodeCatRows(self._cat_context).filter(i['context'] for i in next(tees))

        if not (self._cat_actions and has_actions):
            cat_actions_iter = None
        else:
            rows = (i['actions'] for i in next(tees))

            if len(firsts) == 2 and firsts[0]['actions'] == firsts[1]['actions']:
                def yield_prev_action_on_repeat():
                    prev_row   = None
                    prev_yield = None

                    encoder = pipes.EncodeCatRows(self._cat_actions)

                    for row in rows:
                        if row != prev_row:
                            prev_row = row
                            prev_yield = list(encoder.filter(row))
                        yield prev_yield

                cat_actions_iter = yield_prev_action_on_repeat()
            else:
                def yield_action_lists():
                    r1,r2 = tee(rows,2)
                    actionitr = pipes.EncodeCatRows(self._cat_actions).filter(chain.from_iterable(r2))
                    for row in r1:
                        yield list(islice(actionitr,len(row)))
                cat_actions_iter = yield_action_lists()

        if not (self._cat_actions and has_action):
            cat_action_iter = None
        else:
            cat_action_iter = pipes.EncodeCatRows(self._cat_context).filter(i['action'] for i in next(tees))

        reward_targets = []
        if has_rewards   and callable(first['rewards'])  : reward_targets.append('rewards')
        if has_feedbacks and callable(first['feedbacks']): reward_targets.append('feedbacks')

        for old in next(tees):

            new = old.copy()

            if cat_context_iter:
                new['context'] = next(cat_context_iter)

            if cat_actions_iter:
                new['actions'] = next(cat_actions_iter)

            if cat_action_iter:
                new['action'] = next(cat_action_iter)

            actions_changed = cat_actions_iter and new['actions'] != old['actions']

            if actions_changed:
                for target in reward_targets:
                    if isinstance(old[target],BinaryReward):
                        new_argmax = new['actions'][old['actions'].index(old[target]._argmax)]
                        new[target] = BinaryReward(new_argmax,old[target]._value)
                    elif isinstance(old[target],DiscreteReward):
                        new[target] = DiscreteReward(new['actions'],old[target].rewards)
                    else:
                        new[target] = DiscreteReward(new['actions'],list(map(old[target],old['actions'])))
            yield new

class Batch(EnvironmentFilter):
    """Batch interactions."""
    class List(list):
        is_batch: bool = True

    class Callable(list):
        is_batch: bool = True
        def __call__(self, args: Any) -> Any:
            ndim = getattr(args,'ndim',-1)
            outs = list(map(lambda f,a: f(a), self, args))

            if ndim == -1:
                return Batch.List(outs)
            else:
                try:
                    t  = torch # type: ignore
                    bt = Batch.Tensor
                except:
                    t  = globals()['torch'] = __import__('torch')
                    bt = Batch._make_batch_tensor()
                return bt(t.stack(outs))

    @classmethod
    def _make_batch_tensor(cls):
        try:
            return cls.Tensor
        except:
            import torch
            class BatchTensor(torch.Tensor):
                is_batch: bool = True
            cls.Tensor = BatchTensor
            return cls.Tensor

    def __init__(self, batch_size: Optional[int], batch_type: Literal['list','torch'] = 'list') -> None:
        """Instantiate a Batch filter.

        Args:
            batch_size: The number of interactions in a batched interaction.
            batch_type: The type of batch for interaction values.
        """

        self._batch_size = batch_size or None
        self._batch_type = batch_type

        if batch_type == 'torch': PackageChecker.torch('Batch')

    @property
    def params(self) -> Mapping[str,Any]:
        return {'batch_size': self._batch_size, 'batch_type':self._batch_type}

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        first,interactions = peek_first(interactions)

        if not interactions:
            yield from []

        elif not self._batch_size:
            yield from interactions

        else:
            callable_keys = [k for k,v in first.items() if callable(v)]
            if self._batch_type == 'torch': self._make_batch_tensor()

            batch_data_type = Batch.List if self._batch_type == 'list' else Batch.Tensor
            batch_call_type = Batch.Callable

            for batch in self._batched(interactions, self._batch_size):

                new = {}

                for key in first.keys() & callable_keys:
                    new[key] = batch_call_type(map(itemgetter(key),batch))

                for key in first.keys() - callable_keys:
                    values = list(map(itemgetter(key),batch))
                    try:
                        new[key] = batch_data_type(values)
                    except:
                        new[key] = Batch.List(values)

                yield new

    def _batched(self, iterable, n):
        #Batch data into lists of length n.
        #The last batch may be shorter.
        #Taken from python itertools recipes.
        it = iter(iterable)
        batch = list(islice(it, n))
        while (batch):
            yield batch
            batch = list(islice(it, n))

class Unbatch(EnvironmentFilter):
    """Unbatch interactions."""

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        first, interactions = peek_first(interactions)

        if not interactions: return []

        batched_keys = [k for k,v in first.items() if is_batch(v) ]

        if batched_keys:
            yield from self._unbatch(interactions, batched_keys)
        else:
            yield from interactions

    def _unbatch(self, interactions: Iterable[Interaction], batched_keys:Sequence[str]):
        for interaction in interactions:
            batch_size = len(interaction[batched_keys[0]])
            for i in range(batch_size):
                new = {}
                for k in interaction:
                    try:
                        new[k] = interaction[k][i]
                    except:
                        new[k] = interaction[k]
                yield new

class BatchSafe(EnvironmentFilter):
    """Make a filter agnostic to batching."""

    def __init__(self, filter: EnvironmentFilter) -> None:
        """Instantiate a BatchSafe filter.

        Args:
            filter: The filter to make agnostic to batching.
        """
        self._filter = filter

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        first, interactions = peek_first(interactions)
        if not interactions: return []

        first_val  = next(iter(first.values()))
        batch_size = len(first_val) if is_batch(first_val) else None

        pipe = Pipes.join(Unbatch(),self._filter,Batch(batch_size)) if batch_size else self._filter

        yield from pipe.filter(interactions)

    def __str__(self):
        return f"BatchSafe({self._filter})"

class Harden(EnvironmentFilter):
    """Make all lazy data concrete.

    Remarks:
        Convert lazily evaluated data to concrete types.
    """

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        first, interactions = peek_first(interactions)

        if not interactions: return []

        first_has_context = 'context' in first
        first_has_actions = 'actions' in first and first['actions']
        first_has_action  = 'action'  in first

        if first_has_context:
            is_dense_context  = isinstance(first['context'],primitives.Dense)
            is_sparse_context = isinstance(first['context'],primitives.Sparse)
        if first_has_actions:
            is_dense_actions  = isinstance(first['actions'][0],primitives.Dense)
            is_sparse_actions = isinstance(first['actions'][0],primitives.Sparse)
        if first_has_action:
            is_dense_action  = isinstance(first['action'],primitives.Dense)
            is_sparse_action = isinstance(first['action'],primitives.Sparse)

        context_materialized = not first_has_context or primitives.is_materialized(first['context'])
        actions_materialized = not first_has_actions or primitives.is_materialized(first['actions'][0])
        action_materialized  = not first_has_action  or primitives.is_materialized(first['action'])

        for interaction in interactions:

            new = interaction.copy()

            if not context_materialized and is_dense_context:
                new['context'] = list(new['context'])
            elif not context_materialized and is_sparse_context:
                new['context'] = new['context'].copy()

            if not actions_materialized and is_dense_actions:
                new['actions'] = list(map(list,new['actions']))
            elif not actions_materialized and is_sparse_actions:
                new['actions'] = list(map(methodcaller('copy'),new['actions']))

            if not action_materialized and is_dense_action:
                new['action'] = list(new['action'])
            elif not action_materialized and is_sparse_action:
                new['action'] = new['action'].copy()

            yield new

class Chunk(EnvironmentFilter):
    """Indicate where to chunk.

    Remarks:
        Functionally this filter does nothing. This filter
        only has an effect when environments are passed to
        Experiment. In that case, this is useful if an early
        part of an Environments pipeline takes considerable
        time to evaluate (e.g., `logged`). Placing a chunk
        after a long-running filter means that the filter
        will only be executed once. After that, the results
        will be resued by all pipes downstream of the chunk.
    """

    def filter(self, items: Iterable[Interaction]) -> Iterable[Interaction]:
        return items

class Logged(EnvironmentFilter):
    """Transform simulated interactions to logged interactions."""

    def __init__(self, learner: Learner, seed: Optional[float] = 1.23) -> None:
        """Instantiate a Logged filter.

        Args:
            learners: The learners that will be used as the logging policy.
                An environment will be created for every learner provided.
            seed: The seed for used for all random number generation.

        Remarks:
            Adds 'action', 'reward', and 'probability' to interactions with 'context',
            'actions', and 'rewards'.
        """
        self._learner = learner
        self._seed    = seed

    @property
    def params(self) -> Mapping[str, Any]:
        lrn_params = SafeLearner(self._learner).params
        return { **lrn_params, "learner": lrn_params, "logged":True, "log_seed":self._seed}

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        first, interactions = peek_first(interactions)

        if not interactions:
            return

        if 'actions' not in first or 'rewards' not in first:
            raise CobaException("We were unable to create a logged representation of the interaction.")

        #Avoid circular dependency
        from coba.evaluators import SequentialCB

        seed = self._seed if self._seed is not None else CobaRandom().random()

        I1,I2 = tee(interactions,2)
        interactions = Unbatch().filter(I1)

        env = type("",(),{'read':lambda : I2})
        lrn = copy.deepcopy(self._learner)
        evaluator = SequentialCB(record=['action','reward','probability'],seed=seed)

        for interaction, log in zip(interactions,evaluator.evaluate(env,lrn)):
            out = interaction.copy()
            out.update(log)
            yield out

class Mutable(EnvironmentFilter):
    """Make all feature containers mutable (i.e., list or dict)."""

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        first, interactions = peek_first(interactions)

        if not interactions:
            return

        if 'context' not in first:
            yield from interactions
            return

        first_is_dense   = isinstance(first['context'], primitives.Dense)
        first_is_sparse  = isinstance(first['context'], primitives.Sparse)
        first_is_value   = not (first_is_dense or first_is_sparse)
        first_is_mutable = isinstance(first['context'], (list,dict,SparseDense))

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

class OpeRewards(EnvironmentFilter):
    """Transform logged interactions to simulated interactions."""

    def __init__(self, rwd_type:Literal['IPS','DM','DR']=None, target:str = 'rewards', features=[1,'x','a','xa','xxa']):
        """Instantiate an OpeRewards filter.

        Args:
            rewards_type: How to estimate the rewards function from the logged
            data (i.e., inverse propensity score, direct method, or doubly robust).

        Remarks:
            Adds 'rewards' to interactions with 'context', 'action', 'reward',
            and 'probability'.
        """
        if rwd_type in ['DM','DR']:
            PackageChecker.vowpalwabbit(f"{rwd_type} OpeRewards")
        self._rwd_type = rwd_type
        self._target   = target
        self._features = features

    @property
    def params(self) -> Mapping[str, Any]:
        return {'ope_reward': self._rwd_type or 'None'}

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        rwd_type = self._rwd_type

        if rwd_type is None:
            yield from interactions

        elif rwd_type == "IPS":
            for interaction in interactions:
                interaction = interaction.copy()
                interaction[self._target] = BinaryReward(interaction['action'], interaction['reward']/(interaction.get('probability') or 1))
                yield interaction

        elif rwd_type in ["DM","DR"]:
            from coba.learners.vowpal import VowpalMediator

            class DMReward:
                def __init__(self, vw:VowpalMediator, context:Any) -> None:
                    self._vw = vw
                    self._x  = context

                def __call__(self, action) -> Any:
                    feats   = {"x":self._x, 'a':action} if self._x is not None else {'a':action}
                    return vw.predict(vw.make_example(feats))

            class DRReward:
                def __init__(self,vw:VowpalMediator, context:Any,  off_action, off_reward, off_prob) -> None:
                    self._vw         = vw
                    self._x          = context
                    self._off_action = off_action
                    self._off_reward = off_reward
                    self._off_prob   = (off_prob or 1)

                def __call__(self, action) -> Any:
                    feats   = {"x":self._x, 'a':action} if self._x is not None else {'a':action}
                    dm_part = vw.predict(vw.make_example(feats))
                    dr_part = (self._off_reward-dm_part)/self._off_prob if action == self._off_action else 0
                    return dr_part + dm_part

            rng = CobaRandom(1)
            vw = VowpalMediator()

            # When using cb_adf to estimate rewards for actions instead of the
            # regression we would receive rewards far outside of the expected boundaries.
            #vw.init_learner("--cb_adf --interactions xa --interactions xxa --quiet ", label_type=4)

            args = "--quiet"
            if 'x' not in self._features: args = '--ignore_linear x ' + args
            if 'a' not in self._features: args = '--ignore_linear a ' + args
            if 1 not in self._features  : args = '--noconstant ' + args
            for f in self._features:
                if f not in ['x','a',1]:
                    args = f'--interactions {f} ' + args

            vw.init_learner(args, label_type=1)

            #so that we don't run forever
            #when interactions is re-iterable
            interactions = iter(interactions)

            while True:

                #Batch Shuffling reduces the correlation between
                #the reward functions learned by this VW learner
                #and potential downstream VW learners. If this
                #correlation is strong, we will over estimate
                #the downstream VW learner's performance.
                L = list(islice(interactions,4000))

                if not L: break

                for interaction in rng.shuffle(L):
                    if 'context' in interaction:
                        features = {"x": interaction['context'], "a": interaction['action']}
                    else:
                        features = {"a": interaction['action']}
                    label    = f"{interaction['reward']} {1/(interaction.get('probability') or 1)}"
                    vw.learn(vw.make_example(features,label))

                for interaction in L:
                    new = interaction.copy()

                    if rwd_type=="DR":
                        new[self._target] = DRReward(vw,new.get('context'),new['action'],new['reward'],new['probability'])
                    else:
                        new[self._target] = DMReward(vw,new.get('context'))

                    yield new

class EmptyCheck(EnvironmentFilter):

    def __init__(self) -> None:
        self._isempty = None

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:

        if self._isempty is None:
            interactions = peek_first(interactions)[1]
            self._isempty = interactions == []
            if not interactions:
                CobaContext.logger.log("An environment had nothing to evaluate (this is often due to having too few interactions).")

        if not self._isempty:
            yield from interactions

class Finalize(EnvironmentFilter):
    """Final preparation for built-in Evaluators.

    Remarks:
        This filter does two things:
            1. Materializes all lazy data
            2. Turns categorical types into one-hot encodings
    """

    def __init__(self):
        self._emptycheker = EmptyCheck()

    def filter(self, interactions: Iterable[Interaction]) -> Iterable[Interaction]:
        first,interactions = peek_first(self._emptycheker.filter(interactions))

        rwds_is_list  = first and isinstance(first.get('rewards')  ,list)
        fbks_is_list  = first and isinstance(first.get('feedbacks'),list)

        interactions = Pipes.join(Harden(),Repr("onehot","onehot")).filter(interactions)

        if not (rwds_is_list or fbks_is_list):
            yield from interactions
        else:
            for interaction in interactions:
                new = interaction.copy()
                if rwds_is_list: new['rewards'  ] = DiscreteReward(new['actions'],new['rewards'  ])
                if fbks_is_list: new['feedbacks'] = DiscreteReward(new['actions'],new['feedbacks'])
                yield new
