import re
from itertools import repeat, compress
from sys import platform
from typing import Any, Dict, Union, Sequence, Mapping, Optional, Tuple, Literal

from coba.exceptions import CobaException
from coba.primitives import is_batch, Learner, Context, Action, Actions, Prob, PMF, kwargs, Sparse
from coba.utilities import PackageChecker

Feature       = Union[str,int,float]
Features      = Union[Feature, Sequence[Feature], Dict[str,Union[int,float]]]
Namespaces    = Dict[str,Features]
VW_Features   = Sequence[Union[str,int,Tuple[Union[str,int],Union[int,float]],Dict[str,Union[int,float]]]]
VW_Namespaces = Dict[str,VW_Features]

DEFAULT_NAMESPACE_INTERACTIONS = (1, 'a', 'ax', 'axx')

def make_args(vw_kwargs: Dict[str, Any], namespace_interactions: Sequence[str] = DEFAULT_NAMESPACE_INTERACTIONS) -> Sequence[str]:
    """
    Turn settings into VW command line arguments.

        Args:
        vw_kwargs: Keyword argument dict that gets translated into VW CLI arguments
        namespace_interactions: A sequence of namespace interactions to use during learning.
    """
    args = []
    vw_kwargs['quiet'] = vw_kwargs.get('quiet', True)

    for k,v in vw_kwargs.items():
        if v is not False and v is not None:
            if not k.startswith("-"):
                k = ("-" if len(k)==1 else "--") + k
            args.append(k if (v is True) else f"{k} {v}")

    noconstant = sum([n for n in namespace_interactions if isinstance(n, (int, float))]) == 0
    ignore_linear = set(['x', 'a']) - set(namespace_interactions)
    interactions = [n for n in namespace_interactions if isinstance(n, str) and len(n) > 1]

    if noconstant:
        args.append(f"--noconstant")

    for interaction in interactions:
        args.append(f"--interactions {interaction}")

    for ignore in ignore_linear:
        args.append(f"--ignore_linear {ignore}")

    return args

class VowpalMediator:
    """A class to handle all communication between Coba and VW."""

    def __init__(self) -> None:
        self._vw = None
        self._args = ""
        self._ns_keys = {}
        self._ns_keys_cnt = 0

        PackageChecker.vowpalwabbit('VowpalMediator')

    @property
    def is_initialized(self) -> bool:
        """Indicate whether init_learner has been called previously."""
        return self._vw is not None

    @property
    def params(self) -> dict:
        return {}

    def init_learner(self, args:str, label_type: int) -> 'VowpalMediator':
        """Create a VW learner from a command line arg string.

        Args:
            args: The command line arg string to use for VW learner creation.
            label_type: The label type this VW learner will take.
                - 1 : `simple`__
                - 2 : `multiclass`__
                - 3 : `cost sensitive`__
                - 4 : `contextual bandit`__
                - 5 : max (deprecated)
                - 6 : `conditional contextual bandit`__
                - 7 : `slates`__
                - 8 : `continuous actions`__

        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format#simple
        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format#multiclass
        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format#cost-sensitive
        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format#contextual-bandit
        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Conditional-Contextual-Bandit#vw-text-format
        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Slates#text-format
        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/CATS,-CATS-pdf-for-Continuous-Actions#vw-text-format
        """
        from vowpalwabbit import pyvw, __version__

        if self._vw is not None:
            raise CobaException("We cannot initilaize a VW learner twice in a single mediator.")

        self._version = [ int(v) for v in __version__.split(".") ]
        self._vw = pyvw.Workspace(args) if self._version[0] >= 9 else pyvw.vw(args)
        self._label_type = pyvw.LabelType(label_type) if self._version[0] >= 9 else label_type
        self._example_init = pyvw.Example if self._version[0] >= 9 else pyvw.example
        self._args = args

        return self

    def predict(self, example: Any) -> Any:
        """Predict for an example created by the mediator."""
        pred = self._vw.predict(example)
        self._vw.finish_example(example)
        return pred

    def learn(self, example: Any) -> None:
        """Learn for an example created by the mediator."""
        self._vw.learn(example)
        self._vw.finish_example(example)

    def finish(self):
        if self.is_initialized:
            self._vw.finish()

    def custom_ns(self, namespace: VW_Namespaces) -> VW_Namespaces:
        """Override to transform example namespaces before they get pushed down to VW"""
        return namespace

    def make_example(self, namespaces: Namespaces, label:Optional[str] = None) -> Any:
        """Create a VW example.

        Args:
            ns: The features grouped by namespace in this example.
            label: An optional label (required if this example will be used for learning).
        """
        ns = dict(self._prep_namespaces(namespaces))
        ns = self.custom_ns(ns)

        ex = self._example_init(self._vw, None, self._label_type)
        ex.push_feature_dict(self._vw, ns)
        if label is not None: ex.set_label_string(label)
        ex.setup_example()

        return ex

    def make_examples(self, shared: Namespaces, uniques: Sequence[Namespaces], labels:Optional[Sequence[str]] = None) -> Sequence[Any]:
        """Create a list of VW examples.

        Args:
            shared: The features grouped by namespace in this example.
            separates: The features, grouped by namespace, unique to each each example.
            label: An optional label (required if this example will be used for learning).
        """

        labels     = repeat(None) if labels is None else labels
        vw_shared  = dict(self._prep_namespaces(shared))
        vw_uniques = list(map(dict,map(self._prep_namespaces,uniques)))

        vw_shared  = self.custom_ns(vw_shared)
        vw_uniques = list(map(self.custom_ns,vw_uniques))

        examples = []
        for vw_unique, label in zip(vw_uniques,labels):
            ex = self._example_init(self._vw, None, self._label_type)
            ex.push_feature_dict(self._vw, {**vw_shared, **vw_unique})
            if label: ex.set_label_string(label)
            ex.setup_example()
            examples.append(ex)

        return examples

    def _prep_namespaces(self, namespaces: Namespaces) -> VW_Namespaces:
        """Turn a collection of coba formatted namespaces into VW format."""

        if self._version < [9,2] or (self._version == [9,2,0] and "linux" in platform): #pragma: no cover
            #the strange type checks below were faster than traditional methods when performance testing
            for ns, feats in namespaces.items():
                if not feats and feats != 0 and feats != "":
                    continue
                elif feats.__class__ is str:
                    yield (ns, [f"{self._get_ns_keys(ns,0)}={feats}"])
                elif feats.__class__ is int or feats.__class__ is float:
                    yield (ns, [(self._get_ns_keys(ns,0), feats)])
                else:
                    feats = feats.items() if isinstance(feats, Sparse) else zip(self._get_ns_keys(ns,len(feats)),feats)
                    yield (ns, [f"{k}={v}" if v.__class__ is str else (k, v) for k,v in feats if v!= 0])
        else: #pragma: no cover
            #the strange type checks below were faster than traditional methods when performance testing
            for ns, feats in namespaces.items():
                if not feats and feats != 0 and feats != "":
                    continue
                elif feats.__class__ is str:
                    yield (ns, {f"{self._get_ns_keys(ns,0)}={feats}":1})
                elif feats.__class__ is int or feats.__class__ is float:
                    yield (ns, {self._get_ns_keys(ns,0): feats})
                elif isinstance(feats,Sparse):
                    no_str = not any(map(isinstance,feats.values(),repeat(str))) # most-performant (still 2x slower than no check)
                    #no_str = str not in set(map(attrgetter("__class__"),feats.values()))
                    #no_str = not any(map(is_,map(attrgetter("__class__"),feats.values()),repeat(str)))
                    #no_str = True
                    if no_str:
                        yield (ns, feats.copy())
                    else:
                        new = feats.copy()
                        for k,v in feats.items():
                            if isinstance(v,str):
                                new[f"{k}={v}"] = 1
                                del new[k]
                        yield (ns,new)
                else:
                    no_str = not any(map(isinstance,compress(feats,feats),repeat(str)))

                    K = self._get_ns_keys(ns,len(feats))
                    V = feats

                    if no_str:
                        d = dict(compress(zip(K,V),V))
                    else:
                        d={}
                        for k,v in compress(zip(K,V),V):
                            if v.__class__ is str:
                                d[f"{k}={v}"] = 1
                            else:
                                d[k] = v

                    yield (ns, d)

    def _get_ns_keys(self, ns:str, length: int) -> Union[str,Sequence[str]]:

        if ns not in self._ns_keys:
            if not length:
                self._ns_keys[ns] = str(self._ns_keys_cnt)
                self._ns_keys_cnt += 1
            else:
                self._ns_keys[ns] = list(map(str,range(self._ns_keys_cnt,self._ns_keys_cnt+length)))
                self._ns_keys_cnt += length

        return self._ns_keys[ns]

    def __str__(self) -> str:
        return self._args

class VowpalLearner(Learner):
    """A Vowpal Wabbit wrapper.

    Remarks:
        This learner requires that the Vowpal Wabbit package be installed. This package can be
        installed via `pip install vowpalwabbit`. To learn more about solving contextual bandit
        problems with Vowpal Wabbit see `here`__ and `here`__.

    __ https://vowpalwabbit.org/tutorials/contextual_bandits.html
    __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms
    """

    def __init__(self, args: str = "--cb_explore_adf --epsilon 0.05 --interactions ax --interactions axx --ignore_linear x --random_seed 1 --quiet", vw: VowpalMediator = None) -> None:
        """Instantiate a VowpalLearner.

        Args:
            args: Command line arguments to instantiate a Vowpal Wabbit contextual bandit learner. For
                examples and documentation on how to instantiate VW learners from command line arguments
                see `here`__. We require that either cb, cb_adf, cb_explore, or cb_explore_adf is used.
                When we format examples for VW context features are placed in the 'x' namespace and action
                features, when relevant, are placed in the 'a' namespace.
            vw: A mediator able to communicate with VW. This should not need to ever be changed.
        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms
        """

        if "--cb" not in args:
            raise CobaException("VowpalLearner was instantiated without a cb flag. One of the cb flags must be defined.")

        self._args    = args
        self._explore = "--cb_explore" in args
        self._adf     = "--cb_adf"     in args or "--cb_explore_adf" in args

        self._n_actions = None
        self._actions   = None

        if not self._adf:
            n_action_str = re.match("--cb.*?\s*(\d*)\\b.*", args).group(1)
            self._n_actions = int(n_action_str) if n_action_str else None
            #useful for removing the n_actions from args
            #re.sub("(--cb)\s*(\d+)", '\g<1>', "--cb 123", count=1)

        self._vw = vw or VowpalMediator()

    @property
    def params(self) -> Mapping[str, Any]:
        return {"family": "vw", 'args': self._args.replace("--quiet","").strip(), **self._vw.params}

    def score(self, context: 'Context', actions: 'Actions', action: 'Action') -> 'Prob':
        return self.predict(context,actions)[0][actions.index(action)]

    def predict(self, context: 'Context', actions: 'Actions') -> Tuple['PMF','kwargs']:
        if not self._vw.is_initialized and is_batch(context):#pragma: no cover
            raise CobaException("VW learner does not support batched calls.")

        if not self._vw.is_initialized and self._adf:
            self._vw.init_learner(self._args, 4)

        if not self._vw.is_initialized and not self._adf and self._n_actions:
            self._vw.init_learner(self._args, 4)

        if not self._vw.is_initialized and not self._adf and not self._n_actions:
            self._n_actions = len(actions)
            args            = self._args.replace('--cb_explore','').replace('--cb','')
            args            = f"--cb_explore {len(actions)} " if self._explore else f"--cb {len(actions)} " + args
            args            = args.strip()
            self._vw.init_learner(args,4)

        if not self._adf and not self._actions:
            self._actions = actions

        if not self._adf and actions != self._actions:
            raise CobaException("Actions are only allowed to change between predictions when using `adf`.")

        if not self._adf and len(actions) != self._n_actions:
            raise CobaException("The number of actions doesn't match the `--cb` action count given in args.")

        context = {'x':context}
        adfs    = None if not self._adf else [{'a':action} for action in actions]

        if self._adf and self._explore:
            probs = self._vw.predict(self._vw.make_examples(context, adfs, None))
            if probs is None: probs = [1/len(actions)] * len(actions)

        if self._adf and not self._explore:
            losses    = self._vw.predict(self._vw.make_examples(context,adfs, None))
            min_loss  = min(losses)
            min_bools = [s == min_loss for s in losses]
            min_count = sum(min_bools)
            probs     = [ int(min_indicator)/min_count for min_indicator in min_bools ]

        if not self._adf and self._explore:
            probs = self._vw.predict(self._vw.make_example(context, None))
            if probs is None: probs = [1/len(actions)] * len(actions)

        if not self._adf and not self._explore:
            index = self._vw.predict(self._vw.make_example(context, None))
            probs = [ int(i==index) for i in range(1,len(actions)+1) ]

        return probs, {'actions':actions}

    def learn(self, context: 'Context', action: 'Action', reward: float, probability: float, actions: 'Actions' = None) -> None:
        if not self._vw.is_initialized and self._adf:
            self._vw.init_learner(self._args, 4)

        if not self._vw.is_initialized and not self._adf and self._n_actions:
            self._vw.init_learner(self._args, 4)

        if not self._vw.is_initialized and not self._adf and not self._n_actions:
            raise CobaException("When using `cb` without `adf` predict must be called before learn to initialize the vw learner")

        if actions is None: actions = [action]

        index   = actions.index(action)
        labels  = self._labels(actions, index, reward, probability)
        label   = labels[index]

        context = {'x':context}
        adfs    = None if not self._adf else [{'a':action} for action in actions]

        if self._adf:
            self._vw.learn(self._vw.make_examples(context, adfs, labels))
        else:
            self._vw.learn(self._vw.make_example(context, label))

    def _labels(self,actions,index,reward:float,prob:float) -> Sequence[Optional[str]]:
        labels = [None]*len(actions)
        labels[index] = f"{index+1}:{round(-reward,5)}:{round(prob,5)}"
        return labels

    def _finish(self):
        try:
            self._vw.finish()
        except AttributeError:
            #we do these checks because when mocking up for unittests
            #it is possible that _vw actually isn't ever initialized
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._finish()

    def finish(self):
        """Finish all pending work (e.g., write buffers to disk)."""
        self._finish()

    def __del__(self):
        self._finish()

class VowpalEpsilonLearner(VowpalLearner):
    """Epsilon-greedy exploration with a VW contextual bandit learner.

    More information on VW exploration algorithms can be found `here`__.

        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms
    """

    def __init__(self,
                 epsilon: float = 0.05,
                 features: Sequence[str] = DEFAULT_NAMESPACE_INTERACTIONS,
                 seed: Optional[int] = 1,
                 **kwargs) -> None:
        """Instantiate a VowpalEpsilonLearner.

        Args:
            epsilon: The probability that we will explore instead of exploit.
            features: A list of namespaces and interactions to use when learning reward functions.
            seed: The seed used by VW to generate any necessary random numbers.
            kwargs: Additional key-word args are passed on as VW CLI arguments (unless removed in the function).
        """
        vw_kwargs = {
            "cb_explore_adf": True,
            "epsilon": epsilon,
            "random_seed": seed
        }
        vw = kwargs.pop('vw', None)
        vw_kwargs.update(kwargs)
        vw_args_string = " ".join(make_args(namespace_interactions=features, vw_kwargs=vw_kwargs))
        super().__init__(vw_args_string, vw)

class VowpalSoftmaxLearner(VowpalLearner):
    """Softmax exploration with a VW contextual bandit learner.

    More information on VW exploration algorithms can be found `here`__.

        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms
    """


    def __init__(self,
                 softmax: float=10,
                 features: Sequence[str] = DEFAULT_NAMESPACE_INTERACTIONS,
                 seed: Optional[int] = 1,
                 **kwargs) -> None:
        """Instantiate a VowpalSoftmaxLearner.

        Args:
            softmax: An exploration parameter with 0 indicating predictions should be completely random
                and infinity indicating that predictions should be greedy. For more information see `lambda`__.
            features: A list of namespaces and interactions to use when learning reward functions.
            seed: The seed used by VW to generate any necessary randomness.
            kwargs: Additional key-word args are passed on as VW CLI arguments (unless removed in the function).

        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms
        """
        vw_kwargs = {
            "cb_explore_adf": True,
            "softmax": True,
            "lambda": softmax,
            "random_seed": seed
        }
        vw = kwargs.pop('vw', None)
        vw_kwargs.update(kwargs)
        vw_args_string = " ".join(make_args(namespace_interactions=features, vw_kwargs=vw_kwargs))
        super().__init__(vw_args_string, vw)

class VowpalBagLearner(VowpalLearner):
    """Bootstrap aggregated policy exploration with a VW contextual bandit learner.

    More information on VW exploration algorithms can be found `here`__.

        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms
    """

    def __init__(self,
                 bag: int = 5,
                 features: Sequence[str] = DEFAULT_NAMESPACE_INTERACTIONS,
                 seed: Optional[int] = 1,
                 **kwargs) -> None:
        """Instantiate a VowpalBagLearner.

        Args:
            bag: This value determines the number of policies which will be learned and must be greater
                than 0. Each policy is trained using bootstrap aggregation, making each policy unique. During
                prediction a random policy will be selected according to a uniform distribution and followed.
            features: A list of namespaces and interactions to use when learning reward functions.
            seed: The seed used by VW to generate any necessary random numbers.
            kwargs: Additional key-word args are passed on as VW CLI arguments (unless removed in the function).
        """
        vw_kwargs = {
            "cb_explore_adf": True,
            "bag": bag,
            "random_seed": seed
        }
        vw = kwargs.pop('vw', None)
        vw_kwargs.update(kwargs)
        vw_args_string = " ".join(make_args(namespace_interactions=features, vw_kwargs=vw_kwargs))
        super().__init__(vw_args_string, vw)

class VowpalCoverLearner(VowpalLearner):
    """Online Cover exploration with a VW contextual bandit learner.

    For more information on this algorithm see Agarwal et al. (2014) and `here`__.

    References:
        Agarwal, Alekh, Daniel Hsu, Satyen Kale, John Langford, Lihong Li, and Robert Schapire. "Taming
        the monster: A fast and simple algorithm for contextual bandits." In International Conference on
        Machine Learning, pp. 1638-1646. 2014.

        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms
    """

    def __init__(self,
                 cover: int = 5,
                 features: Sequence[str] = DEFAULT_NAMESPACE_INTERACTIONS,
                 seed: Optional[int] = 1,
                 **kwargs) -> None:
        """Instantiate a VowpalCoverLearner.

        Args:
            cover: The number of policies which will be learned (must be greater than 0).
            features: A list of namespaces and interactions to use when learning reward functions.
            seed: The seed used by VW to generate any necessary random numbers.
            kwargs: Additional key-word args are passed on as VW CLI arguments (unless removed in the function).
        """

        vw_kwargs = {
            "cb_explore_adf": True,
            "cover": cover,
            "random_seed": seed
        }
        vw = kwargs.pop('vw', None)
        vw_kwargs.update(kwargs)
        vw_args_string = " ".join(make_args(namespace_interactions=features, vw_kwargs=vw_kwargs))
        super().__init__(vw_args_string, vw)

class VowpalRndLearner(VowpalLearner):
    """RND exploration with a VW contextual bandit learner.

    Inspired by Random Network Distillation, this explorer constructs an auxiliary prediction
    problem whose expected target value is zero and uses the prediction magnitude to construct
    a confidence interval. In the contextual bandit case this is equivalent to a randomized
    approximation to the LinUCB bound.

    For more information see the `wiki`__.

        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms
    """

    def __init__(self,
                 rnd: int = 3,
                 features: Sequence[str] = DEFAULT_NAMESPACE_INTERACTIONS,
                 epsilon: Optional[float] = 0.025,
                 rnd_alpha: Optional[float] = None,
                 rnd_invlambda: Optional[float] = None,
                 seed: Optional[int] = 1,
                 **kwargs) -> None:
        """Instantiate a VowpalRndLearner.

        Args:
            rnd: Number of predictors
            features: A list of namespaces and interactions  to use when learning reward functions
            epsilon: Uniform exploration term for stabilization
            rnd_alpha: Increase for more exploration on a repeated example
            rnd_invlambda: Increase for more exploration on examples with new features/actions
            seed: The seed used by VW to generate any necessary random numbers
            kwargs: Additional key-word args are passed on as VW CLI arguments (unless removed in the function).
        """
        vw_kwargs = {
            "cb_explore_adf": True,
            "rnd": rnd,
            "epsilon": epsilon,
            "rnd_alpha": rnd_alpha,
            "rnd_invlambda": rnd_invlambda,
            "random_seed": seed
        }
        vw = kwargs.pop('vw', None)
        vw_kwargs.update(kwargs)
        vw_args_string = " ".join(make_args(namespace_interactions=features, vw_kwargs=vw_kwargs))
        super().__init__(vw_args_string, vw)

class VowpalRegcbLearner(VowpalLearner):
    """RegCB exploration with a VW contextual bandit learner.

    For more information on this algorithm see Foster et al. (2014) and `here`__.

    References:
        Foster, D., Agarwal, A., Dudik, M., Luo, H. & Schapire, R.. (2018). Practical Contextual
        Bandits with Regression Oracles. Proceedings of the 35th International Conference on Machine
        Learning, in Proceedings of Machine Learning Research 80:1539-1548.

        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms
    """

    def __init__(self,
                 mode: Literal["optimistic","elimination"] = "elimination",
                 features: Sequence[str] = DEFAULT_NAMESPACE_INTERACTIONS,
                 seed: Optional[int] = 1,
                 **kwargs) -> None:
        """Instantiate a VowpalRegcbLearner.

        Args:
            mode: Indicate that exploration should only predict the optimal upper bound action or
                should use an elimination technique to remove actions that no longer seem plausible
                and pick randomly from the remaining actions.
            features: A list of namespaces and interactions to use when learning reward functions.
            seed: The seed used by VW to generate any necessary random numbers.
            kwargs: Additional key-word args are passed on as VW CLI arguments (unless removed in the function).
        """

        vw_kwargs = {
            "cb_explore_adf": True,
        }
        if mode == "elimination":
            vw_kwargs["regcb"] = True
        else:
            vw_kwargs["regcbopt"] = True
        vw_kwargs["random_seed"] = seed

        vw = kwargs.pop('vw', None)
        vw_kwargs.update(kwargs)
        vw_args_string = " ".join(make_args(namespace_interactions=features, vw_kwargs=vw_kwargs))
        super().__init__(vw_args_string, vw)

class VowpalSquarecbLearner(VowpalLearner):
    """SquareCB exploration with a VW contextual bandit learner.

    For more information on this algorithm see Foster et al. (2020) and `here`__.

    References:
        Foster, D.& Rakhlin, A.. (2020). Beyond UCB: Optimal and Efficient Contextual Bandits with Regression
        Oracles. Proceedings of the 37th International Conference on Machine Learning, in Proceedings of Machine
        Learning Research 119:3199-3210.

        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms
    """

    def __init__(self,
                 mode: Literal["standard","elimination"] = "standard",
                 gamma_scale: float = 10,
                 features: Sequence[str] = DEFAULT_NAMESPACE_INTERACTIONS,
                 seed: Optional[int] = 1,
                 **kwargs) -> None:
        """Instantiate a VowpalSquarecbLearner.

        Args:
            mode: Indicates iwhether all actions should be considered for exploration on each step or actions
                which no longer seem plausible should be eliminated.
            gamma_scale: Controls how quickly squarecb exploration converges to a greedy policy. The larger the
                gamma_scale the faster the algorithm will converge to a greedy policy. This value is the same
                as gamma in the original paper.
            features: A list of namespaces and interactions to use when learning reward functions.
            seed: The seed used by VW to generate any necessary random numbers.
            kwargs: Additional key-word args are passed on as VW CLI arguments (unless removed in the function).
        """
        vw_kwargs = {
            "cb_explore_adf": True,
            "squarecb": True,
            "gamma_scale": gamma_scale,
            "random_seed": seed
        }
        if mode == "elimination":
            vw_kwargs["elim"] = True
        vw = kwargs.pop('vw', None)
        vw_kwargs.update(kwargs)
        vw_args_string = " ".join(make_args(namespace_interactions=features, vw_kwargs=vw_kwargs))
        super().__init__(vw_args_string, vw)

class VowpalOffPolicyLearner(VowpalLearner):
    """No exploration with a VW contextual bandit learner.

    This wrapper performs policy learning without any exploration. This is
    correct when training examples come from a logging policy that controls
    all exploration.

    Remarks:
        More information on VW exploration algorithms can be found `here`__.

        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms
    """

    def __init__(self,
                 features: Sequence[str] = DEFAULT_NAMESPACE_INTERACTIONS,
                 seed: Optional[int] = 1,
                 **kwargs) -> None:
        """Instantiate a VowpalOffPolicyLearner.

        Args:
            features: A list of namespaces and interactions to use when learning reward functions.
            seed: The seed used by VW to generate any necessary random numbers.
            kwargs: Additional key-word args are passed on as VW CLI arguments (unless removed in the function).
        """

        vw_kwargs = {
            "cb_adf": True,
            "random_seed": seed
        }
        vw = kwargs.pop('vw', None)
        vw_kwargs.update(kwargs)
        vw_args_string = " ".join(make_args(namespace_interactions=features, vw_kwargs=vw_kwargs))
        super().__init__(vw_args_string, vw)
