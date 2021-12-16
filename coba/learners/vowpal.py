"""The vowpal module contains classes to make it easier to interact with pyvw."""

import re
import collections.abc

from itertools import repeat
from numbers import Number
from typing_extensions import Literal
from typing import Any, Dict, Union, Sequence, overload, cast, Optional, Tuple

from coba.exceptions import CobaException
from coba.utilities import PackageChecker, KeyDefaultDict
from coba.environments import Context, Action

from coba.learners.core import Learner, Probs, Info

Coba_Feature    = Union[str,float]
Coba_Features   = Union[Coba_Feature, Sequence[Coba_Feature], Sequence[Tuple[str,Coba_Feature]], Dict[str,Coba_Feature]]
Vowpal_Features = Sequence[Union[str,Tuple[str,float]]]

class VowpalMediator:

    _string_cache = KeyDefaultDict(str)

    @staticmethod
    def prep_features(features: Coba_Features) -> Vowpal_Features:
        # one big potential for error here is if our features have float keys      
        # checking for this case though greatly reduces efficiency of the prep operation.

        if features is None or features == [] or features == ():
            return []
        elif isinstance(features, str):
            return [features]
        elif isinstance(features, Number):
            return [("0",float(features))]
        elif isinstance(features,dict):
            features = features.items()
        elif isinstance(features, collections.abc.Sequence) and features and isinstance(features[0], tuple):
            features = features
        elif isinstance(features, collections.abc.Sequence) and features and not isinstance(features[0],tuple):
            features = zip(map(VowpalMediator._string_cache.__getitem__, range(len(features))) ,features)
        else:
            raise CobaException(f"Unrecognized features of type {type(features).__name__} passed to VowpalLearner.")

        return [f"{F[0]}={F[1]}" if isinstance(F[1],str) else (F[0], float(F[1])) for F in features if F[1] != 0]        

    @staticmethod
    def make_learner(args:str):
        PackageChecker.vowpalwabbit('VowpalMediator.make_learner')
        from vowpalwabbit import pyvw
        return pyvw.vw(args)

    @staticmethod
    def make_example(vw, ns:Dict[str,Vowpal_Features], label:Optional[str], label_type:int):
        PackageChecker.vowpalwabbit('VowpalMediator.make_example')
        from vowpalwabbit.pyvw import example

        ns = { k:v for k,v in ns.items() if v != [] }

        ex = example(vw, ns, label_type)
        
        if label: ex.set_label_string(label)

        ex.setup_example()

        return ex

    @staticmethod
    def get_version() -> str:
        PackageChecker.vowpalwabbit('VowpalMediator.get_version')
        from vowpalwabbit.version import __version__
        return __version__

class VowpalLearner(Learner):
    """A learner using Vowpal Wabbit's contextual bandit command line interface.

    Remarks:
        This learner requires that the Vowpal Wabbit package be installed. This package can be
        installed via `pip install vowpalwabbit`. To learn more about solving contextual bandit
        problems with Vowpal Wabbit see https://vowpalwabbit.org/tutorials/contextual_bandits.html
        and https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms.
    """

    @overload
    def __init__(self, *, epsilon: float = 0.05, seed: Optional[int] = 1) -> None:
        """Instantiate a VowpalLearner.
        Args:
            epsilon: A value between 0 and 1. If provided, exploration will follow epsilon-greedy.
            seed: The seed used by VW to generate any necessary random numbers.
        """
        ...

    @overload
    def __init__(self, *, bag: int, seed: Optional[int] = 1) -> None:
        """Instantiate a VowpalLearner.
        Args:
            bag: This value determines the number of policies which will be learned and must be greater
                than 0. Each policy is trained using bootstrap aggregation, making each policy unique. During
                prediction a random policy will be selected according to a uniform distribution and followed.
            adf: Indicate whether cb_explore or cb_explore_adf should be used.
            seed: The seed used by VW to generate any necessary random numbers.
        """
        ...

    @overload
    def __init__(self, *, cover: int, seed: Optional[int] = 1) -> None:
        """Instantiate a VowpalLearner.
        Args:
            cover: This value determines the number of policies which will be learned and must be
                greater than 0. For more information on this algorithm see Agarwal et al. (2014).
            seed: The seed used by VW to generate any necessary random numbers.

        References:
            Agarwal, Alekh, Daniel Hsu, Satyen Kale, John Langford, Lihong Li, and Robert Schapire. "Taming 
            the monster: A fast and simple algorithm for contextual bandits." In International Conference on 
            Machine Learning, pp. 1638-1646. 2014.
        """
        ...

    @overload
    def __init__(self, *, softmax: float, seed: Optional[int] = 1) -> None:
        """Instantiate a VowpalLearner.
        Args:
            softmax: An exploration parameter with 0 indicating predictions should be completely random
                and infinity indicating that predictions should be greedy. For more information see `lambda`
                at https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms.
            seed: The seed used by VW to generate any necessary random numbers.
        """
        ...

    @overload
    def __init__(self, *, regcb: Literal["opt","elim"], seed: Optional[int] = 1) -> None:
        """Instantiate a VowpalLearner.
        Args:
            regcb: Indicates whether exploration should only predict the optimal upper bound action
                or should use an elimination technique to remove actions that no longer seem plausible
                and pick randomly from the remaining actions.
            seed: The seed used by VW to generate any necessary random numbers.

        References:
            Foster, D., Agarwal, A., Dudik, M., Luo, H. & Schapire, R.. (2018). Practical Contextual 
            Bandits with Regression Oracles. Proceedings of the 35th International Conference on Machine 
            Learning, in Proceedings of Machine Learning Research 80:1539-1548.
        """
        ...

    @overload
    def __init__(self, *, squarecb: Literal["all","elim"], gamma_scale: float = 10, seed: Optional[int] = 1) -> None:
        """Instantiate a VowpalLearner.
        Args:
            squarecb: Indicates if all actions should be considered for exploration on each step or if actions
                which no longer seem plausible should be eliminated.
            gamma_scale: Controls how quickly squarecb exploration converges to a greedy policy. The larger the
                gamma_scale the faster the algorithm will converge to a greedy policy. This value is the same as
                gamma in the original paper.
            seed: The seed used by VW to generate any necessary random numbers.

        References:
            Foster, D.& Rakhlin, A.. (2020). Beyond UCB: Optimal and Efficient Contextual Bandits with Regression 
            Oracles. Proceedings of the 37th International Conference on Machine Learning, in Proceedings of Machine 
            Learning Research 119:3199-3210.
        """
        ...

    @overload
    def __init__(self, args:str) -> None:
        ...
        """Instantiate a VowpalLearner.
        Args:
            args: Command line argument to instantiates a Vowpal Wabbit contextual bandit learner. 
                For examples and documentation on how to instantiate VW learners from command line arguments see 
                https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms. It is assumed that
                either the --cb_explore or --cb_explore_adf flag is used. When formatting examples for VW, context
                features are namespaced with `s` and action features, when relevant, are namespaced with `a`.
        """

    def __init__(self, args:str = None, **kwargs) -> None:
        """Instantiate a VowpalLearner with the requested VW learner and exploration."""

        PackageChecker.vowpalwabbit("VowpalLearner")

        if args is not None and len(kwargs) > 0:
            raise CobaException("VowpalLearner expects to be initialized by keyword args alone or cli args alone.")

        if args:
            assert "--cb" in args, "VowpalLearner was instantiated without a cb learner being defined."

            self._exp  = "--cb_explore" in args
            self._adf  = "--cb_adf"     in args or "--cb_explore_adf" in args
            self._args = re.sub("--cb[^-]*", '', args, count=1)

        else:
            self._exp = True
            self._adf = True

            options = []

            kwargs.pop('adf', None) #this is for backwards compatability

            if 'epsilon' in kwargs:
                options.append(f"--epsilon {kwargs.pop('epsilon')}")

            if 'softmax' in kwargs:
                options.append(f"--softmax --lambda {kwargs.pop('softmax')}")

            if 'bag' in kwargs:
               options.append(f"--bag {kwargs.pop('bag')}")

            if 'cover' in kwargs:
                options.append(f"--cover {kwargs.pop('cover')}")

            if 'regcb' in kwargs:
                options.append(f"--regcb")
                if kwargs.pop('regcb') == "opt": options.append("--regcbopt")

            if 'squarecb' in kwargs:
                options.append(f"--squarecb --gamma_scale {kwargs.pop('gamma_scale',10)}")
                if kwargs.pop('squarecb') == "elim": options.append("--elim")

            if 'interactions' not in kwargs and 'cubic' not in kwargs:
                options.append("--interactions xxa --interactions xa --ignore_linear x")

            seed = kwargs.pop('seed',1)
            if seed is not None: options.append(f"--random_seed {seed}")

            options.extend([f"-{k} {v}" if len(k) == 1 else f"--{k} {v}" for k,v in kwargs.items()])
            self._args = " ".join(options)

        self._actions: Sequence[Action] = []
        self._vw                        = None

    @property
    def params(self) -> Dict[str, Any]:
        """The parameters of the learner.

        See the base class for more information
        """

        return {"family": "vw", 'args': self._cli_args(None)}

    def predict(self, context: Context, actions: Sequence[Action]) -> Tuple[Probs, Info]:
        """Determine a PMF with which to select the given actions.

        Args:
            context: The context we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The probability of taking each action. See the base class for more information.
        """

        if self._vw is None:
            # vowpal has an annoying warning that is written to stderr whether or not we provide
            # the --quiet flag. Therefore, we temporarily redirect all stderr output to null so that
            # this warning isn't shown during creation. It should be noted this isn't thread-safe
            # so if you are here because of strange problems with threads we may just need to suck
            # it up and accept that there will be an obnoxious warning message.
            self._vw = VowpalMediator.make_learner(self._cli_args(actions) + " --quiet")

            if not self._adf:
                self._actions = actions

        if not self._adf and (len(actions) != len(self._actions) or any(a not in self._actions for a in actions)):
            raise CobaException("Actions are only allowed to change between predictions with `--cb_explore_adf`.")

        info = (actions if self._adf else self._actions)

        shared = self._shared(context)
        adfs   = self._adfs(actions)

        if self._adf and self._exp:
            probs = self._vw.predict(self._examples(shared, adfs))

        if self._adf and not self._exp:
            loss_values    = self._vw.predict(self._examples(shared, adfs))
            min_loss_value = min(loss_values)
            min_indicators = [int(s == min_loss_value) for s in loss_values]
            min_count      = sum(min_indicators)
            probs          = [ min_indicator/min_count for min_indicator in min_indicators ]

        if not self._adf and self._exp:
            probs = self._vw.predict(self._example(shared))

        if not self._adf and not self._exp:
            index = self._vw.predict(self._example(shared))
            probs = [ int(i==index) for i in range(1,len(actions)+1) ]

        return probs, info

    def learn(self, context: Context, action: Action, reward: float, probability: float, info: Info) -> None:
        """Learn from the given interaction.

        Args:
            key: The key identifying the interaction this observed reward came from.
            context: The context we're learning about. See the base class for more information.
            action: The action that was selected in the context. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
            probability: The probability that the given action was taken.
        """

        assert self._vw is not None, "You must call predict before learn in order to initialize the VW learner"

        actions = info
        shared  = self._shared(context)
        adfs    = self._adfs(actions)
        labels  = self._labels(actions, action, reward, probability)
        label   = labels[actions.index(action)]

        if self._adf:
            self._vw.learn(self._examples(shared, adfs, labels))
        else:
            self._vw.learn(self._example(shared, label))

    def _cli_args(self, actions: Optional[Sequence[Action]]) -> str:

        base_learner = "--cb"

        if self._exp: base_learner += "_explore"
        if self._adf: base_learner += "_adf"
        
        if not self._adf: base_learner += f" {len(actions) if actions else ''}"

        return base_learner + " " + self._args

    def _examples(self, shared: Dict[str,Any], adfs: Sequence[Dict[str,Any]] = None, labels: Sequence[str] = repeat(None)):
        return [ self._example({**shared,**adf}, label) for adf,label in zip(adfs,labels) ]

    def _example(self, ns, label=None):
        return VowpalMediator.make_example(self._vw, ns, label, 4)

    def _labels(self,actions,action,reward:float,prob:float) -> Sequence[Optional[str]]:
        return [ f"{i+1}:{round(1-reward,5)}:{round(prob,5)}" if a == action else None for i,a in enumerate(actions)]

    def _shared(self, context) -> Dict[str,Any]:
        return {} if not context else { 'x': VowpalMediator.prep_features(self._flat(context)) }

    def _adfs(self,actions) -> Sequence[Dict[str,Any]]:
        return [ {'a': VowpalMediator.prep_features(self._flat(a))} for a in actions]

    def _flat(self,features:Any) -> Any:
        if features is None or isinstance(features,(int,float,str)):
            return features
        elif isinstance(features,dict):
            new_items = {}
            for k,v in features.items():
                if v is None or isinstance(v, (int,float,str)):
                    new_items[str(k)] = v
                else:
                    new_items.update( (f"{k}_{i}",f)  for i,f in enumerate(v))
            return new_items

        else:
            return [ff for f in features for ff in (f if isinstance(f,tuple) else [f]) ]