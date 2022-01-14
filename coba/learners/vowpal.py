import re
import collections.abc

from itertools import repeat
from numbers import Number
from typing import Any, Dict, Union, Sequence, Optional, Tuple
from coba.backports import Literal

from coba.exceptions import CobaException
from coba.utilities import PackageChecker, KeyDefaultDict
from coba.environments import Context, Action

from coba.learners.primitives import Learner, Probs, Info

Coba_Feature    = Union[str,float]
Coba_Features   = Union[Coba_Feature, Sequence[Coba_Feature], Sequence[Tuple[str,Coba_Feature]], Dict[str,Coba_Feature]]
Vowpal_Features = Sequence[Union[str,Tuple[str,float]]]

class VowpalMediator:
    """A class to handle all communication between coba and VW."""
    
    _string_cache = KeyDefaultDict(str)

    @staticmethod
    def make_args(options: Sequence[str],
        interactions: Sequence[str],
        ignore_linear:Sequence[str],
        seed: Optional[int], 
        **kwargs) -> str:
        """Turn specific settings into a VW command line arg string.

        Args:
            options: A sequence of string values that represent VW CLI options.
            interactions: A sequence of namespace interactions to use during learning.
            ignore_linear: A sequence of linear namespaces to ignore during learning.
            seed: A random number generator seed to make sure VW behaves consistently.
            kwargs: Any number of additional options to add to the arg string.
        """

        options = list(options)

        for interaction in interactions:
            options.append(f"--interactions {interaction}")

        for ignore in ignore_linear:
            options.append(f"--ignore_linear {ignore}")

        if seed is not None:
            options.append(f"--random_seed {seed}")

        for k,v in kwargs.items():
            k = ("-" if len(k)==1 else "--") + k
            options.append(k if v is None else f"{k} {v}")

        return " ".join(options)

    @staticmethod
    def prep_features(features: Coba_Features) -> Vowpal_Features:
        """Turn a collection of coba formatted features into VW format.
        
        Args:
            features: The features in coba format we wish to prepare for VW.
        """
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
            raise CobaException(f"Unrecognized features of type {type(features).__name__} passed to VowpalMediator.")

        return [f"{F[0]}={F[1]}" if isinstance(F[1],str) else (F[0], float(F[1])) for F in features if F[1] != 0]        

    @staticmethod
    def make_learner(args:str):
        """Create a VW learner from a command line arg string.
        
        Args:
            args: The command line arg string to use for VW learner creation.
        """
        PackageChecker.vowpalwabbit('VowpalMediator.make_learner')
        from vowpalwabbit import pyvw
        return pyvw.vw(args)
    

    @staticmethod
    def make_example(vw, ns:Dict[str,Vowpal_Features], label:Optional[str], label_type:int):
        """Create a VW example using the given features and optional label.
        
        Args:
            vw: The vw learner we are creating an example for.
            ns: The features grouped by namespace in this example.
            label: An optional label (required if this a learning example).
            label_type: The expected VW label_type (4 indicates a CB label).
        """
        PackageChecker.vowpalwabbit('VowpalMediator.make_example')
        from vowpalwabbit.pyvw import example

        ns = { k:v for k,v in ns.items() if v != [] }

        ex = example(vw, ns, label_type)
        
        if label: ex.set_label_string(label)

        ex.setup_example()

        return ex

    @staticmethod
    def get_version() -> str:
        """Return the current version of VW."""
        PackageChecker.vowpalwabbit('VowpalMediator.get_version')
        from vowpalwabbit.version import __version__
        return __version__

class VowpalArgsLearner(Learner):
    """A friendly wrapper around Vowpal Wabbit's python interface to support CB learning.
    
    Remarks: 
        This learner requires that the Vowpal Wabbit package be installed. This package can be
        installed via `pip install vowpalwabbit`. To learn more about solving contextual bandit
        problems with Vowpal Wabbit see `here`__ and `here`__.

    __ https://vowpalwabbit.org/tutorials/contextual_bandits.html
    __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms
    """

    def __init__(self, args: str = "--cb_explore_adf --epsilon 0.05 --interactions xxa --interactions xa --ignore_linear x --random_seed 1") -> None:
        """Instantiate a VowpalArgsLearner.

        Args:
            args: Command line arguments to instantiate a Vowpal Wabbit contextual bandit learner. For 
                examples and documentation on how to instantiate VW learners from command line arguments 
                see `here`__. We require that either cb, cb_adf, cb_explore, or cb_explore_adf is used. 
                When we format examples for VW context features are placed in the 'x' namespace and action 
                features, when relevant, are placed in the 'a' namespace.
        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms
        """

        PackageChecker.vowpalwabbit("VowpalArgsLearner")

        if "--cb" not in args: 
            raise CobaException("VowpalArgsLearner was instantiated without a cb flag. One cb flag must be defined.")

        self._exp  = "--cb_explore" in args
        self._adf  = "--cb_adf"     in args or "--cb_explore_adf" in args
        self._args = re.sub("--cb[^-]*", '', args, count=1)

        self._actions: Sequence[Action] = []
        self._vw                        = None

    @property
    def params(self) -> Dict[str, Any]:

        return {"family": "vw", 'args': self._cli_args(None)}

    def predict(self, context: Context, actions: Sequence[Action]) -> Tuple[Probs, Info]:

        if self._vw is None:
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

class VowpalEpsilonLearner(VowpalArgsLearner):
    """A wrapper around VowpalArgsLearner that provides more documentation. For more 
        information on the types of exploration algorithms availabe in VW see `here`__.

        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms
    """

    def __init__(self,
        epsilon: float = 0.05,
        interactions: Sequence[str] = ["xxa","xa"],
        ignore_linear: Sequence[str] = ["x"],
        seed: Optional[int] = 1,
        **kwargs) -> None:
        """Instantiate a VowpalEpsilonLearner.

        Args:
            epsilon: The probability that we will explore instead of exploit.
            interactions: A list of namespace interactions to use when learning reward functions.
            ignore_linear: A list of namespaces to ignore when learning reward functions.
            seed: The seed used by VW to generate any necessary random numbers.
        """

        options = [ "--cb_explore_adf", f"--epsilon {epsilon}" ]
        super().__init__(VowpalMediator.make_args(options, interactions, ignore_linear, seed, **kwargs))

class VowpalSoftmaxLearner(VowpalArgsLearner):
    """A wrapper around VowpalArgsLearner that provides more documentation. For more 
        information on the types of exploration algorithms availabe in VW see `here`__.

        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms
    """

    def __init__(self,
        softmax: float=10,
        interactions: Sequence[str] = ["xxa","xa"],
        ignore_linear: Sequence[str] = ["x"],
        seed: Optional[int] = 1,
        **kwargs) -> None:
        """Instantiate a VowpalSoftmaxLearner.

        Args:
            softmax: An exploration parameter with 0 indicating predictions should be completely random
                and infinity indicating that predictions should be greedy. For more information see `lambda`__.
            interactions: A list of namespace interactions to use when learning reward functions.
            ignore_linear: A list of namespaces to ignore when learning reward functions.
            seed: The seed used by VW to generate any necessary randomness.
        
        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms.
        """

        options = [ "--cb_explore_adf", "--softmax", f"--lambda {softmax}" ]
        super().__init__(VowpalMediator.make_args(options, interactions, ignore_linear, seed, **kwargs))

class VowpalBagLearner(VowpalArgsLearner):
    """A wrapper around VowpalArgsLearner that provides more documentation. For more 
        information on the types of exploration algorithms availabe in VW see `here`__.

        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms
    """

    def __init__(self,
        bag: int = 5,
        interactions: Sequence[str] = ["xxa","xa"],
        ignore_linear: Sequence[str] = ["x"],
        seed: Optional[int] = 1, 
        **kwargs) -> None:
        """Instantiate a VowpalBagLearner.

        Args:
            bag: This value determines the number of policies which will be learned and must be greater
                than 0. Each policy is trained using bootstrap aggregation, making each policy unique. During
                prediction a random policy will be selected according to a uniform distribution and followed.
            interactions: A list of namespace interactions to use when learning reward functions.
            ignore_linear: A list of namespaces to ignore when learning reward functions.
            seed: The seed used by VW to generate any necessary random numbers.
        """

        options = [ "--cb_explore_adf", f"--bag {bag}" ]
        super().__init__(VowpalMediator.make_args(options, interactions, ignore_linear, seed, **kwargs))

class VowpalCoverLearner(VowpalArgsLearner):
    """A wrapper around VowpalArgsLearner that provides more documentation. For more 
        information on the types of exploration algorithms availabe in VW see `here`__.

    For more information on this algorithm see Agarwal et al. (2014).

    References:
        Agarwal, Alekh, Daniel Hsu, Satyen Kale, John Langford, Lihong Li, and Robert Schapire. "Taming 
        the monster: A fast and simple algorithm for contextual bandits." In International Conference on 
        Machine Learning, pp. 1638-1646. 2014.

        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms
    """

    def __init__(self, 
        cover: int = 5,
        interactions: Sequence[str] = ["xxa","xa"],
        ignore_linear: Sequence[str] = ["x"],
        seed: Optional[int] = 1, 
        **kwargs) -> None:
        """Instantiate a VowpalCoverLearner.

        Args:
            cover: The number of policies which will be learned (must be greater than 0).
            interactions: A list of namespace interactions to use when learning reward functions.
            ignore_linear: A list of namespaces to ignore when learning reward functions.
            seed: The seed used by VW to generate any necessary random numbers.
        """

        options = [ "--cb_explore_adf", f"--cover {cover}" ]
        super().__init__(VowpalMediator.make_args(options, interactions, ignore_linear, seed, **kwargs))

class VowpalRegcbLearner(VowpalArgsLearner):
    """A wrapper around VowpalArgsLearner that provides more documentation. For more 
        information on the types of exploration algorithms availabe in VW see `here`__.

    References:
        Foster, D., Agarwal, A., Dudik, M., Luo, H. & Schapire, R.. (2018). Practical Contextual 
        Bandits with Regression Oracles. Proceedings of the 35th International Conference on Machine 
        Learning, in Proceedings of Machine Learning Research 80:1539-1548.

        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms
    """

    def __init__(self,
        mode: Literal["optimistic","elimination"] = "elimination",
        interactions: Sequence[str] = ["xxa","xa"],
        ignore_linear: Sequence[str] = ["x"],
        seed: Optional[int] = 1,
        **kwargs) -> None:
        """Instantiate a VowpalRegcbLearner.

        Args:
            mode: Indicates whether exploration should only predict the optimal upper bound action or
                should use an elimination technique to remove actions that no longer seem plausible
                and pick randomly from the remaining actions.
            interactions: A list of namespace interactions to use when learning reward functions.
            ignore_linear: A list of namespaces to ignore when learning reward functions.
            seed: The seed used by VW to generate any necessary random numbers.
        """

        options = [ "--cb_explore_adf", "--regcb" if mode=="elimination" else "--regcbopt" ]
        super().__init__(VowpalMediator.make_args(options, interactions, ignore_linear, seed, **kwargs))

class VowpalSquarecbLearner(VowpalArgsLearner):
    """A wrapper around VowpalArgsLearner that provides more documentation. For more 
        information on the types of exploration algorithms availabe in VW see `here`__.

    References:
        Foster, D.& Rakhlin, A.. (2020). Beyond UCB: Optimal and Efficient Contextual Bandits with Regression 
        Oracles. Proceedings of the 37th International Conference on Machine Learning, in Proceedings of Machine 
        Learning Research 119:3199-3210.

        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms
    """

    def __init__(self,
        mode: Literal["standard","elimination"] = "standard",
        gamma_scale: float = 10,
        interactions: Sequence[str] = ["xxa","xa"],
        ignore_linear: Sequence[str] = ["x"],
        seed: Optional[int] = 1,
        **kwargs) -> None:
        """Instantiate a VowpalSquarecbLearner.

        Args:
            mode: Indicates iwhether all actions should be considered for exploration on each step or actions
                which no longer seem plausible should be eliminated.
            gamma_scale: Controls how quickly squarecb exploration converges to a greedy policy. The larger the
                gamma_scale the faster the algorithm will converge to a greedy policy. This value is the same
                as gamma in the original paper.
            interactions: A list of namespace interactions to use when learning reward functions.
            ignore_linear: A list of namespaces to ignore when learning reward functions.
            seed: The seed used by VW to generate any necessary random numbers.
        """

        options = [
            "--cb_explore_adf",
            "--squarecb",
            f"--gamma_scale {gamma_scale}",
        ]

        if mode == "elimination": options.append("--elim")
        super().__init__(VowpalMediator.make_args(options, interactions, ignore_linear, seed, **kwargs))

class VowpalOffPolicyLearner(VowpalArgsLearner):
    """A wrapper around VowpalArgsLearner that provides more documentation. For more 
        information on the types of exploration algorithms availabe in VW see `here`__.

        This wrapper in particular performs policy learning without any exploration. This is
        only correct when training examples come from a logging policy so that any exploration 
        on our part is ignored.

        __ https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms
    """

    def __init__(self,
        interactions: Sequence[str] = ["xxa","xa"],
        ignore_linear: Sequence[str] = ["x"],
        seed: Optional[int] = 1,
        **kwargs) -> None:
        """Instantiate a VowpalOffPolicyLearner.

        Args:
            interactions: A list of namespace interactions to use when learning reward functions.
            ignore_linear: A list of namespaces to ignore when learning reward functions.
            seed: The seed used by VW to generate any necessary random numbers.
        """

        options = ["--cb_adf"]
        super().__init__(VowpalMediator.make_args(options, interactions, ignore_linear, seed, **kwargs))
