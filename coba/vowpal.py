"""The vowpal module contains classes to make it easier to interact with pyvw.

TODO Add unittests
"""

from os import devnull
from typing import Any, Dict, Tuple, Union, Sequence

import coba.random
from coba.execution import redirect_stderr
from coba.utilities import check_vowpal_support
from coba.simulations import Context, Action, Choice

class cb_explore_Formatter:
    @staticmethod
    def predict(context, actions) -> str:
        return f"|s {_features_format(context)}"

    @staticmethod
    def learn(prob, actions, context: Context, action: Action, reward: float) -> str:
        return f"{actions.index(action)+1}:{-reward}:{prob} |s {_features_format(context)}"

class cb_explore_adf_Formatter:
    @staticmethod
    def predict(context: Context, actions:Sequence[Action]) -> str:
        vw_context = None if context is None else f"shared |s {_features_format(context)}"
        vw_actions = [ f"|a {_features_format(a)}" for a in actions]
        return "\n".join(filter(None,[vw_context, *vw_actions]))

    @staticmethod
    def learn(prob, actions, context: Context, action: Action, reward: float) -> str:
        vw_context   = None if context is None else f"shared |s {_features_format(context)}"
        vw_rewards  = [ "" if a != action else f"0:{-reward}:{prob}" for a in actions ]
        vw_actions  = [ _features_format(a) for a in actions]
        vw_observed = [ f"{r} |a {a}" for r,a in zip(vw_rewards,vw_actions) ]
        return "\n".join(filter(None,[vw_context, *vw_observed]))

class pyvw_Wrapper:
    def __init__(self, format: Union[cb_explore_Formatter, cb_explore_adf_Formatter], seed: int = None) -> None:
        check_vowpal_support('VowpalLearner.__init__')
        from vowpalwabbit import pyvw #type: ignore #ignored due to mypy error

        self._vw_init = pyvw.vw
        self._format  = format
        self._created = False
        self._seed    = seed
        self._random  = coba.random.CobaRandom(seed)

    @property
    def created(self):
        return self._created

    def create(self, flags: str):
        assert not self._created, "VW instance has already been created"
        
        seed_flag     = f"--random_seed {self._seed}" if self._seed is not None else ""
        self._created = True

        # vowpal has an annoying warning that is written to stderr whether or not we provide
        # the --quiet flag. Therefore, we temporarily redirect all stderr output to null so that
        # this warning isn't shown during creation. It should be noted this isn't thread-safe
        # so if you are here because of strange problems with threads we may just need to suck
        # it up and accept that there will be an obnoxious warning message.
        with open(devnull, 'w') as f, redirect_stderr(f):
            self._vw = self._vw_init(flags + f" --quiet {seed_flag}")

    def choose(self, context, actions) -> Tuple[Choice, float]:
        pmf  = self._vw.predict(self._format.predict(context, actions))

        assert len(pmf) == len(actions), "An incorrect number of action probabilites was returned by VW."
        assert abs(sum(pmf)-1) < .03   , "An invalid PMF for action probabilites was returned by VW."

        choices = list(range(len(actions)))
        choice = self._random.choice(choices, pmf)

        return choice, pmf[choice]

    def learn(self, actions, prob, context, action, reward):
        self._vw.learn(self._format.learn(actions, prob, context, action, reward))

class cb_explore:
    """A Vowpal Learner that assumes there is a fixed set of actions (aka, `--cb_explore`)."""

    def __init__(self, interactions:Sequence[str] = ["ssa", "sa"], ignore_linear: Sequence[str] = ["s"]) -> None:
        self._interactions  = interactions
        self._ignore_linear = ignore_linear
        self._actions: Sequence[Action] = []

    def params(self) -> Dict[str, Any]:
        #return {"interactions": self._interactions, "ignore_linear": self._ignore_linear}
        return {}

    def flags(self, actions: Sequence[Action]) -> str:
        n_actions    = len(self._actions)
        inter_flags  = " ".join([ f"--interactions {i}"  for i in self._interactions ]).strip()
        ignore_flags = " ".join([ f"--ignore_linear {n}" for n in self._ignore_linear]).strip()

        return f"--cb_explore {n_actions} {inter_flags} {ignore_flags}"

    @property
    def formatter(self) -> cb_explore_Formatter:
        return cb_explore_Formatter()

class cb_explore_adf:
    """A Vowpal Learner that makes no assumptions regarding actions (aka, `--cb_explore_adf`)."""

    def __init__(self, interactions: Sequence[str] = ["ssa", "sa"], ignore_linear: Sequence[str] = ["s"]) -> None:
        self._interactions  = interactions
        self._ignore_linear = ignore_linear

    def params(self) -> Dict[str, Any]:
        #return {"interactions": self._interactions, "ignore_linear": self._ignore_linear}
        return {"x":list(self._interactions) + [ f"-{i}" for i in self._ignore_linear]}

    def flags(self, actions) -> str:
        inter_flags  = " ".join([ f"--interactions {i}"  for i in self._interactions ]).strip()
        ignore_flags = " ".join([ f"--ignore_linear {n}" for n in self._ignore_linear]).strip()

        return f"--cb_explore_adf {inter_flags} {ignore_flags}"

    @property
    def formatter(self) -> cb_explore_adf_Formatter:
        return cb_explore_adf_Formatter()

class epsilongreedy:
    """A Vowpal exploration algorithm."""

    def __init__(self, epsilon:float = 0.025) -> None:
        """Instantiate VW epsilon-greedy exploration.
        
        Args:
            epsilon: A value between [0,1]. Actions will be selected at random with probability 
            epsilon otherwise actions will be selected greedily.
        """
        self._epsilon = epsilon

    def params(self) -> Dict[str, Any]:
        return {"epsilon": self._epsilon}

    def flags(self) -> str:
        """To vowpal wabbit representation."""

        return f"--epsilon {self._epsilon}"

class softmax:
    """A Vowpal exploration algorithm that heuristically considers confidence."""

    def __init__(self, lamda:float) -> None:
        """Instantiate VW softmax exploration.
        
        Args:
            _lambda: A value between [0,inf] that determines explores actions proportional to 
            exp(lamda*score(x,a)). When _lambda=0 this results in uniform exploration. As
            _lambda approaches inf exploration tends toward greedy selection with no exploration.
        """
        self.lamda = lamda
    
    def params(self) -> Dict[str, Any]:
        return {"lamda": self.lamda}

    def flags(self) -> str:
        """To vowpal wabbit representation."""

        return f"--softmax --lambda {self.lamda}"

class cover:
    """A Vowpal exploration algorithm with theoretical optimal guarantees."""

    def __init__(self, n_policies:int) -> None:
        """Instantiate VW cover exploration.
        
        Args:
            n_policies: An integer value greater than 0 determining the number of exploration policies
                to learn. These policies are learned so as to optimize diversity in order to control 
                exploration. On each choice one of the n exploration policies will be selected and followed. 
                For more information see Agarwal et al. (2014).           

        References:
            Agarwal, Alekh, Daniel Hsu, Satyen Kale, John Langford, Lihong Li, and Robert Schapire. "Taming 
            the monster: A fast and simple algorithm for contextual bandits." In International Conference on 
            Machine Learning, pp. 1638-1646. 2014.
        """
        self._n_policies = n_policies

    def params(self) -> Dict[str, Any]:
        return {"cover": self._n_policies}

    def flags(self) -> str:
        """To vowpal wabbit representation."""

        return f"--cover {self._n_policies}"

class bagging:
    """A Vowpal exploration algorithm utilizing bootstrap aggregation."""

    def __init__(self, n_policies:int) -> None:
        """Instantiate VW bagging exploration.

        Args:
            n_policies: An integer value greater than 0 determining the number of exploration policies 
                to learn. Each exploration policy is learned using bootstrap aggregation. On each choice 
                one of the n exploration policies will be selected and followed.
        """
        self._n_policies = n_policies

    def params(self) -> Dict[str, Any]:
        return {"bag": self._n_policies}

    def flags(self) -> str:
        """To vowpal wabbit representation."""
        
        return f"--bag {self._n_policies}"

def _features_format(features: Union[Context,Action]) -> str:
    """convert features into the proper format for pyvw.

    Args:
        features: The feature set we wish to convert to pyvw representation.

    Returns:
        The context in pyvw representation.

    Remarks:
        Note, using the enumeration index for action features below only works if all actions
        have the same number of features. If some actions simply leave out features in their
        feature array a more advanced method may need to be implemented in the future...
    """

    if not isinstance(features, tuple):
        features = (features,)

    if isinstance(features, tuple):
        return " ". join([_feature_format(i,f) for i,f in enumerate(features) if f is not None and f != 0 ])

    raise Exception("We were unable to determine an appropriate vw context format.")

def _feature_format(name: Any, value: Any) -> str:
    """Convert a feature into the proper format for pyvw.

    Args:
        name: The name of the feature.
        value: The value of the feature.

    Remarks:
        In feature formatting we prepend a "name" to each feature. This makes it possible
        to compare features across actions/contexts. See the definition of `Features` at 
        the top of https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format for more info.
    """

    return f"{name}:{value}" if isinstance(value,(int,float)) else f"{value}"
