"""The vowpal module contains classes to make it easier to interact with pyvw.

TODO Add unittests
"""

import collections

from os import devnull
from typing import Any, Dict, Union, Sequence, overload

from coba.random import CobaRandom
from coba.tools import PackageChecker, redirect_stderr
from coba.simulations import Context, Action
from coba.learners.core import Learner, Key

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
        PackageChecker.vowpalwabbit('VowpalLearner.__init__')
        from vowpalwabbit import pyvw #type: ignore #ignored due to mypy error

        self._vw_init = pyvw.vw
        self._format  = format
        self._created = False
        self._seed    = seed
        self._random  = CobaRandom(seed)

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

    def predict(self, context, actions) -> Sequence[float]:
        pmf  = self._vw.predict(self._format.predict(context, actions))

        assert len(pmf) == len(actions), "An incorrect number of action probabilites was returned by VW."
        assert abs(sum(pmf)-1) < .03   , "An invalid PMF for action probabilites was returned by VW."

        return pmf

    def learn(self, prob, actions, context, action, reward):
        self._vw.learn(self._format.learn(prob, actions, context, action, reward))

class cb_explore:
    """A Vowpal Learner that assumes there is a fixed set of actions (aka, `--cb_explore`)."""

    def __init__(self, interactions:Sequence[str] = ["ss"], ignore_linear: Sequence[str] = []) -> None:
        self._interactions  = interactions
        self._ignore_linear = ignore_linear

    def params(self) -> Dict[str, Any]:
        return {"x":list(self._interactions) + [ f"-{i}" for i in self._ignore_linear]}

    def flags(self, actions: Sequence[Action]) -> str:
        n_actions    = len(actions)
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

    if isinstance(features, dict):
        return " ". join([_feature_format(k,v) for k,v in features.items() if v is not None and v != 0 ])

    if not isinstance(features, collections.Sequence):
        features = (features,)

    if isinstance(features, collections.Sequence):
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

class VowpalLearner(Learner):
    """A learner using Vowpal Wabbit's contextual bandit command line interface.

    Remarks:
        This learner requires that the Vowpal Wabbit package be installed. This package can be
        installed via `pip install vowpalwabbit`. To learn more about solving contextual bandit
        problems with Vowpal Wabbit see https://vowpalwabbit.org/tutorials/contextual_bandits.html
        and https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms.
    """

    @overload
    def __init__(self, *, epsilon: float, is_adf: bool = True, seed:int = None) -> None:
        """Instantiate a VowpalLearner.
        Args:
            epsilon: A value between 0 and 1. If provided exploration will follow epsilon-greedy.
        """
        ...

    @overload
    def __init__(self, *, bag: int, is_adf: bool = True, seed:int = None) -> None:
        """Instantiate a VowpalLearner.
        Args:
            bag: An integer value greater than 0. This value determines how many separate policies will be
                learned. Each policy will be learned from bootstrap aggregation, making each policy unique. 
                For each choice one policy will be selected according to a uniform distribution and followed.
        """
        ...

    @overload
    def __init__(self, *, cover: int, seed:int = None) -> None:
        """Instantiate a VowpalLearner.
        Args:
            cover: An integer value greater than 0. This value value determines how many separate policies will be
                learned. These policies are learned in such a way to explicitly optimize policy diversity in order
                to control exploration. For each choice one policy will be selected according to a uniform distribution
                and followed. For more information on this algorithm see Agarwal et al. (2014).
        References:
            Agarwal, Alekh, Daniel Hsu, Satyen Kale, John Langford, Lihong Li, and Robert Schapire. "Taming 
            the monster: A fast and simple algorithm for contextual bandits." In International Conference on 
            Machine Learning, pp. 1638-1646. 2014.
        """
        ...

    @overload
    def __init__(self, *, softmax:float, seed:int = None) -> None:
        """Instantiate a VowpalLearner.
        Args:
            softmax: An exploration parameter with 0 indicating uniform exploration is desired and infinity
                indicating that no exploration is desired (aka, greedy action selection only). For more info
                see `lambda` at https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms.
        """
        ...

    @overload
    def __init__(self,
        learning: cb_explore,
        exploration: Union[epsilongreedy, bagging, cover], *, seed:int = None) -> None:
        ...
    
    @overload
    def __init__(self,
        learning: cb_explore_adf = cb_explore_adf(),
        exploration: Union[epsilongreedy, softmax, bagging] = epsilongreedy(0.025), 
        *, 
        seed:int = None) -> None:
        ...

    def __init__(self, 
        learning: Union[cb_explore,cb_explore_adf] = cb_explore_adf(),
        exploration: Union[epsilongreedy, softmax, bagging, cover] = epsilongreedy(0.025),
        **kwargs) -> None:
        """Instantiate a VowpalLearner with the requested VW learner and exploration."""

        self._learning: Union[cb_explore, cb_explore_adf]
        self._exploration: Union[epsilongreedy, softmax, bagging, cover]

        if 'epsilon' in kwargs:
            self._learning    = cb_explore_adf() if kwargs.get('is_adf',True) else cb_explore()
            self._exploration = epsilongreedy(kwargs['epsilon'])

        elif 'softmax' in kwargs:
            self._learning   = cb_explore_adf()
            self._exploration = softmax(kwargs['softmax'])

        elif 'bag' in kwargs:
            self._learning = cb_explore_adf() if kwargs.get('is_adf',True) else cb_explore()
            self._exploration = bagging(kwargs['bag'])

        elif 'cover' in kwargs:
            self._learning = cb_explore()
            self._exploration = cover(kwargs['cover'])

        else:
            self._learning = learning
            self._exploration = exploration

        self._probs: Dict[Key, Sequence[float]] = {}
        self._actions = self._new_actions(self._learning)

        self._flags = kwargs.get('flags', '')

        self._vw = pyvw_Wrapper(self._learning.formatter, seed=kwargs.get('seed', None))

    @property
    def family(self) -> str:
        """The family of the learner.

        See the base class for more information
        """
        return f"vw_{self._learning.__class__.__name__}_{self._exploration.__class__.__name__}"

    @property
    def params(self) -> Dict[str, Any]:
        """The parameters of the learner.
        
        See the base class for more information
        """

        params = {**self._learning.params(), **self._exploration.params()}

        if self._flags != '':
            params['flags'] = self._flags

        return params

    def predict(self, key: Key, context: Context, actions: Sequence[Action]) -> Sequence[float]:
        """Determine a PMF with which to select the given actions.

        Args:
            key: The key identifying the interaction we are choosing for.
            context: The context we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The probability of taking each action. See the base class for more information.
        """

        if not self._vw.created:
            self._vw.create(self._learning.flags(actions) + " " + self._exploration.flags() + " " + self._flags)

        probs = self._vw.predict(context, actions)

        self._set_actions(key,actions)

        if isinstance(self._learning, cb_explore):
            return [probs[i] for i in sorted(range(len(actions)), key=lambda i: actions.index(self._actions[i])) ]
        else:
            return probs

    def learn(self, key: Key, context: Context, action: Action, reward: float, probability: float) -> None:
        """Learn from the given interaction.

        Args:
            key: The key identifying the interaction this observed reward came from.
            context: The context we're learning about. See the base class for more information.
            action: The action that was selected in the context. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
            probability: The probability that the given action was taken.
        """

        actions = self._get_actions(key)
        self._vw.learn(probability, actions, context, action, reward)

    def _new_actions(self, learning) -> Any:
        if isinstance(learning, cb_explore):
            return []
        else:
            return {}

    def _set_actions(self, key, actions) -> None:
        if self._actions == []:
            self._actions = actions

        if isinstance(self._actions, collections.MutableMapping):
            self._actions[key] = actions

    def _get_actions(self, key) -> Sequence[Action]:
        if isinstance(self._actions, collections.MutableMapping) :
            return self._actions.pop(key)
        else:
            return self._actions