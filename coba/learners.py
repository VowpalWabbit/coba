"""The learners module contains core classes and types for defining learner simulations.

This module contains the abstract interface expected for Learner implementations along
with a number of Learner implementations out of the box for testing and baseline comparisons.

TODO Add VowpalAdfLearner
"""

import math

from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence, Tuple, Optional, Dict, cast, Generic, TypeVar, overload, Union
from itertools import accumulate
from collections import defaultdict
from inspect import signature

import coba.random
from coba.simulations import Context, Action, Reward, Choice, Key
from coba.utilities import check_vowpal_support
from coba.statistics import OnlineVariance

_C_in = TypeVar('_C_in', bound=Context, contravariant=True)
_A_in = TypeVar('_A_in', bound=Action , contravariant=True)

class Learner(Generic[_C_in, _A_in], ABC):
    """The interface for Learner implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """ The name of the learner.

        This value is used for descriptive purposes when creating benchmark results.
        """
        ...

    @abstractmethod
    def choose(self, key: Key, context: _C_in, actions: Sequence[_A_in]) -> Choice:
        """Choose which action to take.

        Args:
            key: A unique identifier for the interaction that the observed reward 
                came from. This identifier allows learners to share information
                between the choose and learn methods while still keeping the overall 
                learner interface consistent and clean.
            context: The current context. This argument will be None when playing 
                a multi-armed bandit simulation and will contain context features 
                when playing a contextual bandit simulation. Context features could 
                be an individual number (e.g. 1.34), a string (e.g., "hot"), or a 
                tuple of strings and numbers (e.g., (1.34, "hot")) depending on the 
                simulation being played.
            actions: The current set of actions to choose from in the given context. 
                Action sets can be lists of numbers (e.g., [1,2,3,4]), a list of 
                strings (e.g. ["high", "medium", "low"]), or a list of lists such 
                as in the case of movie recommendations (e.g., [["action", "oscar"], 
                ["fantasy", "razzie"]]).

        Returns:
            An integer indicating the index of the selected action in the action set.
        """
        ...

    @abstractmethod
    def learn(self, key: Key, context: _C_in, action: _A_in, reward: Reward) -> None:
        """Learn about the result of an action that was taken in a context.

        Args:
            key: A unique identifier for the interaction that the observed reward 
                came from. This identifier allows learners to share information
                between the choose and learn methods while still keeping the overall 
                learner interface consistent and clean.
            context: The current context. This argument will be None when playing 
                a multi-armed bandit simulation and will contain context features 
                when playing a contextual bandit simulation. Context features could 
                be an individual number (e.g. 1.34), a string (e.g., "hot"), or a 
                tuple of strings and numbers (e.g., (1.34, "hot")) depending on the 
                simulation being played.
            action: The action that was selected to play and observe its reward. 
                An Action can be an individual number (e.g., 2), a string (e.g. 
                "medium"), or a list of some combination of numbers or strings
                (e.g., ["action", "oscar"]).
            reward: the reward received for taking the given action in the given context.
        """
        ...

class LambdaLearner(Learner[_C_in, _A_in]):
    """A Learner implementation that chooses and learns according to provided lambda functions."""

    @overload
    def __init__(self, 
                choose: Callable[[_C_in, Sequence[_A_in]], Choice], 
                learn : Optional[Callable[[_C_in, _A_in, Reward],None]] = None,
                name  : str = "Lambda") -> None:
        ...
    
    @overload
    def __init__(self, 
                 choose: Callable[[Key, _C_in, Sequence[_A_in]], Choice], 
                 learn : Optional[Callable[[Key, _C_in, _A_in, Reward],None]] = None,
                 name  : str = "Lambda") -> None:
        ...

    def __init__(self, choose, learn = None, name: str = "Lambda") -> None:
        """Instantiate LambdaLearner.

        Args:
            chooser: a function matching the super().choose() signature. All parameters are passed straight through.
            learner: a function matching the super().learn() signature. If provided all parameters are passed
                straight through. If the function isn't provided then no learning occurs.
        """

        if len(signature(choose).parameters) == 2:
            og_choose = choose
            choose = lambda k,c,A: cast(Callable[[_C_in, Sequence[_A_in]], Choice], og_choose)(c,A)

        if learn is not None and len(signature(learn).parameters) == 3:
            og_learn = learn
            learn = lambda k,c,a,r: cast(Callable[[_C_in, _A_in, Reward], None], og_learn)(c,a,r)

        self._choose = choose
        self._learn  = learn
        self._name   = name

    @property
    def name(self) -> str:
        """The name of the Learner.
        
        See the base class for more information
        """  
        return self._name

    def choose(self, key: Key, context: _C_in, actions: Sequence[_A_in]) -> Choice:
        """Choose via the provided lambda function.

        Args:
            key: The key identifying the interaction we are choosing for.
            context: The context we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The index of the selected action. See the base class for more information.
        """

        return self._choose(key, context, actions)

    def learn(self, index: int, context: _C_in, action: _A_in, reward: Reward) -> None:
        """Learn via the optional lambda function or learn nothing without a lambda function.

        Args:
            key: The key identifying the interaction this observed reward came from.
            context: The context we're learning about. See the base class for more information.
            action: The action that was selected in the context. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
        """
        if self._learn is None:
            pass
        else:
            self._learn(index, context,action,reward)

class RandomLearner(Learner[Context, Action]):
    """A Learner implementation that selects an action at random and learns nothing."""

    @property
    def name(self) -> str:
        """The name of the Learner.

        See the base class for more information
        """  
        return "Random"

    def choose(self, key: Key, context: Context, actions: Sequence[Action]) -> Choice:
        """Choose a random action from the action set.
        
        Args:
            key: The key identifying the interaction we are choosing for.
            context: The context we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The index of the selected action. See the base class for more information.
        """
        return coba.random.randint(0, len(actions)-1)

    def learn(self, key: Key, context: Context, action: Action, reward: Reward) -> None:
        """Learns nothing.

        Args:
            key: The key identifying the interaction this observed reward came from.
            context: The context we're learning about. See the base class for more information.
            action: The action that was selected in the context. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
        """

        pass

class EpsilonLearner(Learner[Context, Action]):
    """A learner using epsilon-greedy searching while smoothing observations into a context/context-action lookup table.

    Remarks:
        This algorithm does not use any function approximation to attempt to generalize observed rewards.
    """

    def __init__(self, epsilon: float, default: Optional[float] = None, include_context: bool = False) -> None:
        """Instantiate an EpsilonLearner.

        Args:
            epsilon: A value between 0 and 1. We explore with probability epsilon and exploit otherwise.
            default: Our initial guess of the expected rewards for all context-action pairs.
            include_context: If true lookups are a function of context-action otherwise they are a function of action.
        """

        self._epsilon       = epsilon
        self._include_context = include_context
        self._N: Dict[Tuple[Context, Action], int            ] = defaultdict(lambda: int(0 if default is None else 1))
        self._Q: Dict[Tuple[Context, Action], Optional[float]] = defaultdict(lambda: default)

    @property
    def name(self) -> str:
        """The name of the Learner.

        See the base class for more information
        """
        return "Epislon-greedy"

    def choose(self, key: Key, context: Context, actions: Sequence[Action]) -> Choice:
        """Choose greedily with probability 1-epsilon. Choose a randomly with probability epsilon.
        
        Args:
            key: The key identifying the interaction we are choosing for.
            context: The context we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The index of the selected action. See the base class for more information.
        """
        if(coba.random.random() <= self._epsilon): return coba.random.randint(0,len(actions)-1)

        keys        = [ self._key(context,action) for action in actions ]
        values      = [ self._Q[key] for key in keys ]
        max_value   = None if set(values) == {None} else max(v for v in values if v is not None)
        max_indexes = [i for i in range(len(values)) if values[i]==max_value]

        return coba.random.choice(max_indexes)

    def learn(self, key: Key, context: Context, action: Action, reward: Reward) -> None:
        """Smooth the observed reward into our current estimate of either E[R|S,A] or E[R|A].

        Args:
            key: The key identifying the interaction this observed reward came from.
            context: The context we're learning about. See the base class for more information.
            action: The action that was selected in the context. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
        """

        sa_key = self._key(context,action)
        alpha  = 1/(self._N[sa_key]+1)

        old_Q = cast(float, 0 if self._Q[sa_key] is None else self._Q[sa_key])

        self._Q[sa_key] = (1-alpha) * old_Q + alpha * reward
        self._N[sa_key] = self._N[sa_key] + 1

    def _key(self, context: Context, action: Action) -> Tuple[Context,Action]:
        return (context, action) if self._include_context else (None, action)

class UcbTunedLearner(Learner[Context, Action]):
    """This is an implementation of Auer et al. (2002) UCB1-Tuned algorithm.

    References:
        Auer, Peter, Nicolo Cesa-Bianchi, and Paul Fischer. "Finite-time analysis of 
        the multiarmed bandit problem." Machine learning 47.2-3 (2002): 235-256.
    """
    def __init__(self):
        """Instantiate a UcbTunedLearner."""

        self._init_a: int = 0
        self._t     : int = 0
        self._s     : Dict[Action,int] = {}
        self._m     : Dict[Action,float] = {}
        self._v     : Dict[Action,OnlineVariance] = defaultdict(OnlineVariance)

    @property
    def name(self) -> str:
        """The name of the Learner.

        See the base class for more information
        """
        return "UCB"

    def choose(self, key: Key, context: Context, actions: Sequence[Action]) -> Choice:
        """Choose an action greedily according to the upper confidence bound estimates.

        Args:
            key: The key identifying the interaction we are choosing for.
            context: The context we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The index of the selected action. See the base class for more information.
        """
        #we initialize by playing every action once
        if self._init_a < len(actions):
            self._init_a += 1
            return self._init_a-1

        else:
            values      = [ self._m[a] + self._Avg_R_UCB(a) if a in self._m else None for a in actions ]
            max_value   = None if set(values) == {None} else max(v for v in values if v is not None)
            max_indexes = [i for i in range(len(values)) if values[i]==max_value]
            return coba.random.choice(max_indexes)

    def learn(self, key: Key, context: Context, action: Action, reward: Reward) -> None:
        """Smooth the observed reward into our current estimate of E[R|S,A].

        Args:
            key: The key identifying the interaction this observed reward came from.
            context: The context we're learning about. See the base class for more information.
            action: The action that was selected in the context. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
        """
        if action not in self._s:
            self._s[action] = 1
        else:
            self._s[action] += 1

        if action not in self._m:
            self._m[action] = reward
        else:
            self._m[action] = (1-1/self._s[action]) * self._m[action] + 1/self._s[action] * reward

        self._t         += 1
        self._s[action] += 1
        self._v[action].update(reward)

    def _Avg_R_UCB(self, action: Action) -> float:
        """Produce the estimated upper confidence bound (UCB) for E[R|A].

        Args:
            action: The action for which we want to retrieve UCB for E[R|A].

        Returns:
            The estimated UCB for E[R|A].

        Remarks:
            See the beginning of section 4 in the algorithm's paper for this equation.
        """
        ln = math.log; n = self._t; n_j = self._s[action]; V_j = self._Var_R_UCB(action)

        return math.sqrt(ln(n)/n_j * min(1/4,V_j))

    def _Var_R_UCB(self, action: Action) -> float:
        """Produce the upper confidence bound (UCB) for Var[R|A].

        Args:
            action: The action for which we want to retrieve UCB for Var[R|A].

        Returns:
            The estimated UCB for Var[R|A].

        Remarks:
            See the beginning of section 4 in the algorithm's paper for this equation.
        """
        ln = math.log; t = self._t; s = self._s[action]; var = self._v[action].variance

        return var + math.sqrt(2*ln(t)/s)

class VowpalLearner(Learner[Context, Action]):
    """A learner using Vowpal Wabbit's contextual bandit command line interface.

    Remarks:
        This learner requires that the Vowpal Wabbit package be installed. This package can be
        installed via `pip install vowpalwabbit`. To learn more about solving contextual bandit
        problems with Vowpal Wabbit see https://vowpalwabbit.org/tutorials/contextual_bandits.html
        and https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms.
    """

    @overload
    def __init__(self, *, epsilon: float = 0.025, force_tabular: bool = False) -> None:
        """Instantiate a VowpalLearner.

        Args:
            epsilon: A value between 0 and 1. If provided exploration will follow epsilon-greedy.
        """
        ...
    
    @overload
    def __init__(self, *, bag: int, force_tabular: bool = False) -> None:
        """Instantiate a VowpalLearner.

        Args:
            bag: An integer value greater than 0. This value determines how many separate policies will be
                learned. Each policy will be learned from bootstrap aggregation making each policy unique. 
                For each choice one policy will be selected according to a uniform distribution and followed.
        """
        ...

    @overload
    def __init__(self, *, cover: int) -> None:
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
    def __init__(self, *, softmax: float) -> None:
        ...

    @overload
    def __init__(self, *, rnd: float, epsilon:float = 0.025, rnd_invlambda: float = 0.1, rnd_alpha:float = 0.1) -> None:
        ...

    def __init__(self, *, 
        epsilon: float = None, 
        bag: int = None, 
        cover: int = None, 
        softmax: float = None,
        rnd: float = None,
        rnd_invlambda: float = 0.1,
        rnd_alpha: float = 0.1,
        force_tabular: bool = False) -> None:
        """Instantiate a VowpalLearner.

        See @overload signatures for more information.
        """

        check_vowpal_support('VowpalLearner.__init__')
        from vowpalwabbit import pyvw #type: ignore #ignored due to mypy error
        self._vw = pyvw.vw

        if all(algorithm == None for algorithm in [bag, cover, softmax, rnd]):
            epsilon = 0.1 if epsilon is None else epsilon 
            self._algorithm = f"--epsilon {epsilon}"

        if bag is not None:
            self._algorithm = f"--bag {bag}"

        if cover is not None:
            self._algorithm = f"--cover {cover}"

        if softmax is not None:
            self._algorithm = f"--softmax {softmax}"

        if rnd is not None:
            epsilon = 0.025 if epsilon is None else epsilon 
            self._algorithm = f"--rnd {rnd} --epsilon {epsilon} --rnd_invlambda {rnd_invlambda} --rnd_alpha {rnd_alpha}"

        self._actions      : Any              = None
        self._vw_learner   : pyvw.vw          = None
        self._prob         : Dict[int, float] = {}
        self._force_tabular: bool             = force_tabular

    @property
    def name(self) -> str:
        """The name of the Learner.
        
        See the base class for more information
        """  
        return "Vowpal"

    def choose(self, key: Key, context: Context, actions: Sequence[Action]) -> Choice:
        """Choose an action according to the explor-exploit parameters passed into the contructor.

        Args:
            key: The key identifying the interaction we are choosing for.
            context: The context we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The index of the selected action. See the base class for more information.

        Remarks:
            We assume that the action set passed in is always the same. This restriction
            is forced on us by Vowpal Wabbit. If your action set is not static then you
            should use VowpalAdfLearner.
        """

        if self._vw_learner is None and self._force_tabular:
            self._actions = actions
            self._mode    = f"--cb_explore {len(actions)}"

        if self._vw_learner is None and not self._force_tabular:
            self._actions = {}
            self._mode = f"--cb_explore_adf"

        if self._vw_learner is None:
            self._vw_learner = self._vw(f"{self._mode} {self._algorithm} --quiet")

        if not self._force_tabular:
            self._actions[key] = actions

        pmf = self._vw_learner.predict(self._vw_predict_format(context, actions))

        cdf    = list(accumulate(pmf))
        rng    = coba.random.random()
        choice = [ rng < c for c in cdf].index(True)
        action = actions[choice]

        self._prob[key] = pmf[choice]

        return choice if not self._force_tabular else self._actions.index(action)

    def learn(self, key: Key, context: Context, action: Action, reward: Reward) -> None:
        """Learn from the obsered reward for the given context action pair.

        Args:
            key: The key identifying the interaction this observed reward came from.
            context: The context we're learning about. See the base class for more information.
            action: The action that was selected in the context. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
        """

        self._vw_learner.learn(self._vw_learn_format(key, context, action, reward))

    def _vw_predict_format(self, context: Context, actions:Sequence[Action]) -> str:
        """Convert context and actions into the proper prediction format for vowpal wabbit.
        
        Args:
            context: The context we wish to convert to vowpal wabbit representation.
            actions: The actions we wish to predict from.

        Returns:
            The proper format for vowpal wabbit prediction.
        """

        if self._force_tabular:
            return f"| {self._vw_features_format(context)}"
        else:
            vw_context = None if context is None else f"shared | {self._vw_features_format(context)}"
            vw_actions = [ f"| {self._vw_features_format(a)}" for a in actions]

            return "\n".join(filter(None,[vw_context, *vw_actions]))

    def _vw_learn_format(self, key: Key, context: Context, action: Action, reward: float) -> str:
        prob    = self._prob.pop(key)
        vw_actions = self._actions if self._force_tabular else self._actions.pop(key)

        if self._force_tabular:
            return f"{vw_actions.index(action)+1}:{-reward}:{prob} | {self._vw_features_format(context)}"
        else:
            vw_context   = None if context is None else f"shared | {self._vw_features_format(context)}"
            vw_rewards  = [ "" if a != action else f"0:{-reward}:{prob}" for a in vw_actions ]
            vw_actions  = [ self._vw_features_format(a) for a in vw_actions]
            vw_observed = [ f"{r} | {a}" for r,a in zip(vw_rewards,vw_actions) ]

            return "\n".join(filter(None,[vw_context, *vw_observed]))

    def _vw_features_format(self, features: Union[Context,Action]) -> str:
        """convert action features into the proper format for pyvw.
        
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
            return " ". join([ self._vw_feature_format(i,f) for i,f in enumerate(features) if f is not None ])

        raise Exception("We were unable to determine an appropriate vw context format.")

    def _vw_feature_format(self, name: Any, value: Any) -> str:
        """Convert a feature into the proper format for pyvw.

        Args:
            name: The name of the feature.
            value: The value of the feature.

        Remarks:
            In feature formatting we prepend a "key" to each feature. This makes it possible
            to compare features across actions in an ADF setting (which is the only setting where
            we'd be sending action features to vowpal wabbit). See the definition of `Features` at 
            the top of https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format for more info.
        """
        return f"{name}:{value}" if isinstance(value,(int,float)) else f"{value}"