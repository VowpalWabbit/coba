"""The learners module contains core classes and types for defining learner simulations.

This module contains the abstract interface expected for Learner implementations along
with a number of Learner implementations out of the box for testing and baseline comparisons.

Todo:
    * Add VowpalAdfLearner
"""

import math
import random

from abc import ABC, abstractmethod
from typing import Callable, Sequence, Tuple, Optional, Dict, cast, Generic, TypeVar
from itertools import accumulate
from collections import defaultdict

from coba.simulations import State, Action, Reward
from coba.utilities import OnlineVariance, check_vowpal_support

_S_in = TypeVar('_S_in', bound=State , contravariant=True)
_A_in = TypeVar('_A_in', bound=Action, contravariant=True)

class Learner(Generic[_S_in, _A_in], ABC):
    """The interface for Learner implementations."""

    @abstractmethod
    def choose(self, state: _S_in, actions: Sequence[_A_in]) -> int:
        """Choose which action to take.

        Args:
            state: The current state. This argument will be None when playing 
                a multi-armed bandit simulation and will contain context features 
                when playing a contextual bandit simulation. Context features could 
                be an individual number (e.g. 1.34), a string (e.g., "hot"), or a 
                tuple of strings and numbers (e.g., (1.34, "hot")) depending on the 
                simulation being played.
            actions: The current set of actions to choose from in the given state. 
                Action sets can be lists of numbers (e.g., [1,2,3,4]), a list of 
                strings (e.g. ["high", "medium", "low"]), or a list of lists such 
                as in the case of movie recommendations (e.g., [["action", "oscar"], 
                ["fantasy", "razzie"]]).

        Returns:
            An integer indicating the index of the selected action in the action set.
        """
        ...
    
    @abstractmethod
    def learn(self, state: _S_in, action: _A_in, reward: Reward) -> None:
        """Learn about the result of an action that was taken in a state.

        Args:
            state: The current state. This argument will be None when playing 
                a multi-armed bandit simulation and will contain context features 
                when playing a contextual bandit simulation. Context features could 
                be an individual number (e.g. 1.34), a string (e.g., "hot"), or a 
                tuple of strings and numbers (e.g., (1.34, "hot")) depending on the 
                simulation being played.
            action: The action that was selected to play and observe its reward. 
                An Action can be an individual number (e.g., 2), a string (e.g. 
                "medium"), or a list of some combination of numbers or strings
                (e.g., ["action", "oscar"]).
            reward: the reward received for taking the given action in the given state.
        """
        ...

class LambdaLearner(Learner[_S_in, _A_in]):
    """A Learner implementation that chooses and learns according to provided lambda functions."""

    def __init__(self, 
                 choose: Callable[[_S_in, Sequence[_A_in]], int], 
                 learn : Optional[Callable[[_S_in, _A_in, Reward],None]] = None) -> None:
        """Instantiate LambdaLearner.

        Args:
            chooser: a function matching the super().choose() signature. All parameters are passed straight through.
            learner: a function matching the super().learn() signature. If provided all parameters are passed
                straight through. If the function isn't provided then no learning occurs.
        """
        self._choose = choose
        self._learn  = learn

    def choose(self, state: _S_in, actions: Sequence[_A_in]) -> int:
        """Choose via the provided lambda function.

        Args:
            state: The state we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The index of the selected action. See the base class for more information.
        """

        return self._choose(state, actions)

    def learn(self, state: _S_in, action: _A_in, reward: Reward) -> None:
        """Learn via the optional lambda function or learn nothing without a lambda function.

        Args:
            state: The state we're learning about. See the base class for more information.
            action: The action that was selected in the state. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
        """
        if self._learn is None:
            pass
        else:
            self._learn(state,action,reward)

class RandomLearner(Learner[State, Action]):
    """A Learner implementation that selects an action at random and learns nothing."""

    def choose(self, state: State, actions: Sequence[Action]) -> int:
        """Choose a random action from the action set.
        
        Args:
            state: The state we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The index of the selected action. See the base class for more information.
        """
        return random.randint(0, len(actions)-1)

    def learn(self, state: State, action: Action, reward: Reward) -> None:
        """Learns nothing.

        Args:
            state: The state we're learning about. See the base class for more information.
            action: The action that was selected in the state. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
        """

        pass

class EpsilonLearner(Learner[State, Action]):
    """A learner using epsilon-greedy searching while smoothing observations into a state/state-action lookup table.

    Remarks:
        This algorithm does not use any function approximation to attempt to generalize observed rewards.
    """

    def __init__(self, epsilon: float, default: Optional[float] = None, include_state: bool = False) -> None:
        """Instantiate an EpsilonLearner.

        Args:
            epsilon: A value between 0 and 1. We explore with probability epsilon and exploit otherwise.
            default: Our initial guess of the expected rewards for all state-action pairs.
            include_state: If true lookups are a function of state-action otherwise they are a function of action.
        """

        self._epsilon       = epsilon
        self._include_state = include_state
        self._N: Dict[Tuple[State, Action], int            ] = defaultdict(lambda: int(0 if default is None else 1))
        self._Q: Dict[Tuple[State, Action], Optional[float]] = defaultdict(lambda: default)

    def choose(self, state: State, actions: Sequence[Action]) -> int:
        """Choose greedily with probability 1-epsilon. Choose a randomly with probability epsilon.
        
        Args:
            state: The state we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The index of the selected action. See the base class for more information.
        """
        if(random.random() <= self._epsilon): return random.randint(0,len(actions)-1)

        keys        = [ self._key(state,action) for action in actions ]
        values      = [ self._Q[key] for key in keys ]
        max_value   = None if set(values) == {None} else max(v for v in values if v is not None)
        max_indexes = [i for i in range(len(values)) if values[i]==max_value]

        return random.choice(max_indexes)

    def learn(self, state: State, action: Action, reward: Reward) -> None:
        """Smooth the observed reward into our current estimate of either E[R|S,A] or E[R|A].

        Args:
            state: The state we're learning about. See the base class for more information.
            action: The action that was selected in the state. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
        """

        key   = self._key(state,action)
        alpha = 1/(self._N[key]+1)

        old_Q = cast(float, 0 if self._Q[key] is None else self._Q[key])

        self._Q[key] = (1-alpha) * old_Q + alpha * reward
        self._N[key] = self._N[key] + 1

    def _key(self, state: State, action: Action) -> Tuple[State,Action]:
        return (state, action) if self._include_state else (None, action)

class VowpalLearner(Learner[State, Action]):
    """A learner using Vowpal Wabbit's contextual bandit command line interface.

    Remarks:
        This learner requires that the Vowpal Wabbit package be installed. This package can be
        installed via `pip install vowpalwabbit`. To learn more about solving contextual bandit
        problems with Vowpal Wabbit see https://vowpalwabbit.org/tutorials/contextual_bandits.html.
    """

    def __init__(self, epsilon: Optional[float] = 0.1, bag: Optional[int] = None, cover: Optional[int] = None) -> None:
        """Instantiate a VowpalLearner.

        Args:
            epsilon: A value between 0 and 1. If provided exploration will follow epsilon-greedy.
            bag: An integer value greater than 0. This value determines how many separate policies will be
                learned. Each policy will be learned from bootstrap aggregation making each policy unique. 
                For each choice one policy will be selected according to a uniform distribution and followed.
            cover: An integer value greater than 0. This value value determines how many separate policies will be
                learned. These policies are learned in such a way to explicitly optimize policy diversity in order
                to control exploration. For each choice one policy will be selected according to a uniform distribution
                and followed. For more information on this algorithm see Agarwal et al. (2014).

        Remarks:
            Only one parameter of epsilon, bag and cover should be set. If more than one parameter is set then 
            only one value is used according to the precedence of first use cover then bag then epsilon.

        References:
            Agarwal, Alekh, Daniel Hsu, Satyen Kale, John Langford, Lihong Li, and Robert Schapire. "Taming 
            the monster: A fast and simple algorithm for contextual bandits." In International Conference on 
            Machine Learning, pp. 1638-1646. 2014.
        """

        check_vowpal_support('VowpalLearner.__init__')
        from vowpalwabbit import pyvw #type: ignore #ignored due to mypy error
        self._vw = pyvw.vw

        if epsilon is not None:
            self._explore = f"--epsilon {epsilon}"

        if bag is not None:
            self._explore = f"--bag {bag}"

        if cover is not None:
            self._explore = f"--cover {cover}"

        self._actions: Sequence[State]                  = []
        self._prob   : Dict[Tuple[State,Action], float] = {}

    def choose(self, state: State, actions: Sequence[Action]) -> int:
        """Choose an action according to the explor-exploit parameters passed into the contructor.

        Args:
            state: The state we're currently in. See the base class for more information.
            actions: The actions to choose from. See the base class for more information.

        Returns:
            The index of the selected action. See the base class for more information.

        Remarks:
            We assume that the action set passed in is always the same. This restriction
            is forced on us by Vowpal Wabbit. If your action set is not static then you
            should use VowpalAdfLearner.
        """

        if len(self._actions) == 0:
            self._actions = actions
            self._vw_learner = self._vw(f"--cb_explore {len(actions)} -q UA  {self._explore} --quiet")

        pmf = self._vw_learner.predict("| " + self._vw_format(state))

        cdf   = list(accumulate(pmf))
        rng   = random.random()
        index = [ rng < c for c in cdf].index(True)

        self._prob[(state, actions[index])] = pmf[index]

        return index

    def learn(self, state: State, action: Action, reward: Reward) -> None:
        """Learn from the obsered reward for the given state action pair.

        Args:
            state: The state we're learning about. See the base class for more information.
            action: The action that was selected in the state. See the base class for more information.
            reward: The reward that was gained from the action. See the base class for more information.
        """
        
        prob  = self._prob[(state,action)]
        cost  = -reward

        vw_state  = self._vw_format(state)
        vw_action = str(self._actions.index(action)+1)

        self._vw_learner.learn( vw_action + ":" + str(cost) + ":" + str(prob) + " | " + vw_state)

    def _vw_format(self, state: State) -> str:
        """convert state into the proper format for pyvw.
        
        Args:
            state: The state we wish to convert to pyvw representation.

        Returns:
            The state in pyvw representation.
        """
        if state is None:  return ""

        if isinstance(state, (int,float,str)):
            return str(state)

        #Right now if a state isn't one of the above types it
        #has to be a tuple. The type checker doesn't know that,
        #however, so we tell it with the explicit cast below.
        #During runtime this cast will do nothing.
        return " ". join(map(str, cast(tuple,state)))

class UcbTunedLearner(Learner[State, Action]):
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
    
    def choose(self, state: State, actions: Sequence[Action]) -> int:
        """Choose an action greedily according to the upper confidence bound estimates.

        Args:
            state: The state we're currently in. See the base class for more information.
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
            return random.choice(max_indexes)
        
    def learn(self, state: State, action: Action, reward: Reward) -> None:
        """Smooth the observed reward into our current estimate of E[R|S,A].

        Args:
            state: The state we're learning about. See the base class for more information.
            action: The action that was selected in the state. See the base class for more information.
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