"""The learners module contains core classes and types for defining learner simulations.

This module contains the abstract interface expected for Learner implementations along
with a number of Learner implementations out of the box for testing and baseline comparisons.

Todo:
    * Add UCBLearner
    * Add VowpalAdfLearner
"""

import math
import random

from abc import ABC, abstractmethod
from typing import Callable, Sequence, Tuple, Union, Optional, Dict, Any, Iterable, Set, cast
from itertools import accumulate
from collections import defaultdict

from bbench.simulations import State, Action, Reward
from bbench.utilities import check_vowpal_support

class Learner(ABC):
    """The interface for Learner implementations."""

    @abstractmethod
    def choose(self, state: State, actions: Sequence[Action]) -> int:
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
    def learn(self, state: State, action: Action, reward: Reward) -> None:
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

class RandomLearner(Learner):
    """A Learner implementation that selects an action at random and learns nothing."""

    def choose(self, state: State, actions: Sequence[Action]) -> int:
        """Choose a random action from the action set.
        
        Args:
            state: See the base class for more information.
            actions: See the base class for more information.

        Returns:
            See the base class for more information.
        """
        return random.randint(0,len(actions)-1)

    def learn(self, state: State, action: Action, reward: Reward) -> None:
        """Learn nothing.

        Args:
            state: See the base class for more information.
            action: See the base class for more information.
            reward: See the base class for more information.
        """

        pass

class LambdaLearner(Learner):
    """A Learner implementation that chooses and learns according to provided lambda functions."""

    def __init__(self, 
                 choose: Callable[[State,Sequence[Action]],int], 
                 learn : Optional[Callable[[State,Action,Reward],None]] = None) -> None:
        """Instantiate LambdaLearner.

        Args:
            chooser: a function matching the super().choose() signature. All parameters are passed straight through.
            learner: an optional function matching the super().learn() signature. If provided all parameters are passed
                straight through. If the function isn't provided then no learning occurs.
        """
        self._choose = choose
        self._learn  = learn

    def choose(self, state: State, actions: Sequence[Action]) -> int:
        """Choose via the provided lambda function.

        Args:
            state: See the base class for more information.
            actions: See the base class for more information.

        Returns:
            See the base class for more information.
        """

        return self._choose(state, actions)

    def learn(self, state: State, action: Action, reward: Reward) -> None:
        """Learn via the optional lambda function or learn nothing without a lambda function.

        Args:
            state: See the base class for more information.
            action: See the base class for more information.
            reward: See the base class for more information.
        """
        if self._learn is None:
            pass
        else:
            self._learn(state,action,reward)

class EpsilonLookupLearner(Learner):

    def __init__(self, epsilon: float, default: Optional[float] = None, include_state: bool = False) -> None:
        self._epsilon       = epsilon
        self._include_state = include_state
        self._N: Dict[Tuple[State, Action], int            ] = defaultdict(lambda: 0 if default is None else 1)
        self._Q: Dict[Tuple[State, Action], Optional[float]] = defaultdict(lambda: default)

    def choose(self, state: State, actions: Sequence[Action]) -> int:

        if(random.random() <= self._epsilon): return random.randint(0,len(actions)-1)

        keys        = [ self._key(state,action) for action in actions ]
        values      = [ self._Q[key] for key in keys ]
        max_value   = None if set(values) == {None} else max(v for v in values if v is not None)
        max_indexes = [i for i in range(len(values)) if values[i]==max_value]

        return random.choice(max_indexes)

    def learn(self, state: State, action: Action, reward: Reward) -> None:

        key   = self._key(state,action)
        alpha = 1/(self._N[key]+1)

        old_Q = cast(float, 0 if self._Q[key] is None else self._Q[key])

        self._Q[key] = (1-alpha) * old_Q + alpha * reward
        self._N[key] = self._N[key] + 1

    def _key(self, state: State, action: Action) -> Tuple[State,Action]:
        return (state, action) if self._include_state else (None, action)

class VowpalLearner(Learner):
    def __init__(self, epsilon: Optional[float] = 0.1, bag: Optional[int] = None, cover: Optional[int] = None) -> None:

        check_vowpal_support('VowpalLearner.__init__')

        if epsilon is not None:
            self._explore = f"--epsilon {epsilon}"

        if bag is not None:
            self._explore = f"--bag {bag}"

        if cover is not None:
            self._explore = f"--cover {cover}"

        self._actions: Sequence[Action]                 = []
        self._prob   : Dict[Tuple[State,Action], float] = {}

    def choose(self, state: State, actions: Sequence[Action]) -> int:
        """
        Remarks:
            We assume that the action set passed in is always the same. This restriction
            is forced on us by Vowpal Wabbit. If your action set is not static then you
            should use VowpalAdfLearner
        """

        if len(self._actions) == 0:
            from vowpalwabbit import pyvw #type: ignore
            self._actions = actions
            self._vw = pyvw.vw(f"--cb_explore {len(actions)} -q UA  {self._explore} --quiet")

        pmf = self._vw.predict("| " + self._vw_format(state))

        cdf   = list(accumulate(pmf))
        rng   = random.random()
        index = [ rng < c for c in cdf].index(True)

        self._prob[self._key(state, actions[index])] = pmf[index]

        return index

    def learn(self, state: State, action: Action, reward: Reward) -> None:
        
        prob  = self._prob[self._key(state,action)]
        cost  = -reward

        vw_state  = self._vw_format(state)
        vw_action = str(self._actions.index(action)+1)

        self._vw.learn( vw_action + ":" + str(cost) + ":" + str(prob) + " | " + vw_state)

    def _vw_format(self, state: State) -> str:
        
        if state is None:  return ""

        if isinstance(state, (int,float,str)):
            return str(state)

        return " ". join(map(str,state))

    def _key(self, state: State, action: Action) -> Tuple[State,Action]:
        return (state, action)

class UcbTunedLearner(Learner):
    """This is an implementation of Auer et al. (2002) UCB1-Tuned algorithm.
    
    References:
        Auer, Peter, Nicolo Cesa-Bianchi, and Paul Fischer. "Finite-time analysis of 
        the multiarmed bandit problem." Machine learning 47.2-3 (2002): 235-256.
    """
    def __init__(self):
        self._init_a: int = 0
        self._t     : int = 0
        self._s     : Dict[Action,int] = {}
        self._v     : Dict[Action,float] = {}
        self._m     : Dict[Action,float] = {}
        self._w     : Dict[Action,Tuple[int,float,float]] = {}
    
    def choose(self, state: State, actions: Sequence[Action]) -> int:

        #we initialize by playing every action once
        if self._init_a < len(actions):
            i = self._init_a
            self._init_a += 1
        else:
            values      = [ self._m[a] + self._UCB(a) if a in self._m else None for a in actions ]
            max_value   = None if set(values) == {None} else max(v for v in values if v is not None)
            max_indexes = [i for i in range(len(values)) if values[i]==max_value]

            i = random.choice(max_indexes)

        return i
        
    def learn(self, state: State, action: Action, reward: Reward) -> None:
        
        if action not in self._s:
            self._s[action] = 1
        else:
            self._s[action] += 1

        if action not in self._m:
            self._m[action] = reward
        else:
            self._m[action] = 1/self._s[action] * reward + (1-1/self._s[action]) * self._m[action]

        self._t         += 1
        self._s[action] += 1
        self._update_v(action, reward)

    def _UCB(self, action: Action) -> float:
        return math.sqrt((math.log(self._t)/self._s[action]) * self._V(action))

    def _V(self, action: Action) -> float:
        return self._v[action] + math.sqrt(2*math.log(self._t)/self._s[action])

    def _update_v(self, action: Action, reward: Reward):

        #Welfords algorithm for online variance
        #taken largely from Wikipedia
        if action not in self._w:
            (count,mean,M2) = (1,reward,0.)
        else:
            (count,mean,M2) = self._w[action]
            count += 1
            delta = reward - mean
            mean += delta / count
            delta2 = reward - mean
            M2 += delta * delta2

        self._w[action] = (count,mean,M2)

        if count == 1:
            self._v[action] = 0
        else:
            self._v[action] = M2 / (count - 1)