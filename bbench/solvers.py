"""The solvers module contains core bandit solving interfaces and implementations.

This module contains the abstract interface expected for Solver implementations along
with a number of simple Solver implementations for testing and baseline comparisons.

Todo:
    * Add more Solver implementations
"""

from abc import ABC, abstractmethod
from typing import Callable, Sequence, Optional
from random import randint

from bbench.games import State, Action, Reward

class Solver(ABC):
    """The interface for Solver implementations."""

    @abstractmethod
    def choose(self, state: Optional[State], actions: Sequence[Action]) -> int:
        """Choose which action to take.

        Args:
            state: The current state. This argument will be None when playing 
                a multi-armed bandit game and will contain context features 
                when playing a contextual bandit game. Context features could 
                be an individual number (e.g. 1.34), a string (e.g., "hot"), 
                or a list of strings and numbers (e.g., [1.34, "hot"]) depending
                on the Game being played.
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
    def learn(self, state: Optional[State], action: Action, reward: Reward) -> None:
        """Learn about the result of an action that was taken in a state.

        Args:
            state: The state in which the action was taken. This argument will be None 
                when playing a multi-armed bandit game and will contain context features 
                when playing a contextual bandit game. Depending on the Game being played 
                context features can be an individual numbers (e.g. 1.34), a string (e.g., 
                "hot"), or a list of strings and numbers (e.g., [1.34, "hot"]).
            action: The action that was selected to play and observe its reward. 
                An Action can be an individual number (e.g., 2), a string (e.g. 
                "medium"), or a list of some combination of numbers or strings
                (e.g., ["action", "oscar"]).
            reward: the reward received for taking the given action in the given state.
        """
        ...

class RandomSolver(Solver):
    """A Solver implementation that selects an action at random and learns nothing."""

    def choose(self, state: Optional[State], actions: Sequence[Action]) -> int:
        """Choose a random action from the action set.
        
        Args:
            state: See the base class for more information.
            actions: See the base class for more information.

        Returns:
            See the base class for more information.
        """
        return randint(0,len(actions)-1)

    def learn(self, state: Optional[State], action: Action, reward: Reward) -> None:
        """Learn nothing.

        Args:
            state: See the base class for more information.
            action: See the base class for more information.
            reward: See the base class for more information.
        """

        pass

class LambdaSolver(Solver):
    """A Solver implementation that chooses and learns according to provided functions."""

    def __init__(self, 
                 chooser: Callable[[Optional[State],Sequence[Action]],int], 
                 learner: Optional[Callable[[Optional[State],Action,Reward],None]] = None) -> None:
        """Instantiate LambdaSolver.

        Args:
            chooser: a function matching the super().choose() signature. All parameters are passed straight through.
            learner: an optional function matching the super().learn() signature. If provided all parameters are passed
                straight through. If the function isn't provided then no learning occurs.
        """
        self._chooser = chooser
        self._learner = learner

    def choose(self, state: Optional[State], actions: Sequence[Action]) -> int:
        """Choose via the provided lambda function.

        Args:
            state: See the base class for more information.
            actions: See the base class for more information.

        Returns:
            See the base class for more information.
        """

        return self._chooser(state, actions)

    def learn(self, state: Optional[State], action: Action, reward: Reward) -> None:
        """Learn via the optional lambda function or learn nothing without a lambda function.

        Args:
            state: See the base class for more information.
            action: See the base class for more information.
            reward: See the base class for more information.
        """
        if self._learner is None:
            pass
        else:
            self._learner(state,action,reward)