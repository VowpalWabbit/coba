"""
"""
from abc import ABC, abstractmethod
from typing import Callable, Sequence, Optional
from random import randint

from bbench.games import State, Action, Reward

class Solver(ABC):

    @abstractmethod
    def choose(self, state: Optional[State], actions: Sequence[Action]) -> int:
        ...
    
    @abstractmethod
    def learn(self, state: Optional[State], action: Action, reward: Reward) -> None:
        ...

class RandomSolver(Solver):
    def choose(self, state: Optional[State], actions: Sequence[Action]) -> int:
        return randint(0,len(actions)-1)

    def learn(self, state: Optional[State], action: Action, reward: Reward) -> None:
        pass

class LambdaSolver(Solver):
    def __init__(self, 
                 chooser: Callable[[Optional[State],Sequence[Action]],int], 
                 learner: Optional[Callable[[Optional[State],Action,Reward],None]] = None) -> None:
        self._chooser = chooser
        self._learner = learner

    def choose(self, state: Optional[State], actions: Sequence[Action]) -> int:
        return self._chooser(state, actions)

    def learn(self, state: Optional[State], action: Action, reward: Reward) -> None:
        if self._learner is None:
            pass
        else:
            self._learner(state,action,reward)