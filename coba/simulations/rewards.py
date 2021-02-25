import collections
import json

from abc import ABC, abstractmethod
from typing import (
    Optional, Sequence, List, Callable, 
    Hashable, Any, Tuple, overload, cast,
    Dict
)
from coba.simulations import Key, Context, Action

class Reward(ABC):

    @abstractmethod
    def observe(self, choices: Sequence[Tuple[Key,Action]] ) -> Sequence[float]:
        ...

class MemoryReward(Reward):
    def __init__(self, rewards: Sequence[Tuple[Key,Action,float]] = []) -> None:
        self._rewards: Dict[Tuple[Key,Action], float] = {}

        for reward in rewards:
            self._rewards[(reward[0],reward[1])] = reward[2]

    def add_observation(self, observation: Tuple[Key,Action,float]) -> None:
        choice = (observation[0],observation[1])
        reward = observation[2]

        if choice in self._rewards: raise Exception("Unable to add an existing observation.")
        
        self._rewards[choice] = reward

    def observe(self, choices: Sequence[Tuple[Key,Action]] ) -> Sequence[float]:
        return [ self._rewards[choice] for choice in choices ]

class ClassificationReward(Reward):

    def __init__(self, labels: Sequence[Tuple[Key,Action]]) -> None:
        self._labels = dict(labels)

    def observe(self, choices: Sequence[Tuple[Key,Action]] ) -> Sequence[float]:
        return [ float(self._labels[choice[0]] == choice[1]) for choice in choices ]
