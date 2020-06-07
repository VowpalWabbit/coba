"""The games module contains core classes and types for defining bandit games.

This module contains the abstract interface expected for bandit game implementations along
with the class defining a Round within a bandit game. This module also contains the type hints 
for State, Action and Reward. These type hints don't contain any functionality. Rather, they 
simply make it possible to use static type checking for any project that desires to do so.

Todo:
    * Add more Solver implementations
"""

from itertools import repeat
from typing import Optional, Iterable, Sequence, List, Union, Callable

#state, action, reward types
State  = Union[str,float,Sequence[Union[str,float]]]
Action = Union[str,float,Sequence[Union[str,float]]]
Reward = float

class Round:
    def __init__(self, 
                 state  : Optional[State], 
                 actions: Sequence[Action],
                 rewards: Sequence[Reward]) -> None:
        
        assert len(actions) == len(rewards), "Mismatched lengths of actions and rewards"

        self._state   = state
        self._actions = actions
        self._rewards = rewards

    @property
    def state(self) -> Optional[State]:
        return self._state

    @property
    def actions(self) -> Sequence[Action]:
        return self._actions

    @property
    def rewards(self) -> Sequence[Reward]:
        return self._rewards

class Game:
    @staticmethod
    def from_classifier_data(features: Sequence[Union[str,float,Sequence[Union[str,float]]]],
                             labels  : Sequence[Union[str,float]]) -> 'Game':

        assert len(features) == len(labels), "Mismatched lengths of features and labels"

        states  = features
        actions = list(set(labels)) #todo: make this also work for labels that are lists of features
        rewards = [ [int(l==a) for a in actions] for l in labels ]

        return Game(list(map(lambda s,r: Round(s,actions,r), states, rewards)))

    @staticmethod
    def from_csv_reader(csv_reader: Iterable[List[str]], label_col: str) -> 'Game':
        features: List[Sequence[str]] = []
        labels  : List[str]           = []

        for row_num, row_vals in enumerate(csv_reader):
            if row_num == 0:
                label_index = row_vals.index(label_col)
            else:
                features.append(row_vals[:label_index] + row_vals[(label_index+1):])
                labels  .append(row_vals[label_index])

        return Game.from_classifier_data(features, labels)

    @staticmethod
    def from_callable(S: Callable[[],State], 
                      A: Callable[[State],Sequence[Action]], 
                      R: Callable[[State,Action],float]) -> 'Game':

        return Game.from_iterable((S() for _ in repeat(1)), A, R)
    
    @staticmethod
    def from_iterable(S: Iterable[State], 
                      A: Callable[[State],Sequence[Action]], 
                      R: Callable[[State,Action],float]) -> 'Game':

        def round_generator() -> Iterable[Round]:
            for s in S: yield Round(s, A(s), [R(s,a) for a in A(s)])
        
        return Game(round_generator())

    def __init__(self, rounds: Iterable[Round]) -> None:
        self._rounds = rounds

    @property
    def rounds(self) -> Iterable[Round]:
        return self._rounds