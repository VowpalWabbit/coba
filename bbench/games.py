"""
    The games module contains classes and functions for creating bandit games. A bandit
    game is a repeated decision making problem where each decision is known as a round.
    
    Classes:
        > ...
"""
from abc import ABC
from typing import Optional, Generic, Iterable, Sequence, List, TypeVar, Any, Union, cast

#reward type
R  = float

#feature type
F = Union[str,float,Sequence[Union[str,float]]]

class Round:
    def __init__(self, state: Optional[F], actions: Sequence[F], rewards: Sequence[R]):
        
        assert len(actions) == len(rewards), "Mismatched lengths of actions and rewards"

        self._state   = state
        self._actions = actions
        self._rewards = rewards

    @property
    def state(self) -> Optional[F]:
        return self._state

    @property
    def actions(self) -> Sequence[F]:
        return self._actions

    @property
    def rewards(self) -> Sequence[R]:
        return self._rewards

class Game:
    @staticmethod
    def from_classifier_data(features: Sequence[F], labels: Sequence[Union[str,float]]) -> 'Game':
        
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

    def __init__(self, rounds: Sequence[Round]) -> None:
        self._rounds = rounds

    @property
    def rounds(self) -> Sequence[Round]:
        return self._rounds