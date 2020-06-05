"""
    The games module contains classes and functions for creating bandit games. A bandit
    game is a repeated decision making problem where each decision is known as a round.
    
    Classes:
        > ...
"""
from abc import ABC
from typing import Optional, Iterable, Sequence, List, Union, Callable

#state, action, reward types
S = Union[str,float,Sequence[Union[str,float]]]
A = Union[str,float,Sequence[Union[str,float]]]
R = float

class Round:
    def __init__(self, state: Optional[S], actions: Sequence[A], rewards: Sequence[R]):
        
        assert len(actions) == len(rewards), "Mismatched lengths of actions and rewards"

        self._state   = state
        self._actions = actions
        self._rewards = rewards

    @property
    def state(self) -> Optional[S]:
        return self._state

    @property
    def actions(self) -> Sequence[A]:
        return self._actions

    @property
    def rewards(self) -> Sequence[R]:
        return self._rewards

class Game:
    @staticmethod
    def from_classifier_data(features: Sequence[S], labels: Sequence[Union[str,float]]) -> 'Game':
        
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
    def from_generator(S: Iterable[S], A: Callable[[S],Sequence[A]], R: Callable[[S,A],float]) -> 'Game':
        
        def round_generator() -> Iterable[Round]:
            for s in S: yield Round(s, A(s), [R(s,a) for a in A(s)])
        
        return Game(round_generator())

    def __init__(self, rounds: Iterable[Round]) -> None:
        self._rounds = rounds

    @property
    def rounds(self) -> Iterable[Round]:
        return self._rounds