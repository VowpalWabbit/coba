"""
    The games module contains classes and functions for creating bandit games. A bandit
    game is a repeated decision making problem where each decision is known as a round.
    
    Classes:
        > ...
"""

from typing import Iterable, Sequence, List, Generic, TypeVar, Any, Union, cast

#reward type
R  = float

#feature type
F = Union[str,float,Sequence[Union[str,float]]]

class Round:
    def __init__(self, actions: Sequence[F], rewards: Sequence[R]) -> None:
        self._actions = actions
        self._rewards = rewards

    @property
    def actions(self) -> Sequence[F]:
        return self._actions

    @property
    def rewards(self) -> Sequence[R]:
        return self._rewards

class ContextRound(Round):
    def __init__(self, context: F, actions: Sequence[F], rewards: Sequence[R]):
        super().__init__(actions, rewards)
        self._context = context

    @property
    def context(self) -> F:
        return self._context

class Game:
    def __init__(self, rounds: Iterable[Round]) -> None:
        self._rounds = rounds

    @property
    def rounds(self) -> Iterable[Round]:
        return self._rounds

class ContextGame(Game):
    @staticmethod
    def from_classifier_data(features: Sequence[F], labels: Sequence[Union[str,float]]) -> 'ContextGame':
        
        assert len(features) == len(labels), "Mismatched lengths of features and labels"

        rounds  = []
        actions = list(set(labels)) #todo: make this also work for labels that are lists of features        

        for context_features, rewarded_action in zip(features, labels):
            rounds.append(ContextRound(context_features, actions, [int(rewarded_action==a) for a in actions] ))

        return ContextGame(rounds)

    @staticmethod
    def from_csv_reader(csv_reader: Iterable[List[str]], label_col: str) -> 'ContextGame':
        features: List[Sequence[str]] = []
        labels  : List[str]           = []

        for row_num, row_vals in enumerate(csv_reader):
            if row_num == 0:
                label_index = row_vals.index(label_col)
            else:
                features.append(row_vals[:label_index] + row_vals[(label_index+1):])
                labels  .append(row_vals[label_index])

        return ContextGame.from_classifier_data(features, labels)

    def __init__(self, rounds: Iterable[ContextRound]) -> None: 
        super().__init__(rounds)
        
        #MyPy doesn't infer type correctly when overriding so explicit type is necessary
        self._rounds: Iterable[ContextRound] = rounds 

    @property
    def rounds(self) -> Iterable[ContextRound]:
        return self._rounds
