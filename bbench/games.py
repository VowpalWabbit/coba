"""
    The games module contains classes and functions for creating bandit games. A bandit
    game is a repeated decision making problem where each decision is known as a round.
    
    Classes:
        > ...
"""

from typing import Iterable, List, Generic, TypeVar, Any

CF = List[Any]
AF = List[List[Any]]
AR = List[float]
T = TypeVar('T', bound=BanditRound) # pylint: disable=used-before-assignment

class BanditRound:
    def __init__(self, action_features: AF, action_rewards: List[float]) -> None:
        self._action_features = action_features
        self._action_rewards = action_rewards

    @property
    def action_features(self) -> AF:
        return self._action_features

    @property
    def action_rewards(self) -> AR:
        return self._action_rewards

class ContextualBanditRound(BanditRound):
    def __init__(self, context_features: CF, action_features: AF, action_rewards: AR):
        super().__init__(action_features, action_rewards)
        self._context_features = context_features

    @property
    def context_features(self) -> CF:
        return self._context_features

class BanditGame(Generic[T]):

    @staticmethod
    def from_classifier_data(features: List[List[Any]], labels: List[Any]) -> BanditGame: # pylint: disable=undefined-variable
        rounds  = []
        actions = [ [a] for a in list(set(labels)) ]

        for context_features, rewarded_action in zip(features, labels):
            rounds.append(ContextualBanditRound(context_features, actions, [int(rewarded_action==a) for a in actions] ))

        return BanditGame(rounds)

    @staticmethod
    def from_csv_reader(csv_reader: Iterable[List[str]], label_col: str) -> BanditGame: # pylint: disable=undefined-variable
        features = []
        labels = []

        for row_num, row_vals in enumerate(csv_reader):
            if row_num == 0:
                label_index = row_vals.index(label_col)
            else:
                features.append(row_vals[:label_index] + row_vals[(label_index+1):])
                labels.append(row_vals[label_index])
        
        return BanditGame.from_classifier_data(features, labels)

    def __init__(self, rounds: Iterable[T]) -> None:
        self._rounds = rounds

    @property
    def rounds(self) -> Iterable[T]:
        return self._rounds