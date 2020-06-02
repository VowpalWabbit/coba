"""
    The games module contains classes and functions for creating bandit games. A bandit
    game is a repeated decision making problem where each decision is known as a round.
    
    Classes:
        > ...
"""

from typing import Iterable, Sequence, List, Generic, TypeVar, Any, Union, cast

#reward type
R  = float

#feature type (in theory we chould probably be more specific than Any but using unions with base types is a huge headache)
#scratch what I say above, implicit type conversion works with union when F is used inside of a "Sequence" rather than a "List"
F = Union[str,float,Sequence[Union[str,float]]]

#game round
T = TypeVar('T', bound=BanditRound) # pylint: disable=used-before-assignment

class BanditRound:
    def __init__(self, action_features: Sequence[F], action_rewards: Sequence[R]) -> None:
        self._action_features = action_features
        self._action_rewards = action_rewards

    @property
    def action_features(self) -> Sequence[F]:
        return self._action_features

    @property
    def action_rewards(self) -> Sequence[R]:
        return self._action_rewards

class ContextualBanditRound(BanditRound):
    def __init__(self, context_features: F, action_features: Sequence[F], action_rewards: Sequence[R]):
        super().__init__(action_features, action_rewards)
        self._context_features = context_features

    @property
    def context_features(self) -> F:
        return self._context_features

class BanditGame(Generic[T]):

    @staticmethod
    def from_classifier_data(features: Sequence[F], labels: Sequence[Union[str,float]]) -> BanditGame: # pylint: disable=undefined-variable
        rounds  = []
        actions = list(set(labels))

        for context_features, rewarded_action in zip(features, labels):
            rounds.append(ContextualBanditRound(context_features, actions, [int(rewarded_action==a) for a in actions] ))

        return BanditGame(rounds)

    @staticmethod
    def from_csv_reader(csv_reader: Iterable[List[str]], label_col: str) -> BanditGame: # pylint: disable=undefined-variable
        features: List[Sequence[str]] = []
        labels  : List[str]           = []

        for row_num, row_vals in enumerate(csv_reader):
            if row_num == 0:
                label_index = row_vals.index(label_col)
            else:
                features.append(row_vals[:label_index] + row_vals[(label_index+1):])
                labels  .append(row_vals[label_index])

        return BanditGame.from_classifier_data(features, labels)

    def __init__(self, rounds: Iterable[T]) -> None:
        self._rounds = rounds

    @property
    def rounds(self) -> Iterable[T]:
        return self._rounds