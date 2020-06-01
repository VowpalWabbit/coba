from typing import NoReturn, Iterable, List

"""
    The games module. This module contains classes and functions for creating bandit games. A bandit
    game is a repeated decision making problem where each decision is known as a round.
    
    Classes:
        > 

"""

CF = List[float]
AF = List[List[float]]
AR = List[float]

class BanditRound:
    def __init__(self, action_features: AF, action_rewards: List[float]) -> NoReturn:
        self._action_features = action_features
        self._action_rewards = action_rewards
    
    @property
    def action_features(self) -> AF:
        return self._action_features

    @property
    def action_rewards(self) -> AR:
        return self._action_rewards

class ContextualBanditRound(BanditRound):
    def __init__(self, context_features, action_features, action_rewards):
        super().__init__(action_features, action_rewards)
        self._context_features = context_features

    @property
    def context_features(self):
        return self._context_features

IBR = Iterable[BanditRound]
ICR = Iterable[ContextualBanditRound]

class BanditGame:
    def __init__(self, rounds: IBR) -> NoReturn:
        self._rounds = rounds

    @property
    def rounds(self) -> IBR:
        return self._rounds

class ContextualBanditGame(BanditGame):
    def __init__(self, rounds: ICR) -> NoReturn:
        super().__init__(rounds)

    @property
    def rounds(self) -> ICR:
        return self._rounds