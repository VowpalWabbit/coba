"""
"""

from random import randint

class Solver:

    def choose(self, state, actions) -> int:
        return randint(0,len(actions)-1)

    def learn(self, state, action, reward) -> None:
        pass