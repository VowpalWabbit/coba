"""
"""

from random import randint

class Solver:

    def Choose(self, state, actions) -> int:
        return randint(0,len(actions)-1)

    def Learn(self, state, action, reward) -> None:
        pass