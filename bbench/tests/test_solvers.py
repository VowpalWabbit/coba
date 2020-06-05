import unittest
import itertools
import random

from typing import Tuple, Sequence, Iterable, Optional, cast
from abc import ABC, abstractmethod

from bbench.games import State, Action
from bbench.solvers import Solver, RandomSolver, LambdaSolver

class Solver_Interface_Tests(ABC):

    @abstractmethod
    def make_solver(self) -> Solver:
        ...

    @property
    def n_samples(self) -> int:
        # many of these tests assume the solver's choices are non-deterministic
        # we can never guarantee via test that a non-deterministic result will
        # never return a bad value. Therefore we merely show that the maximum 
        # likelihood of solver returns being wrong is less than below
        return 1000

    @property
    def state_action_population(self) -> Sequence[Tuple[Optional[State],Sequence[Action]]]:
        states: Sequence[Optional[State]] = [None, 1, [1,2,3], "ABC", ["A","B","C"], ["A",2,"C"]]
        actions: Sequence[Sequence[Action]] = [[1,2], [1,2,3,4], [[1,2],[3,4]], ["A","B","C"], ["A",2]]
        
        return [ (s,a) for s in states for a in actions]

    def test_choose_in_range(self) -> None:
        solver = self.make_solver()
        for state,actions in random.choices(self.state_action_population, k=self.n_samples):
            actual   = solver.choose(state, actions)
            expected = range(len(actions))
            self.assertIn(actual, expected) # type: ignore #pylint: disable=no-member

    def test_learn_no_exceptions(self) -> None:
        solver = self.make_solver()

        for state,actions in self.state_action_population:
            for action in actions:
                solver.learn(state, action, random.uniform(-2,2))

class RandomSolver_Tests(Solver_Interface_Tests, unittest.TestCase):
    def make_solver(self) -> Solver:
        return RandomSolver()

class LambdaSolver_Tests(Solver_Interface_Tests, unittest.TestCase):
    def make_solver(self) -> Solver:
        return LambdaSolver(lambda s,a: 0, lambda s,a,r:None)

if __name__ == '__main__':
    unittest.main()
