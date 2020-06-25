import unittest
import itertools
import random

from typing import Tuple, List, Iterable, Optional, cast
from abc import ABC, abstractmethod

from bbench.simulations import State, Action
from bbench.learners import Learner, RandomLearner, LambdaLearner, EpsilonLookupLearner, VowpalLearner

class Learner_Interface_Tests(ABC):

    @abstractmethod
    def _make_learner(self) -> Learner:
        ...

    def _state_actions_pairs(self) -> List[Tuple[Optional[State],List[Action]]]:
        states: List[State] = [None, 1, (1,2,3), "ABC", ("A","B","C"), ("A",2.,"C")]
        actions: List[List[Action]] = [[1,2], [1,2,3,4], [(1,2),(3,4)], ["A","B","C"], ["A",2.]]
 
        return [ (s,a) for s in states for a in actions]

    def setUp(self):
        # If a learner's `choose()` function is non-deterministic then it is not
        # possible to guarantee via a single test that it will never return a bad
        # value. Therefore we sample `choose()` repeatedly to show that the maximum 
        # likelihood distribution for a `choose()` being wrong is less than 1/n_samples.
        self._n_samples = 1000
        self._learner = self._make_learner()

    def test_choose_index_in_actions_range(self) -> None:
        for state,actions in random.choices(self._state_actions_pairs(), k=self._n_samples):
            actual   = self._learner.choose(state, actions)
            expected = range(len(actions))
            self.assertIn(actual, expected) #type: ignore #pylint: disable=no-member

    def test_learn_throws_no_exceptions(self) -> None:
        for state,actions in self._state_actions_pairs():
            for action in actions:
                self._learner.learn(state, action, random.uniform(-2,2))

class RandomLearner_Tests(Learner_Interface_Tests, unittest.TestCase):
    def _make_learner(self) -> Learner:
        return RandomLearner()

class LambdaLearner_Tests(Learner_Interface_Tests, unittest.TestCase):
    def _make_learner(self) -> Learner:
        return LambdaLearner(lambda s,a: 0, lambda s,a,r:None)

class EpsilonLookupLearner_Tests(Learner_Interface_Tests, unittest.TestCase):
    def _make_learner(self) -> Learner:
        return EpsilonLookupLearner(1/10, 0)

if __name__ == '__main__':
    unittest.main()
