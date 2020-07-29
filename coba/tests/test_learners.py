import unittest
import random

from typing import Tuple, List, cast
from abc import ABC, abstractmethod

from coba.utilities import check_vowpal_support
from coba.simulations import State, Action
from coba.learners import Learner, RandomLearner, LambdaLearner, EpsilonLearner, VowpalLearner, UcbTunedLearner

class Learner_Interface_Tests(ABC):

    @abstractmethod
    def _make_learner(self) -> Learner:
        ...

    def _states_actions_pairs(self) -> List[Tuple[List[State],List[Action]]]:
        states: List[State] = [None, 1, (1,2,3), "ABC", ("A","B","C"), ("A",2.,"C")]
        actions: List[List[Action]] = [[1,2], [1,2,3,4], [(1,2),(3,4)], ["A","B","C"], ["A",2.]]
 
        return [ (states,a) for a in actions ]

    def _n_samples(self) -> int:
        # If a learner's `choose()` function is non-deterministic then it is not
        # possible to guarantee via a single test that it will never return a bad
        # value. Therefore we sample `choose()` repeatedly to show that the maximum 
        # likelihood distribution for a `choose()` being wrong is less than 1/n_samples.
        return 1000

    def test_choose_index_in_actions_range(self) -> None:
        for states, actions in self._states_actions_pairs():
            learner = self._make_learner()
            for state in states:
                for _ in range(self._n_samples()):
                    choice = learner.choose(state, actions)
                    cast(unittest.TestCase, self).assertIn(choice, range(len(actions)))

    def test_learn_throws_no_exceptions(self) -> None:
        for states,actions in self._states_actions_pairs():
            learner = self._make_learner()
            for state in states:
                try:
                    learner.learn(state, actions[learner.choose(state,actions)], random.uniform(-2,2))
                except:
                    cast(unittest.TestCase, self).fail("An exception was raised on a call to learn.")

class RandomLearner_Tests(Learner_Interface_Tests, unittest.TestCase):
    def _make_learner(self) -> Learner:
        return RandomLearner()

class LambdaLearner_Tests(Learner_Interface_Tests, unittest.TestCase):
    def _make_learner(self) -> Learner:
        return LambdaLearner(lambda s,a: 0, lambda s,a,r:None)

class EpsilonLearner_Tests(Learner_Interface_Tests, unittest.TestCase):
    def _make_learner(self) -> Learner:
        return EpsilonLearner(1/10, 0)

class UcbTunedLearner_Tests(Learner_Interface_Tests, unittest.TestCase):
    def _make_learner(self) -> Learner:
        return UcbTunedLearner()

class VowpalLearner_Tests(Learner_Interface_Tests, unittest.TestCase):
    def _make_learner(self) -> Learner:
        try:
            check_vowpal_support('VowpalLearner_Tests._make_learner')
            return VowpalLearner()
        except ImportError:
            #if somebody is using the package with no intention of
            #using the VowpalLearner we don't want them to see failed
            #tests or the VowpalLearner and think something is wrong
            #so we return a different learner for the sake of passing
            return RandomLearner()
        

if __name__ == '__main__':
    unittest.main()