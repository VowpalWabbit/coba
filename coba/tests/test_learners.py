import unittest
import random

from typing import Tuple, List, cast
from abc import ABC, abstractmethod

from coba.utilities import check_vowpal_support
from coba.simulations import Context, Action
from coba.learners import Learner, RandomLearner, LambdaLearner, EpsilonLearner, VowpalLearner, UcbTunedLearner
from coba.json import CobaJsonDecoder, CobaJsonEncoder, JsonSerializable

class Learner_Interface_Tests(ABC):

    @abstractmethod
    def _make_learner(self) -> Learner:
        ...

    def _contexts_actions_pairs(self) -> List[Tuple[List[Context],List[Action]]]:
        contexts: List[Context] = [None, 1, (1,2,3), "ABC", ("A","B","C"), ("A",2.,"C")]
        actions: List[List[Action]] = [[1,2], [1,2,3,4], [(1,2),(3,4)], ["A","B","C"], ["A",2.]]
 
        return [ (contexts,a) for a in actions ]

    def _n_samples(self) -> int:
        # If a learner's `choose()` function is non-deterministic then it is not
        # possible to guarantee via a single test that it will never return a bad
        # value. Therefore we sample `choose()` repeatedly to show that the maximum 
        # likelihood distribution for a `choose()` being wrong is less than 1/n_samples.
        return 1000

    def test_choose_index_in_actions_range(self) -> None:
        for contexts, actions in self._contexts_actions_pairs():
            learner = self._make_learner()
            for context in contexts:
                for i in range(self._n_samples()):
                    choice = learner.choose(i, context, actions)
                    cast(unittest.TestCase, self).assertIn(choice, range(len(actions)))

    def test_learn_throws_no_exceptions(self) -> None:
        for contexts,actions in self._contexts_actions_pairs():
            learner = self._make_learner()
            for k, context in enumerate(contexts):
                learner.learn(k, context, actions[learner.choose(k, context,actions)], random.uniform(-2,2))

    def test_to_from_json(self) -> None:
        learner = self._make_learner()

        if isinstance(learner, JsonSerializable):
            a = CobaJsonEncoder().encode(learner)
            b = CobaJsonDecoder().decode(a)

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