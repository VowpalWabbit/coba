import math
import importlib.util
from typing import Any, Iterable, Mapping, Optional, Union
import unittest
from coba.environments import Environment

from coba.learners import EpsilonBanditLearner, Learner, VowpalSoftmaxLearner, SafeLearner
from coba.evaluators.primitives import Evaluator, SafeEvaluator, get_ope_loss

class Evaluator_Tests(unittest.TestCase):

    def test_params(self):
        class SubEvaluator(Evaluator):
            def evaluate(self, environment: Environment, learner: Learner):
                pass
        self.assertEqual(SubEvaluator().params,{})

class SafeEvaluator_Tests(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(1, SafeEvaluator(lambda env,lrn: 1).evaluate(None,None))

    def test_params_callable(self):
        def test_eval(env,lrn): return 1
        self.assertEqual({'type':'test_eval'}, SafeEvaluator(test_eval).params)

class Helper_Tests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW is not installed")
    def test_get_ope_loss(self):

        #VW learner
        learner = VowpalSoftmaxLearner()
        learner.learn(1, [1], 1, 1, 1.0)
        self.assertEqual(get_ope_loss(SafeLearner(learner)), -1.0)

        # Non-VW learner
        self.assertTrue(math.isnan(get_ope_loss(SafeLearner(EpsilonBanditLearner()))))

if __name__ == '__main__':
    unittest.main()
