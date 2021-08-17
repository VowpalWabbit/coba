import unittest

from coba.learners import RandomLearner, FixedLearner, SafeLearner

class FixedLearner_Tests(unittest.TestCase):

    def test_pmf_assert(self):
        learner = FixedLearner([1/3,1/3,1/3])
        
        with self.assertRaises(AssertionError):
            FixedLearner([1/3, 1/2])

        with self.assertRaises(AssertionError):
            FixedLearner([-1, 2])

    def test_learn(self):
        learner = FixedLearner([1/3,1/3,1/3])
        self.assertEqual([1/3,1/3,1/3], learner.predict(None, [1,2,3]))

class RandomLearner_Tests(unittest.TestCase):

    def test_predict(self):
        learner = RandomLearner()
        self.assertEqual([0.25, 0.25, 0.25, 0.25], learner.predict(None, [1,2,3,4]))

    def test_learn(self):
        learner = RandomLearner()
        learner.learn(2, None, 1, 1, 1)

class SafeLearner_Tests(unittest.TestCase):

    class NoFamilyOrParamLearner:
        def predict(self, context, actions):
            pass
    
        def learn(self, context, action, reward, probability, info):
           pass

    class UncheckedFixedLearner:
        def __init__(self, pmf, info):
            self._pmf = pmf
            self._info = info
        
        def predict(self, context, actions):
            if self._info is None:
                return self._pmf
            else:
                return self._pmf, self._info

    def test_no_family(self):
        learner = SafeLearner(SafeLearner_Tests.NoFamilyOrParamLearner())
        self.assertEqual("NoFamilyOrParamLearner", learner.family)

    def test_no_params(self):
        learner = SafeLearner(SafeLearner_Tests.NoFamilyOrParamLearner())
        self.assertEqual({}, learner.params)

    def test_no_sum_one_no_info_action_match_predict(self):
        learner = SafeLearner(SafeLearner_Tests.UncheckedFixedLearner([1/3,1/2], None))

        with self.assertRaises(AssertionError):
            learner.predict(None, [1,2])

    def test_no_sum_one_info_action_match_predict(self):
        learner = SafeLearner(SafeLearner_Tests.UncheckedFixedLearner([1/3,1/2], 1))

        with self.assertRaises(AssertionError):
            learner.predict(None, [1,2])

    def test_sum_one_no_info_action_mismatch_predict(self):
        learner = SafeLearner(SafeLearner_Tests.UncheckedFixedLearner([1/2,1/2], None))

        with self.assertRaises(AssertionError):
            learner.predict(None, [1,2,3])

    def test_sum_one_info_action_mismatch_predict(self):
        learner = SafeLearner(SafeLearner_Tests.UncheckedFixedLearner([1/2,1/2], 1))

        with self.assertRaises(AssertionError):
            learner.predict(None, [1,2,3])

    def test_sum_one_no_info_action_match_predict(self):
        learner = SafeLearner(SafeLearner_Tests.UncheckedFixedLearner([1/2,1/2], None))

        predict = learner.predict(None, [1,2])

        self.assertEqual([1/2,1/2], predict[0])
        self.assertEqual(None, predict[1])

    def test_sum_one_info_action_match_predict(self):
        learner = SafeLearner(SafeLearner_Tests.UncheckedFixedLearner([1/2,1/2], 1))

        predict = learner.predict(None, [1,2])

        self.assertEqual([1/2,1/2], predict[0])
        self.assertEqual(1, predict[1])

if __name__ == '__main__':
    unittest.main()