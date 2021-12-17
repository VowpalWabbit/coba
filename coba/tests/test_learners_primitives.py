import unittest

from coba.learners import SafeLearner

class ParamsLearner:
    def __init__(self, params):
        self._params = params
    
    @property
    def params(self):
        return self._params

    def predict(self, context, actions):
        pass

    def learn(self, context, action, reward, probability, info):
        pass

class NoParamsLearner:
    def predict(self, context, actions):
        pass

    def learn(self, context, action, reward, probability, info):
        pass

class UnsafeFixedLearner:
    def __init__(self, pmf, info):
        self._pmf = pmf
        self._info = info
    
    def predict(self, context, actions):
        if self._info is None:
            return self._pmf
        else:
            return self._pmf, self._info

class SafeLearner_Tests(unittest.TestCase):

    def test_no_params(self):
        learner = SafeLearner(NoParamsLearner())
        self.assertEqual("NoParamsLearner", learner.params["family"])

    def test_params_no_family(self):
        learner = SafeLearner(ParamsLearner({'a':"A"}))
        self.assertDictEqual({"family":"ParamsLearner", "a":"A"}, learner.params)

    def test_params_family(self):
        learner = SafeLearner(ParamsLearner({'a':"A", "family":"B"}))
        self.assertDictEqual({"family":"B", "a":"A"}, learner.params)

    def test_params_fullname_0_params(self):
        learner = SafeLearner(ParamsLearner({"family":"B"}))
        
        self.assertEqual("B", learner.full_name)

    def test_params_fullname_1_param(self):
        learner = SafeLearner(ParamsLearner({'a':"A", "family":"B"}))
        
        self.assertEqual("B(a=A)", learner.full_name)

    def test_params_fullname_2_params(self):
        learner = SafeLearner(ParamsLearner({'a':"A","b":"B", "family":"B"}))
        
        self.assertEqual("B(a=A,b=B)", learner.full_name)

    def test_no_sum_one_no_info_action_match_predict(self):
        learner = SafeLearner(UnsafeFixedLearner([1/3,1/2], None))

        with self.assertRaises(AssertionError):
            learner.predict(None, [1,2])

    def test_no_sum_one_info_action_match_predict(self):
        learner = SafeLearner(UnsafeFixedLearner([1/3,1/2], 1))

        with self.assertRaises(AssertionError):
            learner.predict(None, [1,2])

    def test_sum_one_no_info_action_mismatch_predict(self):
        learner = SafeLearner(UnsafeFixedLearner([1/2,1/2], None))

        with self.assertRaises(AssertionError):
            learner.predict(None, [1,2,3])

    def test_sum_one_info_action_mismatch_predict(self):
        learner = SafeLearner(UnsafeFixedLearner([1/2,1/2], 1))

        with self.assertRaises(AssertionError):
            learner.predict(None, [1,2,3])

    def test_sum_one_no_info_action_match_predict(self):
        learner = SafeLearner(UnsafeFixedLearner([1/2,1/2], None))

        predict = learner.predict(None, [1,2])

        self.assertEqual([1/2,1/2], predict[0])
        self.assertEqual(None, predict[1])

    def test_sum_one_info_action_match_predict(self):
        learner = SafeLearner(UnsafeFixedLearner([1/2,1/2], 1))

        predict = learner.predict(None, [1,2])

        self.assertEqual([1/2,1/2], predict[0])
        self.assertEqual(1, predict[1])

if __name__ == '__main__':
    unittest.main()