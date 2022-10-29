import unittest

from coba.learners import SafeLearner, FixedLearner, ActionScore, Probs

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

class ParamsNotPropertyLearner:
    def __init__(self, params):
        self._params = params

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

class NoParamsLearnerNoInfo:
    def predict(self, context, actions):
        pass

    def learn(self, context, actions, action, reward, probability, **kwargs):
        pass

class UnsafeFixedLearner:
    def __init__(self, pmf, info):
        self._pmf = pmf
        self._info = info

    def predict(self, context, actions):
        if self._info is None:
            return self._pmf
        else:
            return self._pmf, {'info':self._info}

class AmbiguousPredictionLearner:

    def predict(self, context, actions):
        return [1,0]

    def learn(self, context, actions, action, reward, probability) -> None:
        pass

class SafeLearner_Tests(unittest.TestCase):

    def test_no_params(self):
        learner = SafeLearner(NoParamsLearner())
        self.assertEqual("NoParamsLearner", learner.params["family"])

    def test_params_not_property(self):
        learner = SafeLearner(ParamsNotPropertyLearner({'a':"A"}))
        self.assertDictEqual({"family":"ParamsNotPropertyLearner", "a":"A"}, learner.params)

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

        self.assertEqual(1  , predict[0])
        self.assertEqual(1/2, predict[1])
        self.assertEqual({} , predict[2])

    def test_sum_one_info_action_match_predict(self):
        learner = SafeLearner(UnsafeFixedLearner([1/2,1/2], 1))

        predict = learner.predict(None, [1,2])

        self.assertEqual(1         , predict[0])
        self.assertEqual(1/2       , predict[1])
        self.assertEqual({'info':1}, predict[2])

    def test_no_exception_without_info_args(self):
        SafeLearner(NoParamsLearnerNoInfo()).learn(1,[1,2],2,3,4)
        SafeLearner(NoParamsLearnerNoInfo()).learn(1,[1,2],2,3,4,**{})

    def test_no_exception_without_info_kwargs(self):
        SafeLearner(NoParamsLearnerNoInfo()).learn(context=1,actions=[1,2],action=2,reward=3,probability=4)
        SafeLearner(NoParamsLearnerNoInfo()).learn(context=1,actions=[1,2],action=2,reward=3,probability=4,info=None)

    def test_ambiguous_prediction(self):
        with self.assertWarns(UserWarning) as w:
            SafeLearner(AmbiguousPredictionLearner()).predict(None,[0,1])

    def test_pdf_prediction(self):
        action,score,kwargs = SafeLearner(FixedLearner(lambda a: 1 if a == 0 else 0)).predict(None,[0,1])
        self.assertEqual(0, action)
        self.assertEqual(1, score)


class ActionScore_Tests(unittest.TestCase):

    def test_simple(self):
        self.assertEqual((1,.5), ActionScore(1,.5))
        self.assertIsInstance(ActionScore(1,2), ActionScore)

class Probs_Tests(unittest.TestCase):

    def test_simple(self):
        self.assertEqual([1/4,1/2,1/4], Probs([1/4,1/2,1/4]))
        self.assertIsInstance(Probs([1/4,1/2,1/4]), Probs)

if __name__ == '__main__':
    unittest.main()
