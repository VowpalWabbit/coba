import unittest

from statistics import mean

from coba.contexts import CobaContext
from coba.learners import CorralLearner, FixedLearner

class ReceivedLearnFixedLearner(FixedLearner):

    def __init__(self, fixed_pmf, key='a') -> None:
        self.received_learn = None
        self.key            = key
        super().__init__(fixed_pmf)

    def learn(self, context, action, reward, probability) -> None:
        self.received_learn = (context, action, reward, probability)
        CobaContext.learning_info.update({self.key:1})

class FamilyLearner:
    def __init__(self, family):
        self._family = family

    @property
    def params(self):
        return {'family': self._family}

class CorralLearner_Tests(unittest.TestCase):

    def test_importance_request(self):

        learner = CorralLearner([FixedLearner([1/2,1/2]), FixedLearner([1/4,3/4])], eta=0.5, mode="importance")

        mean_request = list(map(mean, zip(*[learner.request(None, [1,2],[1,2]) for _ in range(10000)])) )

        self.assertAlmostEqual(1/2*1/2+1/2*1/4, mean_request[0], 2)
        self.assertAlmostEqual(1/2*1/2+1/2*3/4, mean_request[1], 2)

    def test_importance_predict(self):

        learner = CorralLearner([FixedLearner([1/2,1/2]), FixedLearner([1/4,3/4])], eta=0.5, mode="importance")

        mean_predict = list(map(mean, zip(*[learner.predict(None, [1,2])[0] for _ in range(10000)])) )

        self.assertAlmostEqual(1/2*1/2+1/2*1/4, mean_predict[0], 2)
        self.assertAlmostEqual(1/2*1/2+1/2*3/4, mean_predict[1], 2)

    def test_importance_learn(self):
        actions      = [1,2]
        base1        = ReceivedLearnFixedLearner([1,0])
        base2        = ReceivedLearnFixedLearner([0,1])
        learner      = CorralLearner([base1, base2], eta=0.5, mode="importance")
        predict,info = learner.predict(None, actions)

        probability = predict[0]
        reward      = 1/2

        learner.learn(None, actions[0], reward, probability, **info)

        self.assertEqual((None, actions[0], 1, 1), base1.received_learn)
        self.assertEqual((None, actions[1], 0, 1), base2.received_learn)

    def test_off_policy_request(self):

        learner = CorralLearner([FixedLearner([1/2,1/2]), FixedLearner([1/4,3/4])], eta=0.5, mode="off-policy")

        mean_request = list(map(mean, zip(*[learner.request(None, [1,2], [2,1]) for _ in range(10000)])) )

        self.assertAlmostEqual(1/2*1/2+1/2*1/4, mean_request[1], 2)
        self.assertAlmostEqual(1/2*1/2+1/2*3/4, mean_request[0], 2)

    def test_off_policy_predict(self):

        learner = CorralLearner([FixedLearner([1/2,1/2]), FixedLearner([1/4,3/4])], eta=0.5, mode="off-policy")

        mean_predict = list(map(mean, zip(*[learner.predict(None, [1,2])[0] for _ in range(10000)])) )

        self.assertAlmostEqual(1/2*1/2+1/2*1/4, mean_predict[0], 2)
        self.assertAlmostEqual(1/2*1/2+1/2*3/4, mean_predict[1], 2)

    def test_off_policy_learn(self):

        actions      = [1,2]
        base1        = ReceivedLearnFixedLearner([1/2,1/2], 'a')
        base2        = ReceivedLearnFixedLearner([1/4,3/4], 'b')
        learner      = CorralLearner([base1, base2], eta=0.5, mode="off-policy")
        predict,info = learner.predict(None, actions)

        action      = actions[0]
        probability = predict[0]
        reward      = 1

        learner.learn(None, action, reward, probability, **info)

        self.assertEqual((None, action, reward, predict[0]), base1.received_learn)
        self.assertEqual((None, action, reward, predict[0]), base2.received_learn)

    def test_params(self):

        base1_name = 'A'
        base2_name = 'B'

        expected = {'family': 'corral', 'eta': 0.075, 'mode': 'importance', 'T': float('inf'), 'B': [base1_name, base2_name], 'seed': 1}
        actual   = CorralLearner([FamilyLearner("A"), FamilyLearner("B")]).params

        self.assertEqual(expected, actual)

if __name__ == '__main__':
    unittest.main()