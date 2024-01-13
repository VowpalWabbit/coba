import unittest

from statistics import mean
from collections import Counter

from coba.context import CobaContext
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

    def test_importance_score(self):
        learner = CorralLearner([FixedLearner([1/2,1/2]), FixedLearner([1/4,3/4])], eta=0.5, mode="importance")
        mean_score = list(map(mean, zip(*[ [learner.score(None,[1,2],1),learner.score(None,[1,2],2)] for _ in range(20000)])) )
        self.assertAlmostEqual(1/2*1/2+1/2*1/4, mean_score[0], 2)
        self.assertAlmostEqual(1/2*1/2+1/2*3/4, mean_score[1], 2)

    def test_importance_predict(self):
        learner = CorralLearner([FixedLearner([1/2,1/2]), FixedLearner([1/4,3/4])], eta=0.5, mode="importance")
        counts = Counter([learner.predict(None, [1,2])[0] for _ in range(20000)])
        self.assertEqual(len(counts),2)
        self.assertAlmostEqual(counts[1]/sum(counts.values()), 1/2*1/2+1/2*1/4, delta=.05)
        self.assertAlmostEqual(counts[2]/sum(counts.values()), 1/2*1/2+1/2*3/4, delta=.05)

    def test_importance_learn(self):
        actions      = [1,2]
        base1        = ReceivedLearnFixedLearner([1,0])
        base2        = ReceivedLearnFixedLearner([0,1])
        learner      = CorralLearner([base1, base2], eta=0.5, mode="importance")
        act,scr,info = learner.predict(None, actions)
        reward       = 1/2
        learner.learn(None, act, reward, scr, **info)
        self.assertEqual((None, actions[0], 1, 1), base1.received_learn)
        self.assertEqual((None, actions[1], 0, 1), base2.received_learn)

    def test_off_policy_score(self):
        learner = CorralLearner([FixedLearner([1/2,1/2]), FixedLearner([1/4,3/4])], eta=0.5, mode="off-policy")
        mean_score = list(map(mean, zip(*[ [learner.score(None,[1,2],1),learner.score(None,[1,2],2)] for _ in range(20000)])) )
        self.assertAlmostEqual(1/2*1/2+1/2*1/4, mean_score[0], 2)
        self.assertAlmostEqual(1/2*1/2+1/2*3/4, mean_score[1], 2)

    def test_off_policy_predict(self):
        learner = CorralLearner([FixedLearner([1/2,1/2]), FixedLearner([1/4,3/4])], eta=0.5, mode="off-policy")
        counts = Counter([learner.predict(None, [1,2])[0] for _ in range(20000)])
        self.assertEqual(len(counts),2)
        self.assertAlmostEqual(counts[1]/sum(counts.values()), 1/2*1/2+1/2*1/4, delta=.05)
        self.assertAlmostEqual(counts[2]/sum(counts.values()), 1/2*1/2+1/2*3/4, delta=.05)

    def test_off_policy_learn(self):
        actions      = [0,1]
        base1        = ReceivedLearnFixedLearner([1/2,1/2], 'a')
        base2        = ReceivedLearnFixedLearner([1/4,3/4], 'b')
        learner      = CorralLearner([base1, base2], eta=0.5, mode="off-policy")
        act,scr,info = learner.predict(None, actions)
        reward       = 1
        learner.learn(None, act, reward, scr, **info)
        self.assertEqual((None, act, reward, scr), base1.received_learn)
        self.assertEqual((None, act, reward, scr), base2.received_learn)

    def test_params(self):
        base1_name = 'A'
        base2_name = 'B'
        expected = {'family': 'corral', 'eta': 0.075, 'mode': 'importance', 'T': float('inf'), 'B': [base1_name, base2_name], 'seed': 1}
        actual   = CorralLearner([FamilyLearner("A"), FamilyLearner("B")]).params
        self.assertEqual(expected, actual)

if __name__ == '__main__':
    unittest.main()
