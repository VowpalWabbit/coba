import unittest

from statistics import mean
from typing import Dict, Any

from coba.pipes import ListIO, NullIO
from coba.contexts import LearnerContext
from coba.random import CobaRandom
from coba.learners import CorralLearner, FixedLearner, LinUCBLearner

class ReceivedLearnFixedLearner(FixedLearner):

    def __init__(self, fixed_pmf, key='a') -> None:
        self.received_learn = None
        self.key            = key
        super().__init__(fixed_pmf)

    def learn(self, context, action, reward, probability, info) -> None:
        self.received_learn = (context, action, reward, probability)
        LearnerContext.logger.write({self.key:1})

class FamilyLearner:
    def __init__(self, family):
        self._family = family
    
    @property
    def params(self):
        return {'family': self._family}

class CorralLearner_Tests(unittest.TestCase):

    def test_importance_predict(self):

        learner = CorralLearner([FixedLearner([1/2,1/2]), FixedLearner([1/4,3/4])], eta=0.5, mode="importance")

        mean_predict = list(map(mean, zip(*[learner.predict(None, [1,2])[0] for _ in range(1000)])) )

        self.assertAlmostEqual(1/2*1/2+1/2*1/4, mean_predict[0], 2)
        self.assertAlmostEqual(1/2*1/2+1/2*3/4, mean_predict[1], 2)

    def test_importance_learn(self):
        actions      = [1,2]
        base1        = ReceivedLearnFixedLearner([1/2,1/2])
        base2        = ReceivedLearnFixedLearner([1/4,3/4])
        learner      = CorralLearner([base1, base2], eta=0.5, mode="importance")
        predict,info = learner.predict(None, actions)

        action      = actions[0]
        probability = predict[0]
        reward      = 1/2

        learner.learn(None, action, reward, probability, info)

        self.assertEqual((None, 1, 1, 1/2), base1.received_learn)
        self.assertEqual((None, 2, 0, 3/4), base2.received_learn)

    def test_off_policy_predict(self):

        learner = CorralLearner([FixedLearner([1/2,1/2]), FixedLearner([1/4,3/4])], eta=0.5, mode="off-policy")
        
        predict = learner.predict(None, [1,2])[0]

        self.assertEqual(.375, predict[0])
        self.assertEqual(.625, predict[1])

    def test_off_policy_learn(self):
        
        actions      = [1,2]
        base1        = ReceivedLearnFixedLearner([1/2,1/2], 'a')
        base2        = ReceivedLearnFixedLearner([1/4,3/4], 'b')
        learner      = CorralLearner([base1, base2], eta=0.5, mode="off-policy")
        predict,info = learner.predict(None, actions)

        action      = actions[0]
        probability = predict[0]
        reward      = 1

        LearnerContext.logger = ListIO[Dict[str,Any]]()
        learner.learn(None, action, reward, probability, info)
        info = { k:v for item in LearnerContext.logger.read() for k,v in item.items() }
        LearnerContext.logger = NullIO()

        self.assertDictEqual({'a':1,'b':1, **info}, info)
        self.assertEqual((None, action, reward, predict[0]), base1.received_learn)
        self.assertEqual((None, action, reward, predict[0]), base2.received_learn)

    def test_rejection_predict(self):

        learner = CorralLearner([FixedLearner([1/2,1/2]), FixedLearner([1/4,3/4])], eta=0.5, mode="rejection")
        
        predict = learner.predict(None, [1,2])[0]

        self.assertEqual(.375, predict[0])
        self.assertEqual(.625, predict[1])

    def test_rejection_learn(self):

        actions      = [0,1]
        base1        = ReceivedLearnFixedLearner([1/2,1/2], 'a')
        base2        = ReceivedLearnFixedLearner([1/4,3/4], 'b')
        learner      = CorralLearner([base1, base2], eta=0.5, mode="rejection")
        predict,info = learner.predict(None, actions)

        action      = actions[0]
        probability = predict[0]
        reward      = 1

        base1_learn_cnt = [0,0]
        base2_learn_cnt = [0,0]

        random = CobaRandom(1)

        for _ in range(1000):

            action      = random.choice(actions, predict)
            probability = predict[actions.index(action)] 

            learner.learn(None, action, reward, probability, info)
            base1_learn_cnt[action] += int(base1.received_learn is not None)
            base2_learn_cnt[action] += int(base2.received_learn is not None)

            base1.received_learn = None
            base2.received_learn = None

        self.assertLessEqual(abs(base1_learn_cnt[0]/sum(base1_learn_cnt) - 1/2), .02)
        self.assertLessEqual(abs(base1_learn_cnt[1]/sum(base1_learn_cnt) - 1/2), .02)

        self.assertLessEqual(abs(base2_learn_cnt[0]/sum(base2_learn_cnt) - 1/4), .02)
        self.assertLessEqual(abs(base2_learn_cnt[1]/sum(base2_learn_cnt) - 3/4), .02)

    def test_params(self):

        LinUCBLearner

        base1_name = 'A'
        base2_name = 'B'

        expected = {'family': 'corral', 'eta': 0.075, 'mode': 'importance', 'T': float('inf'), 'B': [base1_name, base2_name], 'seed': 1}
        actual   = CorralLearner([FamilyLearner("A"), FamilyLearner("B")]).params

        self.assertEqual(expected, actual)

if __name__ == '__main__':
    unittest.main()