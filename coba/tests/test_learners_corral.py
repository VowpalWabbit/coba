import unittest

from statistics import mean

from coba.random import CobaRandom
from coba.learners import CorralLearner, FixedLearner

class CorallLearner_Tests(unittest.TestCase):

    class ReceivedLearnFixedLearner(FixedLearner):

        def __init__(self, fixed_pmf) -> None:
            self._received_learn = None
            super().__init__(fixed_pmf)

        def received_learn(self):
            receieved = self._received_learn
            self._received_learn = None
            return receieved

        def learn(self, context, action, reward, probability, info) -> None:
            self._received_learn = (context, action, reward, probability)
            return super().learn(context, action, reward, probability, info)

    def test_importance_predict(self):

        learner = CorralLearner([FixedLearner([1/2,1/2]), FixedLearner([1/4,3/4])], eta=0.5, type="importance")
        
        mean_predict = list(map(mean, zip(*[learner.predict(None, [1,2])[0] for _ in range(1000)])) )

        self.assertAlmostEqual(.375, mean_predict[0], 2)
        self.assertAlmostEqual(.625, mean_predict[1], 2)

    def test_importance_learn(self):
        actions      = [1,2]
        base1        = CorallLearner_Tests.ReceivedLearnFixedLearner([1/2,1/2])
        base2        = CorallLearner_Tests.ReceivedLearnFixedLearner([1/4,3/4])
        learner      = CorralLearner([base1, base2], eta=0.5, type="importance")
        predict,info = learner.predict(None, actions)

        action      = actions[0]
        probability = predict[0]
        reward      = 1

        learner.learn(None, action, reward, probability, info)

        self.assertEqual((None, 1, 2, 1/2), base1.received_learn())
        self.assertEqual((None, 2, 0, 3/4), base2.received_learn())

    def test_off_policy_predict(self):

        learner = CorralLearner([FixedLearner([1/2,1/2]), FixedLearner([1/4,3/4])], eta=0.5, type="off-policy")
        
        predict = learner.predict(None, [1,2])[0]

        self.assertEqual(.375, predict[0])
        self.assertEqual(.625, predict[1])
    
    def test_off_policy_learn(self):
        
        actions      = [1,2]
        base1        = CorallLearner_Tests.ReceivedLearnFixedLearner([1/2,1/2])
        base2        = CorallLearner_Tests.ReceivedLearnFixedLearner([1/4,3/4])
        learner      = CorralLearner([base1, base2], eta=0.5, type="off-policy")
        predict,info = learner.predict(None, actions)

        action      = actions[0]
        probability = predict[0]
        reward      = 1

        learner.learn(None, action, reward, probability, info)

        self.assertEqual((None, action, reward, predict[0]), base1.received_learn())
        self.assertEqual((None, action, reward, predict[0]), base2.received_learn())

    def test_rejection_predict(self):

        learner = CorralLearner([FixedLearner([1/2,1/2]), FixedLearner([1/4,3/4])], eta=0.5, type="rejection")
        
        predict = learner.predict(None, [1,2])[0]

        self.assertEqual(.375, predict[0])
        self.assertEqual(.625, predict[1])

    def test_off_rejection_learn(self):

        actions      = [0,1]
        base1        = CorallLearner_Tests.ReceivedLearnFixedLearner([1/2,1/2])
        base2        = CorallLearner_Tests.ReceivedLearnFixedLearner([1/4,3/4])
        learner      = CorralLearner([base1, base2], eta=0.5, type="rejection")
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
            base1_learn_cnt[action] += int(base1.received_learn() is not None)
            base2_learn_cnt[action] += int(base2.received_learn() is not None)

        self.assertLessEqual(abs(base1_learn_cnt[0]/sum(base1_learn_cnt) - 1/2), .02)
        self.assertLessEqual(abs(base1_learn_cnt[1]/sum(base1_learn_cnt) - 1/2), .02)

        self.assertLessEqual(abs(base2_learn_cnt[0]/sum(base2_learn_cnt) - 1/4), .02)
        self.assertLessEqual(abs(base2_learn_cnt[1]/sum(base2_learn_cnt) - 3/4), .02)

if __name__ == '__main__':
    unittest.main()