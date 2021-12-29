import unittest

from coba.learners import EpsilonBanditLearner, UcbBanditLearner, RandomLearner, FixedLearner

class EpsilonBanditLearner_Tests(unittest.TestCase):
    
    def test_params(self):
        learner = EpsilonBanditLearner(epsilon=0.5)
        self.assertEqual({"family":"epsilon_bandit", "epsilon":0.5}, learner.params)

    def test_predict_no_learn(self):
        learner = EpsilonBanditLearner(epsilon=0.5)

        self.assertEqual([.25,.25,.25,.25],learner.predict(None, [1,2,3,4]))
        self.assertEqual([.25,.25,.25,.25],learner.predict(None, [1,2,3,4]))

    def test_predict_learn_no_epsilon(self):
        learner = EpsilonBanditLearner(epsilon=0)

        learner.learn(None, 2, 1, None, None)
        learner.learn(None, 1, 2, None, None)
        learner.learn(None, 3, 3, None, None)

        self.assertEqual([0,0,1],learner.predict(None, [1,2,3]))

    def test_predict_learn_epsilon(self):
        learner = EpsilonBanditLearner(epsilon=0.1)

        learner.learn(None, 2, 1, None, None)
        learner.learn(None, 1, 2, None, None)
        learner.learn(None, 2, 1, None, None)

        self.assertEqual([.95,.05],learner.predict(None, [1,2]))
    
    def test_predict_learn_epsilon_all_equal(self):
        learner = EpsilonBanditLearner(epsilon=0.1)

        learner.learn(None, 2, 1, None, None)
        learner.learn(None, 1, 2, None, None)
        learner.learn(None, 2, 3, None, None)

        self.assertEqual([.5,.5],learner.predict(None, [1,2]))

class UcbBanditLearner_Tests(unittest.TestCase):
    
    def test_params(self):
        learner = UcbBanditLearner()
        self.assertEqual({ "family": "UCB_bandit" }, learner.params)

    def test_predict_all_actions_first(self):

        learner = UcbBanditLearner()
        actions = [1,2,3]

        self.assertEqual([1/3, 1/3, 1/3],learner.predict(None, actions))
        learner.learn(None, actions[0], 0, 0, None)

        self.assertEqual([0,1/2,1/2],learner.predict(None, actions))
        learner.learn(None, actions[1], 0, 0, None)
        
        self.assertEqual([0,  0,  1],learner.predict(None, actions))
        learner.learn(None, actions[2], 0, 0, None)

        #the last time all actions have the same value so we pick randomly
        self.assertEqual([1/3, 1/3, 1/3],learner.predict(None, actions))

    def test_learn_predict_best1(self):
        learner = UcbBanditLearner()
        actions = [1,2,3,4]
        
        learner.predict(None, actions)
        learner.predict(None, actions)
        learner.predict(None, actions)
        learner.predict(None, actions)

        learner.learn(None, actions[0], 1, None, None)
        learner.learn(None, actions[1], 1, None, None)        
        learner.learn(None, actions[2], 1, None, None)
        learner.learn(None, actions[3], 1, None, None)

        self.assertEqual([0.25,0.25,0.25,0.25], learner.predict(None, actions))

    def test_learn_predict_best2(self):
        learner = UcbBanditLearner()
        actions = [1,2,3,4]
        
        learner.predict(None, actions)
        learner.predict(None, actions)
        learner.predict(None, actions)
        learner.predict(None, actions)
        
        learner.learn(None, actions[0], 0, None, None)
        learner.learn(None, actions[1], 0, None, None)
        learner.learn(None, actions[2], 0, None, None)
        learner.learn(None, actions[3], 1, None, None)

        self.assertEqual([0, 0, 0, 1], learner.predict(None, actions))

    def test_learn_predict_best2(self):
        learner = UcbBanditLearner()
        actions = [1,2,3,4]
        
        learner.predict(None, actions)
        learner.predict(None, actions)
        learner.predict(None, actions)
        learner.predict(None, actions)
        
        learner.learn(None, actions[0], 0, None, None)
        learner.learn(None, actions[1], 0, None, None)
        learner.learn(None, actions[2], 0, None, None)
        learner.learn(None, actions[3], 1, None, None)
        learner.learn(None, actions[0], 0, None, None)
        learner.learn(None, actions[1], 0, None, None)
        learner.learn(None, actions[2], 0, None, None)
        learner.learn(None, actions[3], 1, None, None)

        self.assertEqual([0, 0, 0, 1], learner.predict(None, actions))

class FixedLearner_Tests(unittest.TestCase):

    def test_params(self):

        self.assertEqual({"family":"fixed"}, FixedLearner([1/2,1/2]).params)

    def test_create_errors(self):
        with self.assertRaises(AssertionError):
            FixedLearner([1/3, 1/2])

        with self.assertRaises(AssertionError):
            FixedLearner([-1, 2])

    def test_predict(self):
        learner = FixedLearner([1/3,1/3,1/3])
        self.assertEqual([1/3,1/3,1/3], learner.predict(None, [1,2,3]))

    def test_learn(self):
        FixedLearner([1/3,1/3,1/3]).learn(None, 1, 1, .5, None)

class RandomLearner_Tests(unittest.TestCase):

    def test_params(self):
        self.assertEqual({"family":"random"}, RandomLearner().params)

    def test_predict(self):
        learner = RandomLearner()
        self.assertEqual([0.25, 0.25, 0.25, 0.25], learner.predict(None, [1,2,3,4]))

    def test_learn(self):
        learner = RandomLearner()
        learner.learn(2, None, 1, 1, 1)

if __name__ == '__main__':
    unittest.main()
