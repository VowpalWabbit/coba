import unittest
from unittest.case import SkipTest

from coba.tools import PackageChecker
from coba.learners import RandomLearner, EpsilonBanditLearner, VowpalLearner, UcbBanditLearner

class RandomLearner_Tests(unittest.TestCase):
    
    def test_predict(self):
        learner = RandomLearner()
        self.assertEqual([0.25, 0.25, 0.25, 0.25], learner.predict(1, None, [1,2,3,4]))

    def test_learn(self):
        learner = RandomLearner()
        learner.learn(2, None, 1, 1, 1)

class EpsilonBanditLearner_Tests(unittest.TestCase):
    def test_predict_no_learn(self):
        learner = EpsilonBanditLearner(epsilon=0.5)

        self.assertEqual([.25,.25,.25,.25],learner.predict(1, None, [1,2,3,4]))
        self.assertEqual([.25,.25,.25,.25],learner.predict(1, None, [1,2,3,4]))

    def test_predict_learn_no_epsilon(self):
        learner = EpsilonBanditLearner(epsilon=0)

        learner.learn(1, None, 2, 1, 1)
        learner.learn(2, None, 1, 2, 1)
        learner.learn(3, None, 3, 3, 1)

        self.assertEqual([0,0,1],learner.predict(4, None, [1,2,3]))

class UcbBanditLearner_Tests(unittest.TestCase):
    def test_predict_all_actions_first(self):

        learner = UcbBanditLearner()

        self.assertEqual([1,0,0],learner.predict(1, None, [1,2,3]))
        self.assertEqual([0,1,0],learner.predict(1, None, [1,2,3]))
        self.assertEqual([0,0,1],learner.predict(1, None, [1,2,3]))

    def test_learn_predict_best1(self):
        learner = UcbBanditLearner()
        actions = [1,2,3,4]
        
        learner.predict(0, None, actions)
        learner.learn(0, None, actions[0], 1, 1)
        
        learner.predict(1, None, actions)
        learner.learn(1, None, actions[1], 1, 1)
        
        learner.predict(2, None, actions)
        learner.learn(2, None, actions[2], 1, 1)
        
        learner.predict(3, None, actions)
        learner.learn(2, None, actions[3], 1, 1)

        self.assertEqual([0.25,0.25,0.25,0.25], learner.predict(3, None, actions))

    def test_learn_predict_best2(self):
        learner = UcbBanditLearner()
        actions = [1,2,3,4]
        
        learner.predict(0, None, actions)
        learner.learn(0, None, actions[0], 0, 1)
        
        learner.predict(1, None, actions)
        learner.learn(1, None, actions[1], 0, 1)
        
        learner.predict(2, None, actions)
        learner.learn(2, None, actions[2], 0, 1)
        
        learner.predict(3, None, actions)
        learner.learn(2, None, actions[3], 1, 1)

        self.assertEqual([0, 0, 0, 1], learner.predict(3, None, actions))

class VowpalLearner_Tests(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        try:
            PackageChecker.vowpalwabbit('VowpalLearner_Tests._make_learner')
        except ImportError:
            #if somebody is using the package with no intention of
            #using the VowpalLearner we don't want them to see failed
            #tests and think something is wrong so we skip these tests
            raise SkipTest("Vowpal Wabbit is not installed so no need to test VowpalLearner")

    def test_predict_epsilon_adf(self):
        learner = VowpalLearner(epsilon=0.05, is_adf=True, seed=20) 

        self.assertEqual([0.25,0.25,0.25,0.25],learner.predict(1, None, [1,2,3,4]))

    def test_predict_epsilon_dict_context_adf(self):
        learner = VowpalLearner(epsilon=0.05, is_adf=True, seed=20) 

        self.assertEqual([0.25,0.25,0.25,0.25],learner.predict(1, {1:10.2, 2:3.5}, [1,2,3,4]))

    def test_predict_epsilon_not_adf(self):
        learner = VowpalLearner(epsilon=0.75, is_adf=False, seed=30) 

        self.assertEqual([0.25+0.25*0.75,0.25*0.75,0.25*0.75,0.25*0.75],learner.predict(1, None, [1,2,3,4]))

    def test_predict_bag_adf(self):
        learner = VowpalLearner(bag=5, is_adf=True, seed=30)

        self.assertEqual([0.25,0.25,0.25,0.25],learner.predict(1, None, [1,2,3,4]))

    def test_predict_bag_not_adf(self):
        learner = VowpalLearner(bag=5, is_adf=False, seed=30)

        self.assertEqual([1,0,0,0], learner.predict(1, None, [1,2,3,4]))

    def test_predict_cover_not_adf(self):
        learner = VowpalLearner(cover=5, seed=30)

        self.assertEqual([1,0,0,0], learner.predict(1, None, [1,2,3,4]))
        
if __name__ == '__main__':
    unittest.main()