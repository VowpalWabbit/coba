
import unittest
from unittest.case import SkipTest

from coba.utilities import check_vowpal_support
from coba.learners import RandomLearner, EpsilonLearner, VowpalLearner, UcbTunedLearner

class RandomLearner_Tests(unittest.TestCase):
    
    def test_choose(self):
        learner = RandomLearner(seed=20) #this seed always makes the first three choices [0,1,2]

        choice1 = learner.choose(1, None, [1,2,3])
        choice2 = learner.choose(1, None, [1,2,3])
        choice3 = learner.choose(1, None, [1,2,3])

        self.assertEqual(choice1,0)
        self.assertEqual(choice2,1)
        self.assertEqual(choice3,2)

    def test_learn(self):
        learner = RandomLearner()

        learner.learn(2, None, 1, 1)

class EpsilonLearner_Tests(unittest.TestCase):
    def test_choose_no_learn(self):
        learner = EpsilonLearner(epsilon=0.5,seed=75) #this seed always makes the first three choices [1,2,0]

        choice1 = learner.choose(1, None, [1,2,3])
        choice2 = learner.choose(1, None, [1,2,3])
        choice3 = learner.choose(1, None, [1,2,3])

        self.assertEqual(choice1,1)
        self.assertEqual(choice2,2)
        self.assertEqual(choice3,0)

    def test_choose_learn_no_epsilon(self):
        learner = EpsilonLearner(epsilon=0,seed=75)

        learner.learn(1, 1, 2, 1)
        learner.learn(1, 1, 1, 2)
        learner.learn(1, 1, 3, 3)

        choice1 = learner.choose(1, None, [1,2,3])
        choice2 = learner.choose(1, None, [1,2,3])
        choice3 = learner.choose(1, None, [1,2,3])

        self.assertEqual(choice1,2)
        self.assertEqual(choice2,2)
        self.assertEqual(choice3,2)

class UcbTunedLearner_Tests(unittest.TestCase):
    def test_choose_all_actions_first(self):
        for seed in [20,30,40]:
            learner = UcbTunedLearner(seed=seed)

            choice1 = learner.choose(1, None, [1,2,3])
            choice2 = learner.choose(1, None, [1,2,3])
            choice3 = learner.choose(1, None, [1,2,3])

            self.assertEqual(choice1,0)
            self.assertEqual(choice2,1)
            self.assertEqual(choice3,2)

    def test_learn_choose_best(self):
        learner = UcbTunedLearner()
        actions = [1,2,3]

        choice = learner.choose(0, None, actions)
        learner.learn(0, None, actions[choice], 1)

        choice = learner.choose(1, None, actions)
        learner.learn(1, None, actions[choice], 0)

        choice = learner.choose(2, None, actions)
        learner.learn(2, None, actions[choice], 0)

        choice1 = learner.choose(3, None, actions)
        choice2 = learner.choose(4, None, actions)
        choice3 = learner.choose(5, None, actions)

        self.assertEqual(choice1, 0)
        self.assertEqual(choice2, 0)
        self.assertEqual(choice3, 0)

class VowpalLearner_Tests(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        try:
            check_vowpal_support('VowpalLearner_Tests._make_learner')
        except ImportError:
            #if somebody is using the package with no intention of
            #using the VowpalLearner we don't want them to see failed
            #tests and think something is wrong so we skip these tests
            raise SkipTest("Vowpal Wabbit is not installed so no need to test VowpalLearner")

    def test_choose_epsilon_adf(self):
        # this seed always makes the first three choices [1,2,0]
        # however, VW is outside of our control so it is possible
        # future updates of VW will change how seeds affect 
        # randomness thus causing this test to break unexpectedly.
        learner = VowpalLearner(epsilon=0.05, is_adf=True, seed=20) 

        choice1 = learner.choose(1, None, [1,2,3])
        choice2 = learner.choose(1, None, [1,2,3])
        choice3 = learner.choose(1, None, [1,2,3])

        self.assertEqual(choice1,0)
        self.assertEqual(choice2,1)
        self.assertEqual(choice3,2)

    def test_choose_epsilon_not_adf(self):
        # this seed always makes the first three choices [1,0,0]
        # however, VW is outside of our control so it is possible
        # future updates of VW will change how seeds affect 
        # randomness thus causing this test to break unexpectedly.
        learner = VowpalLearner(epsilon=0.75, is_adf=False, seed=30) 

        choice1 = learner.choose(1, None, [1,2,3])
        choice2 = learner.choose(1, None, [1,2,3])
        choice3 = learner.choose(1, None, [1,2,3])

        self.assertEqual(choice1,0)
        self.assertEqual(choice2,1)
        self.assertEqual(choice3,2)

    def test_choose_bag_adf(self):
        # this seed always makes the first three choices [1,0,0]
        # however, VW is outside of our control so it is possible
        # future updates of VW will change how seeds affect 
        # randomness thus causing this test to break unexpectedly.
        learner = VowpalLearner(bag=5, is_adf=True, seed=30)

        choice1 = learner.choose(1, None, [1,2,3])
        choice2 = learner.choose(1, None, [1,2,3])
        choice3 = learner.choose(1, None, [1,2,3])

        self.assertEqual(choice1,0)
        self.assertEqual(choice2,1)
        self.assertEqual(choice3,2)

    def test_choose_bag_not_adf(self):
        # this seed always makes the first three choices [1,0,0]
        # however, VW is outside of our control so it is possible
        # future updates of VW will change how seeds affect 
        # randomness thus causing this test to break unexpectedly.
        learner = VowpalLearner(bag=5, is_adf=False, seed=30)

        choice1 = learner.choose(1, None, [1,2,3])
        choice2 = learner.choose(1, None, [1,2,3])
        choice3 = learner.choose(1, None, [1,2,3])

        self.assertEqual(choice1,0)
        self.assertEqual(choice2,0)
        self.assertEqual(choice3,0)

    def test_choose_cover_not_adf(self):
        # this seed always makes the first three choices [1,0,0]
        # however, VW is outside of our control so it is possible
        # future updates of VW will change how seeds affect 
        # randomness thus causing this test to break unexpectedly.
        learner = VowpalLearner(cover=5, seed=30)

        choice1 = learner.choose(1, None, [1,2,3])
        choice2 = learner.choose(1, None, [1,2,3])
        choice3 = learner.choose(1, None, [1,2,3])

        self.assertEqual(choice1,0)
        self.assertEqual(choice2,0)
        self.assertEqual(choice3,0)

if __name__ == '__main__':
    unittest.main()