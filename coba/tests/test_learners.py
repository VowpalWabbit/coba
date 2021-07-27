import unittest
from unittest.case import SkipTest

from coba.utilities import PackageChecker
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
        learner = VowpalLearner(epsilon=0.05, adf=True, seed=20) 
        self.assertEqual([0.25,0.25,0.25,0.25],learner.predict(1, None, [1,2,3,4]))

    def test_predict_epsilon_adf_args(self):
        learner = VowpalLearner("--cb_explore_adf --epsilon 0.05 --random_seed 20") 
        self.assertEqual([0.25,0.25,0.25,0.25],learner.predict(1, None, [1,2,3,4]))

    def test_predict_epsilon_dict_context_adf(self):
        learner = VowpalLearner(epsilon=0.05, adf=True, seed=20) 

        self.assertEqual([0.25,0.25,0.25,0.25],learner.predict(1, {1:10.2, 2:3.5}, [1,2,3,4])) #type: ignore

    def test_predict_epsilon_tuple_context_adf(self):
        learner = VowpalLearner(epsilon=0.05, adf=True, seed=20) 

        self.assertEqual([0.25,0.25,0.25,0.25],learner.predict(1, ((1,2),(10.2,3.5)), [1,2,3,4])) #type: ignore

    def test_predict_epsilon_not_adf(self):
        learner = VowpalLearner(epsilon=0.75, adf=False, seed=30) 

        self.assertEqual([0.25+0.25*0.75,0.25*0.75,0.25*0.75,0.25*0.75],learner.predict(1, None, [1,2,3,4]))

    def test_predict_epsilon_not_adf_args(self):
        learner = VowpalLearner("--cb_explore 20 --epsilon 0.75 --random_seed 20") 
        self.assertEqual([0.25+0.25*0.75,0.25*0.75,0.25*0.75,0.25*0.75],learner.predict(1, None, [1,2,3,4]))

    def test_predict_epsilon_not_adf_args_error_1(self):
        learner = VowpalLearner("--cb_explore --epsilon 0.75 --random_seed 20")
        self.assertEqual([0.25+0.25*0.75,0.25*0.75,0.25*0.75,0.25*0.75],learner.predict(1, None, [1,2,3,4]))

        with self.assertRaises(Exception) as e:
            self.assertEqual([0.25+0.25*0.75,0.25*0.75,0.25*0.75,0.25*0.75],learner.predict(1, None, [1,2,3,4,5]))

        self.assertTrue("--cb_explore_adf" in str(e.exception))

    def test_predict_epsilon_not_adf_args_error_2(self):
        learner = VowpalLearner("--cb_explore --epsilon 0.75 --random_seed 20")
        self.assertEqual([0.25+0.25*0.75,0.25*0.75,0.25*0.75,0.25*0.75],learner.predict(1, None, [1,2,3,4]))

        with self.assertRaises(Exception) as e:
            self.assertEqual([0.25+0.25*0.75,0.25*0.75,0.25*0.75,0.25*0.75],learner.predict(1, None, [1,2,3]))

        self.assertTrue("--cb_explore_adf" in str(e.exception))

    def test_predict_bag_adf(self):
        learner = VowpalLearner(bag=5, adf=True, seed=30)

        self.assertEqual([0.25,0.25,0.25,0.25],learner.predict(1, None, ['1','2','3','4']))

    def test_predict_bag_not_adf(self):
        learner = VowpalLearner(bag=5, adf=False, seed=30)

        self.assertEqual([1,0,0,0], learner.predict(1, None, [1,2,3,4]))

    def test_predict_cover_not_adf(self):
        learner = VowpalLearner(cover=5, seed=30)

        self.assertEqual([0.25,0.25,0.25,0.25], learner.predict(1, None, [1,2,3,4]))

    def test_predict_dense_cb_explore(self):
        actual   = VowpalLearner._predict_format(False, (1,2,3), ["A","B","C"])
        expected = "|s 0:1 1:2 2:3"

        self.assertEqual(actual, expected)

    def test_predict_sparse_cb_explore_1(self):
        actual   = VowpalLearner._predict_format(False, {'a':1, 3:2}, ["A","B","C"])
        expected = "|s a:1 3:2"

        self.assertEqual(actual, expected)

    def test_predict_sparse_cb_explore_2(self):
        actual   = VowpalLearner._predict_format(False, ((1,5,7), (2,2,3)), ["A","B","C"])
        expected = "|s 1:2 5:2 7:3"

        self.assertEqual(actual, expected)

    def test_predict_dense_cb_explore_adf(self):
        actual   = VowpalLearner._predict_format(True, (1,2,3), [(1,0,0),(0,1,0),(0,0,1)])
        expected = "shared |s 0:1 1:2 2:3\n|a 0:1\n|a 1:1\n|a 2:1"

        self.assertEqual(actual, expected)

    def test_predict_sparse_cb_explore_adf_1(self):
        actual   = VowpalLearner._predict_format(True, (1,2,3), ["A","B","C"])
        expected = "shared |s 0:1 1:2 2:3\n|a A:1\n|a B:1\n|a C:1"

        self.assertEqual(actual, expected)

    def test_predict_sparse_cb_explore_adf_2(self):
        actual   = VowpalLearner._predict_format(True, {"D":1,"C":2}, ["A","B","C"])
        expected = "shared |s D:1 C:2\n|a A:1\n|a B:1\n|a C:1"

        self.assertEqual(actual, expected)
    
    def test_predict_sparse_cb_explore_adf_3(self):
        actual   = VowpalLearner._predict_format(True, ((1,3),(1,2)), ["A","B","C"])
        expected = "shared |s 1:1 3:2\n|a A:1\n|a B:1\n|a C:1"

        self.assertEqual(actual, expected)

    def test_learn_dense_cb_explore_1(self):
        actual   = VowpalLearner._learn_format(False, 0.25, [(1,0,0),(0,1,0),(0,0,1)], (1,2,3), (0,1,0), 1)
        expected = "2:-1:0.25 |s 0:1 1:2 2:3"

        self.assertEqual(actual, expected)
    
    def test_learn_dense_cb_explore_2(self):
        actual   = VowpalLearner._learn_format(False, 0.33, [(1,0,0),(0,1,0),(0,0,1)], (1,2,3), (1,0,0), .5)
        expected = "1:-0.5:0.33 |s 0:1 1:2 2:3"

        self.assertEqual(actual, expected)

    def test_learn_sparse_cb_explore_1(self):
        actual   = VowpalLearner._learn_format(False, 0.33, [(1,0,0),(0,1,0),(0,0,1)], {"A":1,"B":2}, (1,0,0), .25)
        expected = "1:-0.25:0.33 |s A:1 B:2"

        self.assertEqual(actual, expected)
    
    def test_learn_sparse_cb_explore_2(self):
        actual   = VowpalLearner._learn_format(False, 0.33, [(1,0,0),(0,1,0),(0,0,1)], ((1,5),(3,4)), (1,0,0), 1)
        expected = "1:-1:0.33 |s 1:3 5:4"

        self.assertEqual(actual, expected)

    def test_learn_dense_cb_explore_adf_1(self):
        actual   = VowpalLearner._learn_format(True, 0.25, [(1,0,0),(0,1,0),(0,0,1)], (1,2,3), (0,1,0), 1)
        expected = "shared |s 0:1 1:2 2:3\n|a 0:1\n2:-1:0.25 |a 1:1\n|a 2:1"

        self.assertEqual(actual, expected)
    
    def test_learn_dense_cb_explore_adf_2(self):
        actual   = VowpalLearner._learn_format(True, 0.33, [(1,0,0),(0,1,0),(0,0,1)], (1,2,3), (0,0,1), 0.25)
        expected = "shared |s 0:1 1:2 2:3\n|a 0:1\n|a 1:1\n3:-0.25:0.33 |a 2:1"

        self.assertEqual(actual, expected)
    
    def test_learn_sparse_cb_explore_adf_1(self):
        actual   = VowpalLearner._learn_format(True, 0.11, ["A","B","C"], {"D":1,"E":2}, "C", 0.33)
        expected = "shared |s D:1 E:2\n|a A:1\n|a B:1\n3:-0.33:0.11 |a C:1"

        self.assertEqual(actual, expected)
    
    def test_learn_sparse_cb_explore_adf_2(self):
        actual   = VowpalLearner._learn_format(True, 0.1111111, ["A","B","C"], ((3,5), (2,3)), "B", 0.3333)
        expected = "shared |s 3:2 5:3\n|a A:1\n2:-0.3333:0.11111 |a B:1\n|a C:1"

        self.assertEqual(actual, expected)

    def test_create_epsilon(self):
        actual   = VowpalLearner(epsilon=0.1)._create_format([1,2,3])
        expected = "--cb_explore_adf --interactions ssa --interactions sa --ignore_linear s --epsilon 0.1 --random_seed 1"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner(epsilon=0.1,seed=10)._create_format([1,2,3])
        expected = "--cb_explore_adf --interactions ssa --interactions sa --ignore_linear s --epsilon 0.1 --random_seed 10"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner(epsilon=0.1, seed=None)._create_format([1,2,3])
        expected = "--cb_explore_adf --interactions ssa --interactions sa --ignore_linear s --epsilon 0.1"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner(epsilon=0.1, adf=False)._create_format([1,2,3])
        expected = "--cb_explore 3 --interactions ssa --interactions sa --ignore_linear s --epsilon 0.1 --random_seed 1"

        self.assertEqual(actual, expected)

    def test_create_bag(self):
        actual   = VowpalLearner(bag=2)._create_format([1,2,3])
        expected = "--cb_explore_adf --interactions ssa --interactions sa --ignore_linear s --bag 2 --random_seed 1"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner(bag=2,seed=10)._create_format([1,2,3])
        expected = "--cb_explore_adf --interactions ssa --interactions sa --ignore_linear s --bag 2 --random_seed 10"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner(bag=2, seed=None)._create_format([1,2,3])
        expected = "--cb_explore_adf --interactions ssa --interactions sa --ignore_linear s --bag 2"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner(bag=2, adf=False)._create_format([1,2,3])
        expected = "--cb_explore 3 --interactions ssa --interactions sa --ignore_linear s --bag 2 --random_seed 1"

        self.assertEqual(actual, expected)

    def test_create_cover(self):
        actual   = VowpalLearner(cover=2)._create_format([1,2,3])
        expected = "--cb_explore 3 --interactions ssa --interactions sa --ignore_linear s --cover 2 --random_seed 1"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner(cover=2,seed=10)._create_format([1,2,3])
        expected = "--cb_explore 3 --interactions ssa --interactions sa --ignore_linear s --cover 2 --random_seed 10"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner(cover=2, seed=None)._create_format([1,2,3])
        expected = "--cb_explore 3 --interactions ssa --interactions sa --ignore_linear s --cover 2"

        self.assertEqual(actual, expected)

    def test_create_softmax(self):
        actual   = VowpalLearner(softmax=0.5)._create_format([1,2,3])
        expected = "--cb_explore_adf --interactions ssa --interactions sa --ignore_linear s --softmax --lambda 0.5 --random_seed 1"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner(softmax=0.5,seed=10)._create_format([1,2,3])
        expected = "--cb_explore_adf --interactions ssa --interactions sa --ignore_linear s --softmax --lambda 0.5 --random_seed 10"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner(softmax=0.5, seed=None)._create_format([1,2,3])
        expected = "--cb_explore_adf --interactions ssa --interactions sa --ignore_linear s --softmax --lambda 0.5"

        self.assertEqual(actual, expected)

    def test_create_args(self):
        actual   = VowpalLearner("--cb_explore_adf --interactions sa --ignore_linear s --bag 2 --random_seed 1")._create_format([1,2,3])
        expected = "--cb_explore_adf --interactions sa --ignore_linear s --bag 2 --random_seed 1"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner("--cb_explore 10 --interactions sa --ignore_linear s --bag 2 --random_seed 1")._create_format([1,2,3])
        expected = "--cb_explore 3 --interactions sa --ignore_linear s --bag 2 --random_seed 1"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner("--cb_explore --interactions sa --ignore_linear s --bag 2 --random_seed 1")._create_format([1,2,3])
        expected = "--cb_explore 3 --interactions sa --ignore_linear s --bag 2 --random_seed 1"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner("--cb_explore_adf --interactions sa --ignore_linear s --bag 2")._create_format([1,2,3])
        expected = "--cb_explore_adf --interactions sa --ignore_linear s --bag 2"

        self.assertEqual(actual, expected)

if __name__ == '__main__':
    unittest.main()