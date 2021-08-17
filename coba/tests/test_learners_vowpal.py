import unittest

from coba.utilities import PackageChecker
from coba.learners import VowpalLearner

class VowpalLearner_Tests(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        try:
            PackageChecker.vowpalwabbit('VowpalLearner_Tests._make_learner')
        except ImportError:
            #if somebody is using the package with no intention of
            #using the VowpalLearner we don't want them to see failed
            #tests and think something is wrong so we skip these tests
            raise unittest.SkipTest("Vowpal Wabbit is not installed so no need to test VowpalLearner")

    def test_predict_epsilon_adf(self):
        learner = VowpalLearner(epsilon=0.05, adf=True, seed=20) 
        self.assertEqual([0.25,0.25,0.25,0.25],learner.predict(None, [1,2,3,4])[0])

    def test_predict_epsilon_adf_args(self):
        learner = VowpalLearner("--cb_explore_adf --epsilon 0.05 --random_seed 20") 
        self.assertEqual([0.25,0.25,0.25,0.25],learner.predict(None, [1,2,3,4])[0])

    def test_predict_epsilon_dict_context_adf(self):
        learner = VowpalLearner(epsilon=0.05, adf=True, seed=20) 

        self.assertEqual([0.25,0.25,0.25,0.25],learner.predict({1:10.2, 2:3.5}, [1,2,3,4])[0]) #type: ignore

    def test_predict_epsilon_tuple_context_adf(self):
        learner = VowpalLearner(epsilon=0.05, adf=True, seed=20) 

        self.assertEqual([0.25,0.25,0.25,0.25],learner.predict(((1,2),(10.2,3.5)), [1,2,3,4])[0]) #type: ignore

    def test_predict_epsilon_not_adf(self):
        learner = VowpalLearner(epsilon=0.75, adf=False, seed=30) 

        self.assertEqual([0.25+0.25*0.75,0.25*0.75,0.25*0.75,0.25*0.75],learner.predict(None, [1,2,3,4])[0])

    def test_predict_epsilon_not_adf_args(self):
        learner = VowpalLearner("--cb_explore 20 --epsilon 0.75 --random_seed 20") 
        self.assertEqual([0.25+0.25*0.75,0.25*0.75,0.25*0.75,0.25*0.75],learner.predict(None, [1,2,3,4])[0])

    def test_predict_epsilon_not_adf_args_error_1(self):
        learner = VowpalLearner("--cb_explore --epsilon 0.75 --random_seed 20")
        self.assertEqual([0.25+0.25*0.75,0.25*0.75,0.25*0.75,0.25*0.75],learner.predict(None, [1,2,3,4])[0])

        with self.assertRaises(Exception) as e:
            self.assertEqual([0.25+0.25*0.75,0.25*0.75,0.25*0.75,0.25*0.75],learner.predict(None, [1,2,3,4,5])[0])

        self.assertTrue("--cb_explore_adf" in str(e.exception))

    def test_predict_epsilon_not_adf_args_error_2(self):
        learner = VowpalLearner("--cb_explore --epsilon 0.75 --random_seed 20")
        self.assertEqual([0.25+0.25*0.75,0.25*0.75,0.25*0.75,0.25*0.75],learner.predict(None, [1,2,3,4])[0])

        with self.assertRaises(Exception) as e:
            self.assertEqual([0.25+0.25*0.75,0.25*0.75,0.25*0.75,0.25*0.75],learner.predict(None, [1,2,3])[0])

        self.assertTrue("--cb_explore_adf" in str(e.exception))

    def test_predict_bag_adf(self):
        learner = VowpalLearner(bag=5, adf=True, seed=30)

        self.assertEqual([0.25,0.25,0.25,0.25],learner.predict(None, ['1','2','3','4'])[0])

    def test_predict_bag_not_adf(self):
        learner = VowpalLearner(bag=5, adf=False, seed=30)

        self.assertEqual([1,0,0,0], learner.predict(None, [1,2,3,4])[0])

    def test_predict_cover_not_adf(self):
        learner = VowpalLearner(cover=5, seed=30)

        self.assertEqual([0.25,0.25,0.25,0.25], learner.predict(None, [1,2,3,4])[0])

    def test_predict_dense_cb_explore(self):
        actual   = VowpalLearner(adf=False)._predict_format((1,2,3), ["A","B","C"])
        expected = "|s 0:1 1:2 2:3"

        self.assertEqual(actual, expected)

    def test_predict_sparse_cb_explore_1(self):
        actual   = VowpalLearner(adf=False)._predict_format({'a':1, 3:2}, ["A","B","C"])
        expected = "|s a:1 3:2"

        self.assertEqual(actual, expected)

    def test_predict_sparse_cb_explore_2(self):
        actual   = VowpalLearner(adf=False)._predict_format(((1,5,7), (2,2,3)), ["A","B","C"])
        expected = "|s 1:2 5:2 7:3"

        self.assertEqual(actual, expected)

    def test_predict_dense_cb_explore_adf(self):
        actual   = VowpalLearner(adf=True)._predict_format((1,2,3), [(1,0,0),(0,1,0),(0,0,1)])
        expected = "shared |s 0:1 1:2 2:3\n|a 0:1\n|a 1:1\n|a 2:1"

        self.assertEqual(actual, expected)

    def test_predict_sparse_cb_explore_adf_1(self):
        actual   = VowpalLearner(adf=True)._predict_format((1,2,3), ["A","B","C"])
        expected = "shared |s 0:1 1:2 2:3\n|a A:1\n|a B:1\n|a C:1"

        self.assertEqual(actual, expected)

    def test_predict_sparse_cb_explore_adf_2(self):
        actual   = VowpalLearner(adf=True)._predict_format({"D":1,"C":2}, ["A","B","C"])
        expected = "shared |s D:1 C:2\n|a A:1\n|a B:1\n|a C:1"

        self.assertEqual(actual, expected)
    
    def test_predict_sparse_cb_explore_adf_3(self):
        actual   = VowpalLearner(adf=True)._predict_format(((1,3),(1,2)), ["A","B","C"])
        expected = "shared |s 1:1 3:2\n|a A:1\n|a B:1\n|a C:1"

        self.assertEqual(actual, expected)

    def test_learn_dense_cb_explore_1(self):
        actual   = VowpalLearner(adf=False)._learn_format(0.25, [(1,0,0),(0,1,0),(0,0,1)], (1,2,3), (0,1,0), 1)
        expected = "2:-1:0.25 |s 0:1 1:2 2:3"

        self.assertEqual(actual, expected)
    
    def test_learn_dense_cb_explore_2(self):
        actual   = VowpalLearner(adf=False)._learn_format(0.33, [(1,0,0),(0,1,0),(0,0,1)], (1,2,3), (1,0,0), .5)
        expected = "1:-0.5:0.33 |s 0:1 1:2 2:3"

        self.assertEqual(actual, expected)

    def test_learn_sparse_cb_explore_1(self):
        actual   = VowpalLearner(adf=False)._learn_format(0.33, [(1,0,0),(0,1,0),(0,0,1)], {"A":1,"B":2}, (1,0,0), .25)
        expected = "1:-0.25:0.33 |s A:1 B:2"

        self.assertEqual(actual, expected)
    
    def test_learn_sparse_cb_explore_2(self):
        actual   = VowpalLearner(adf=False)._learn_format(0.33, [(1,0,0),(0,1,0),(0,0,1)], ((1,5),(3,4)), (1,0,0), 1)
        expected = "1:-1:0.33 |s 1:3 5:4"

        self.assertEqual(actual, expected)

    def test_learn_dense_cb_explore_adf_1(self):
        actual   = VowpalLearner(adf=True)._learn_format(0.25, [(1,0,0),(0,1,0),(0,0,1)], (1,2,3), (0,1,0), 1)
        expected = "shared |s 0:1 1:2 2:3\n|a 0:1\n2:-1:0.25 |a 1:1\n|a 2:1"

        self.assertEqual(actual, expected)
    
    def test_learn_dense_cb_explore_adf_2(self):
        actual   = VowpalLearner(adf=True)._learn_format(0.33, [(1,0,0),(0,1,0),(0,0,1)], (1,2,3), (0,0,1), 0.25)
        expected = "shared |s 0:1 1:2 2:3\n|a 0:1\n|a 1:1\n3:-0.25:0.33 |a 2:1"

        self.assertEqual(actual, expected)
    
    def test_learn_sparse_cb_explore_adf_1(self):
        actual   = VowpalLearner(adf=True)._learn_format(0.11, ["A","B","C"], {"D":1,"E":2}, "C", 0.33)
        expected = "shared |s D:1 E:2\n|a A:1\n|a B:1\n3:-0.33:0.11 |a C:1"

        self.assertEqual(actual, expected)
    
    def test_learn_sparse_cb_explore_adf_2(self):
        actual   = VowpalLearner(adf=True)._learn_format(0.1111111, ["A","B","C"], ((3,5), (2,3)), "B", 0.3333)
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