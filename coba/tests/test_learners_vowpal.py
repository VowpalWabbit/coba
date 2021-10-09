import unittest

from coba.utilities import PackageChecker
from coba.learners import VowpalLearner, vowpal_feature_prepper

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

    def test_create_epsilon(self):
        actual   = VowpalLearner(epsilon=0.1)._cli_args([1,2,3])
        expected = "--cb_explore_adf --interactions ssa --interactions sa --ignore_linear s --epsilon 0.1 --random_seed 1"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner(epsilon=0.1,seed=10)._cli_args([1,2,3])
        expected = "--cb_explore_adf --interactions ssa --interactions sa --ignore_linear s --epsilon 0.1 --random_seed 10"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner(epsilon=0.1, seed=None)._cli_args([1,2,3])
        expected = "--cb_explore_adf --interactions ssa --interactions sa --ignore_linear s --epsilon 0.1"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner(epsilon=0.1, adf=False)._cli_args([1,2,3])
        expected = "--cb_explore 3 --interactions ssa --interactions sa --ignore_linear s --epsilon 0.1 --random_seed 1"

        self.assertEqual(actual, expected)

    def test_create_bag(self):
        actual   = VowpalLearner(bag=2)._cli_args([1,2,3])
        expected = "--cb_explore_adf --interactions ssa --interactions sa --ignore_linear s --bag 2 --random_seed 1"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner(bag=2,seed=10)._cli_args([1,2,3])
        expected = "--cb_explore_adf --interactions ssa --interactions sa --ignore_linear s --bag 2 --random_seed 10"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner(bag=2, seed=None)._cli_args([1,2,3])
        expected = "--cb_explore_adf --interactions ssa --interactions sa --ignore_linear s --bag 2"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner(bag=2, adf=False)._cli_args([1,2,3])
        expected = "--cb_explore 3 --interactions ssa --interactions sa --ignore_linear s --bag 2 --random_seed 1"

        self.assertEqual(actual, expected)

    def test_create_cover(self):
        actual   = VowpalLearner(cover=2)._cli_args([1,2,3])
        expected = "--cb_explore 3 --interactions ssa --interactions sa --ignore_linear s --cover 2 --random_seed 1"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner(cover=2,seed=10)._cli_args([1,2,3])
        expected = "--cb_explore 3 --interactions ssa --interactions sa --ignore_linear s --cover 2 --random_seed 10"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner(cover=2, seed=None)._cli_args([1,2,3])
        expected = "--cb_explore 3 --interactions ssa --interactions sa --ignore_linear s --cover 2"

        self.assertEqual(actual, expected)

    def test_create_softmax(self):
        actual   = VowpalLearner(softmax=0.5)._cli_args([1,2,3])
        expected = "--cb_explore_adf --interactions ssa --interactions sa --ignore_linear s --softmax --lambda 0.5 --random_seed 1"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner(softmax=0.5,seed=10)._cli_args([1,2,3])
        expected = "--cb_explore_adf --interactions ssa --interactions sa --ignore_linear s --softmax --lambda 0.5 --random_seed 10"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner(softmax=0.5, seed=None)._cli_args([1,2,3])
        expected = "--cb_explore_adf --interactions ssa --interactions sa --ignore_linear s --softmax --lambda 0.5"

        self.assertEqual(actual, expected)

    def test_create_args(self):
        actual   = VowpalLearner("--cb_explore_adf --interactions sa --ignore_linear s --bag 2 --random_seed 1")._cli_args([1,2,3])
        expected = "--cb_explore_adf --interactions sa --ignore_linear s --bag 2 --random_seed 1"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner("--cb_explore 10 --interactions sa --ignore_linear s --bag 2 --random_seed 1")._cli_args([1,2,3])
        expected = "--cb_explore 3 --interactions sa --ignore_linear s --bag 2 --random_seed 1"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner("--cb_explore --interactions sa --ignore_linear s --bag 2 --random_seed 1")._cli_args([1,2,3])
        expected = "--cb_explore 3 --interactions sa --ignore_linear s --bag 2 --random_seed 1"

        self.assertEqual(actual, expected)

        actual   = VowpalLearner("--cb_explore_adf --interactions sa --ignore_linear s --bag 2")._cli_args([1,2,3])
        expected = "--cb_explore_adf --interactions sa --ignore_linear s --bag 2"

        self.assertEqual(actual, expected)

class vowpal_feature_prep_Tests(unittest.TestCase):
    def test_string(self):
        actual   = vowpal_feature_prepper('a')
        expected = [ 'a' ]
        self.assertEqual(actual, expected)

    def test_numeric(self):
        actual   = vowpal_feature_prepper(2)
        expected = [ (0,2) ]
        self.assertEqual(actual, expected)

    def test_dense_numeric_sequence(self):
        actual   = vowpal_feature_prepper((1,2,3))
        expected = [ (0,1), (1,2), (2,3) ]
        self.assertEqual(actual, expected)

    def test_dense_string_sequence(self):
        actual   = vowpal_feature_prepper((1,'a',3))
        expected = [ (0,1), 'a', (2,3) ]
        self.assertEqual(actual, expected)

    def test_sparse_dict_numeric_key_numeric_value(self):
        actual   = vowpal_feature_prepper({1:1,2:2})
        expected = [ (1,1), (2,2) ]
        self.assertEqual(actual, expected)

    def test_sparse_dict_string_key_numeric_value(self):
        actual   = vowpal_feature_prepper({'a':1,'b':2})
        expected = [ ('a',1), ('b',2) ]
        self.assertEqual(actual, expected)

    def test_sparse_dict_numeric_key_string_value(self):
        actual   = vowpal_feature_prepper({1:'a',2:'b'})
        expected = [ 'a', 'b' ]
        self.assertEqual(actual, expected)

    def test_sparse_dict_string_key_string_value(self):
        actual   = vowpal_feature_prepper({'c':'a','d':'b'})
        expected = [ 'a', 'b' ]
        self.assertEqual(actual, expected)

    def test_sparse_tuple(self):
        actual   = vowpal_feature_prepper([('a',1),('b',2)])
        expected = [ ('a',1), ('b',2) ]
        self.assertEqual(actual, expected)

if __name__ == '__main__':
    unittest.main()