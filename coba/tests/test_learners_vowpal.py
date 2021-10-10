import unittest

from coba.learners import VowpalLearner, VowpalMediator

class VowpalMockLearner:

    def __init__(self,args):
        self.args            = args
        self.predict_example = None
        self.learn_example   = None

    def predict(self, example):
        self.predict_example = example

    def learn(self, example):
        self.learn_example = example

class VowpalMockExample:
    
    def __init__(self,vw,ns,label,label_type):
        self.vw = vw
        self.ns = ns
        self.label = label
        self.label_type = label_type

class VowpalLearner_Tests(unittest.TestCase):

    def setUp(self):

        self.mock_learner: VowpalMockLearner = None

        def _make_learner(args):
            self.mock_learner = VowpalMockLearner(args)
            return self.mock_learner

        VowpalMediator.make_learner = _make_learner
        VowpalMediator.make_example = VowpalMockExample

    def test_epsilon_adf_create_args(self):

        VowpalLearner(epsilon=0.05, adf=True, seed=20).predict(None, ['yes','no'])

        expected_args = [
            "--cb_explore_adf",
            "--interactions ssa", 
            "--interactions sa",
            "--ignore_linear s",
            "--epsilon 0.05",
            "--random_seed 20",
            "--quiet"
        ]

        self.assertEqual(" ".join(expected_args), self.mock_learner.args)

    def test_epsilon_not_adf_create_args(self):

        VowpalLearner(epsilon=0.05, adf=False, seed=20).predict(None, ['yes','no'])

        expected_args = [
            "--cb_explore 2",
            "--interactions ssa", 
            "--interactions sa",
            "--ignore_linear s",
            "--epsilon 0.05",
            "--random_seed 20",
            "--quiet"
        ]

        self.assertEqual(" ".join(expected_args), self.mock_learner.args)

    def test_bag_adf_create_args(self):

        VowpalLearner(bag=2, adf=True, seed=20).predict(None, ['yes','no'])

        expected_args = [
            "--cb_explore_adf",
            "--interactions ssa", 
            "--interactions sa",
            "--ignore_linear s",
            "--bag 2",
            "--random_seed 20",
            "--quiet"
        ]

        self.assertEqual(" ".join(expected_args), self.mock_learner.args)

    def test_bag_not_adf_create_args(self):

        VowpalLearner(bag=2, adf=False, seed=20).predict(None, ['yes','no'])

        expected_args = [
            "--cb_explore 2",
            "--interactions ssa", 
            "--interactions sa",
            "--ignore_linear s",
            "--bag 2",
            "--random_seed 20",
            "--quiet"
        ]

        self.assertEqual(" ".join(expected_args), self.mock_learner.args)

    def test_bag_adf_create_args(self):

        VowpalLearner(bag=2, adf=True, seed=20).predict(None, ['yes','no'])

        expected_args = [
            "--cb_explore_adf",
            "--interactions ssa", 
            "--interactions sa",
            "--ignore_linear s",
            "--bag 2",
            "--random_seed 20",
            "--quiet"
        ]

        self.assertEqual(" ".join(expected_args), self.mock_learner.args)

    def test_cover_not_adf_create_args(self):

        VowpalLearner(cover=3, seed=20).predict(None, ['yes','no'])

        expected_args = [
            "--cb_explore 2",
            "--interactions ssa", 
            "--interactions sa",
            "--ignore_linear s",
            "--cover 3",
            "--random_seed 20",
            "--quiet"
        ]

        self.assertEqual(" ".join(expected_args), self.mock_learner.args)

    def test_softmax_adf_create_args(self):

        VowpalLearner(softmax=0.2, seed=20).predict(None, ['yes','no'])

        expected_args = [
            "--cb_explore_adf",
            "--interactions ssa", 
            "--interactions sa",
            "--ignore_linear s",
            "--softmax",
            "--lambda 0.2",
            "--random_seed 20",
            "--quiet"
        ]

        self.assertEqual(" ".join(expected_args), self.mock_learner.args)

    def test_adf_explicit_args(self):
        VowpalLearner("--cb_explore_adf --epsilon 0.75 --random_seed 20").predict(None, ['yes','no'])
        
        expected_args = [
            "--cb_explore_adf",
            "--epsilon 0.75",
            "--random_seed 20",
            "--quiet"
        ]

    def test_not_adf_explicit_args(self):
        VowpalLearner("--cb_explore 20 --epsilon 0.75 --random_seed 20").predict(None, ['yes','no'])
        
        expected_args = [
            "--cb_explore 2",
            "--epsilon 0.75",
            "--random_seed 20",
            "--quiet"
        ]
        
        self.assertEqual(" ".join(expected_args), self.mock_learner.args)

    def test_no_cb_explicit_args(self):
        with self.assertRaises(Exception) as e:
            VowpalLearner("--epsilon 0.75 --random_seed 20").predict(None, ['yes','no'])
        
        self.assertTrue("VowpalLearner was instantiated" in str(e.exception))

    def test_adf_predict_sans_context_str_actions(self):
        VowpalLearner(epsilon=0.05, adf=True, seed=20).predict(None, ['yes','no'])

        mock_learner = self.mock_learner    

        self.assertEqual(2, len(mock_learner.predict_example))
        
        self.assertEqual(mock_learner , mock_learner.predict_example[0].vw)
        self.assertEqual({'a':['yes']}, mock_learner.predict_example[0].ns)
        self.assertEqual(None         , mock_learner.predict_example[0].label)
        self.assertEqual(4            , mock_learner.predict_example[0].label_type)

        self.assertEqual(mock_learner , mock_learner.predict_example[1].vw)
        self.assertEqual({'a':['no']} , mock_learner.predict_example[1].ns)
        self.assertEqual(None         , mock_learner.predict_example[1].label)
        self.assertEqual(4            , mock_learner.predict_example[1].label_type)
        
        self.assertEqual(None         , mock_learner.learn_example)

    def test_adf_predict_with_str_context_str_actions(self):
        VowpalLearner(epsilon=0.05, adf=True, seed=20).predict('b', ['yes','no'])

        mock_learner = self.mock_learner    

        self.assertEqual(2, len(mock_learner.predict_example))
        
        self.assertEqual(mock_learner           , mock_learner.predict_example[0].vw)
        self.assertEqual({'s':['b'],'a':['yes']}, mock_learner.predict_example[0].ns)
        self.assertEqual(None                   , mock_learner.predict_example[0].label)
        self.assertEqual(4                      , mock_learner.predict_example[0].label_type)

        self.assertEqual(mock_learner , mock_learner.predict_example[1].vw)
        self.assertEqual({'s':['b'],'a':['no']} , mock_learner.predict_example[1].ns)
        self.assertEqual(None                   , mock_learner.predict_example[1].label)
        self.assertEqual(4                      , mock_learner.predict_example[1].label_type)
        
        self.assertEqual(None, mock_learner.learn_example)

    def test_adf_predict_with_dict_context_str_actions(self):
        VowpalLearner(epsilon=0.05, adf=True, seed=20).predict({'c':2}, ['yes','no'])

        mock_learner = self.mock_learner    

        self.assertEqual(2, len(mock_learner.predict_example))
        
        self.assertEqual(mock_learner           , mock_learner.predict_example[0].vw)
        self.assertEqual({'s':[('c',2)],'a':['yes']}, mock_learner.predict_example[0].ns)
        self.assertEqual(None                   , mock_learner.predict_example[0].label)
        self.assertEqual(4                      , mock_learner.predict_example[0].label_type)

        self.assertEqual(mock_learner , mock_learner.predict_example[1].vw)
        self.assertEqual({'s':[('c',2)],'a':['no']} , mock_learner.predict_example[1].ns)
        self.assertEqual(None                   , mock_learner.predict_example[1].label)
        self.assertEqual(4                      , mock_learner.predict_example[1].label_type)
        
        self.assertEqual(None, mock_learner.learn_example)

    def test_no_adf_predict_sans_context_str_actions(self):
        VowpalLearner(epsilon=0.05, adf=False, seed=20).predict(None, ['yes','no'])

        mock_learner = self.mock_learner    

        self.assertIsInstance(mock_learner.predict_example, VowpalMockExample)
        self.assertEqual(mock_learner , mock_learner.predict_example.vw)
        self.assertEqual({}           , mock_learner.predict_example.ns)
        self.assertEqual(None         , mock_learner.predict_example.label)
        self.assertEqual(4            , mock_learner.predict_example.label_type)

        self.assertEqual(None         , mock_learner.learn_example)

    def test_no_adf_predict_with_str_context_str_actions(self):
        VowpalLearner(epsilon=0.05, adf=False, seed=20).predict('b', ['yes','no'])

        mock_learner = self.mock_learner    

        self.assertIsInstance(mock_learner.predict_example, VowpalMockExample)
        self.assertEqual(mock_learner , mock_learner.predict_example.vw)
        self.assertEqual({'s':['b']}  , mock_learner.predict_example.ns)
        self.assertEqual(None         , mock_learner.predict_example.label)
        self.assertEqual(4            , mock_learner.predict_example.label_type)
        
        self.assertEqual(None         , mock_learner.learn_example)

    def test_adf_learn_sans_context_str_actions(self):
        learner = VowpalLearner(epsilon=0.05, adf=True, seed=20)
        learner.predict(None, ['yes','no'])
        learner.learn(None, 'yes', 1, 0.2, ['yes','no'])

        mock_learner = self.mock_learner    

        self.assertEqual(2, len(mock_learner.learn_example))
        
        self.assertEqual(mock_learner , mock_learner.learn_example[0].vw)
        self.assertEqual({'a':['yes']}, mock_learner.learn_example[0].ns)
        self.assertEqual("1:0:0.2"    , mock_learner.learn_example[0].label)
        self.assertEqual(4            , mock_learner.learn_example[0].label_type)

        self.assertEqual(mock_learner , mock_learner.learn_example[1].vw)
        self.assertEqual({'a':['no']} , mock_learner.learn_example[1].ns)
        self.assertEqual(None         , mock_learner.learn_example[1].label)
        self.assertEqual(4            , mock_learner.learn_example[1].label_type)
        
    def test_adf_learn_with_str_context_str_actions(self):
        learner = VowpalLearner(epsilon=0.05, adf=True, seed=20)
        learner.predict('b', ['yes','no'])
        learner.learn('b', 'no', .5, 0.2, ['yes','no'])

        mock_learner = self.mock_learner    

        self.assertEqual(2, len(mock_learner.learn_example))
        
        self.assertEqual(mock_learner           , mock_learner.learn_example[0].vw)
        self.assertEqual({'s':['b'],'a':['yes']}, mock_learner.learn_example[0].ns)
        self.assertEqual(None                   , mock_learner.learn_example[0].label)
        self.assertEqual(4                      , mock_learner.learn_example[0].label_type)

        self.assertEqual(mock_learner , mock_learner.predict_example[1].vw)
        self.assertEqual({'s':['b'],'a':['no']} , mock_learner.learn_example[1].ns)
        self.assertEqual("2:0.5:0.2"            , mock_learner.learn_example[1].label)
        self.assertEqual(4                      , mock_learner.learn_example[1].label_type)
        
    def test_adf_learn_with_dict_context_str_actions(self):
        learner = VowpalLearner(epsilon=0.05, adf=True, seed=20)
        learner.predict({'c':2}, ['yes','no'])
        learner.learn({'c':2}, 'no', .5, 0.2, ['yes','no'])

        mock_learner = self.mock_learner    

        self.assertEqual(2, len(mock_learner.learn_example))
        
        self.assertEqual(mock_learner           , mock_learner.learn_example[0].vw)
        self.assertEqual({'s':[('c',2)],'a':['yes']}, mock_learner.learn_example[0].ns)
        self.assertEqual(None                   , mock_learner.learn_example[0].label)
        self.assertEqual(4                      , mock_learner.learn_example[0].label_type)

        self.assertEqual(mock_learner , mock_learner.learn_example[1].vw)
        self.assertEqual({'s':[('c',2)],'a':['no']} , mock_learner.learn_example[1].ns)
        self.assertEqual("2:0.5:0.2"            , mock_learner.learn_example[1].label)
        self.assertEqual(4                      , mock_learner.learn_example[1].label_type)

    def test_no_adf_learn_sans_context_str_actions(self):
        learner = VowpalLearner(epsilon=0.05, adf=False, seed=20)
        learner.predict(None, ['yes','no'])
        learner.learn(None, 'no', .5, 0.2, ['yes','no'])

        mock_learner = self.mock_learner    

        self.assertIsInstance(mock_learner.learn_example, VowpalMockExample)
        self.assertEqual(mock_learner , mock_learner.learn_example.vw)
        self.assertEqual({}           , mock_learner.learn_example.ns)
        self.assertEqual("2:0.5:0.2"  , mock_learner.learn_example.label)
        self.assertEqual(4            , mock_learner.learn_example.label_type)

    def test_no_adf_learn_with_str_context_str_actions(self):
        learner = VowpalLearner(epsilon=0.05, adf=False, seed=20)
        learner.predict('b', ['yes','no'])
        learner.learn('b', 'yes', .25, 0.2, ['yes','no'])

        mock_learner = self.mock_learner    

        self.assertIsInstance(mock_learner.learn_example, VowpalMockExample)
        self.assertEqual(mock_learner , mock_learner.learn_example.vw)
        self.assertEqual({'s':['b']}  , mock_learner.learn_example.ns)
        self.assertEqual("1:0.75:0.2" , mock_learner.learn_example.label)
        self.assertEqual(4            , mock_learner.learn_example.label_type)

    def test_predict_epsilon_not_adf_args_error_1(self):
        learner = VowpalLearner("--cb_explore --epsilon 0.75 --random_seed 20")
        learner.predict(None, [1,2,3,4])

        with self.assertRaises(Exception) as e:
            learner.predict(None, [1,2,3,4,5])

        self.assertTrue("--cb_explore_adf" in str(e.exception))

    def test_predict_epsilon_not_adf_args_error_2(self):
        learner = VowpalLearner("--cb_explore --epsilon 0.75 --random_seed 20")
        learner.predict(None, [1,2,3,4])

        with self.assertRaises(Exception) as e:
            learner.predict(None, [1,2,3])

        self.assertTrue("--cb_explore_adf" in str(e.exception))

class prep_features_Tests(unittest.TestCase):
    def test_string(self):
        actual   = VowpalMediator.prep_features('a')
        expected = [ 'a' ]
        self.assertEqual(actual, expected)

    def test_numeric(self):
        actual   = VowpalMediator.prep_features(2)
        expected = [ (0,2) ]
        self.assertEqual(actual, expected)

    def test_dense_numeric_sequence(self):
        actual   = VowpalMediator.prep_features((1,2,3))
        expected = [ (0,1), (1,2), (2,3) ]
        self.assertEqual(actual, expected)

    def test_dense_string_sequence(self):
        actual   = VowpalMediator.prep_features((1,'a',3))
        expected = [ (0,1), 'a', (2,3) ]
        self.assertEqual(actual, expected)

    def test_sparse_dict_numeric_key_numeric_value(self):
        actual   = VowpalMediator.prep_features({1:1,2:2})
        expected = [ (1,1), (2,2) ]
        self.assertEqual(actual, expected)

    def test_sparse_dict_string_key_numeric_value(self):
        actual   = VowpalMediator.prep_features({'a':1,'b':2})
        expected = [ ('a',1), ('b',2) ]
        self.assertEqual(actual, expected)

    def test_sparse_dict_numeric_key_string_value(self):
        actual   = VowpalMediator.prep_features({1:'a',2:'b'})
        expected = [ 'a', 'b' ]
        self.assertEqual(actual, expected)

    def test_sparse_dict_string_key_string_value(self):
        actual   = VowpalMediator.prep_features({'c':'a','d':'b'})
        expected = [ 'a', 'b' ]
        self.assertEqual(actual, expected)

    def test_sparse_tuple(self):
        actual   = VowpalMediator.prep_features([('a',1),('b',2)])
        expected = [ ('a',1), ('b',2) ]
        self.assertEqual(actual, expected)

if __name__ == '__main__':
    unittest.main()