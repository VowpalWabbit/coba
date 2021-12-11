import importlib.util
import unittest
import unittest.mock

from coba.utilities import KeyDefaultDict
from coba.exceptions import CobaException
from coba.learners import VowpalLearner, VowpalMediator

class VowpalMockExample:
    
    def __init__(self,vw,ns,label,label_type):
        self.vw = vw
        self.ns = ns
        self.label = label
        self.label_type = label_type

class VowpalLearner_Tests(unittest.TestCase):

    def tearDown(self) -> None:
        self.module.stop()
        self.mediat.start()

    def setUp(self) -> None:
        self.module = unittest.mock.patch('importlib.import_module')
        self.mediat = unittest.mock.patch('coba.learners.vowpal.VowpalMediator')
        self.module.start()
        self.mocked  = self.mediat.start()

        def _prep_features(features):
            return VowpalMediator.prep_features(features)

        def _get_version() -> str:
            return "8.11.0"

        def _make_example(vw, ns, label, label_type):
            return VowpalMockExample(vw, ns, label, label_type)

        self.mocked._string_cache = KeyDefaultDict(str)
        self.mocked.prep_features = _prep_features
        self.mocked.get_version   = _get_version
        self.mocked.make_example  = _make_example

    def tearDown(self) -> None:
        self.mediat.stop()

    def test_epsilon_adf_create_args(self):

        VowpalLearner(epsilon=0.05, seed=20).predict(None, ['yes','no'])

        expected_args = [
            "--cb_explore_adf",
            "--epsilon 0.05",
            "--interactions xxa",
            "--interactions xa",
            "--ignore_linear x",
            "--random_seed 20",
            "--quiet"
        ]

        self.mocked.make_learner.assert_called_with(" ".join(expected_args))

    def test_epsilon_adf_create_args2(self):

        VowpalLearner(epsilon=0.05, seed=20, b=10).predict(None, ['yes','no'])

        expected_args = [
            "--cb_explore_adf",
            "--epsilon 0.05",
            "--interactions xxa",
            "--interactions xa",
            "--ignore_linear x",
            "--random_seed 20",
            "-b 10",
            "--quiet"
        ]

        self.mocked.make_learner.assert_called_with(" ".join(expected_args))

    def test_epsilon_adf_create_args3(self):

        VowpalLearner(epsilon=0.05, seed=20, bit_precision=10).predict(None, ['yes','no'])

        expected_args = [
            "--cb_explore_adf",
            "--epsilon 0.05",
            "--interactions xxa",
            "--interactions xa",
            "--ignore_linear x",
            "--random_seed 20",
            "--bit_precision 10",
            "--quiet"
        ]

        self.mocked.make_learner.assert_called_with(" ".join(expected_args))

    def test_epsilon_adf_create_args4(self):

        VowpalLearner().predict(None, ['yes','no'])

        expected_args = [
            "--cb_explore_adf",
            "--epsilon 0.1",
            "--interactions xxa",
            "--interactions xa",
            "--ignore_linear x",
            "--random_seed 1",
            "--quiet"
        ]

        self.mocked.make_learner.assert_called_with(" ".join(expected_args))

    def test_bag_adf_create_args(self):

        VowpalLearner(bag=2, adf=True, seed=20).predict(None, ['yes','no'])

        expected_args = [
            "--cb_explore_adf",
            "--bag 2",
            "--interactions xxa",
            "--interactions xa",
            "--ignore_linear x",
            "--random_seed 20",
            "--quiet"
        ]

        self.mocked.make_learner.assert_called_with(" ".join(expected_args))

    def test_bag_not_adf_create_args(self):

        VowpalLearner(bag=2, adf=False, seed=20).predict(None, ['yes','no'])

        expected_args = [
            "--cb_explore 2",
            "--bag 2",
            "--interactions xxa",
            "--interactions xa",
            "--ignore_linear x",
            "--random_seed 20",
            "--quiet"
        ]

        self.mocked.make_learner.assert_called_with(" ".join(expected_args))

    def test_cover_adf_create_args(self):

        VowpalLearner(cover=3, seed=20).predict(None, ['yes','no'])

        expected_args = [
            "--cb_explore_adf",
            "--cover 3",
            "--interactions xxa",
            "--interactions xa",
            "--ignore_linear x",
            "--random_seed 20",
            "--quiet"
        ]

        self.mocked.make_learner.assert_called_with(" ".join(expected_args))

    def test_softmax_adf_create_args(self):

        VowpalLearner(softmax=0.2, seed=20).predict(None, ['yes','no'])

        expected_args = [
            "--cb_explore_adf",
            "--softmax",
            "--lambda 0.2",
            "--interactions xxa", 
            "--interactions xa",
            "--ignore_linear x",
            "--random_seed 20",
            "--quiet"
        ]

        self.mocked.make_learner.assert_called_with(" ".join(expected_args))

    def test_regcb_opt_create_args(self):

        VowpalLearner(regcb="opt").predict(None, ['yes','no'])

        expected_args = [
            "--cb_explore_adf",
            "--regcb",
            "--interactions xxa",
            "--interactions xa",
            "--ignore_linear x",
            "--regcbopt",
            "--random_seed 1",
            "--quiet"
        ]

        self.mocked.make_learner.assert_called_with(" ".join(expected_args))
    
    def test_regcb_elim_create_args(self):

        VowpalLearner(regcb="elim").predict(None, ['yes','no'])

        expected_args = [
            "--cb_explore_adf",
            "--regcb",
            "--interactions xxa",
            "--interactions xa",
            "--ignore_linear x",
            "--random_seed 1",
            "--quiet"
        ]

        self.mocked.make_learner.assert_called_with(" ".join(expected_args))


    def test_squarecb_all_create_args(self):

        VowpalLearner(squarecb="all").predict(None, ['yes','no'])

        expected_args = [
            "--cb_explore_adf",
            "--squarecb",
            "--gamma_scale 10",
            "--interactions xxa",
            "--interactions xa",
            "--ignore_linear x",
            "--random_seed 1",
            "--quiet"
        ]

        self.mocked.make_learner.assert_called_with(" ".join(expected_args))

    def test_squarecb_elim_create_args(self):

        VowpalLearner(squarecb="elim").predict(None, ['yes','no'])

        expected_args = [
            "--cb_explore_adf",
            "--squarecb",
            "--gamma_scale 10",
            "--interactions xxa",
            "--interactions xa",
            "--ignore_linear x",
            "--elim",
            "--random_seed 1",
            "--quiet"
        ]

        self.mocked.make_learner.assert_called_with(" ".join(expected_args))

    def test_params(self):

        learners = VowpalLearner(squarecb="all")
        learners.predict(None, ['yes','no'])

        expected_args = [
            "--cb_explore_adf",
            "--squarecb",
            "--gamma_scale 10",
            "--interactions xxa",
            "--interactions xa",
            "--ignore_linear x",
            "--random_seed 1"
        ]

        self.assertEqual(learners.params['family'], "vw")
        self.assertEqual(learners.params["args"], " ".join(expected_args))

    def test_adf_explicit_args(self):
        VowpalLearner("--cb_explore_adf --epsilon 0.75 --random_seed 20").predict(None, ['yes','no'])

        expected_args = [
            "--cb_explore_adf",
            "--epsilon 0.75",
            "--random_seed 20",
            "--quiet"
        ]

        self.mocked.make_learner.assert_called_with(" ".join(expected_args))

    def test_not_adf_explicit_args(self):
        VowpalLearner("--cb_explore 20 --epsilon 0.75 --random_seed 20").predict(None, ['yes','no'])

        expected_args = [
            "--cb_explore 2",
            "--epsilon 0.75",
            "--random_seed 20",
            "--quiet"
        ]

        self.mocked.make_learner.assert_called_with(" ".join(expected_args))

    def test_no_cb_explicit_args(self):
        with self.assertRaises(Exception) as e:
            VowpalLearner("--epsilon 0.75 --random_seed 20").predict(None, ['yes','no'])

        self.assertTrue("VowpalLearner was instantiated" in str(e.exception))

    def test_adf_predict_sans_context_str_actions(self):
        VowpalLearner(epsilon=0.05, adf=True, seed=20).predict(None, ['yes','no'])

        mock_learner = self.mocked.make_learner()

        self.assertEqual(2, len(mock_learner.predict.call_args[0][0]))

        self.assertEqual(mock_learner , mock_learner.predict.call_args[0][0][0].vw)
        self.assertEqual({'a':['yes']}, mock_learner.predict.call_args[0][0][0].ns)
        self.assertEqual(None         , mock_learner.predict.call_args[0][0][0].label)
        self.assertEqual(4            , mock_learner.predict.call_args[0][0][0].label_type)

        self.assertEqual(mock_learner , mock_learner.predict.call_args[0][0][1].vw)
        self.assertEqual({'a':['no']} , mock_learner.predict.call_args[0][0][1].ns)
        self.assertEqual(None         , mock_learner.predict.call_args[0][0][1].label)
        self.assertEqual(4            , mock_learner.predict.call_args[0][0][1].label_type)

        self.assertEqual(0, mock_learner.call_count)

    def test_adf_predict_sans_context_mixed_actions(self):
        VowpalLearner(epsilon=0.05, adf=True, seed=20).predict(None, [(1,'a'),(2,'b')])

        mock_learner = self.mocked.make_learner()

        self.assertEqual(2, len(mock_learner.predict.call_args[0][0]))

        self.assertEqual(mock_learner         , mock_learner.predict.call_args[0][0][0].vw)
        self.assertEqual({'a':[('0',1),'1=a']}, mock_learner.predict.call_args[0][0][0].ns)
        self.assertEqual(None                 , mock_learner.predict.call_args[0][0][0].label)
        self.assertEqual(4                    , mock_learner.predict.call_args[0][0][0].label_type)

        self.assertEqual(mock_learner         , mock_learner.predict.call_args[0][0][1].vw)
        self.assertEqual({'a':[('0',2),'1=b']}, mock_learner.predict.call_args[0][0][1].ns)
        self.assertEqual(None                 , mock_learner.predict.call_args[0][0][1].label)
        self.assertEqual(4                    , mock_learner.predict.call_args[0][0][1].label_type)

        self.assertEqual(0, mock_learner.call_count)

    def test_adf_predict_with_str_context_str_actions(self):
        VowpalLearner(epsilon=0.05, adf=True, seed=20).predict('b', ['yes','no'])

        mock_learner = self.mocked.make_learner()

        self.assertEqual(2, len(mock_learner.predict.call_args[0][0]))

        self.assertEqual(mock_learner           , mock_learner.predict.call_args[0][0][0].vw)
        self.assertEqual({'x':['b'],'a':['yes']}, mock_learner.predict.call_args[0][0][0].ns)
        self.assertEqual(None                   , mock_learner.predict.call_args[0][0][0].label)
        self.assertEqual(4                      , mock_learner.predict.call_args[0][0][0].label_type)

        self.assertEqual(mock_learner           , mock_learner.predict.call_args[0][0][1].vw)
        self.assertEqual({'x':['b'],'a':['no']} , mock_learner.predict.call_args[0][0][1].ns)
        self.assertEqual(None                   , mock_learner.predict.call_args[0][0][1].label)
        self.assertEqual(4                      , mock_learner.predict.call_args[0][0][1].label_type)

        self.assertEqual(0, mock_learner.call_count)

    def test_adf_predict_with_dict_context_str_actions(self):
        VowpalLearner(epsilon=0.05, adf=True, seed=20).predict({'c':2}, ['yes','no'])

        mock_learner = self.mocked.make_learner()

        self.assertEqual(2, len(mock_learner.predict.call_args[0][0]))

        self.assertEqual(mock_learner               , mock_learner.predict.call_args[0][0][0].vw)
        self.assertEqual({'x':[('c',2)],'a':['yes']}, mock_learner.predict.call_args[0][0][0].ns)
        self.assertEqual(None                       , mock_learner.predict.call_args[0][0][0].label)
        self.assertEqual(4                          , mock_learner.predict.call_args[0][0][0].label_type)

        self.assertEqual(mock_learner              , mock_learner.predict.call_args[0][0][1].vw)
        self.assertEqual({'x':[('c',2)],'a':['no']}, mock_learner.predict.call_args[0][0][1].ns)
        self.assertEqual(None                      , mock_learner.predict.call_args[0][0][1].label)
        self.assertEqual(4                         , mock_learner.predict.call_args[0][0][1].label_type)

        self.assertEqual(0, mock_learner.call_count)

    def test_no_adf_predict_sans_context_str_actions(self):
        VowpalLearner(bag=2, adf=False, seed=20).predict(None, ['yes','no'])

        mock_learner = self.mocked.make_learner()

        self.assertEqual(mock_learner , mock_learner.predict.call_args[0][0].vw)
        self.assertEqual({}           , mock_learner.predict.call_args[0][0].ns)
        self.assertEqual(None         , mock_learner.predict.call_args[0][0].label)
        self.assertEqual(4            , mock_learner.predict.call_args[0][0].label_type)

        self.assertEqual(0, mock_learner.call_count)

    def test_no_adf_predict_with_str_context_str_actions(self):
        VowpalLearner(bag=2, adf=False, seed=20).predict('b', ['yes','no'])

        mock_learner = self.mocked.make_learner()

        self.assertEqual(mock_learner, mock_learner.predict.call_args[0][0].vw)
        self.assertEqual({'x':['b']} , mock_learner.predict.call_args[0][0].ns)
        self.assertEqual(None        , mock_learner.predict.call_args[0][0].label)
        self.assertEqual(4           , mock_learner.predict.call_args[0][0].label_type)

        self.assertEqual(0, mock_learner.call_count)

    def test_adf_learn_sans_context_str_actions(self):
        learner = VowpalLearner(epsilon=0.05, adf=True, seed=20)
        learner.predict(None, ['yes','no'])
        learner.learn(None, 'yes', 1, 0.2, ['yes','no'])

        mock_learner = self.mocked.make_learner()

        self.assertEqual(2, len(mock_learner.learn.call_args[0][0]))

        self.assertEqual(mock_learner , mock_learner.learn.call_args[0][0][0].vw)
        self.assertEqual({'a':['yes']}, mock_learner.learn.call_args[0][0][0].ns)
        self.assertEqual("1:0:0.2"    , mock_learner.learn.call_args[0][0][0].label)
        self.assertEqual(4            , mock_learner.learn.call_args[0][0][0].label_type)

        self.assertEqual(mock_learner , mock_learner.learn.call_args[0][0][1].vw)
        self.assertEqual({'a':['no']} , mock_learner.learn.call_args[0][0][1].ns)
        self.assertEqual(None         , mock_learner.learn.call_args[0][0][1].label)
        self.assertEqual(4            , mock_learner.learn.call_args[0][0][1].label_type)

    def test_adf_learn_with_str_context_str_actions(self):
        learner = VowpalLearner(epsilon=0.05, adf=True, seed=20)
        learner.predict('b', ['yes','no'])
        learner.learn('b', 'no', .5, 0.2, ['yes','no'])

        mock_learner = self.mocked.make_learner()

        self.assertEqual(2, len(mock_learner.learn.call_args[0][0]))

        self.assertEqual(mock_learner           , mock_learner.learn.call_args[0][0][0].vw)
        self.assertEqual({'x':['b'],'a':['yes']}, mock_learner.learn.call_args[0][0][0].ns)
        self.assertEqual(None                   , mock_learner.learn.call_args[0][0][0].label)
        self.assertEqual(4                      , mock_learner.learn.call_args[0][0][0].label_type)

        self.assertEqual(mock_learner           , mock_learner.learn.call_args[0][0][1].vw)
        self.assertEqual({'x':['b'],'a':['no']} , mock_learner.learn.call_args[0][0][1].ns)
        self.assertEqual("2:0.5:0.2"            , mock_learner.learn.call_args[0][0][1].label)
        self.assertEqual(4                      , mock_learner.learn.call_args[0][0][1].label_type)

    def test_adf_learn_with_dict_context_str_actions(self):
        learner = VowpalLearner(epsilon=0.05, adf=True, seed=20)
        learner.predict({'c':2}, ['yes','no'])
        learner.learn({'c':2}, 'no', .5, 0.2, ['yes','no'])

        mock_learner = self.mocked.make_learner()

        self.assertEqual(2, len(mock_learner.learn.call_args[0][0]))

        self.assertEqual(mock_learner               , mock_learner.learn.call_args[0][0][0].vw)
        self.assertEqual({'x':[('c',2)],'a':['yes']}, mock_learner.learn.call_args[0][0][0].ns)
        self.assertEqual(None                       , mock_learner.learn.call_args[0][0][0].label)
        self.assertEqual(4                          , mock_learner.learn.call_args[0][0][0].label_type)

        self.assertEqual(mock_learner              , mock_learner.learn.call_args[0][0][1].vw)
        self.assertEqual({'x':[('c',2)],'a':['no']}, mock_learner.learn.call_args[0][0][1].ns)
        self.assertEqual("2:0.5:0.2"               , mock_learner.learn.call_args[0][0][1].label)
        self.assertEqual(4                         , mock_learner.learn.call_args[0][0][1].label_type)

    def test_adf_learn_with_dict_context_str_actions2(self):
        learner = VowpalLearner(epsilon=0.05, adf=True, seed=20)
        learner.predict({1:(0,1)}, ['yes','no'])
        learner.learn({1:(0,1)}, 'no', .5, 0.2, ['yes','no'])

        mock_learner = self.mocked.make_learner()

        self.assertEqual(2, len(mock_learner.learn.call_args[0][0]))

        self.assertEqual(mock_learner                 , mock_learner.learn.call_args[0][0][0].vw)
        self.assertEqual({'x':[('1_1',1)],'a':['yes']}, mock_learner.learn.call_args[0][0][0].ns)
        self.assertEqual(None                         , mock_learner.learn.call_args[0][0][0].label)
        self.assertEqual(4                            , mock_learner.learn.call_args[0][0][0].label_type)

        self.assertEqual(mock_learner                , mock_learner.learn.call_args[0][0][1].vw)
        self.assertEqual({'x':[('1_1',1)],'a':['no']}, mock_learner.learn.call_args[0][0][1].ns)
        self.assertEqual("2:0.5:0.2"                 , mock_learner.learn.call_args[0][0][1].label)
        self.assertEqual(4                           , mock_learner.learn.call_args[0][0][1].label_type)

    def test_adf_learn_with_mixed_dense_context_str_actions(self):
        learner = VowpalLearner(epsilon=0.05, adf=True, seed=20)
        learner.predict([1,'a',(0,1)], ['yes','no'])
        learner.learn([1,'a',(0,1)], 'no', .5, 0.2, ['yes','no'])

        mock_learner = self.mocked.make_learner()

        self.assertEqual(2, len(mock_learner.learn.call_args[0][0]))

        self.assertEqual(mock_learner                             , mock_learner.learn.call_args[0][0][0].vw)
        self.assertEqual({'x':[('0',1),'1=a',("3",1)],'a':['yes']}, mock_learner.learn.call_args[0][0][0].ns)
        self.assertEqual(None                                     , mock_learner.learn.call_args[0][0][0].label)
        self.assertEqual(4                                        , mock_learner.learn.call_args[0][0][0].label_type)

        self.assertEqual(mock_learner                            , mock_learner.learn.call_args[0][0][1].vw)
        self.assertEqual({'x':[('0',1),'1=a',("3",1)],'a':['no']}, mock_learner.learn.call_args[0][0][1].ns)
        self.assertEqual("2:0.5:0.2"                             , mock_learner.learn.call_args[0][0][1].label)
        self.assertEqual(4                                       , mock_learner.learn.call_args[0][0][1].label_type)

    def test_adf_learn_with_no_context_mixed_dense_actions(self):
        learner = VowpalLearner(epsilon=0.05, adf=True, seed=20)
        learner.predict(None, [(1,'a'),(2,'b')])
        learner.learn(None, (2,'b'), .5, 0.2, [(1,'a'),(2,'b')])

        mock_learner = self.mocked.make_learner()

        self.assertEqual(2, len(mock_learner.learn.call_args[0][0]))

        self.assertEqual(mock_learner         , mock_learner.learn.call_args[0][0][0].vw)
        self.assertEqual({'a':[('0',1),'1=a']}, mock_learner.learn.call_args[0][0][0].ns)
        self.assertEqual(None                 , mock_learner.learn.call_args[0][0][0].label)
        self.assertEqual(4                    , mock_learner.learn.call_args[0][0][0].label_type)

        self.assertEqual(mock_learner         , mock_learner.learn.call_args[0][0][1].vw)
        self.assertEqual({'a':[('0',2),'1=b']}, mock_learner.learn.call_args[0][0][1].ns)
        self.assertEqual("2:0.5:0.2"          , mock_learner.learn.call_args[0][0][1].label)
        self.assertEqual(4                    , mock_learner.learn.call_args[0][0][1].label_type)

    def test_no_adf_learn_sans_context_str_actions(self):
        learner = VowpalLearner(bag=2, adf=False, seed=20)
        learner.predict(None, ['yes','no'])
        learner.learn(None, 'no', .5, 0.2, ['yes','no'])

        mock_learner = self.mocked.make_learner()

        self.assertEqual(mock_learner , mock_learner.learn.call_args[0][0].vw)
        self.assertEqual({}           , mock_learner.learn.call_args[0][0].ns)
        self.assertEqual("2:0.5:0.2"  , mock_learner.learn.call_args[0][0].label)
        self.assertEqual(4            , mock_learner.learn.call_args[0][0].label_type)

    def test_no_adf_learn_with_str_context_str_actions(self):
        learner = VowpalLearner(bag=2, adf=False, seed=20)
        learner.predict('b', ['yes','no'])
        learner.learn('b', 'yes', .25, 0.2, ['yes','no'])

        mock_learner = self.mocked.make_learner()

        self.assertEqual(mock_learner , mock_learner.learn.call_args[0][0].vw)
        self.assertEqual({'x':['b']}  , mock_learner.learn.call_args[0][0].ns)
        self.assertEqual("1:0.75:0.2" , mock_learner.learn.call_args[0][0].label)
        self.assertEqual(4            , mock_learner.learn.call_args[0][0].label_type)

    def test_predict_epsilon_not_adf_args_error_1(self):
        learner = VowpalLearner("--cb_explore --epsilon 0.75 --random_seed 20 --quiet")
        learner.predict(None, [1,2,3,4])

        with self.assertRaises(Exception) as e:
            learner.predict(None, [1,2,3,4,5])

        self.assertTrue("--cb_explore_adf" in str(e.exception))

    def test_predict_epsilon_not_adf_args_error_2(self):
        learner = VowpalLearner("--cb_explore --epsilon 0.75 --random_seed 20 --quiet")
        learner.predict(None, [1,2,3,4])

        with self.assertRaises(Exception) as e:
            learner.predict(None, [1,2,3])

        self.assertTrue("--cb_explore_adf" in str(e.exception))

class VowpalMediator_Tests(unittest.TestCase):

    def test_empty(self):
        self.assertEqual([],VowpalMediator.prep_features(None))
        self.assertEqual([],VowpalMediator.prep_features([]))
        self.assertEqual([],VowpalMediator.prep_features(()))

    def test_string(self):
        actual   = VowpalMediator.prep_features('a')
        expected = [ 'a' ]
        self.assertEqual(actual, expected)

    def test_numeric(self):
        actual   = VowpalMediator.prep_features(2)
        expected = [ ('0',2) ]
        self.assertEqual(actual, expected)
        self.assertIsInstance(actual[0][1], float)

    def test_dense_numeric_sequence(self):
        actual   = VowpalMediator.prep_features((1,2,3))
        expected = [ ('0',1), ('1',2), ('2',3) ]
        
        self.assertEqual(actual, expected)
        self.assertIsInstance(actual[0][1], float)
        self.assertIsInstance(actual[1][1], float)
        self.assertIsInstance(actual[2][1], float)

    def test_dense_string_sequence(self):
        actual   = VowpalMediator.prep_features((1,'a',3))
        expected = [ ('0',1), '1=a', ('2',3) ]
        
        self.assertEqual(actual, expected)
        self.assertIsInstance(actual[0][1], float)
        self.assertIsInstance(actual[2][1], float)

    def test_sparse_dict_numeric_key_numeric_value(self):
        actual   = VowpalMediator.prep_features({1:1,2:2})
        expected = [ (1,1), (2,2) ]
        self.assertEqual(actual, expected)
        self.assertIsInstance(actual[0][1], float)
        self.assertIsInstance(actual[1][1], float)

    def test_sparse_dict_string_key_numeric_value(self):
        actual   = VowpalMediator.prep_features({'a':1,'b':2})
        expected = [ ('a',1), ('b',2) ]
        self.assertEqual(actual, expected)
        self.assertIsInstance(actual[0][1], float)
        self.assertIsInstance(actual[1][1], float)

    def test_sparse_dict_numeric_key_string_value(self):
        actual   = VowpalMediator.prep_features({1:'a',2:'b'})
        expected = [ '1=a', '2=b' ]
        self.assertEqual(actual, expected)

    def test_sparse_dict_string_key_string_value(self):
        actual   = VowpalMediator.prep_features({'c':'a','d':'b'})
        expected = [ 'c=a', 'd=b' ]
        self.assertEqual(actual, expected)

    def test_sparse_tuple(self):
        actual   = VowpalMediator.prep_features([('a',1),('b',2)])
        expected = [ ('a',1), ('b',2) ]
        self.assertEqual(actual, expected)
        self.assertIsInstance(actual[0][1], float)
        self.assertIsInstance(actual[1][1], float)

    def test_sparse_tuple_with_str_val(self):
        actual   = VowpalMediator.prep_features([('a','c'),('b','d')])
        expected = [ 'a=c', 'b=d' ]
        self.assertEqual(actual, expected)

    def test_sparse_tuple_with_float_val(self):
        actual   = VowpalMediator.prep_features([(1.,1.23),(2.,2.23)])
        expected = [ (1.,1.23), (2.,2.23) ]

        #this return is incorrect for VW because VW requires the first value to be a string or int
        #unfortunately checking for this makes the code considerably slower so we don't look for it
        self.assertEqual(actual, expected)
        self.assertIsInstance(actual[0][0],float)
        self.assertIsInstance(actual[1][0],float)

    def test_bad_features(self):
        with self.assertRaises(CobaException):
            VowpalMediator.prep_features(object())

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW not installed")
    def test_make_learner(self):
        vw = VowpalMediator.make_learner("--cb_explore_adf --epsilon .1 --quiet")

        self.assertIn("--cb_explore_adf", vw.get_arguments())
        self.assertIn("--epsilon", vw.get_arguments())

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW not installed")
    def test_make_example(self):
        
        vw = VowpalMediator.make_learner("--cb_explore_adf --epsilon .1 --noconstant --quiet")
        ex = VowpalMediator.make_example(vw, {'a': [('0',1.)]}, None, 4)

        self.assertEqual(1, ex.get_feature_number())

        vw = VowpalMediator.make_learner("--cb_explore_adf --epsilon .1 --quiet")
        ex = VowpalMediator.make_example(vw, {'a': [('0',1.)]}, None, 4)

        self.assertEqual(2, ex.get_feature_number())

        vw = VowpalMediator.make_learner("--cb_explore_adf --epsilon .1 --interactions aa --interactions aaa --quiet")
        ex = VowpalMediator.make_example(vw, {'a': [('0',1)]}, None, 4)

        self.assertEqual(2, ex.get_feature_number()) #for some reason interactions aren't counted until predict
        P = vw.predict([ex])
        self.assertEqual(4, ex.get_feature_number()) #for some reason interactions aren't counted until predict

        #providing integer keys doesn't seem to impact VW performance 
        vw = VowpalMediator.make_learner("--cb_explore_adf --epsilon .1 --interactions xa --quiet")
        ex = VowpalMediator.make_example(vw, {'x': [(1,1),(2,1),(3,1)], 'a': [(1,2),(2,2),(3,2)]}, "2:0:1", 4)

        self.assertEqual(7, ex.get_feature_number()) #for some reason interactions aren't counted until predict
        P = vw.predict([ex])
        self.assertEqual(16, ex.get_feature_number()) #for some reason interactions aren't counted until predict

        from vowpalwabbit.pyvw import cbandits_label

        label = cbandits_label(ex)
        self.assertEqual(2, label.costs[0].action)
        self.assertEqual(0, label.costs[0].cost)
        self.assertEqual(1, label.costs[0].probability)

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW not installed")
    def test_get_version(self):
        VowpalMediator.get_version()

if __name__ == '__main__':
    unittest.main()