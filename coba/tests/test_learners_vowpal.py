import importlib.util
import unittest
import unittest.mock

from coba.utilities import KeyDefaultDict
from coba.exceptions import CobaException
from coba.learners import VowpalLearner, VowpalMediator
from coba.learners import (
    VowpalEpsilonLearner, VowpalSoftmaxLearner, VowpalBagLearner, 
    VowpalCoverLearner, VowpalRegcbLearner, VowpalSquarecbLearner,
    VowpalOffPolicyLearner
)

class VowpalMockExample:
    
    def __init__(self,vw,ns,label,label_type):
        self.vw = vw
        self.ns = ns
        self.label = label
        self.label_type = label_type

class VowpalEpsilonLearner_Tests(unittest.TestCase):
    
    @unittest.mock.patch('coba.learners.vowpal.VowpalLearner.__init__')
    def test_defaults(self, mock) -> None:
        VowpalEpsilonLearner()
        mock.assert_called_once_with("--cb_explore_adf --epsilon 0.05 --interactions xxa --interactions xa --ignore_linear x --random_seed 1")

    @unittest.mock.patch('coba.learners.vowpal.VowpalLearner.__init__')
    def test_specifics(self, mock) -> None:
        VowpalEpsilonLearner(epsilon=0.1, interactions=["xa"], ignore_linear=[], seed=None)
        mock.assert_called_once_with("--cb_explore_adf --epsilon 0.1 --interactions xa")

class VowpalSoftmaxLearner_Tests(unittest.TestCase):
    
    @unittest.mock.patch('coba.learners.vowpal.VowpalLearner.__init__')
    def test_defaults(self, mock) -> None:
        VowpalSoftmaxLearner()
        mock.assert_called_once_with("--cb_explore_adf --softmax --lambda 10 --interactions xxa --interactions xa --ignore_linear x --random_seed 1")

    @unittest.mock.patch('coba.learners.vowpal.VowpalLearner.__init__')
    def test_specifics(self, mock) -> None:
        VowpalSoftmaxLearner(softmax=5, interactions=["xa"], ignore_linear=[], seed=None)
        mock.assert_called_once_with("--cb_explore_adf --softmax --lambda 5 --interactions xa")

class VowpalSoftmaxLearner_Tests(unittest.TestCase):

    @unittest.mock.patch('coba.learners.vowpal.VowpalLearner.__init__')
    def test_defaults(self, mock) -> None:
        VowpalSoftmaxLearner()
        mock.assert_called_once_with("--cb_explore_adf --softmax --lambda 10 --interactions xxa --interactions xa --ignore_linear x --random_seed 1")

    @unittest.mock.patch('coba.learners.vowpal.VowpalLearner.__init__')
    def test_specifics(self, mock) -> None:
        VowpalSoftmaxLearner(softmax=5, interactions=["xa"], ignore_linear=[], seed=None)
        mock.assert_called_once_with("--cb_explore_adf --softmax --lambda 5 --interactions xa")

class VowpalBagLearner_Tests(unittest.TestCase):

    @unittest.mock.patch('coba.learners.vowpal.VowpalLearner.__init__')
    def test_defaults(self, mock) -> None:
        VowpalBagLearner()
        mock.assert_called_once_with("--cb_explore_adf --bag 5 --interactions xxa --interactions xa --ignore_linear x --random_seed 1")

    @unittest.mock.patch('coba.learners.vowpal.VowpalLearner.__init__')
    def test_specifics(self, mock) -> None:
        VowpalBagLearner(bag=10, interactions=["xa"], ignore_linear=[], seed=None)
        mock.assert_called_once_with("--cb_explore_adf --bag 10 --interactions xa")

class VowpalCoverLearner_Tests(unittest.TestCase):

    @unittest.mock.patch('coba.learners.vowpal.VowpalLearner.__init__')
    def test_defaults(self, mock) -> None:
        VowpalCoverLearner()
        mock.assert_called_once_with("--cb_explore_adf --cover 5 --interactions xxa --interactions xa --ignore_linear x --random_seed 1")

    @unittest.mock.patch('coba.learners.vowpal.VowpalLearner.__init__')
    def test_specifics(self, mock) -> None:
        VowpalCoverLearner(cover=10, interactions=["xa"], ignore_linear=[], seed=None)
        mock.assert_called_once_with("--cb_explore_adf --cover 10 --interactions xa")

class VowpalRegcbLearner_Tests(unittest.TestCase):

    @unittest.mock.patch('coba.learners.vowpal.VowpalLearner.__init__')
    def test_defaults(self, mock) -> None:
        VowpalRegcbLearner()
        mock.assert_called_once_with("--cb_explore_adf --regcb --interactions xxa --interactions xa --ignore_linear x --random_seed 1")

    @unittest.mock.patch('coba.learners.vowpal.VowpalLearner.__init__')
    def test_specifics(self, mock) -> None:
        VowpalRegcbLearner(mode="optimistic", interactions=["xa"], ignore_linear=[], seed=None)
        mock.assert_called_once_with("--cb_explore_adf --regcbopt --interactions xa")

class VowpalSquarecbLearner_Tests(unittest.TestCase):

    @unittest.mock.patch('coba.learners.vowpal.VowpalLearner.__init__')
    def test_defaults(self, mock) -> None:
        VowpalSquarecbLearner()
        mock.assert_called_once_with("--cb_explore_adf --squarecb --gamma_scale 10 --interactions xxa --interactions xa --ignore_linear x --random_seed 1")

    @unittest.mock.patch('coba.learners.vowpal.VowpalLearner.__init__')
    def test_specifics(self, mock) -> None:
        VowpalSquarecbLearner(mode="elimination", gamma_scale=5, interactions=["xa"], ignore_linear=[], seed=None)
        mock.assert_called_once_with("--cb_explore_adf --squarecb --gamma_scale 5 --elim --interactions xa")

class VowpalOffpolicyLearner_Tests(unittest.TestCase):

    @unittest.mock.patch('coba.learners.vowpal.VowpalLearner.__init__')
    def test_defaults(self, mock) -> None:
        VowpalOffPolicyLearner()
        mock.assert_called_once_with("--cb_adf --interactions xxa --interactions xa --ignore_linear x --random_seed 1")

    @unittest.mock.patch('coba.learners.vowpal.VowpalLearner.__init__')
    def test_specifics(self, mock) -> None:
        VowpalOffPolicyLearner(interactions=["xa"], ignore_linear=[], seed=None)
        mock.assert_called_once_with("--cb_adf --interactions xa")

class VowpalLearner_Tests(unittest.TestCase):

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
        self.module.stop()
        self.mediat.stop()

    def test_params(self):

        learners = VowpalLearner()

        expected_args = [
            "--cb_explore_adf",
            "--epsilon 0.05",
            "--interactions xxa",
            "--interactions xa",
            "--ignore_linear x",
            "--random_seed 1",
        ]

        self.assertEqual(learners.params['family'], "vw")
        self.assertEqual(learners.params["args"], " ".join(expected_args))

    def test_init_bad(self):
        with self.assertRaises(CobaException):
            VowpalLearner('--epsilon .1')

    def test_init_default(self):

        VowpalLearner().predict(None, ['yes','no'])

        expected_args = [
            "--cb_explore_adf",
            "--epsilon 0.05",
            "--interactions xxa",
            "--interactions xa",
            "--ignore_linear x",
            "--random_seed 1",
            "--quiet"
        ]

        self.mocked.make_learner.assert_called_with(" ".join(expected_args))

    def test_init_cb_explore_adf(self):
        VowpalLearner("--cb_explore_adf --epsilon 0.75 --random_seed 20").predict(None, ['yes','no'])

        expected_args = [
            "--cb_explore_adf",
            "--epsilon 0.75",
            "--random_seed 20",
            "--quiet"
        ]

        self.mocked.make_learner.assert_called_with(" ".join(expected_args))

    def test_init_cb_adf(self):
        
        self.mocked.make_learner().predict.return_value = [.25, .75]

        p = VowpalLearner("--cb_adf --random_seed 20").predict(None, ['yes','no'])[0]

        expected_args = [
            "--cb_adf",
            "--random_seed 20",
            "--quiet"
        ]

        self.assertEqual([1,0], p)
        self.mocked.make_learner.assert_called_with(" ".join(expected_args))

    def test_init_cb_explore(self):
        VowpalLearner("--cb_explore 20 --epsilon 0.75 --random_seed 20").predict(None, ['yes','no'])

        expected_args = [
            "--cb_explore 2",
            "--epsilon 0.75",
            "--random_seed 20",
            "--quiet"
        ]

        self.mocked.make_learner.assert_called_with(" ".join(expected_args))

    def test_init_cb(self):

        self.mocked.make_learner().predict.return_value = 1
        p = VowpalLearner("--cb 20 --epsilon 0.75 --random_seed 20").predict(None, ['yes','no'])[0]

        expected_args = [
            "--cb 2",
            "--epsilon 0.75",
            "--random_seed 20",
            "--quiet"
        ]

        self.assertEqual([1,0], p)
        self.mocked.make_learner.assert_called_with(" ".join(expected_args))

    def test_predict_adf_sans_context_str_actions(self):
        VowpalLearner("--cb_explore_adf").predict(None, ['yes','no'])

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

    def test_predict_adf_sans_context_mixed_actions(self):
        VowpalLearner("--cb_explore_adf").predict(None, [(1,'a'),(2,'b')])

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

    def test_predict_adf_with_str_context_str_actions(self):
        VowpalLearner("--cb_explore_adf").predict('b', ['yes','no'])

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

    def test_predict_adf_with_dict_context_str_actions(self):
        VowpalLearner("--cb_explore_adf").predict({'c':2}, ['yes','no'])

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

    def test_learn_adf_sans_context_str_actions(self):
        learner = VowpalLearner("--cb_explore_adf")
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

    def test_learn_adf_with_str_context_str_actions(self):
        learner = VowpalLearner("--cb_explore_adf")
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

    def test_learn_adf_with_dict_context_str_actions(self):
        learner = VowpalLearner("--cb_explore_adf")
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

    def test_learn_adf_with_dict_context_str_actions2(self):
        learner = VowpalLearner("--cb_explore_adf")
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

    def test_learn_adf_with_mixed_dense_context_str_actions(self):
        learner = VowpalLearner("--cb_explore_adf")
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

    def test_learn_adf_with_no_context_mixed_dense_actions(self):
        learner = VowpalLearner("--cb_explore_adf")
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

    def test_learn_no_adf_sans_context_str_actions(self):
        learner = VowpalLearner("--cb_explore")
        learner.predict(None, ['yes','no'])
        learner.learn(None, 'no', .5, 0.2, ['yes','no'])

        mock_learner = self.mocked.make_learner()

        self.assertEqual(mock_learner , mock_learner.learn.call_args[0][0].vw)
        self.assertEqual({}           , mock_learner.learn.call_args[0][0].ns)
        self.assertEqual("2:0.5:0.2"  , mock_learner.learn.call_args[0][0].label)
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

    def test_make_args(self):
        args = VowpalMediator.make_args(["--cb_explore_adf"], ["xxa", "xa"], ["x"], 1, bits=20, b=20, c=None)

        expected_options = [
            "--cb_explore_adf",
            "--interactions xxa",
            "--interactions xa",
            "--ignore_linear x",
            "--random_seed 1",
            "--bits 20",
            "-b 20",
            "-c"
        ]

        self.assertEqual(" ".join(expected_options), args)

    def test_make_args_no_seed(self):
        args = VowpalMediator.make_args(["--cb_explore_adf"], ["xxa", "xa"], ["x"], None, bits=20, b=20, c=None)

        expected_options = [
            "--cb_explore_adf",
            "--interactions xxa",
            "--interactions xa",
            "--ignore_linear x",
            "--bits 20",
            "-b 20",
            "-c"
        ]

        self.assertEqual(" ".join(expected_options), args)

    def test_prep_features_empty(self):
        self.assertEqual([],VowpalMediator.prep_features(None))
        self.assertEqual([],VowpalMediator.prep_features([]))
        self.assertEqual([],VowpalMediator.prep_features(()))

    def test_prep_features_string(self):
        actual   = VowpalMediator.prep_features('a')
        expected = [ 'a' ]
        self.assertEqual(actual, expected)

    def test_prep_features_numeric(self):
        actual   = VowpalMediator.prep_features(2)
        expected = [ ('0',2) ]
        self.assertEqual(actual, expected)
        self.assertIsInstance(actual[0][1], float)

    def test_prep_features_dense_numeric_sequence(self):
        actual   = VowpalMediator.prep_features((1,2,3))
        expected = [ ('0',1), ('1',2), ('2',3) ]
        
        self.assertEqual(actual, expected)
        self.assertIsInstance(actual[0][1], float)
        self.assertIsInstance(actual[1][1], float)
        self.assertIsInstance(actual[2][1], float)

    def test_prep_features_dense_string_sequence(self):
        actual   = VowpalMediator.prep_features((1,'a',3))
        expected = [ ('0',1), '1=a', ('2',3) ]
        
        self.assertEqual(actual, expected)
        self.assertIsInstance(actual[0][1], float)
        self.assertIsInstance(actual[2][1], float)

    def test_prep_features_sparse_dict_numeric_key_numeric_value(self):
        actual   = VowpalMediator.prep_features({1:1,2:2})
        expected = [ (1,1), (2,2) ]
        self.assertEqual(actual, expected)
        self.assertIsInstance(actual[0][1], float)
        self.assertIsInstance(actual[1][1], float)

    def test_prep_features_sparse_dict_string_key_numeric_value(self):
        actual   = VowpalMediator.prep_features({'a':1,'b':2})
        expected = [ ('a',1), ('b',2) ]
        self.assertEqual(actual, expected)
        self.assertIsInstance(actual[0][1], float)
        self.assertIsInstance(actual[1][1], float)

    def test_prep_features_sparse_dict_numeric_key_string_value(self):
        actual   = VowpalMediator.prep_features({1:'a',2:'b'})
        expected = [ '1=a', '2=b' ]
        self.assertEqual(actual, expected)

    def test_prep_features_sparse_dict_string_key_string_value(self):
        actual   = VowpalMediator.prep_features({'c':'a','d':'b'})
        expected = [ 'c=a', 'd=b' ]
        self.assertEqual(actual, expected)

    def test_prep_features_sparse_tuple(self):
        actual   = VowpalMediator.prep_features([('a',1),('b',2)])
        expected = [ ('a',1), ('b',2) ]
        self.assertEqual(actual, expected)
        self.assertIsInstance(actual[0][1], float)
        self.assertIsInstance(actual[1][1], float)

    def test_prep_features_sparse_tuple_with_str_val(self):
        actual   = VowpalMediator.prep_features([('a','c'),('b','d')])
        expected = [ 'a=c', 'b=d' ]
        self.assertEqual(actual, expected)

    def test_prep_features_sparse_tuple_with_float_val(self):
        actual   = VowpalMediator.prep_features([(1.,1.23),(2.,2.23)])
        expected = [ (1.,1.23), (2.,2.23) ]

        #this return is incorrect for VW because VW requires the first value to be a string or int
        #unfortunately checking for this makes the code considerably slower so we don't look for it
        self.assertEqual(actual, expected)
        self.assertIsInstance(actual[0][0],float)
        self.assertIsInstance(actual[1][0],float)

    def test_prep_features_bad_features(self):
        with self.assertRaises(CobaException):
            VowpalMediator.prep_features(object())

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW is not installed")
    def test_make_learner(self):
        vw = VowpalMediator.make_learner("--cb_explore_adf --epsilon .1 --quiet")

        self.assertIn("--cb_explore_adf", vw.get_arguments())
        self.assertIn("--epsilon", vw.get_arguments())

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW is not installed")
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

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW is not installed")
    def test_get_version(self):
        VowpalMediator.get_version()

if __name__ == '__main__':
    unittest.main()