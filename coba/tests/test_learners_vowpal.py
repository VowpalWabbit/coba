import unittest
import unittest.mock
import importlib.util
import pickle

from typing import Sequence, cast

from coba.random import CobaRandom
from coba.exceptions import CobaException
from coba.learners import VowpalMediator
from coba.learners import (
    VowpalArgsLearner, VowpalEpsilonLearner, VowpalSoftmaxLearner,
    VowpalBagLearner, VowpalCoverLearner, VowpalRegcbLearner,
    VowpalSquarecbLearner, VowpalOffPolicyLearner
)

class VowpalInherited(VowpalArgsLearner):

    @property
    def params(self):
        return {"Shadow":True}

class VowpalEaxmpleMock:

    def __init__(self,ns,label):
        self.ns         = ns
        self.label      = label

class VowpalMediatorMocked:

    def __init__(self, predict_returns = None) -> None:
        self._init_learner_calls  = []
        self._predict_calls       = []
        self._learn_calls         = []
        self._make_example_calls  = []
        self._make_examples_calls = []
        self._predict_returns     = predict_returns

    @property
    def is_initialized(self):
        return len(self._init_learner_calls) > 0

    def init_learner(self, args:str, label_type: int):
        self._init_learner_calls.append((args, label_type))
        return self

    def predict(self, example):
        self._predict_calls.append(example)
        return self._predict_returns

    def learn(self, example):
        self._learn_calls.append(example)

    def make_example(self, namespaces, label):
        return VowpalEaxmpleMock(namespaces,label)

    def make_examples(self, shared, distincts, labels):
        if labels is None: labels = [None]*len(distincts)
        return [ VowpalEaxmpleMock((shared,d),l) for d,l in zip(distincts,labels)]

class VowpalEpsilonLearner_Tests(unittest.TestCase):

    @unittest.mock.patch('coba.learners.vowpal.VowpalArgsLearner.__init__')
    def test_defaults(self, mock) -> None:
        VowpalEpsilonLearner()
        mock.assert_called_once_with("--cb_explore_adf --epsilon 0.05 --interactions ax --interactions axx --ignore_linear x --random_seed 1 --quiet",None)

    @unittest.mock.patch('coba.learners.vowpal.VowpalArgsLearner.__init__')
    def test_specifics(self, mock) -> None:
        VowpalEpsilonLearner(epsilon=0.1, features = ['a','x','ax'], seed=None)
        mock.assert_called_once_with("--cb_explore_adf --epsilon 0.1 --noconstant --interactions ax --quiet",None)

    @unittest.mock.patch('coba.learners.vowpal.VowpalArgsLearner.__init__')
    def test_custom_flag(self, mock) -> None:
        VowpalEpsilonLearner(epsilon=0.1, features = ['a','x','ax'], seed=None, b=20)
        mock.assert_called_once_with("--cb_explore_adf --epsilon 0.1 --noconstant --interactions ax -b 20 --quiet",None)
    
    def test_pickle(self) -> None:
        self.assertIsInstance(pickle.loads(pickle.dumps(VowpalEpsilonLearner(vw=VowpalMediatorMocked()))), VowpalEpsilonLearner)

class VowpalSoftmaxLearner_Tests(unittest.TestCase):

    @unittest.mock.patch('coba.learners.vowpal.VowpalArgsLearner.__init__')
    def test_defaults(self, mock) -> None:
        VowpalSoftmaxLearner()
        mock.assert_called_once_with("--cb_explore_adf --softmax --lambda 10 --interactions ax --interactions axx --ignore_linear x --random_seed 1 --quiet",None)

    @unittest.mock.patch('coba.learners.vowpal.VowpalArgsLearner.__init__')
    def test_specifics(self, mock) -> None:
        VowpalSoftmaxLearner(softmax=5, features=[1, "x", "a", "ax"], seed=None)
        mock.assert_called_once_with("--cb_explore_adf --softmax --lambda 5 --interactions ax --quiet",None)

    def test_pickle(self) -> None:
        self.assertIsInstance(pickle.loads(pickle.dumps(VowpalSoftmaxLearner(vw=VowpalMediatorMocked()))), VowpalSoftmaxLearner)

class VowpalBagLearner_Tests(unittest.TestCase):

    @unittest.mock.patch('coba.learners.vowpal.VowpalArgsLearner.__init__')
    def test_defaults(self, mock) -> None:
        VowpalBagLearner()
        mock.assert_called_once_with("--cb_explore_adf --bag 5 --interactions ax --interactions axx --ignore_linear x --random_seed 1 --quiet",None)

    @unittest.mock.patch('coba.learners.vowpal.VowpalArgsLearner.__init__')
    def test_specifics(self, mock) -> None:
        VowpalBagLearner(bag=10, features=['x', 'a', "ax"], seed=None)
        mock.assert_called_once_with("--cb_explore_adf --bag 10 --noconstant --interactions ax --quiet",None)

    def test_pickle(self) -> None:
        self.assertIsInstance(pickle.loads(pickle.dumps(VowpalBagLearner(vw=VowpalMediatorMocked()))), VowpalBagLearner)

class VowpalCoverLearner_Tests(unittest.TestCase):

    @unittest.mock.patch('coba.learners.vowpal.VowpalArgsLearner.__init__')
    def test_defaults(self, mock) -> None:
        VowpalCoverLearner()
        mock.assert_called_once_with("--cb_explore_adf --cover 5 --interactions ax --interactions axx --ignore_linear x --random_seed 1 --quiet",None)

    @unittest.mock.patch('coba.learners.vowpal.VowpalArgsLearner.__init__')
    def test_specifics(self, mock) -> None:
        VowpalCoverLearner(cover=10, features = ['a','x','ax'], seed=None)
        mock.assert_called_once_with("--cb_explore_adf --cover 10 --noconstant --interactions ax --quiet",None)

    def test_pickle(self) -> None:
        self.assertIsInstance(pickle.loads(pickle.dumps(VowpalCoverLearner(vw=VowpalMediatorMocked()))), VowpalCoverLearner)

class VowpalRegcbLearner_Tests(unittest.TestCase):

    @unittest.mock.patch('coba.learners.vowpal.VowpalArgsLearner.__init__')
    def test_defaults(self, mock) -> None:
        VowpalRegcbLearner()
        mock.assert_called_once_with("--cb_explore_adf --regcb --interactions ax --interactions axx --ignore_linear x --random_seed 1 --quiet",None)

    @unittest.mock.patch('coba.learners.vowpal.VowpalArgsLearner.__init__')
    def test_specifics(self, mock) -> None:
        VowpalRegcbLearner(mode="optimistic", features = ['a','x','ax'], seed=None)
        mock.assert_called_once_with("--cb_explore_adf --regcbopt --noconstant --interactions ax --quiet",None)

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW is not installed")
    def test_pickle(self) -> None:
        self.assertIsInstance(pickle.loads(pickle.dumps(VowpalRegcbLearner(vw=VowpalMediatorMocked()))), VowpalRegcbLearner)

class VowpalSquarecbLearner_Tests(unittest.TestCase):

    @unittest.mock.patch('coba.learners.vowpal.VowpalArgsLearner.__init__')
    def test_defaults(self, mock) -> None:
        VowpalSquarecbLearner()
        mock.assert_called_once_with("--cb_explore_adf --squarecb --gamma_scale 10 --interactions ax --interactions axx --ignore_linear x --random_seed 1 --quiet",None)

    @unittest.mock.patch('coba.learners.vowpal.VowpalArgsLearner.__init__')
    def test_specifics(self, mock) -> None:
        VowpalSquarecbLearner(mode="elimination", gamma_scale=5, features = ['a','x','ax'], seed=None)
        mock.assert_called_once_with("--cb_explore_adf --squarecb --gamma_scale 5 --elim --noconstant --interactions ax --quiet",None)

    def test_pickle(self) -> None:
        self.assertIsInstance(pickle.loads(pickle.dumps(VowpalSquarecbLearner(vw=VowpalMediatorMocked()))), VowpalSquarecbLearner)

class VowpalOffpolicyLearner_Tests(unittest.TestCase):

    @unittest.mock.patch('coba.learners.vowpal.VowpalArgsLearner.__init__')
    def test_defaults(self, mock) -> None:
        VowpalOffPolicyLearner()
        mock.assert_called_once_with("--cb_adf --interactions ax --interactions axx --ignore_linear x --random_seed 1 --quiet",None)

    @unittest.mock.patch('coba.learners.vowpal.VowpalArgsLearner.__init__')
    def test_specifics(self, mock) -> None:
        VowpalOffPolicyLearner(features=['a','x',"ax"], seed=None)
        mock.assert_called_once_with("--cb_adf --noconstant --interactions ax --quiet",None)

    def test_pickle(self) -> None:
        self.assertIsInstance(pickle.loads(pickle.dumps(VowpalOffPolicyLearner(vw=VowpalMediatorMocked()))), VowpalOffPolicyLearner)

class VowpalArgsLearner_Tests(unittest.TestCase):

    def test_inheritance_after_pickle(self):
        learner = pickle.loads(pickle.dumps(VowpalInherited(vw=VowpalMediatorMocked())))
        self.assertEqual(learner.params, {"Shadow":True})

    def test_params(self):
        learner = VowpalArgsLearner(vw=VowpalMediatorMocked())

        expected_args = [
            "--cb_explore_adf",
            "--epsilon 0.05",
            "--interactions ax",
            "--interactions axx",
            "--ignore_linear x",
            "--random_seed 1",
        ]

        self.assertEqual(learner.params['family'], "vw")
        self.assertEqual(learner.params["args"], " ".join(expected_args))

    def test_init_no_cb_term(self):
        with self.assertRaises(CobaException):
            VowpalArgsLearner('--epsilon .1')

    def test_init_learner_default(self):

        l = VowpalArgsLearner(vw=VowpalMediatorMocked())
        l.predict(None, [1,2,3])

        expected_args = [
            "--cb_explore_adf",
            "--epsilon 0.05",
            "--interactions ax",
            "--interactions axx",
            "--ignore_linear x",
            "--random_seed 1",
            "--quiet"
        ]

        self.assertEqual((" ".join(expected_args), 4), l._vw._init_learner_calls[0])

    def test_init_learner_cb_explore_adf(self):
        l = VowpalArgsLearner("--cb_explore_adf", VowpalMediatorMocked())
        l.predict(None,[1,2,3])
        self.assertEqual(("--cb_explore_adf", 4), l._vw._init_learner_calls[0])

    def test_init_learner_cb_adf(self):
        l = VowpalArgsLearner("--cb_adf",VowpalMediatorMocked([1,2,3]))
        l.predict(None, [1,2,3])

        self.assertEqual(("--cb_adf", 4), l._vw._init_learner_calls[0])

    def test_init_learner_cb_explore_action_infer(self):
        vw = VowpalMediatorMocked()
        VowpalArgsLearner("--cb_explore", vw).predict(None, ['yes','no'])
        self.assertEqual(("--cb_explore 2", 4), vw._init_learner_calls[0])

    def test_init_learner_cb_action_infer(self):
        vw = VowpalMediatorMocked()
        VowpalArgsLearner("--cb", vw).predict(None, ['yes','no'])
        self.assertEqual(("--cb 2", 4), vw._init_learner_calls[0])

    def test_predict_cb_explore_adf(self):

        vw = VowpalMediatorMocked([.25, .75])
        p  = VowpalArgsLearner("--cb_explore_adf",vw).predict(None, ['yes','no'])[0]

        self.assertEqual(2, len(vw._predict_calls[0]))

        self.assertEqual({'x':None }, vw._predict_calls[0][0].ns[0])
        self.assertEqual({'a':'yes'}, vw._predict_calls[0][0].ns[1])
        self.assertEqual(None       , vw._predict_calls[0][0].label)

        self.assertEqual({'x':None }, vw._predict_calls[0][1].ns[0])
        self.assertEqual({'a':'no'} , vw._predict_calls[0][1].ns[1])
        self.assertEqual(None       , vw._predict_calls[0][1].label)

        self.assertEqual([.25, .75], p)

    def test_predict_cb_adf(self):

        vw = VowpalMediatorMocked([.25, .75])
        p  = VowpalArgsLearner("--cb_adf",vw).predict(None, ['yes','no'])[0]

        self.assertEqual(2, len(vw._predict_calls[0]))

        self.assertEqual({'x':None }, vw._predict_calls[0][0].ns[0])
        self.assertEqual({'a':'yes'}, vw._predict_calls[0][0].ns[1])
        self.assertEqual(None       , vw._predict_calls[0][0].label)

        self.assertEqual({'x':None }, vw._predict_calls[0][1].ns[0])
        self.assertEqual({'a':'no'} , vw._predict_calls[0][1].ns[1])
        self.assertEqual(None       , vw._predict_calls[0][1].label)

        self.assertEqual([1,0], p)

    def test_predict_cb_explore(self):

        vw = VowpalMediatorMocked([0.25, 0.75])
        p = VowpalArgsLearner("--cb_explore 2", vw).predict(None, ['yes','no'])[0]

        self.assertIsInstance(vw._predict_calls[0], VowpalEaxmpleMock)

        self.assertEqual({'x':None }, vw._predict_calls[0].ns)
        self.assertEqual(None       , vw._predict_calls[0].label)

        self.assertEqual([.25, .75], p)

    def test_predict_cb(self):

        vw = VowpalMediatorMocked(2)
        p = VowpalArgsLearner("--cb 2", vw).predict(None, ['yes','no'])[0]

        self.assertEqual([0,1], p)

    def test_learn_cb_adf(self):

        vw = VowpalMediatorMocked()
        learner = VowpalArgsLearner("--cb_explore_adf",vw)

        learner.predict(None, ['yes','no'])
        learner.learn(None, 'yes', 1, 0.2, ['yes','no'])

        self.assertEqual(2, len(vw._learn_calls[0]))

        self.assertEqual({'x':None }, vw._learn_calls[0][0].ns[0])
        self.assertEqual({'a':'yes'}, vw._learn_calls[0][0].ns[1])
        self.assertEqual("1:-1:0.2"  , vw._learn_calls[0][0].label)

        self.assertEqual({'x':None }, vw._learn_calls[0][1].ns[0])
        self.assertEqual({'a':'no'} , vw._learn_calls[0][1].ns[1])
        self.assertEqual(None       , vw._learn_calls[0][1].label)

    def test_learn_cb(self):

        vw = VowpalMediatorMocked()
        learner = VowpalArgsLearner("--cb_explore", vw)

        learner.predict(None, ['yes','no'])
        learner.learn(None, 'no', .5, 0.2, ['yes','no'])

        self.assertIsInstance(vw._learn_calls[0], VowpalEaxmpleMock)

        self.assertEqual({'x':None}, vw._learn_calls[0].ns)
        self.assertEqual("2:-0.5:0.2", vw._learn_calls[0].label)

    def test_flatten_tuples(self):

        vw = VowpalMediatorMocked()
        learner = VowpalArgsLearner("--cb_explore", vw)

        learner.predict([(0,0,1)], ['yes','no'])
        learner.learn({'l':(0,0,1), 'j':1 }, 'no', .5, 0.2, ['yes','no'])

        self.assertIsInstance(vw._learn_calls[0], VowpalEaxmpleMock)

        self.assertEqual({'x':[0,0,1]}, vw._predict_calls[0].ns)
        self.assertEqual(None, vw._predict_calls[0].label)

        self.assertEqual({'x':{'l_0':0, 'l_1':0, 'l_2':1, 'j':1}}, vw._learn_calls[0].ns)
        self.assertEqual("2:-0.5:0.2", vw._learn_calls[0].label)

    def test_cb_no_predict_for_inference(self):
        with self.assertRaises(CobaException):
            VowpalArgsLearner("--cb", VowpalMediatorMocked()).learn(None,1,.2,.2,None)

    def test_cb_no_predict_for_inference(self):
        with self.assertRaises(CobaException):
            VowpalArgsLearner("--cb", VowpalMediatorMocked()).learn(None,1,.2,.2,None)

    def test_cb_predict_action_change(self):

        learner = VowpalArgsLearner("--cb", VowpalMediatorMocked())
        learner.predict(None, [1,2,3])

        with self.assertRaises(CobaException) as e:
            learner.predict(None, [4,5,6])

        self.assertTrue("`adf`" in str(e.exception))

    def test_cb_predict_action_count_mismatch(self):

        learner = VowpalArgsLearner("--cb 3", VowpalMediatorMocked())

        with self.assertRaises(CobaException) as e:
            learner.predict(None, [1,2,3,4])

    def test_cb_learn_before_predict(self):
        learner = VowpalArgsLearner("--cb 3", VowpalMediatorMocked())
        learner.learn(None,1,1,1,[1,2,3])

    def test_cb_adf_learn_before_predict(self):
        learner = VowpalArgsLearner("--cb_adf", VowpalMediatorMocked())
        learner.learn(None,1,1,1,[1,2,3])

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW is not installed.")
    def test_cb_adf_learning(self):
        learner = VowpalArgsLearner()

        n_actions  = 3
        n_features = 10
        n_examples = 2000

        rng = CobaRandom(11111)

        contexts = [ rng.randoms(n_features) for _ in range(n_examples) ]

        pre_learn_rewards = []
        for context in contexts[:int(.9*n_examples)]:

            actions = [ rng.randoms(n_features) for _ in range(n_actions) ]
            rewards = [ sum([a*c for a,c in zip(action,context)]) for action in actions ]
            rewards = [ int(r == max(rewards)) for r in rewards ]

            pre_learn_rewards.append(rng.choice(rewards,learner.predict(context, actions)[0]))

        for context in contexts[:int(.9*n_examples)]:

            actions = [ rng.randoms(n_features) for _ in range(n_actions) ]
            rewards = [ sum([a*c for a,c in zip(action,context)]) for action in actions ]
            rewards = [ int(r == max(rewards)) for r in rewards ]

            probs,info = learner.predict(context, actions)
            choice = rng.choice(list(range(3)), probs)

            learner.learn(context, actions[choice], rewards[choice], probs[choice], info)

        post_learn_rewards = []

        for context in contexts[int(.9*n_examples):]:
            actions = [ rng.randoms(n_features) for _ in range(n_actions) ]
            rewards = [ sum([a*c for a,c in zip(action,context)]) for action in actions ]
            rewards = [ int(r == max(rewards)) for r in rewards ]

            post_learn_rewards.append(rng.choice(rewards,learner.predict(context, actions)[0]))

        average_pre_learn_reward  = sum(pre_learn_rewards)/len(pre_learn_rewards)
        average_post_learn_reward = sum(post_learn_rewards)/len(post_learn_rewards)

        self.assertAlmostEqual(.33, average_pre_learn_reward, places=2)
        self.assertAlmostEqual(.775, average_post_learn_reward, places=2)

    def test_pickle(self) -> None:
        self.assertIsInstance(pickle.loads(pickle.dumps(VowpalArgsLearner(vw=VowpalMediatorMocked()))), VowpalArgsLearner)

@unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW is not installed")
class VowpalMediator_Tests(unittest.TestCase):

    def test_make_example_setup_done(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --noconstant --quiet",4)
        ex = vw.make_example({'x':'a'}, None)

        self.assertTrue(hasattr(ex, "setup_done"))

    def test_make_example_single_string_value(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --noconstant --quiet",4)
        ex = vw.make_example({'x':'a'}, None)

        self.assertTrue(hasattr(ex, "setup_done"))
        self.assertEqual([(ex.get_feature_id("x","0=a"),1)],list(ex.iter_features()))

    def test_make_example_single_numeric_value(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --noconstant --quiet",4)
        ex = vw.make_example({'x':5}, None)

        self.assertEqual([(ex.get_feature_id("x",'0'),5)],list(ex.iter_features()))

    def test_make_example_dict_numeric_value(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --noconstant --quiet",4)
        ex = vw.make_example({'x':{'a':5}}, None)

        self.assertEqual([(ex.get_feature_id("x","a"),5)],list(ex.iter_features()))

    @unittest.skip("Supporting this use case made dict processing take nearly 4x as long, so we removed it.")
    def test_make_example_dict_string_value(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --noconstant --quiet",4)
        ex = vw.make_example({'x':{'a':'b'}}, None)

        self.assertEqual([(ex.get_feature_id("x","a=b"),1)],list(ex.iter_features()))

    def test_make_example_list_numeric_value(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --noconstant --quiet",4)
        ex = vw.make_example({'x':[2]}, None)

        self.assertEqual([(ex.get_feature_id("x",'0'),2)],list(ex.iter_features()))

    def test_make_example_empty_list(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --noconstant --quiet",4)
        ex = vw.make_example({'x':[2], 'y':[]}, None)

        self.assertEqual([(ex.get_feature_id("x",'0'),2)],list(ex.iter_features()))

    def test_make_example_list_string_value(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --noconstant --quiet",4)
        ex = vw.make_example({'x':['a']}, None)

        self.assertEqual([(ex.get_feature_id("x","0=a"),1)],list(ex.iter_features()))

    def test_make_example_list_mixed_value(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --noconstant --quiet",4)
        ex = vw.make_example({'x':['a',3.2,'c']}, None)

        expected = [
            (ex.get_feature_id("x","0=a"),1),
            (ex.get_feature_id("x",'1'),3.2),
            (ex.get_feature_id("x","2=c"),1)
        ]

        actual = list(ex.iter_features())

        self.assertEqual(expected[0],actual[0])
        self.assertEqual(expected[1],(actual[1][0],round(actual[1][1],4)))
        self.assertEqual(expected[2],actual[2])

    def test_make_example_label(self):

        from vowpalwabbit import pyvw, __version__

        vw = VowpalMediator()
        vw.init_learner("--cb_explore 2 --noconstant --quiet",4)
        ex = vw.make_example({'x':1}, "0:.5:1")
        self.assertEqual(4, ex.labelType)

        if __version__[0] != '8':
            self.assertEqual("0:0.5:1.0", str(ex.get_label(pyvw.LabelType.CONTEXTUAL_BANDIT)))
        else:
            self.assertEqual("0:0.5:1.0", str(pyvw.cbandits_label(ex)))

    def test_make_examples_setup_done(self):
        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --noconstant --quiet",4)
        exs = vw.make_examples({'x':1}, [{'a':2}, {'a':3}], None)

        for ex in exs:
            self.assertTrue(hasattr(ex, "setup_done"))

    def test_make_examples_namespaces(self):

        from vowpalwabbit.pyvw import example

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --noconstant --quiet",4)
        ex = cast(Sequence[example],vw.make_examples({'x':3}, [{'a':1},{'a':2}], None))

        self.assertEqual(2, len(ex))

        expected_0 = [
            (ex[0].get_feature_id("x",'0'),3),
            (ex[0].get_feature_id("a",'1'),1)
        ]

        expected_1 = [
            (ex[1].get_feature_id("x",'0'),3),
            (ex[1].get_feature_id("a",'1'),2)
        ]

        self.assertEqual(expected_0,list(ex[0].iter_features()))
        self.assertEqual(expected_1,list(ex[1].iter_features()))

    def test_make_examples_labels(self):
        from vowpalwabbit import __version__, pyvw

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --noconstant --quiet", 4)
        exs = vw.make_examples({'x':1}, [{'a':1},{'a':2}], ["0:.5:1",""])

        self.assertEqual(4, exs[0].labelType)
        self.assertEqual(4, exs[1].labelType)

        if __version__[0] == '9':
            self.assertEqual("0:0.5:1.0", str(exs[0].get_label(pyvw.LabelType.CONTEXTUAL_BANDIT)))
            self.assertEqual("", str(exs[1].get_label(pyvw.LabelType.CONTEXTUAL_BANDIT)))
        else:
            self.assertEqual("0:0.5:1.0", str(pyvw.cbandits_label(exs[0])))
            self.assertEqual("", str(pyvw.cbandits_label(exs[1])))

    def test_init_learner(self):
        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --epsilon .1 --quiet", 4)
        self.assertIn("--cb_explore_adf", vw._vw.get_arguments())
        self.assertIn("--epsilon", vw._vw.get_arguments())

    def test_init_twice_exception(self):
        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --epsilon .1 --quiet", 4)

        with self.assertRaises(CobaException) as ex:
            vw.init_learner("--cb_explore_adf --epsilon .1 --quiet", 4)

    def test_predict(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --epsilon .1 --noconstant --quiet", 4)
        ex = vw.make_examples({}, [{'a':1}, {'a':2}], None)

        self.assertEqual(2, len(vw.predict(ex)))

    def test_learn(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --epsilon .1 --noconstant --quiet", 4)
        ex = vw.make_examples({'x':2}, [{'a':1}, {'a':2}], ["0:.5:1", ""])

        self.assertEqual(0, vw._vw.get_weight(ex[0].get_feature_id("x",'0')))
        self.assertEqual(0, vw._vw.get_weight(ex[0].get_feature_id("a",'1')))
        vw.learn(ex)
        self.assertNotEqual(0, vw._vw.get_weight(ex[0].get_feature_id("x",'0')))
        self.assertNotEqual(0, vw._vw.get_weight(ex[0].get_feature_id("a",'1')))
        self.assertEqual(2, len([vw._vw.get_weight(i) for i in range(vw._vw.num_weights()) if vw._vw.get_weight(i) != 0]))

    def test_regression_learning(self):
        vw = VowpalMediator().init_learner("--quiet", 1)

        n_features = 10
        n_examples = 1000

        rng = CobaRandom(1)

        weights = rng.randoms(n_features)
        rows    = [ rng.randoms(n_features) for _ in range(n_examples) ]
        labels  = [ sum([w*r for w,r in zip(weights,row)]) for row in rows ]

        examples = list(zip(rows,labels))

        self.assertEqual(0,vw.predict(vw.make_example({'x': rows[0]}, None)))

        pred_errs = []
        for row,label in examples[int(.9*n_examples):]:
            pred_errs.append(vw.predict(vw.make_example({"x": row}, None))-label)

        pre_learn_mse = sum([e**2 for e in pred_errs])//len(pred_errs)

        for row,label in examples[0:int(.9*n_examples)]:
            vw.learn(vw.make_example({"x": row}, str(label)))

        pred_errs = []

        for row,label in examples[int(.9*n_examples):]:
            pred_errs.append(vw.predict(vw.make_example({"x": row}, None))-label)

        post_learn_mse = sum([e**2 for e in pred_errs])/len(pred_errs)

        self.assertNotAlmostEqual(0,pre_learn_mse, places=2)
        self.assertAlmostEqual(0,post_learn_mse, places=2)

if __name__ == '__main__':
    unittest.main()
