import unittest
import pickle

from collections import OrderedDict
from coba.exceptions import CobaException

from coba.primitives import Sparse, Dense, HashableSparse, HashableDense, Sparse_, Dense_, Categorical
from coba.primitives import Learner, Environment, Evaluator, Interaction, SimulatedInteraction, LoggedInteraction

class DummySparse(Sparse):

    def __init__(self, row) -> None:
        self._row = row

    def __getitem__(self, key):
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self):
        ...

    def keys(self):
        ...

    def items(self):
        return self._row.items()

class DummyDense(Dense):

    def __init__(self, row) -> None:
        self._row = row

    def __getitem__(self, key):
        ...

    def __len__(self) -> int:
        return len(self._row)

    def __iter__(self):
        return iter(self._row)

class DummySparse_(Sparse_):

    def __init__(self, row) -> None:
        self._row = row

    def __getitem__(self, key):
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self):
        ...

    def keys(self):
        ...

    def items(self):
        return self._row.items()

class DummyDense_(Dense_):

    def __init__(self, row) -> None:
        self._row = row

    def __getitem__(self, key):
        ...

    def __len__(self) -> int:
        return len(self._row)

    def __iter__(self):
        return iter(self._row)

class Dense_Tests(unittest.TestCase):
    def test_getattr(self):
        class DummyClass:
            def __init__(self) -> None:
                self.missing = True

        self.assertEqual(True, DummyDense(DummyClass()).missing)
        with self.assertRaises(AttributeError):
            self.assertEqual(True, DummyDense({'a':1}).missing)

    def test_eq(self):
        self.assertEqual(DummyDense([1,2,3]),[1,2,3])

    def test_bad_eq(self):
        self.assertNotEqual(DummyDense([1,2,3]),1)

    def test_copy(self):
        dense = DummyDense((1,2,3))
        dense_copy = dense.copy()
        self.assertEqual(dense,dense_copy)
        self.assertIsNot(dense,dense_copy)

    def test_isinstance(self):
        self.assertTrue(isinstance([],Dense))
        self.assertTrue(isinstance((),Dense))

    def test_not_isinstance(self):
        self.assertFalse(isinstance({},Dense))
        self.assertFalse(isinstance("",Dense))
        self.assertFalse(isinstance(1 ,Dense))

class Sparse_Tests(unittest.TestCase):
    def test_getattr(self):
        class DummyClass:
            def __init__(self) -> None:
                self.missing = True
        self.assertEqual(True, DummySparse(DummyClass()).missing)
        with self.assertRaises(AttributeError):
            self.assertEqual(True, DummySparse({'a':1}).missing)

    def test_eq(self):
        self.assertEqual(DummySparse({'a':1}),{'a':1})

    def test_bad_eq(self):
        self.assertNotEqual(DummySparse({'a':1}),1)

    def test_copy(self):
        sparse = DummySparse({'a':1})
        sparse_copy = sparse.copy()
        self.assertEqual(sparse,sparse_copy)
        self.assertIsNot(sparse,sparse_copy)

    def test_isinstance(self):
        self.assertTrue(isinstance({},Sparse))

    def test_not_isinstance(self):
        self.assertFalse(isinstance([],Sparse))
        self.assertFalse(isinstance("",Sparse))
        self.assertFalse(isinstance(1 ,Sparse))

class Dense__Tests(unittest.TestCase):
    def test_getattr(self):
        class DummyClass:
            def __init__(self) -> None:
                self.missing = True

        self.assertEqual(True, DummyDense_(DummyClass()).missing)

        with self.assertRaises(AttributeError):
            self.assertEqual(True, DummyDense_({'a':1}).missing)

    def test_eq(self):
        self.assertEqual(DummyDense_([1,2,3]),[1,2,3])

    def test_bad_eq(self):
        self.assertNotEqual(DummyDense_([1,2,3]),1)

    def test_copy(self):
        dense = DummyDense_((1,2,3))
        dense_copy = dense.copy()
        self.assertEqual(dense,dense_copy)
        self.assertIsNot(dense,dense_copy)

    def test_isinstance(self):
        self.assertIsInstance(DummyDense_((1,2,3)),Dense)

class Sparse__Tests(unittest.TestCase):
    def test_getattr(self):
        class DummyClass:
            def __init__(self) -> None:
                self.missing = True

        self.assertEqual(True, DummySparse_(DummyClass()).missing)

        with self.assertRaises(AttributeError):
            self.assertEqual(True, DummySparse_({'a':1}).missing)

    def test_eq(self):
        self.assertEqual(DummySparse_({'a':1}),{'a':1})

    def test_bad_eq(self):
        self.assertNotEqual(DummySparse_({'a':1}),1)

    def test_copy(self):
        sparse = DummySparse_({'a':1})
        sparse_copy = sparse.copy()
        self.assertEqual(sparse,sparse_copy)
        self.assertIsNot(sparse,sparse_copy)

    def test_isinstance(self):
        self.assertIsInstance(DummySparse_({'a':1}),Sparse)

class HashableSparse_Tests(unittest.TestCase):
    def test_get(self):
        hash_dict = HashableSparse({'a':1,'b':2})
        self.assertEqual(1,hash_dict['a'])

    def test_iter(self):
        hash_dict = HashableSparse({'a':1,'b':2})
        self.assertEqual(['a','b'],list(hash_dict))

    def test_len(self):
        hash_dict = HashableSparse({'a':1,'b':2})
        self.assertEqual(2,len(hash_dict))

    def test_hash(self):
        hash_dict = HashableSparse({'a':1,'b':2})
        self.assertEqual(hash(hash_dict), hash(hash_dict))
        self.assertEqual(hash_dict,hash_dict)

    def test_hash_is_order_agnostic(self):
        hash_dict1 = HashableSparse(OrderedDict({'a':1,'b':2}))
        hash_dict2 = HashableSparse(OrderedDict({'b':2,'a':1}))

        self.assertEqual(hash(hash_dict1), hash(hash_dict2))
        self.assertEqual(hash_dict1,hash_dict2)

    def test_eq_good(self):
        hash_dict = HashableSparse({'a':1,'b':2})
        self.assertEqual({'a':1,'b':2},hash_dict)

    def test_eq_bad(self):
        hash_dict = HashableSparse({'a':1,'b':2})
        self.assertNotEqual(1,hash_dict)

    def test_repr(self):
        hash_dict = HashableSparse({'a':1,'b':2})
        self.assertEqual("{'a': 1, 'b': 2}",repr(hash_dict))

    def test_str(self):
        hash_dict = HashableSparse({'a':1,'b':2})
        self.assertEqual("{'a': 1, 'b': 2}",str(hash_dict))

    def test_copy(self):
        hash_dict = HashableSparse({'a':1,'b':2})
        hash_dict_copy = hash_dict.copy()
        self.assertEqual(hash_dict,hash_dict_copy)
        self.assertIsNot(hash_dict,hash_dict_copy)

    def test_pickle(self):
        dump_dict = HashableSparse(OrderedDict({'a':1,'b':2}))
        load_dict = pickle.loads(pickle.dumps(dump_dict))

class HashableDense_Tests(unittest.TestCase):

    def test_get(self):
        hash_seq = HashableDense([1,2,3])
        self.assertEqual(2,hash_seq[1])

    def test_len(self):
        hash_seq = HashableDense([1,2,3])
        self.assertEqual(3,len(hash_seq))

    def test_hash(self):
        hash_seq = HashableDense([1,2,3])
        self.assertEqual(hash(hash_seq), hash(hash_seq))
        self.assertEqual(hash_seq,hash_seq)

    def test_eq(self):
        hash_seq = HashableDense([1,2,3])
        self.assertEqual([1,2,3],hash_seq)
        self.assertEqual((1,2,3),hash_seq)

    def test_ne(self):
        hash_seq = HashableDense([1,2,3])
        self.assertNotEqual([1,2,4],hash_seq)
        self.assertNotEqual([1,2,3,4],hash_seq)
        self.assertNotEqual(1,hash_seq)

    def test_repr(self):
        hash_seq = HashableDense([1,2,3])
        self.assertEqual("[1, 2, 3]",repr(hash_seq))

    def test_str(self):
        hash_seq = HashableDense([1,2,3])
        self.assertEqual("[1, 2, 3]",str(hash_seq))

    def test_pickle(self):
        dump = HashableDense([1,2,3])
        load = pickle.loads(pickle.dumps(dump))

class Categorical_Tests(unittest.TestCase):
    def test_value(self):
        self.assertEqual("A", Categorical("A",["A","B"]))

    def test_levels(self):
        self.assertEqual(["A","B"], Categorical("A",["A","B"]).levels)

    def test_eq(self):
        self.assertEqual(Categorical("A",["A","B"]), Categorical("A",["A","B"]))

    def test_ne(self):
        self.assertNotEqual(1, Categorical("A",["A","B"]))

    def test_str(self):
        self.assertEqual("A", str(Categorical("A",["A","B"])))

    def test_repr(self):
        self.assertEqual("Categorical('A',['A', 'B'])", repr(Categorical("A",["A","B"])))

    def test_pickle(self):
        out = pickle.loads(pickle.dumps(Categorical("A",["A","B"])))

        self.assertIsInstance(out,Categorical)
        self.assertEqual(out.levels, ['A',"B"])

    def test_cast(self):
        a = Categorical("A",["A","B"])
        out = pickle.loads(pickle.dumps(a))

        self.assertIsInstance(out,Categorical)
        self.assertEqual(out.levels, ['A',"B"])

class Learner_Tests(unittest.TestCase):

    def test_params_empty(self):
        class MyLearner(Learner):
            pass

        self.assertEqual(MyLearner().params,{})

    def test_score_not_implemented(self):
        class MyLearner(Learner):
            pass

        with self.assertRaises(NotImplementedError) as ex:
            MyLearner().score(None,[],[])

        self.assertIn("`score`", str(ex.exception))

    def test_predict_not_implemented(self):
        class MyLearner(Learner):
            pass

        with self.assertRaises(NotImplementedError) as ex:
            MyLearner().predict(None,[])

        self.assertIn("`predict`", str(ex.exception))

    def test_learn_not_implemented(self):
        class MyLearner(Learner):
            pass

        with self.assertRaises(NotImplementedError) as ex:
            MyLearner().learn(None,None,None,None)

        self.assertIn("`learn`", str(ex.exception))

class Environment_Tests(unittest.TestCase):

    def test_str_with_params(self):
        class TestEnvironment(Environment):
            def read(self):
                return []
            @property
            def params(self):
                return {'a':1}
        self.assertEqual("{'a': 1}", str(TestEnvironment()))

    def test_str_sans_params(self):
        class TestEnvironment(Environment):
            def read(self):
                return []
            @property
            def params(self):
                return {}

        self.assertEqual("TestEnvironment", str(TestEnvironment()))

class Interaction_Tests(unittest.TestCase):
    def test_simulated_dict(self):
        mapping = {'context':1,'actions':[1,2],'rewards':[3,4]}
        self.assertEqual(Interaction.from_dict(mapping),mapping)

    def test_logged_dict(self):
        given    = {'context':1,'actions':[1,2],'action':1,'reward':4,'probability':.1}
        expected = {'context':1,'actions':[1,2],'action':1,'reward':4,'probability':.1}
        self.assertEqual(Interaction.from_dict(given),expected)

    def test_grounded_dict(self):
        given    = {'context':1,'actions':[1,2],'action':1,'rewards':[3,4],'feedbacks':[5,6]}
        expected = given
        self.assertEqual(Interaction.from_dict(given),expected)

    def test_unknown_dict(self):
        given    = {'context':1,'actions':[1,2]}
        expected = given
        self.assertEqual(Interaction.from_dict(given),expected)

class LoggedInteraction_Tests(unittest.TestCase):

    def test_simple_with_actions(self):
        interaction = LoggedInteraction(1, 2, 3, probability=.2, actions=[1,2,3])

        self.assertEqual(1, interaction['context'])
        self.assertEqual(2, interaction['action'])
        self.assertEqual(3, interaction['reward'])
        self.assertEqual(.2, interaction['probability'])
        self.assertEqual([1,2,3], interaction['actions'])

    def test_simple_sans_actions(self):
        interaction = LoggedInteraction(1, 2, 3, probability=.2)

        self.assertEqual(1, interaction['context'])
        self.assertEqual(2, interaction['action'])
        self.assertEqual(3, interaction['reward'])
        self.assertEqual(.2, interaction['probability'])

class SimulatedInteraction_Tests(unittest.TestCase):
    def test_context_none(self):
        self.assertEqual(None, SimulatedInteraction(None, (1,2,3), (4,5,6))['context'])

    def test_context_str(self):
        self.assertEqual("A", SimulatedInteraction("A", (1,2,3), (4,5,6))['context'])

    def test_context_dense(self):
        self.assertEqual((1,2,3), SimulatedInteraction((1,2,3), (1,2,3), (4,5,6))['context'])

    def test_context_dense_2(self):
        self.assertEqual((1,2,3,(0,0,1)), SimulatedInteraction((1,2,3,(0,0,1)), (1,2,3), (4,5,6))['context'])

    def test_context_sparse_dict(self):
        self.assertEqual({1:0}, SimulatedInteraction({1:0}, (1,2,3), (4,5,6))['context'])

    def test_actions_correct_1(self) -> None:
        self.assertSequenceEqual([1,2], SimulatedInteraction(None, [1,2], [1,2])['actions'])

    def test_actions_correct_2(self) -> None:
        self.assertSequenceEqual(["A","B"], SimulatedInteraction(None, ["A","B"], [1,2])['actions'])

    def test_actions_correct_3(self) -> None:
        self.assertSequenceEqual([(1,2), (3,4)], SimulatedInteraction(None, [(1,2), (3,4)], [1,2])['actions'])

    def test_rewards_correct(self):
        self.assertEqual([4,5,6], SimulatedInteraction((1,2), (1,2,3), [4,5,6])['rewards'])

    @unittest.skip("Old behavior")
    def test_rewards_actions_mismatch(self):
        with self.assertRaises(CobaException):
            SimulatedInteraction((1,2), (1,2,3), [4,5])

class Evaluator_Tests(unittest.TestCase):
    def test_params(self):
        class SubEvaluator(Evaluator):
            def evaluate(self, environment: Environment, learner: Learner):
                pass
        self.assertEqual(SubEvaluator().params,{})

if __name__ == '__main__':
    unittest.main()
