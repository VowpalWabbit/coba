import unittest
import pickle

from collections import OrderedDict

from coba.json import dumps,loads
from coba.exceptions import CobaException
from coba.utilities import PackageChecker

from coba.primitives import Sparse, Dense, HashableSparse, HashableDense, Sparse_, Dense_, Categorical
from coba.primitives import Learner, Environment, Evaluator
from coba.primitives import L1Reward, HammingReward, BinaryReward, DiscreteReward
from coba.primitives import SimulatedInteraction, LoggedInteraction

#For testing purposes
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
#For testing purposes

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

class Evaluator_Tests(unittest.TestCase):
    def test_params(self):
        class SubEvaluator(Evaluator):
            def evaluate(self, environment: Environment, learner: Learner):
                pass
        self.assertEqual(SubEvaluator().params,{})

class L1Reward_Tests(unittest.TestCase):
    def test_simple(self):
        rwd = L1Reward(1)
        self.assertEqual(-1, rwd(2))
        self.assertEqual(0 , rwd(1))
        self.assertEqual(-1, rwd(0))

    def test_pickle(self):
        dumped = pickle.dumps(L1Reward(1))
        loaded = pickle.loads(dumped)
        self.assertIsInstance(loaded, L1Reward)
        self.assertEqual(loaded._argmax,1)

    @unittest.skipUnless(PackageChecker.torch(strict=False), "This test requires pytorch")
    def test_squeezed_single_torch(self):
        import torch
        expected = torch.tensor(-1).float()
        actual   = L1Reward(1)(torch.tensor(2))
        self.assertEqual(expected,actual)
        actual   = L1Reward(1)(torch.tensor(0))
        self.assertEqual(expected,actual)

    @unittest.skipUnless(PackageChecker.torch(strict=False), "This test requires pytorch")
    def test_not_squeezed_single_torch(self):
        import torch
        expected = torch.tensor([-1]).float()
        actual   = L1Reward(1)(torch.tensor([2]))
        self.assertEqual(expected,actual)
        actual   = L1Reward(1)(torch.tensor([0]))
        self.assertEqual(expected,actual)

    def test_json_serialization(self):
        obj = loads(dumps(L1Reward(2)))
        self.assertIsInstance(obj,L1Reward)
        self.assertEqual(obj._argmax, 2)

    def test_repr(self):
        self.assertEqual(repr(L1Reward(1)),'L1Reward(1)')
        self.assertEqual(repr(L1Reward(1.123456)),'L1Reward(1.12346)')

class BinaryReward_Tests(unittest.TestCase):
    def test_binary(self):
        rwd = BinaryReward(1)
        self.assertEqual(0, rwd(2))
        self.assertEqual(1, rwd(1))
        self.assertEqual(0, rwd(0))

    def test_binary_with_value(self):
        rwd = BinaryReward(1,2)
        self.assertEqual(0, rwd(2))
        self.assertEqual(2, rwd(1))
        self.assertEqual(0, rwd(0))

    def test_pickle(self):
        dumped = pickle.dumps(BinaryReward(1))
        loaded = pickle.loads(dumped)
        self.assertIsInstance(loaded, BinaryReward)
        self.assertEqual(loaded._argmax,1)

    @unittest.skipUnless(PackageChecker.torch(strict=False), "This test requires pytorch")
    def test_simple_numeric_argmax_torch_numeric_action(self):
        import torch
        rwd = BinaryReward(1,2)
        expected = torch.tensor(0)
        actual   = rwd(torch.tensor(2))
        self.assertTrue(torch.equal(expected,actual))
        expected = torch.tensor(2)
        actual   = rwd(torch.tensor(1))
        self.assertTrue(torch.equal(expected,actual))

    @unittest.skipUnless(PackageChecker.torch(strict=False), "This test requires pytorch")
    def test_simple_numeric_argmax_torch_sequence_action(self):
        import torch
        rwd = BinaryReward(1,2)
        expected = torch.tensor([0])
        actual   = rwd(torch.tensor([2]))
        self.assertTrue(torch.equal(expected,actual))
        expected = torch.tensor([2])
        actual   = rwd(torch.tensor([1]))
        self.assertTrue(torch.equal(expected,actual))
        expected = torch.tensor([[2]])
        actual   = rwd(torch.tensor([[1]]))
        self.assertTrue(torch.equal(expected,actual))

    @unittest.skipUnless(PackageChecker.torch(strict=False), "This test requires pytorch")
    def test_simple_sequence_argmax_torch_sequence_action(self):
        import torch
        rwd = BinaryReward([1],2)
        expected = torch.tensor(0)
        actual   = rwd(torch.tensor([2]))
        self.assertTrue(torch.equal(expected,actual))
        expected = torch.tensor(2)
        actual   = rwd(torch.tensor([1]))
        self.assertTrue(torch.equal(expected,actual))
        expected = torch.tensor([2])
        actual   = rwd(torch.tensor([[1]]))
        self.assertTrue(torch.equal(expected,actual))

    @unittest.skipUnless(PackageChecker.torch(strict=False), "This test requires pytorch")
    def test_torch_sequence_argmax_torch_sequence_action(self):
        import torch
        rwd = BinaryReward(torch.tensor([1]),2)
        expected = torch.tensor(0)
        actual   = rwd(torch.tensor([2]))
        self.assertTrue(torch.equal(expected,actual))
        expected = torch.tensor(2)
        actual   = rwd(torch.tensor([1]))
        self.assertTrue(torch.equal(expected,actual))
        expected = torch.tensor([2])
        actual   = rwd(torch.tensor([[1]]))
        self.assertTrue(torch.equal(expected,actual))

    @unittest.skipUnless(PackageChecker.torch(strict=False), "This test requires pytorch")
    def test_torch_sequence_argmax_simple_sequence_action(self):
        import torch
        rwd = BinaryReward(torch.tensor([1]),2)
        expected = 0
        actual   = rwd([2])
        self.assertEqual(expected,actual)
        self.assertFalse(torch.is_tensor(actual))
        expected = 2
        actual   = rwd([1])
        self.assertEqual(expected,actual)
        self.assertFalse(torch.is_tensor(actual))

    def test_json_serialization(self):
        obj = loads(dumps(BinaryReward('a')))
        self.assertIsInstance(obj,BinaryReward)
        self.assertEqual(obj._argmax, 'a')
        self.assertEqual(obj._value, 1)

        obj = loads(dumps(BinaryReward((0,1),2)))
        self.assertIsInstance(obj,BinaryReward)
        self.assertEqual(obj._argmax, (0,1))
        self.assertEqual(obj._value, 2)

    def test_repr(self):
        self.assertEqual(repr(BinaryReward([1,2])),'BinaryReward([1, 2])')
        self.assertEqual(repr(BinaryReward({1,2})),'BinaryReward({1, 2})')

class HammingReward_Tests(unittest.TestCase):
    def test_sequence(self):
        rwd = HammingReward([1,2,3,4])
        self.assertEqual(2/4, rwd([1,3]))
        self.assertEqual(1/4, rwd([4]))
        self.assertEqual(0  , rwd([5,6,7]))
        self.assertEqual(1/2, rwd([1,2,3,4,5,6,7,8]))

    def test_tuple(self):
        rwd = HammingReward((1,2,3,4))
        self.assertEqual(.5, rwd([1,3]))
        self.assertEqual(.25, rwd([4]))
        self.assertEqual(1, rwd((1,2,3,4)))

    def test_pickle(self):
        dumped = pickle.dumps(HammingReward([1,2,3]))
        loaded = pickle.loads(dumped)

        self.assertIsInstance(loaded, HammingReward)
        self.assertEqual(set(loaded._argmax),{1,2,3})

    @unittest.skipUnless(PackageChecker.torch(strict=False), "This test requires pytorch")
    def test_simple_numeric_argmax_torch_numeric_action(self):
        import torch
        rwd = HammingReward([1,2,3,4])
        self.assertTrue(torch.equal(torch.tensor(2/4), rwd(torch.tensor([1,3]))))
        self.assertTrue(torch.equal(torch.tensor(1/4), rwd(torch.tensor([4]))))
        self.assertTrue(torch.equal(torch.tensor(0  ), rwd(torch.tensor([5,6,7]))))
        self.assertTrue(torch.equal(torch.tensor(1/2), rwd(torch.tensor([1,2,3,4,5,6,7,8]))))

    @unittest.skipUnless(PackageChecker.torch(strict=False), "This test requires pytorch")
    def test_torch_numeric_argmax_torch_numeric_action(self):
        import torch
        rwd = HammingReward(torch.tensor([1,2,3,4]))
        self.assertTrue(torch.equal(torch.tensor(2/4), rwd(torch.tensor([1,3]))))
        self.assertTrue(torch.equal(torch.tensor(1/4), rwd(torch.tensor([4]))))
        self.assertTrue(torch.equal(torch.tensor(0  ), rwd(torch.tensor([5,6,7]))))
        self.assertTrue(torch.equal(torch.tensor(1/2), rwd(torch.tensor([1,2,3,4,5,6,7,8]))))
        self.assertTrue(torch.equal(torch.tensor([2/4]), rwd(torch.tensor([[1,3]]))))
        self.assertTrue(torch.equal(torch.tensor([1/4]), rwd(torch.tensor([[4]]))))
        self.assertTrue(torch.equal(torch.tensor([0  ]), rwd(torch.tensor([[5,6,7]]))))
        self.assertTrue(torch.equal(torch.tensor([1/2]), rwd(torch.tensor([[1,2,3,4,5,6,7,8]]))))

    @unittest.skipUnless(PackageChecker.torch(strict=False), "This test requires pytorch")
    def test_torch_sequence_argmax_torch_sequence_action(self):
        import torch
        rwd = HammingReward(torch.tensor([[1],[2],[3],[4]]))
        self.assertTrue(torch.equal(torch.tensor(2/4), rwd(torch.tensor([[1],[3]]))))
        self.assertTrue(torch.equal(torch.tensor(1/4), rwd(torch.tensor([[4]]))))
        self.assertTrue(torch.equal(torch.tensor(0  ), rwd(torch.tensor([[5],[6],[7]]))))
        self.assertTrue(torch.equal(torch.tensor(1/2), rwd(torch.tensor([[1],[2],[3],[4],[5],[6],[7],[8]]))))
        self.assertTrue(torch.equal(torch.tensor([2/4]), rwd(torch.tensor([[[1],[3]]]))))
        self.assertTrue(torch.equal(torch.tensor([1/4]), rwd(torch.tensor([[[4]]]))))
        self.assertTrue(torch.equal(torch.tensor([0  ]), rwd(torch.tensor([[[5],[6],[7]]]))))
        self.assertTrue(torch.equal(torch.tensor([1/2]), rwd(torch.tensor([[[1],[2],[3],[4],[5],[6],[7],[8]]]))))

    def test_json_serialization(self):
        obj = loads(dumps(HammingReward([1,2,3])))
        self.assertIsInstance(obj,HammingReward)
        self.assertEqual(obj._argmax, [1,2,3])

    def test_repr(self):
        self.assertEqual(repr(HammingReward([1,2])),'HammingReward([1, 2])')

class DiscreteReward_Tests(unittest.TestCase):
    def test_mapping(self):
        rwd = DiscreteReward({0:4,1:5,2:6})
        self.assertEqual(4,rwd(0))
        self.assertEqual(5,rwd(1))
        self.assertEqual(6,rwd(2))
        self.assertEqual(rwd,rwd)
        self.assertEqual([0,1,2],rwd.actions)
        self.assertEqual([4,5,6],rwd.rewards)

    def test_sequence(self):
        rwd = DiscreteReward([0,1,2],[4,5,6])
        self.assertEqual(4,rwd(0))
        self.assertEqual(5,rwd(1))
        self.assertEqual(6,rwd(2))
        self.assertEqual(rwd,rwd)
        self.assertEqual([0,1,2],rwd.actions)
        self.assertEqual([4,5,6],rwd.rewards)

    def test_pickle(self):
        reward = DiscreteReward({0:4,1:5,2:6})
        dumped = pickle.dumps(reward)
        loaded = pickle.loads(dumped)
        self.assertIsInstance(loaded, DiscreteReward)
        self.assertEqual(loaded._state, reward._state)

    @unittest.skipUnless(PackageChecker.torch(strict=False), "This test requires pytorch")
    def test_mapping_torch_numeric_actions_torch_numeric_action(self):
        import torch
        rwd = DiscreteReward({1:4,2:5,3:6})
        self.assertTrue(torch.equal(torch.tensor(5)  , rwd(torch.tensor(2))))
        self.assertTrue(torch.equal(torch.tensor([5]), rwd(torch.tensor([2]))))

    @unittest.skipUnless(PackageChecker.torch(strict=False), "This test requires pytorch")
    def test_mapping_torch_sequence_actions_torch_sequence_action(self):
        import torch
        rwd = DiscreteReward({(1,):4,(2,):5,(3,):6})
        self.assertTrue(torch.equal(torch.tensor(5)  , rwd(torch.tensor([2]))))
        self.assertTrue(torch.equal(torch.tensor([5]), rwd(torch.tensor([[2]]))))

    @unittest.skipUnless(PackageChecker.torch(strict=False), "This test requires pytorch")
    def test_sequence_torch_numeric_actions_torch_numeric_action(self):
        import torch
        rwd = DiscreteReward([1,2,3],[4,5,6])
        self.assertTrue(torch.equal(torch.tensor(5)  , rwd(torch.tensor(2))))
        self.assertTrue(torch.equal(torch.tensor([5]), rwd(torch.tensor([2]))))

    @unittest.skipUnless(PackageChecker.torch(strict=False), "This test requires pytorch")
    def test_sequence_torch_sequence_actions_torch_sequence_action(self):
        import torch
        rwd = DiscreteReward([(1,),(2,),(3,)],[4,5,6])
        self.assertTrue(torch.equal(torch.tensor(5)  , rwd(torch.tensor([2]))))
        self.assertTrue(torch.equal(torch.tensor([5]), rwd(torch.tensor([[2]]))))

    def test_json_serialization(self):
        obj = loads(dumps(DiscreteReward({0:1,1:2})))
        self.assertIsInstance(obj,DiscreteReward)
        self.assertEqual(obj._state, {0:1,1:2})

    def test_bad_actions(self):
        with self.assertRaises(CobaException) as r:
            DiscreteReward([1,2],[1,2,3])
        self.assertEqual(str(r.exception),"The given actions and rewards did not line up.")

    def test_repr(self):
        self.assertEqual(repr(DiscreteReward([1,2],[4,5])),'DiscreteReward([[1, 2], [4, 5]])')
        self.assertEqual(repr(DiscreteReward([1.123456],[4.123456])),'DiscreteReward([[1.12346], [4.12346]])')
        self.assertEqual(repr(DiscreteReward({1:4})),'DiscreteReward({1: 4})')

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

if __name__ == '__main__':
    unittest.main()
