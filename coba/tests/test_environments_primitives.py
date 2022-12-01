import unittest

from coba.environments import SafeEnvironment, HashableMap, HashableSeq
from coba.environments import SimulatedInteraction, LoggedInteraction
from coba.environments import L1Reward, HammingReward, ScaleReward, BinaryReward, SequenceReward
from coba.environments import SequenceFeedback, MulticlassReward
from coba.exceptions import CobaException
from coba.pipes import Pipes, Shuffle

class DummyEnvironment:

    def read(self):
        return []

class HashableMap_Tests(unittest.TestCase):

    def test_get(self):
        hash_dict = HashableMap({'a':1,'b':2})
        self.assertEqual(1,hash_dict['a'])

    def test_len(self):
        hash_dict = HashableMap({'a':1,'b':2})
        self.assertEqual(2,len(hash_dict))

    def test_iter(self):
        hash_dict = HashableMap({'a':1,'b':2})
        self.assertEqual(['a','b'],list(hash_dict))

    def test_len(self):
        hash_dict = HashableMap({'a':1,'b':2})
        self.assertEqual(2,len(hash_dict))

    def test_hash(self):
        hash_dict = HashableMap({'a':1,'b':2})
        self.assertEqual(hash(hash_dict), hash(hash_dict))
        self.assertEqual(hash_dict,hash_dict)

    def test_eq(self):
        hash_dict = HashableMap({'a':1,'b':2})
        self.assertEqual({'a':1,'b':2},hash_dict)

    def test_repr(self):
        hash_dict = HashableMap({'a':1,'b':2})
        self.assertEqual("{'a': 1, 'b': 2}",repr(hash_dict))
    
    def test_str(self):
        hash_dict = HashableMap({'a':1,'b':2})
        self.assertEqual("{'a': 1, 'b': 2}",str(hash_dict))


class HashableSeq_Tests(unittest.TestCase):

    def test_get(self):
        hash_seq = HashableSeq([1,2,3])
        self.assertEqual(2,hash_seq[1])

    def test_len(self):
        hash_seq = HashableSeq([1,2,3])
        self.assertEqual(3,len(hash_seq))

    def test_hash(self):
        hash_seq = HashableSeq([1,2,3])
        self.assertEqual(hash(hash_seq), hash(hash_seq))
        self.assertEqual(hash_seq,hash_seq)

    def test_eq(self):
        hash_seq = HashableSeq([1,2,3])
        self.assertEqual([1,2,3],hash_seq)
        self.assertEqual((1,2,3),hash_seq)

    def test_neq(self):
        hash_seq = HashableSeq([1,2,3])
        self.assertNotEqual([1,2,4],hash_seq)
        self.assertNotEqual([1,2,3,4],hash_seq)
        self.assertNotEqual(1,hash_seq)
    
    def test_repr(self):
        hash_seq = HashableSeq([1,2,3])
        self.assertEqual("[1, 2, 3]",repr(hash_seq))
    
    def test_str(self):
        hash_seq = HashableSeq([1,2,3])
        self.assertEqual("[1, 2, 3]",str(hash_seq))

class SafeEnvironment_Tests(unittest.TestCase):

    def test_params(self):
        self.assertEqual({'type': 'DummyEnvironment'}, SafeEnvironment(DummyEnvironment()).params)

    def test_read(self):
        self.assertEqual([], SafeEnvironment(DummyEnvironment()).read())

    def test_str(self):
        self.assertEqual('DummyEnvironment(shuffle=1)', str(SafeEnvironment(Pipes.join(DummyEnvironment(), Shuffle(1)))))
        self.assertEqual('DummyEnvironment', str(SafeEnvironment(DummyEnvironment())))

    def test_with_nesting(self):
        self.assertIsInstance(SafeEnvironment(SafeEnvironment(DummyEnvironment()))._environment, DummyEnvironment)

    def test_with_pipes(self):
        self.assertEqual({'type': 'DummyEnvironment', "shuffle":1}, SafeEnvironment(Pipes.join(DummyEnvironment(), Shuffle(1))) .params)

class LoggedInteraction_Tests(unittest.TestCase):
    def test_IPS_sequence(self):
        interaction = LoggedInteraction(1,2,3,probability=1/2,actions=[1,2,3])
        self.assertEqual([0,6,0], interaction['rewards'])

    def test_simple_with_actions(self):
        interaction = LoggedInteraction(1, 2, 3, probability=.2,actions=[1,2,3])

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

    def test_rewards_actions_mismatch(self):
        with self.assertRaises(CobaException):
            SimulatedInteraction((1,2), (1,2,3), [4,5])

class L1Reward_Tests(unittest.TestCase):

    def test_simple(self):
        rwd = L1Reward(1)

        self.assertEqual(1 , rwd.argmax())
        self.assertEqual(-1, rwd.eval(2))
        self.assertEqual(0 , rwd.eval(1))
        self.assertEqual(-1, rwd.eval(0))

class BinaryReward_Tests(unittest.TestCase):

    def test_binary_argmax(self):
        rwd = BinaryReward(1)

        self.assertEqual(1, rwd.argmax())
        self.assertEqual(0, rwd.eval(2))
        self.assertEqual(1, rwd.eval(1))
        self.assertEqual(0, rwd.eval(0))

class HammingReward_Tests(unittest.TestCase):

    def test_sequence(self):
        rwd = HammingReward([1,2,3,4])
        self.assertEqual({1,2,3,4}, rwd.argmax())
        self.assertEqual(2/4, rwd.eval([1,3]))
        self.assertEqual(1/4, rwd.eval([4]))
        self.assertEqual(0  , rwd.eval([5,6,7]))

    def test_tuple(self):
        rwd = HammingReward((1,2,3,4))
        self.assertEqual({1,2,3,4}, rwd.argmax())
        self.assertEqual(.5, rwd.eval([1,3]))
        self.assertEqual(.25, rwd.eval([4]))
        self.assertEqual(1, rwd.eval((1,2,3,4)))

class ScaleReward_Tests(unittest.TestCase):

    def test_identity_value(self):
        rwd = ScaleReward(L1Reward(1),0,1,'value')
        self.assertEqual(1 , rwd.argmax())
        self.assertEqual(-1, rwd.eval(2))
        self.assertEqual( 0, rwd.eval(1))
        self.assertEqual(-1, rwd.eval(0))

    def test_scale_shift_value(self):
        rwd = ScaleReward(L1Reward(1),-2,1/2,'value')
        self.assertEqual(1, rwd.argmax())
        self.assertEqual(-1  , rwd.eval(1))
        self.assertEqual(-3/2, rwd.eval(2))
        self.assertEqual(-3/2, rwd.eval(0))

    def test_identity_argmax(self):
        rwd = ScaleReward(L1Reward(1),0,1,'value')
        self.assertEqual(1 , rwd.argmax())
        self.assertEqual(-1, rwd.eval(2))
        self.assertEqual(-1, rwd.eval(0))
        self.assertEqual( 0, rwd.eval(1))

    def test_scale_shift_argmax(self):
        rwd = ScaleReward(L1Reward(1),-2,1/2,'argmax')
        self.assertEqual(-1/2, rwd.argmax())
        self.assertEqual(-1/2, rwd.eval(-1))
        self.assertEqual(-1/2, rwd.eval( 0))
        self.assertEqual(   0, rwd.eval(-1/2))

    def test_bad_target(self):
        with self.assertRaises(CobaException) as e:
            ScaleReward(2,2,'abc',L1Reward(1))

class SequenceReward_Tests(unittest.TestCase):
    def test_sequence(self):
        rwd = SequenceReward([4,5,6])

        self.assertEqual(3,len(rwd))
        self.assertEqual([4,5,6],rwd)
        self.assertEqual(4,rwd[0])
        self.assertEqual(6,rwd.max())
        self.assertEqual(2,rwd.argmax())
        self.assertEqual(4,rwd.eval(0))
        self.assertEqual(5,rwd.eval(1))
        self.assertEqual(6,rwd.eval(2))
        self.assertEqual(rwd,rwd)

    def test_bad_eq(self):
        rwd = SequenceReward([4,5,6])
        self.assertNotEqual(1,rwd)

class MulticlassReward_Tests(unittest.TestCase):
    def test_simple(self):
        rwd = MulticlassReward([1,2,3],1)

        self.assertEqual(3,len(rwd))
        self.assertEqual([0,1,0],rwd)
        self.assertEqual(1,rwd.max())
        self.assertEqual(1,rwd.argmax())
        self.assertEqual(0,rwd.eval(0))
        self.assertEqual(1,rwd.eval(1))
        self.assertEqual(0,rwd.eval(2))
        self.assertEqual(0,rwd[0])
        self.assertEqual(1,rwd[1])
        self.assertEqual(0,rwd[2])


class SequenceFeedback_Tests(unittest.TestCase):
    def test_sequence(self):
        fb = SequenceFeedback([4,5,6])

        self.assertEqual(3,len(fb))
        self.assertEqual([4,5,6],fb)
        self.assertEqual(4,fb[0])
        self.assertEqual(4,fb.eval(0))
        self.assertEqual(5,fb.eval(1))
        self.assertEqual(6,fb.eval(2))

if __name__ == '__main__':
    unittest.main()