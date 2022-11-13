import unittest

from coba.environments import SafeEnvironment
from coba.environments import SimulatedInteraction, LoggedInteraction
from coba.environments import L1Reward, HammingReward, ScaleReward, BinaryReward, SequenceReward
from coba.environments import SequenceFeedback, MulticlassReward
from coba.exceptions import CobaException
from coba.pipes import Pipes, Shuffle

class DummyEnvironment:

    def read(self):
        return []

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
        interaction = LoggedInteraction(1,2,3,1/2,[1,2,3])
        self.assertEqual([0,6,0], interaction.rewards)

    def test_IPS_function(self):
        interaction = LoggedInteraction(1,2,3,1/2,1)
        self.assertEqual([0,6,0], [interaction.rewards(a) for a in [1,2,3]])

    def test_simple_with_actions(self):
        interaction = LoggedInteraction(1, 2, 3, .2, [1,2,3])

        self.assertEqual(1, interaction.context)
        self.assertEqual(2, interaction.action)
        self.assertEqual(3, interaction.reward)
        self.assertEqual(.2, interaction.probability)
        self.assertEqual([1,2,3], interaction.actions)

    def test_simple_sans_actions(self):
        interaction = LoggedInteraction(1, 2, 3, .2)

        self.assertEqual(1, interaction.context)
        self.assertEqual(2, interaction.action)
        self.assertEqual(3, interaction.reward)
        self.assertEqual(.2, interaction.probability)

class SimulatedInteraction_Tests(unittest.TestCase):
    def test_context_none(self):
        self.assertEqual(None, SimulatedInteraction(None, (1,2,3), (4,5,6)).context)

    def test_context_str(self):
        self.assertEqual("A", SimulatedInteraction("A", (1,2,3), (4,5,6)).context)

    def test_context_dense(self):
        self.assertEqual((1,2,3), SimulatedInteraction((1,2,3), (1,2,3), (4,5,6)).context)

    def test_context_dense_2(self):
        self.assertEqual((1,2,3,(0,0,1)), SimulatedInteraction((1,2,3,(0,0,1)), (1,2,3), (4,5,6)).context)

    def test_context_sparse_dict(self):
        self.assertEqual({1:0}, SimulatedInteraction({1:0}, (1,2,3), (4,5,6)).context)

    def test_actions_correct_1(self) -> None:
        self.assertSequenceEqual([1,2], SimulatedInteraction(None, [1,2], [1,2]).actions)

    def test_actions_correct_2(self) -> None:
        self.assertSequenceEqual(["A","B"], SimulatedInteraction(None, ["A","B"], [1,2]).actions)

    def test_actions_correct_3(self) -> None:
        self.assertSequenceEqual([(1,2), (3,4)], SimulatedInteraction(None, [(1,2), (3,4)], [1,2]).actions)

    def test_rewards_correct(self):
        self.assertEqual([4,5,6], SimulatedInteraction((1,2), (1,2,3), [4,5,6]).rewards)

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

    def test_not_sequence(self):
        rwd = HammingReward(1)
        self.assertEqual(1 , rwd.argmax())
        self.assertEqual(0, rwd.eval(2))
        self.assertEqual(1, rwd.eval(1))
        self.assertEqual(0, rwd.eval(0))

    def test_sequence(self):
        rwd = HammingReward([1,2,3,4])
        self.assertEqual({1,2,3,4}, rwd.argmax())
        self.assertEqual(2/4, rwd.eval([1,3]))
        self.assertEqual(1/4, rwd.eval([4]))
        self.assertEqual(0  , rwd.eval([5,6,7]))

    def test_tuple(self):
        rwd = HammingReward((1,2,3,4))
        self.assertEqual((1,2,3,4), rwd.argmax())
        self.assertEqual(0, rwd.eval([1,3]))
        self.assertEqual(0, rwd.eval([4]))
        self.assertEqual(1, rwd.eval((1,2,3,4)))

    def test_string(self):
        rwd = HammingReward("abc")
        self.assertEqual("abc", rwd.argmax())
        self.assertEqual(0, rwd.eval('a'))
        self.assertEqual(0, rwd.eval('ab'))
        self.assertEqual(1, rwd.eval('abc'))

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
        rwd = SequenceReward([1,2,3],[4,5,6])

        self.assertEqual(3,len(rwd))
        self.assertEqual([4,5,6],rwd)
        self.assertEqual(6,rwd.max())
        self.assertEqual(3,rwd.argmax())
        self.assertEqual(4,rwd.eval(1))
        self.assertEqual(5,rwd.eval(2))
        self.assertEqual(6,rwd.eval(3))

class MulticlassReward_Tests(unittest.TestCase):
    def test_simple(self):
        rwd = MulticlassReward([1,2,3],2)

        self.assertEqual(3,len(rwd))
        self.assertEqual([0,1,0],rwd)
        self.assertEqual(1,rwd.max())
        self.assertEqual(2,rwd.argmax())
        self.assertEqual(0,rwd.eval(1))
        self.assertEqual(1,rwd.eval(2))
        self.assertEqual(0,rwd.eval(3))

class SequenceFeedback_Tests(unittest.TestCase):
    def test_sequence(self):
        fb = SequenceFeedback([1,2,3],[4,5,6])

        self.assertEqual(3,len(fb))
        self.assertEqual([4,5,6],fb)
        self.assertEqual(4,fb.eval(1))
        self.assertEqual(5,fb.eval(2))
        self.assertEqual(6,fb.eval(3))

if __name__ == '__main__':
    unittest.main()