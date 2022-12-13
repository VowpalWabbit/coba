import unittest

from coba.primitives import L1Reward, HammingReward, ScaleReward, BinaryReward, SequenceReward, MulticlassReward, BatchReward
from coba.exceptions import CobaException

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

class BatchReward_Tests(unittest.TestCase):
    def test_eval(self):
        rwd = BatchReward([SequenceReward([4,5,6]),SequenceReward([7,8,9])])
        self.assertEqual(rwd.eval([1,2]), [5,9])

    def test_argmax(self):
        rwd = BatchReward([SequenceReward([4,5,6]),SequenceReward([7,8,9])])
        self.assertEqual(rwd.argmax(), [2,2])

    def test_max(self):
        rwd = BatchReward([SequenceReward([4,5,6]),SequenceReward([7,8,9])])
        self.assertEqual(rwd.max(), [6,9])

if __name__ == '__main__':
    unittest.main()
