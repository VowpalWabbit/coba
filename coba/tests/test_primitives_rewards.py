import unittest
import pickle

from coba.primitives import L1Reward, HammingReward, BinaryReward, SequenceReward, MulticlassReward
from coba.primitives import BatchReward, IPSReward, MappingReward, argmax

class argmax_Tests(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(1,argmax([1,2,3],SequenceReward([1,2,3],[9,8,7])))

class IPSReward_Tests(unittest.TestCase):
    def test_eval(self):
        rwd = IPSReward(1,1,1/2)
        self.assertEqual(2,rwd.eval(1))
        self.assertEqual(0,rwd.eval(2))

    def test_eq(self):
        rwd1 = IPSReward(1,1,1/2)
        rwd2 = IPSReward(1,1,1/2)
        rwd3 = IPSReward(2,1,1/2)

        self.assertEqual(rwd1,rwd2)
        self.assertNotEqual(rwd1,rwd3)
        self.assertNotEqual(rwd1,1)

    def test_pickle(self):
        dumped = pickle.dumps(IPSReward(1,2,1))
        loaded = pickle.loads(dumped)

        self.assertIsInstance(loaded, IPSReward)
        self.assertEqual(loaded._reward,1)
        self.assertEqual(loaded._action,2)

class L1Reward_Tests(unittest.TestCase):

    def test_simple(self):
        rwd = L1Reward(1)

        self.assertEqual(-1, rwd.eval(2))
        self.assertEqual(0 , rwd.eval(1))
        self.assertEqual(-1, rwd.eval(0))

    def test_pickle(self):
        dumped = pickle.dumps(L1Reward(1))
        loaded = pickle.loads(dumped)

        self.assertIsInstance(loaded, L1Reward)
        self.assertEqual(loaded._label,1)

    def test_pickle_size(self):
        self.assertLess(len(pickle.dumps(L1Reward(1))), 80)

class BinaryReward_Tests(unittest.TestCase):

    def test_binary_argmax(self):
        rwd = BinaryReward(1)

        self.assertEqual(0, rwd.eval(2))
        self.assertEqual(1, rwd.eval(1))
        self.assertEqual(0, rwd.eval(0))
        self.assertEqual(rwd, rwd)
        self.assertEqual(1, rwd)

    def test_pickle(self):
        dumped = pickle.dumps(BinaryReward(1))
        loaded = pickle.loads(dumped)

        self.assertIsInstance(loaded, BinaryReward)
        self.assertEqual(loaded._maxarg,1)

    def test_pickle_size(self):
        self.assertLess(len(pickle.dumps(BinaryReward(1))), 80)

class HammingReward_Tests(unittest.TestCase):

    def test_sequence(self):
        rwd = HammingReward([1,2,3,4])

        self.assertEqual(2/4, rwd.eval([1,3]))
        self.assertEqual(1/4, rwd.eval([4]))
        self.assertEqual(0  , rwd.eval([5,6,7]))
        self.assertEqual(1/2, rwd.eval([1,2,3,4,5,6,7,8]))

    def test_tuple(self):
        rwd = HammingReward((1,2,3,4))
        self.assertEqual(.5, rwd.eval([1,3]))
        self.assertEqual(.25, rwd.eval([4]))
        self.assertEqual(1, rwd.eval((1,2,3,4)))

    def test_pickle(self):
        dumped = pickle.dumps(HammingReward([1,2,3]))
        loaded = pickle.loads(dumped)

        self.assertIsInstance(loaded, HammingReward)
        self.assertEqual(loaded._labels,{1,2,3})

    def test_pickle_size(self):
        self.assertLess(len(pickle.dumps(HammingReward([1,2]))), 80)

class SequenceReward_Tests(unittest.TestCase):
    def test_sequence(self):
        rwd = SequenceReward([1,2,3],[4,5,6])

        self.assertEqual([4,5,6],rwd)
        self.assertEqual(4,rwd.eval(1))
        self.assertEqual(5,rwd.eval(2))
        self.assertEqual(6,rwd.eval(3))
        self.assertEqual(rwd,rwd)

    def test_bad_eq(self):
        rwd = SequenceReward([1,2,3],[4,5,6])
        self.assertNotEqual(1,rwd)

    def test_pickle(self):
        dumped = pickle.dumps(SequenceReward([1,2,3],[4,5,6]))
        loaded = pickle.loads(dumped)

        self.assertIsInstance(loaded, SequenceReward)
        self.assertSequenceEqual(loaded._actions,[1,2,3])
        self.assertSequenceEqual(loaded._rewards,[4,5,6])

    def test_pickle_size(self):
        self.assertLess(len(pickle.dumps(SequenceReward([3,4],[5,6]))), 80)

class MappingReward_Tests(unittest.TestCase):
    def test_mapping(self):
        rwd = MappingReward({0:4,1:5,2:6})

        self.assertEqual(4,rwd.eval(0))
        self.assertEqual(5,rwd.eval(1))
        self.assertEqual(6,rwd.eval(2))
        self.assertEqual(rwd,rwd)

    def test_bad_eq(self):
        rwd = MappingReward({0:4,1:5,2:6})
        self.assertNotEqual(1,rwd)

    def test_pickle(self):
        dumped = pickle.dumps(MappingReward({0:4,1:5,2:6}))
        loaded = pickle.loads(dumped)

        self.assertIsInstance(loaded, MappingReward)
        self.assertEqual(loaded,MappingReward({0:4,1:5,2:6}))

    def test_pickle_size(self):
        self.assertLess(len(pickle.dumps(MappingReward({0:4,1:5,2:6}))), 80)

class MulticlassReward_Tests(unittest.TestCase):
    def test_simple(self):
        rwd = MulticlassReward(1)

        self.assertEqual(0,rwd.eval(0))
        self.assertEqual(1,rwd.eval(1))
        self.assertEqual(0,rwd.eval(2))

    def test_pickle(self):
        dumped = pickle.dumps(MulticlassReward(1))
        loaded = pickle.loads(dumped)

        self.assertIsInstance(loaded, MulticlassReward)
        self.assertEqual(loaded._label,1)

    def test_pickle_size(self):
        self.assertLess(len(pickle.dumps(MulticlassReward(1))), 80)

class BatchReward_Tests(unittest.TestCase):
    def test_eval_single(self):
        rwd = BatchReward([SequenceReward([0,1,2],[4,5,6]),SequenceReward([0,1,2],[7,8,9])])
        self.assertEqual(rwd.eval([1,2]), [5,9])

if __name__ == '__main__':
    unittest.main()
