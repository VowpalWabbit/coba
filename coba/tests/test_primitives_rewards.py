import unittest
import pickle

from coba.utilities import PackageChecker
from coba.primitives import L1Reward, HammingReward, BinaryReward, DiscreteReward

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

if __name__ == '__main__':
    unittest.main()
