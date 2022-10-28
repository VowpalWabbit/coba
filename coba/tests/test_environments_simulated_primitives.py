import unittest

from coba.exceptions   import CobaException
from coba.contexts     import CobaContext, NullLogger
from coba.environments import SimulatedInteraction

CobaContext.logger = NullLogger()

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

if __name__ == '__main__':
    unittest.main()
