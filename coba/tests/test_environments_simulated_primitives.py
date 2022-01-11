import unittest


from coba.contexts     import CobaContext, NullLogger
from coba.environments import SimulatedInteraction, MemorySimulation

CobaContext.logger = NullLogger()

class SimulatedInteraction_Tests(unittest.TestCase):
    def test_context_none(self):
        interaction = SimulatedInteraction(None, (1,2,3), rewards=(4,5,6))

        self.assertEqual(None, interaction.context)

    def test_context_str(self):
        interaction = SimulatedInteraction("A", (1,2,3), rewards=(4,5,6))

        self.assertEqual("A", interaction.context)

    def test_context_dense(self):
        interaction = SimulatedInteraction((1,2,3), (1,2,3), rewards=(4,5,6))

        self.assertEqual((1,2,3), interaction.context)

    def test_context_dense_2(self):
        interaction = SimulatedInteraction((1,2,3,(0,0,1)), (1,2,3), rewards=(4,5,6))

        self.assertEqual((1,2,3,(0,0,1)), interaction.context)

    def test_context_sparse_dict(self):
        interaction = SimulatedInteraction({1:0}, (1,2,3), rewards=(4,5,6))

        self.assertEqual({1:0}, interaction.context)

    def test_actions_correct_1(self) -> None:
        self.assertSequenceEqual([1,2], SimulatedInteraction(None, [1,2], rewards=[1,2]).actions)

    def test_actions_correct_2(self) -> None:
        self.assertSequenceEqual(["A","B"], SimulatedInteraction(None, ["A","B"], rewards=[1,2]).actions)

    def test_actions_correct_3(self) -> None:
        self.assertSequenceEqual([(1,2), (3,4)], SimulatedInteraction(None, [(1,2), (3,4)], rewards=[1,2]).actions)

    def test_custom_rewards(self):
        interaction = SimulatedInteraction((1,2), (1,2,3), rewards=[4,5,6])

        self.assertEqual((1,2), interaction.context)
        self.assertCountEqual((1,2,3), interaction.actions)
        self.assertEqual({"rewards":[4,5,6] }, interaction.kwargs)

    def test_reveals_results(self):
        interaction = SimulatedInteraction((1,2), (1,2,3), reveals=[(1,2),(3,4),(5,6)],rewards=[4,5,6])

        self.assertEqual((1,2), interaction.context)
        self.assertCountEqual((1,2,3), interaction.actions)
        self.assertEqual({"reveals":[(1,2),(3,4),(5,6)], "rewards":[4,5,6]}, interaction.kwargs)

class MemorySimulation_Tests(unittest.TestCase):

    def test_interactions(self):
        simulation   = MemorySimulation([SimulatedInteraction(1, [1,2,3], rewards=[0,1,2]), SimulatedInteraction(2, [4,5,6], rewards=[2,3,4])])
        interactions = list(simulation.read())

        self.assertEqual(interactions[0], interactions[0])
        self.assertEqual(interactions[1], interactions[1])

    def test_params(self):
        self.assertEqual({}, MemorySimulation([]).params)

    def test_str(self):
        self.assertEqual("MemorySimulation", str(MemorySimulation([])))

if __name__ == '__main__':
    unittest.main()
