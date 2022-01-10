import unittest
import pickle

from typing import List

from coba.exceptions   import CobaException
from coba.contexts     import CobaContext, NullLogger
from coba.environments import SimulatedInteraction, MemorySimulation, LambdaSimulation
from coba.random       import CobaRandom

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

class LambdaSimulation_Tests(unittest.TestCase):

    def test_n_interactions_2_seed_none(self):
        
        def C(i:int) -> int:
            return [1,2][i]

        def A(i:int,c:int) -> List[int]:
            return [[1,2,3],[4,5,6]][i]

        def R(i:int,c:int,a:int) -> int:
            return a-c

        simulation = LambdaSimulation(2,C,A,R)
        interactions = list(simulation.read())

        self.assertEqual(len(interactions), 2)

        self.assertEqual(1      , interactions[0].context)
        self.assertEqual([1,2,3], interactions[0].actions)
        self.assertEqual([0,1,2], interactions[0].kwargs["rewards"])

        self.assertEqual(2      , interactions[1].context)
        self.assertEqual([4,5,6], interactions[1].actions)
        self.assertEqual([2,3,4], interactions[1].kwargs["rewards"])

    def test_n_interactions_none_seed_none(self):
        def C(i:int) -> int:
            return [1,2][i]

        def A(i:int,c:int) -> List[int]:
            return [[1,2,3],[4,5,6]][i]

        def R(i:int,c:int,a:int) -> int:
            return a-c

        simulation   = LambdaSimulation(None,C,A,R)
        interactions = iter(simulation.read())

        interaction = next(interactions)

        self.assertEqual(1      , interaction.context)
        self.assertEqual([1,2,3], interaction.actions)
        self.assertEqual([0,1,2], interaction.kwargs["rewards"])

        interaction = next(interactions)

        self.assertEqual(2      , interaction.context)
        self.assertEqual([4,5,6], interaction.actions)
        self.assertEqual([2,3,4], interaction.kwargs["rewards"])

    def test_n_interactions_2_seed_1(self):
        
        def C(i:int, rng: CobaRandom) -> int:
            return [1,2][i]

        def A(i:int,c:int, rng: CobaRandom) -> List[int]:
            return [[1,2,3],[4,5,6]][i]

        def R(i:int,c:int,a:int, rng: CobaRandom) -> int:
            return a-c

        simulation = LambdaSimulation(2,C,A,R,seed=1)
        interactions = list(simulation.read())

        self.assertEqual(len(interactions), 2)

        self.assertEqual(1      , interactions[0].context)
        self.assertEqual([1,2,3], interactions[0].actions)
        self.assertEqual([0,1,2], interactions[0].kwargs["rewards"])

        self.assertEqual(2      , interactions[1].context)
        self.assertEqual([4,5,6], interactions[1].actions)
        self.assertEqual([2,3,4], interactions[1].kwargs["rewards"])

    def test_params(self):
        def C(i:int) -> int:
            return [1,2][i]

        def A(i:int,c:int) -> List[int]:
            return [[1,2,3],[4,5,6]][i]

        def R(i:int,c:int,a:int) -> int:
            return a-c

        self.assertEqual({}, LambdaSimulation(2,C,A,R).params)

    def test_pickle_n_interactions_2(self):
        def C(i:int) -> int:
            return [1,2][i]

        def A(i:int,c:int) -> List[int]:
            return [[1,2,3],[4,5,6]][i]
        
        def R(i:int,c:int,a:int) -> int:
            return a-c

        simulation = pickle.loads(pickle.dumps(LambdaSimulation(2,C,A,R)))
        interactions = list(simulation.read())

        self.assertEqual("LambdaSimulation",str(simulation))
        self.assertEqual({}, simulation.params)

        self.assertEqual(len(interactions), 2)

        self.assertEqual(1      , interactions[0].context)
        self.assertEqual([1,2,3], interactions[0].actions)
        self.assertEqual([0,1,2], interactions[0].kwargs["rewards"])

        self.assertEqual(2      , interactions[1].context)
        self.assertEqual([4,5,6], interactions[1].actions)
        self.assertEqual([2,3,4], interactions[1].kwargs["rewards"])

    def test_pickle_n_interactions_none(self):
        def C(i:int) -> int:
            return [1,2][i]

        def A(i:int,c:int) -> List[int]:
            return [[1,2,3],[4,5,6]][i]
        
        def R(i:int,c:int,a:int) -> int:
            return a-c

        with self.assertRaises(CobaException) as e:
            pickle.loads(pickle.dumps(LambdaSimulation(None,C,A,R)))
        
        self.assertIn("In general LambdaSimulation", str(e.exception))

if __name__ == '__main__':
    unittest.main()
