import unittest

from typing import List

from coba.pipes import MemoryIO
from coba.config import CobaConfig, NullLogger
from coba.environments import (
    SimulatedInteraction, MemorySimulation, ClassificationSimulation,
    LambdaSimulation, CsvSimulation, ArffSimulation, LibsvmSimulation,
    ValidationSimulation
)

CobaConfig.logger = NullLogger()

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

class ClassificationSimulation_Tests(unittest.TestCase):

    def assert_simulation_for_data(self, simulation, features, answers) -> None:

        interactions = list(simulation.read())

        self.assertEqual(len(interactions), len(features))

        #first we make sure that all the labels are included 
        #in the first interactions actions without any concern for order
        self.assertCountEqual(interactions[0].actions, set(answers))

        #then we set our expected actions to the first interaction
        #to make sure that every interaction has the exact same actions
        #with the exact same order
        expected_actions = interactions[0].actions

        for f,l,i in zip(features, answers, interactions):

            expected_context = f
            expected_rewards = [ int(a == l) for a in i.actions]

            actual_context = i.context
            actual_actions = i.actions
            actual_rewards = i.kwargs["rewards"]

            self.assertEqual(actual_context, expected_context)            
            self.assertSequenceEqual(actual_actions, expected_actions)
            self.assertSequenceEqual(actual_rewards, expected_rewards)

    def test_constructor_with_good_features_and_labels1(self) -> None:
        features   = [1,2,3,4]
        labels     = [1,1,0,0]
        simulation = ClassificationSimulation(zip(features, labels))

        self.assert_simulation_for_data(simulation, features, labels)
    
    def test_constructor_with_good_features_and_labels2(self) -> None:
        features   = ["a","b"]
        labels     = ["good","bad"]
        simulation = ClassificationSimulation(zip(features, labels))

        self.assert_simulation_for_data(simulation, features, labels)

    def test_constructor_with_good_features_and_labels3(self) -> None:
        features   = [(1,2),(3,4)]
        labels     = ["good","bad"]
        simulation = ClassificationSimulation(zip(features, labels))

        self.assert_simulation_for_data(simulation, features, labels)

    def test_sparse(self) -> None:
        feature_rows = [
            {0:10, 1:11},
            {1:20, 2:30},
            {2:30, 3:40},
            {2:30, 3:40}
        ]

        label_column = [1,1,0,2]

        simulation   = ClassificationSimulation(zip(feature_rows, label_column))
        interactions = list(simulation.read())

        self.assertEqual(feature_rows[0], interactions[0].context)
        self.assertEqual(feature_rows[1], interactions[1].context)
        self.assertEqual(feature_rows[2], interactions[2].context)
        self.assertEqual(feature_rows[3], interactions[3].context)

        self.assertEqual([0,2,1], interactions[0].actions)
        self.assertEqual([0,2,1], interactions[1].actions)
        self.assertEqual([0,2,1], interactions[2].actions)
        self.assertEqual([0,2,1], interactions[3].actions)

        self.assertEqual([0,0,1], interactions[0].kwargs["rewards"])
        self.assertEqual([0,0,1], interactions[1].kwargs["rewards"])
        self.assertEqual([1,0,0], interactions[2].kwargs["rewards"])
        self.assertEqual([0,1,0], interactions[3].kwargs["rewards"])

class MemorySimulation_Tests(unittest.TestCase):

    def test_interactions(self):
        simulation   = MemorySimulation([SimulatedInteraction(1, [1,2,3], rewards=[0,1,2]), SimulatedInteraction(2, [4,5,6], rewards=[2,3,4])])
        interactions = list(simulation.read())

        self.assertEqual(interactions[0], interactions[0])
        self.assertEqual(interactions[1], interactions[1])

class LambdaSimulation_Tests(unittest.TestCase):

    def test_interactions(self):
        
        def C(i:int) -> int:
            return [1,2][i]

        def A(i:int,c:int) -> List[int]:
            return [[1,2,3],[4,5,6]][i]
        
        def R(i:int,c:int,a:int) -> int:
            return a-c

        simulation = LambdaSimulation(2,C,A,R)
        interactions = list(simulation.read())

        self.assertEqual(1      , interactions[0].context)
        self.assertEqual([1,2,3], interactions[0].actions)
        self.assertEqual([0,1,2], interactions[0].kwargs["rewards"])

        self.assertEqual(2      , interactions[1].context)
        self.assertEqual([4,5,6], interactions[1].actions)
        self.assertEqual([2,3,4], interactions[1].kwargs["rewards"])

    def test_interactions_len(self):
        def C(i:int) -> int:
            return [1,2][i]

        def A(i:int,c:int) -> List[int]:
            return [[1,2,3],[4,5,6]][i]

        def R(i:int,c:int,a:int) -> int:
            return a-c

        simulation = LambdaSimulation(2,C,A,R)
        interactions = list(simulation.read())
        self.assertEqual(len(interactions), 2)

class ValidationSimulation_Tests(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(500, len(list(ValidationSimulation().read())))

class CsvSimulation_Tests(unittest.TestCase):

    def test_simple(self):
        source       = MemoryIO(['a,b,c','1,2,3','4,5,6','7,8,6'])
        simulation   = CsvSimulation(source,'c')
        interactions = list(simulation.read())

        self.assertEqual(3, len(interactions))
        
        self.assertEqual(('1','2'), interactions[0].context)
        self.assertEqual(('4','5'), interactions[1].context)
        self.assertEqual(('7','8'), interactions[2].context)

        self.assertEqual(['3','6'], interactions[0].actions)
        self.assertEqual(['3','6'], interactions[1].actions)

        self.assertEqual([1,0], interactions[0].kwargs["rewards"])
        self.assertEqual([0,1], interactions[1].kwargs["rewards"])

class ArffSimulation_Tests(unittest.TestCase):

    def test_simple(self):

        lines = [
            "@relation news20",
            "@attribute a numeric",
            "@attribute B numeric",
            "@attribute c {0, class_B, class_C, class_D}",
            "@data",
            "1,2,class_B",
            "2,3,0",
        ]

        source       = MemoryIO(lines)
        simulation   = ArffSimulation(source,'c')
        interactions = list(simulation.read())

        self.assertEqual(2, len(interactions))
        
        self.assertEqual((1,2), interactions[0].context)
        self.assertEqual((2,3), interactions[1].context)

        self.assertEqual(['0','class_B'], interactions[0].actions)
        self.assertEqual(['0','class_B'], interactions[1].actions)

        self.assertEqual([0,1], interactions[0].kwargs["rewards"])
        self.assertEqual([1,0], interactions[1].kwargs["rewards"])

    def test_one_hot(self):

        lines = [
            "@relation news20",
            "@attribute a numeric",
            "@attribute B {0, 1, 2, 3}",
            "@attribute c {0, class_B, class_C, class_D}",
            "@data",
            "1,2,class_B",
            "2,3,0",
            "3,1,0"
        ]

        source       = MemoryIO(lines)
        simulation   = ArffSimulation(source,'c',)
        interactions = list(simulation.read())

        self.assertEqual(3, len(interactions))
        
        self.assertEqual((1,(0,0,1,0)), interactions[0].context)
        self.assertEqual((2,(0,0,0,1)), interactions[1].context)
        self.assertEqual((3,(0,1,0,0)), interactions[2].context)

        self.assertEqual(['0','class_B'], interactions[0].actions)
        self.assertEqual(['0','class_B'], interactions[1].actions)
        self.assertEqual(['0','class_B'], interactions[2].actions)

        self.assertEqual([0,1], interactions[0].kwargs["rewards"])
        self.assertEqual([1,0], interactions[1].kwargs["rewards"])
        self.assertEqual([1,0], interactions[2].kwargs["rewards"])

class LibsvmSimulation_Tests(unittest.TestCase):
    
    def test_simple(self):

        lines = [
            "0 4:2 5:3",
            "1 1:1 2:1",
            "1 3:4"
        ]

        source       = MemoryIO(lines)
        simulation   = LibsvmSimulation(source)
        interactions = list(simulation.read())

        self.assertEqual(3, len(interactions))

        self.assertEqual({4:2,5:3}, interactions[0].context)
        self.assertEqual({1:1,2:1}, interactions[1].context)
        self.assertEqual({3:4    }, interactions[2].context)

        self.assertEqual(['0','1'], interactions[0].actions)
        self.assertEqual(['0','1'], interactions[1].actions)

        self.assertEqual([1,0], interactions[0].kwargs["rewards"])
        self.assertEqual([0,1], interactions[1].kwargs["rewards"])

if __name__ == '__main__':
    unittest.main()