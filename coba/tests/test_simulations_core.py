from coba.simulations.core import ArffSimulation
import unittest

from typing import Sequence, Tuple, Optional, List

from coba.pipes import MemorySource
from coba.config import CobaConfig, NoneLogger
from coba.simulations import (
    Key, Action, Context, Interaction, MemoryReward,
    MemorySimulation, ClassificationSimulation, LambdaSimulation, CsvSimulation, ArffSimulation, LibsvmSimulation
)

CobaConfig.Logger = NoneLogger()

def _choices(interaction: Interaction) -> Sequence[Tuple[Key, Optional[Context], Action]]:
    return [  (interaction.key, interaction.context, a) for a in interaction.actions]

class ClassificationSimulation_Tests(unittest.TestCase):

    def assert_simulation_for_data(self, simulation, features, answers) -> None:

        self.assertEqual(len(simulation.interactions), len(features))

        #first we make sure that all the labels are included 
        #in the first interactions actions without any concern for order
        self.assertCountEqual(simulation.interactions[0].actions, set(answers))

        #then we set our expected actions to the first interaction
        #to make sure that every interaction has the exact same actions
        #with the exact same order
        expected_actions = simulation.interactions[0].actions

        for f,l,i in zip(features, answers, simulation.interactions):

            expected_context = f
            expected_rewards = [ int(a == l) for a in i.actions]

            actual_context = i.context
            actual_actions = i.actions
            
            actual_rewards  = simulation.reward.observe(_choices(i))

            self.assertEqual(actual_context, expected_context)            
            self.assertSequenceEqual(actual_actions, expected_actions)
            self.assertSequenceEqual(actual_rewards, expected_rewards)

    def test_constructor_with_good_features_and_labels1(self) -> None:
        features   = [1,2,3,4]
        labels     = [1,1,0,0]
        simulation = ClassificationSimulation(features, labels)

        self.assert_simulation_for_data(simulation, features, labels)
    
    def test_constructor_with_good_features_and_labels2(self) -> None:
        features   = ["a","b"]
        labels     = ["good","bad"]
        simulation = ClassificationSimulation(features, labels)

        self.assert_simulation_for_data(simulation, features, labels)

    def test_constructor_with_good_features_and_labels3(self) -> None:
        features   = [(1,2),(3,4)]
        labels     = ["good","bad"]
        simulation = ClassificationSimulation(features, labels)

        self.assert_simulation_for_data(simulation, features, labels)

    def test_constructor_with_too_few_features(self) -> None:
        with self.assertRaises(AssertionError): 
            ClassificationSimulation([1], [1,1])

    def test_constructor_with_too_few_labels(self) -> None:
        with self.assertRaises(AssertionError): 
            ClassificationSimulation([1,1], [1])

    def test_sparse(self) -> None:
        feature_rows = [
            ( (0,1), (10,11) ),
            ( (1,2), (20,30) ),
            ( (2,3), (30,40) ),
            ( (2,3), (30,40) )
        ]

        label_column = (1,1,0,2)

        sim = ClassificationSimulation(feature_rows, label_column)

        self.assertEqual(feature_rows[0], sim.interactions[0].context)
        self.assertEqual(feature_rows[1], sim.interactions[1].context)
        self.assertEqual(feature_rows[2], sim.interactions[2].context)
        self.assertEqual(feature_rows[3], sim.interactions[3].context)

        self.assertEqual([1,0,2], sim.interactions[0].actions)
        self.assertEqual([1,0,2], sim.interactions[1].actions)
        self.assertEqual([1,0,2], sim.interactions[2].actions)
        self.assertEqual([1,0,2], sim.interactions[3].actions)

        self.assertEqual([1,0,0], sim.reward.observe(_choices(sim.interactions[0])))
        self.assertEqual([1,0,0], sim.reward.observe(_choices(sim.interactions[1])))
        self.assertEqual([0,1,0], sim.reward.observe(_choices(sim.interactions[2])))
        self.assertEqual([0,0,1], sim.reward.observe(_choices(sim.interactions[3])))

class MemorySimulation_Tests(unittest.TestCase):

    def test_interactions(self):
        interactions = [Interaction(0, 1, [1,2,3]), Interaction(1, 2, [4,5,6])]
        reward       = MemoryReward([ (0,1,0), (0,2,1), (0,3,2), (1,4,2), (1,5,3), (1,6,4) ])

        simulation = MemorySimulation(interactions, reward)

        self.assertEqual(interactions[0], simulation.interactions[0])
        self.assertEqual(interactions[1], simulation.interactions[1])
        self.assertEqual(reward         , simulation.reward)

class LambdaSimulation_Tests(unittest.TestCase):

    def test_interactions(self):
        
        def C(i:int) -> int:
            return [1,2][i]

        def A(i:int,c:int) -> List[int]:
            return [[1,2,3],[4,5,6]][i]
        
        def R(i:int,c:int,a:int) -> int:
            return a-c

        simulation = LambdaSimulation(2,C,A,R).read() #type: ignore

        self.assertEqual(1      , simulation.interactions[0].context)
        self.assertEqual([1,2,3], simulation.interactions[0].actions)
        self.assertEqual([0,1,2], simulation.reward.observe([(0,1,1),(0,1,2),(0,1,3)]))

        self.assertEqual(2      , simulation.interactions[1].context)
        self.assertEqual([4,5,6], simulation.interactions[1].actions)
        self.assertEqual([2,3,4], simulation.reward.observe([(1,1,4),(1,1,5),(1,1,6)]))

    def test_interactions_len(self):
        def C(i:int) -> int:
            return [1,2][i]

        def A(i:int,c:int) -> List[int]:
            return [[1,2,3],[4,5,6]][i]

        def R(i:int, c:int,a:int) -> int:
            return a-c

        simulation = LambdaSimulation(2,C,A,R).read() #type: ignore
        self.assertEqual(len(simulation.interactions), 2)

class CsvSimulation_Tests(unittest.TestCase):

    def test_simple(self):
        source = MemorySource(['a,b,c','1,2,3','4,5,6','7,8,6'])
        simulation = CsvSimulation(source,'c').read()

        self.assertEqual(3, len(simulation.interactions))
        
        self.assertEqual(('1','2'), simulation.interactions[0].context)
        self.assertEqual(('4','5'), simulation.interactions[1].context)
        self.assertEqual(('7','8'), simulation.interactions[2].context)

        self.assertEqual(['3','6'], simulation.interactions[0].actions)
        self.assertEqual(['3','6'], simulation.interactions[1].actions)

        self.assertEqual([1,0], simulation.reward.observe( _choices(simulation.interactions[0]) ))
        self.assertEqual([0,1], simulation.reward.observe( _choices(simulation.interactions[1]) ))

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

        source = MemorySource(lines)
        simulation = ArffSimulation(source,'c',).read()

        self.assertEqual(2, len(simulation.interactions))
        
        self.assertEqual((1,2), simulation.interactions[0].context)
        self.assertEqual((2,3), simulation.interactions[1].context)

        self.assertEqual(['class_B','0'], simulation.interactions[0].actions)
        self.assertEqual(['class_B','0'], simulation.interactions[1].actions)

        self.assertEqual([1,0], simulation.reward.observe( _choices(simulation.interactions[0]) ))
        self.assertEqual([0,1], simulation.reward.observe( _choices(simulation.interactions[1]) ))

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

        source     = MemorySource(lines)
        simulation = ArffSimulation(source,'c',).read()

        self.assertEqual(3, len(simulation.interactions))
        
        self.assertEqual((1,0,0,1,0), simulation.interactions[0].context)
        self.assertEqual((2,0,0,0,1), simulation.interactions[1].context)
        self.assertEqual((3,0,1,0,0), simulation.interactions[2].context)

        self.assertEqual(['class_B','0'], simulation.interactions[0].actions)
        self.assertEqual(['class_B','0'], simulation.interactions[1].actions)
        self.assertEqual(['class_B','0'], simulation.interactions[2].actions)

        self.assertEqual([1,0], simulation.reward.observe(_choices(simulation.interactions[0])))
        self.assertEqual([0,1], simulation.reward.observe(_choices(simulation.interactions[1])))
        self.assertEqual([0,1], simulation.reward.observe(_choices(simulation.interactions[2])))

class LibsvmSimulation_Tests(unittest.TestCase):
    
    def test_simple(self):

        lines = [
            "0 4:2 5:3",
            "1 1:1 2:1",
            "1 3:4"
        ]

        source     = MemorySource(lines)
        simulation = LibsvmSimulation(source).read()

        self.assertEqual(3, len(simulation.interactions))

        self.assertEqual(((0,1),(2,3)), simulation.interactions[0].context)
        self.assertEqual(((2,3),(1,1)), simulation.interactions[1].context)
        self.assertEqual(((4, ),(4, )), simulation.interactions[2].context)

        self.assertEqual(['0', '1'], simulation.interactions[0].actions)
        self.assertEqual(['0', '1'], simulation.interactions[1].actions)

        self.assertEqual([1,0], simulation.reward.observe( _choices(simulation.interactions[0]) ))
        self.assertEqual([0,1], simulation.reward.observe( _choices(simulation.interactions[1]) ))

if __name__ == '__main__':
    unittest.main()