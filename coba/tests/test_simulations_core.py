import unittest

from typing import List, Sequence, Tuple, Optional

from coba.pipes import MemorySource
from coba.config import CobaConfig, NoneLogger
from coba.simulations import (
    Key, Action, Context, Interaction, MemorySimulation, ClassificationSimulation, LambdaSimulation, CsvSimulation, MemoryReward
)

CobaConfig.Logger = NoneLogger()

def _choices(interaction: Interaction) -> Sequence[Tuple[Key, Optional[Context], Action]]:
    return [  (interaction.key, interaction.context, a) for a in interaction.actions]

class ClassificationSimulation_Tests(unittest.TestCase):

    def assert_simulation_for_data(self, simulation, features, answers) -> None:

        self.assertEqual(len(simulation.interactions), len(features))

        answers = simulation.one_hot_encoder.encode(answers)

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
        source = MemorySource(['a,b,c','1,2,3','4,5,6'])
        simulation = CsvSimulation(source,'c',).read()

        self.assertEqual(2, len(simulation.interactions))
        
        self.assertEqual(('1','2'), simulation.interactions[0].context)
        self.assertEqual(('4','5'), simulation.interactions[1].context)

        self.assertEqual([(1,0),(0,1)], simulation.interactions[0].actions)
        self.assertEqual([(1,0),(0,1)], simulation.interactions[1].actions)

        self.assertEqual([1,0], simulation.reward.observe( [(0, ('1','2'), (1,0)), (0, ('1','2'), (0,1) )] ))
        self.assertEqual([0,1], simulation.reward.observe( [(1, ('4','5'), (1,0)), (1, ('4','5'), (0,1) )] ))

if __name__ == '__main__':
    unittest.main()