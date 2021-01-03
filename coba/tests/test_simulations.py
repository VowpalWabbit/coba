import unittest

import timeit

from typing import List, Sequence, Tuple, cast

from coba.execution import ExecutionContext, NoneCache, NoneLogger, MemoryCache
from coba.data.encoders import OneHotEncoder
from coba.simulations import (
    Key, Choice, Interaction, ClassificationSimulation, MemorySimulation, LambdaSimulation, OpenmlSimulation, OpenmlClassificationSource
)

ExecutionContext.Logger = NoneLogger()

def _choices(interaction: Interaction) -> Sequence[Tuple[Key,Choice]]:
    return [  (interaction.key, a) for a in range(len(interaction.actions))]

class ClassificationSimulation_Tests(unittest.TestCase):

    def assert_simulation_for_data(self, simulation, features, answers) -> None:

        self.assertEqual(len(simulation.interactions), len(features))

        answers = OneHotEncoder(simulation.label_set).encode(answers)

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
            
            actual_rewards  = simulation.rewards(_choices(i))

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

    def test_simple_openml_source(self) -> None:
        #this test requires interet acess to download the data

        ExecutionContext.FileCache = NoneCache()

        simulation = ClassificationSimulation(*OpenmlClassificationSource(1116).read())
        #simulation = ClassificationSimulation.from_source(OpenmlSource(273))

        self.assertEqual(len(simulation.interactions), 6598)

        for rnd in simulation.interactions:

            hash(rnd.context)    #make sure these are hashable
            hash(rnd.actions[0]) #make sure these are hashable
            hash(rnd.actions[1]) #make sure these are hashable

            self.assertEqual(len(cast(Tuple,rnd.context)), 268)
            self.assertIn((1,0), rnd.actions)
            self.assertIn((0,1), rnd.actions)
            self.assertEqual(len(rnd.actions),2)
            
            actual_rewards  = simulation.rewards(_choices(rnd))

            self.assertIn(1, actual_rewards)
            self.assertIn(0, actual_rewards)

    @unittest.skip("much of what makes this openml set slow is now tested locally in `test_large_from_table`")
    def test_large_from_openml(self) -> None:
        #this test requires interet acess to download the data

        ExecutionContext.FileCache = MemoryCache()
        OpenmlClassificationSource(154).read() #this will cause it to read and cache in memory so we don't measure read time

        time = min(timeit.repeat(lambda:ClassificationSimulation(*OpenmlClassificationSource(154).read()), repeat=1, number=1))

        print(time)

        #with caching took approximately 17 seconds to encode
        self.assertLess(time, 30)

class MemorySimulation_Tests(unittest.TestCase):

    def test_interactions(self):
        interactions = [Interaction(1, [1,2,3], 0), Interaction(2,[4,5,6],1)]
        reward_sets  = [[0,1,2], [2,3,4]]

        simulation = MemorySimulation(interactions, reward_sets)

        self.assertEqual(1      , simulation.interactions[0].context)
        self.assertEqual([1,2,3], simulation.interactions[0].actions)
        self.assertEqual([0,1,2], simulation.rewards([(0,0),(0,1),(0,2)]))

        self.assertEqual(2      , simulation.interactions[1].context)
        self.assertEqual([4,5,6], simulation.interactions[1].actions)
        self.assertEqual([2,3,4], simulation.rewards([(1,0),(1,1),(1,2)]))

class LambdaSimulation_Tests(unittest.TestCase):

    def test_interactions(self):
        def C(t:int) -> int:
            return [1,2][t]

        def A(t:int) -> List[int]:
            return [[1,2,3],[4,5,6]][t]
        
        def R(c:int,a:int) -> int:
            return a-c

        with LambdaSimulation(2,C,A,R) as simulation:
            self.assertEqual(1      , simulation.interactions[0].context)
            self.assertEqual([1,2,3], simulation.interactions[0].actions)
            self.assertEqual([0,1,2], simulation.rewards([(0,0),(0,1),(0,2)]))

            self.assertEqual(2      , simulation.interactions[1].context)
            self.assertEqual([4,5,6], simulation.interactions[1].actions)
            self.assertEqual([2,3,4], simulation.rewards([(1,0),(1,1),(1,2)]))

    def test_interactions_len(self):
        def C(t:int) -> int:
            return [1,2][t]

        def A(t:int) -> List[int]:
            return [[1,2,3],[4,5,6]][t]

        def R(c:int,a:int) -> int:
            return a-c

        with LambdaSimulation(2,C,A,R) as simulation:
            self.assertEqual(len(simulation.interactions), 2)

class OpenmlSimulation_Tests(unittest.TestCase):

    def test_simple(self):
        simulation = OpenmlSimulation(150)
        self.assertTrue(True)

class Interaction_Tests(unittest.TestCase):

    def test_constructor_no_context(self) -> None:
        Interaction(None, [1, 2])

    def test_constructor_context(self) -> None:
        Interaction((1,2,3,4), [1, 2])

    def test_context_correct_1(self) -> None:
        self.assertEqual(None, Interaction(None, [1, 2]).context)

    def test_actions_correct_1(self) -> None:
        self.assertSequenceEqual([1, 2], Interaction(None, [1, 2]).actions)

    def test_actions_correct_2(self) -> None:
        self.assertSequenceEqual(["A", "B"], Interaction(None, ["A", "B"]).actions)

    def test_actions_correct_3(self) -> None:
        self.assertSequenceEqual([(1,2), (3,4)], Interaction(None, [(1,2), (3,4)]).actions)

if __name__ == '__main__':
    unittest.main()