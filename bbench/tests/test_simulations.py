import unittest
import itertools
import random
import io

from itertools import cycle
from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple

from bbench.simulations import State, Round, Simulation, ClassificationSimulation, MemorySimulation, LambdaSimulation, ShuffleSimulation

class Simulation_Interface_Tests(ABC):

    @abstractmethod
    def _make_simulation(self) -> Tuple[Simulation, List[Round]]:
        ...

    def test_rounds_is_correct(self) -> None:
        #pylint: disable=no-member

        simulation, expected_rounds = self._make_simulation()

        actual_rounds = list(simulation.rounds)

        self.assertEqual(len(actual_rounds), len(expected_rounds)) #type: ignore

        for actual_round, expected_round in zip(actual_rounds, expected_rounds):
            self.assertEqual(actual_round.state  , expected_round.state  ) #type: ignore
            self.assertEqual(actual_round.actions, expected_round.actions) #type: ignore
            self.assertEqual(actual_round.rewards, expected_round.rewards) #type: ignore

    def test_rounds_is_reiterable(self) -> None:
        #pylint: disable=no-member

        simulation, _ = self._make_simulation()

        for round1,round2 in zip(simulation.rounds, simulation.rounds):
            self.assertEqual(round1.state  , round2.state  ) #type: ignore
            self.assertEqual(round1.actions, round2.actions) #type: ignore
            self.assertEqual(round1.rewards, round2.rewards) #type: ignore

    def test_rounds_is_readonly(self) -> None:
        #pylint: disable=no-member

        simulation, _ = self._make_simulation()

        with self.assertRaises(AttributeError): #type: ignore
            simulation.rounds = []

class Round_Tests(unittest.TestCase):

    def test_constructor_no_state(self) -> None:
        Round(None, [1, 2], [1, 0])

    def test_constructor_state(self) -> None:
        Round((1,2,3,4), [1, 2], [1, 0])

    def test_constructor_mismatch_actions_rewards_1(self) -> None:
        with self.assertRaises(AssertionError):
            Round(None, [1, 2, 3], [1, 0])
   
    def test_constructor_mismatch_actions_rewards_2(self) -> None:
        with self.assertRaises(AssertionError): 
            Round(None, [1, 2], [1, 0, 2])

    def test_state_correct_1(self) -> None:
        self.assertEqual(None, Round(None, [1, 2], [1, 0]).state)

    def test_actions_correct_1(self) -> None:
        self.assertEqual([1, 2], Round(None, [1, 2], [1, 0]).actions)

    def test_actions_correct_2(self) -> None:
        self.assertEqual(["A", "B"], Round(None, ["A", "B"], [1, 0]).actions)

    def test_actions_correct_3(self) -> None:
        self.assertEqual([(1,2), (3,4)], Round(None, [(1,2), (3,4)], [1, 0]).actions)

    def test_rewards_correct(self) -> None:
        self.assertEqual([1, 0], Round(None, [1, 2], [1, 0]).rewards)
    
    def test_actions_readonly(self) -> None:
        with self.assertRaises(AttributeError):
            Round(None, [[1],[2],[3]], [1, 0, 1]).actions = [2,0,1]
    
    def test_rewards_readonly(self) -> None:
        with self.assertRaises(AttributeError):
            Round(None, [[1],[2],[3]], [1, 0, 1]).rewards = [2,0,1]

class ClassificationSimulation_Tests(Simulation_Interface_Tests, unittest.TestCase):
    def _make_simulation(self) -> Tuple[Simulation, List[Round]]:
        rounds = [Round(1, [1,2], [0,1]), Round(2, [1,2], [1,0]) ]
        return ClassificationSimulation([1,2], [2,1]), rounds

    def assert_simulation_for_data(self, simulation, features, labels) -> None:

        self.assertEqual(sum(1 for _ in simulation.rounds), len(features))

        for f,l,r in zip(features, labels, simulation.rounds):

            expected_state   = f
            expected_actions = sorted(list(set(labels)))
            expected_rewards = [int(a == l) for a in r.actions]

            self.assertEqual(r.state, expected_state)            
            self.assertEqual(r.actions, expected_actions)
            self.assertEqual(r.rewards, expected_rewards)

    def test_constructor_with_good_features_and_labels1(self) -> None:
        features = [1,2,3,4]
        labels   = [1,1,0,0]
        simulation     = ClassificationSimulation(features, labels)

        self.assert_simulation_for_data(simulation, features, labels)
    
    def test_constructor_with_good_features_and_labels2(self) -> None:
        features = ["a","b"]
        labels   = ["good","bad"]
        simulation     = ClassificationSimulation(features, labels)

        self.assert_simulation_for_data(simulation, features, labels)

    def test_constructor_with_good_features_and_labels3(self) -> None:
        features = [(1,2),(3,4)]
        labels   = ["good","bad"]
        simulation     = ClassificationSimulation(features, labels)

        self.assert_simulation_for_data(simulation, features, labels)
    
    def test_constructor_with_too_few_features(self) -> None:
        with self.assertRaises(AssertionError): 
            ClassificationSimulation([1], [1,1])

    def test_constructor_with_too_few_labels(self) -> None:
        with self.assertRaises(AssertionError): 
            ClassificationSimulation([1,1], [1])

    def test_simple_from_csv_rows(self) -> None:

        simulation = ClassificationSimulation.from_csv_rows([['a','b','c'],['1','2','3'],['4','5','6']],'b')

        self.assert_simulation_for_data(simulation, [('1','3'),('4','6')],('2','5'))

    def test_simple_from_csv_rows_with_stater(self) -> None:

        rows = [['a','b','c'],['s1','2','3'],['s2','5','6']]
        stater = lambda row: (row[0] == "s1", row[0] == "s2", int(row[1]))

        simulation = ClassificationSimulation.from_csv_rows(rows, 'b', csv_stater=stater)

        self.assert_simulation_for_data(simulation, [(1,0,3),(0,1,6)], ['2','5'])

    def test_simple_from_openml(self) -> None:
        #this test requires interet acess to download the data

        simulation = ClassificationSimulation.from_openml(1116)

        self.assertEqual(len(simulation.rounds), 6598)
        
        for rnd in simulation.rounds:
            hash(rnd.state)      #make sure these are hashable
            hash(rnd.actions[0]) #make sure these are hashable
            hash(rnd.actions[1]) #make sure these are hashable
            self.assertEqual(len(rnd.state), 268) #type: ignore #(in this case we know rnd.state will be sizable)
            self.assertIn('0', rnd.actions)
            self.assertIn('1', rnd.actions)
            self.assertEqual(len(rnd.actions),2)
            self.assertIn(1, rnd.rewards)
            self.assertIn(0, rnd.rewards)
            self.assertEqual(len(rnd.rewards),2)

class MemorySimulation_Tests(Simulation_Interface_Tests, unittest.TestCase):

    def _make_simulation(self) -> Tuple[Simulation, List[Round]]:
        
        rounds = [Round(1, [1,2,3], [0,1,2]), Round(2, [4,5,6], [2,3,4])]
        return MemorySimulation(rounds), rounds

class LambdaSimulation_Tests(Simulation_Interface_Tests, unittest.TestCase):
    
    def _make_simulation(self) -> Tuple[Simulation, List[Round]]:
        rounds = [Round(1, [1,2,3], [0,1,2]), Round(2, [4,5,6], [2,3,4])]
        
        def S(i:int) -> int:
            return [1,2][i]

        def A(s:int) -> List[int]:
            return [1,2,3] if s == 1 else [4,5,6]
        
        def R(s:int,a:int) -> int:
            return a-s

        return LambdaSimulation(2,S,A,R), rounds

    def test_correct_number_of_rounds_created(self):
        def S(i:int) -> int:
            return [1,2][i]

        def A(s:int) -> List[int]:
            return [1,2,3] if s == 1 else [4,5,6]
        
        def R(s:int,a:int) -> int:
            return a-s

        simulation = LambdaSimulation(2,S,A,R)

        self.assertEqual(len(simulation.rounds), 2)

class ShuffleSimulation_Tests(Simulation_Interface_Tests, unittest.TestCase):
    
    def _make_simulation(self) -> Tuple[Simulation, List[Round]]:
        rounds = [Round(1, [1,2,3], [0,1,2]), Round(1, [1,2,3], [0,1,2])]

        return ShuffleSimulation(MemorySimulation(rounds)), rounds

    def test_rounds_not_duplicated_in_memory(self):
        rounds = [Round(1, [1,2,3], [0,1,2]), Round(2, [4,5,6], [2,3,4])]

        simulation = ShuffleSimulation(MemorySimulation(rounds))

        self.assertEqual(len(simulation.rounds),2)

        self.assertEqual(sum(1 for r in simulation.rounds if r.state == 1),1)
        self.assertEqual(sum(1 for r in simulation.rounds if r.state == 2),1)

        simulation.rounds[0]._state = 3
        simulation.rounds[1]._state = 3

        self.assertEqual(sum(1 for r in simulation.rounds if r.state == 3),2)
    
if __name__ == '__main__':
    unittest.main()