import unittest

import timeit

from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple, cast, Dict

from coba.contexts import ExecutionContext, NoneCache
from coba.preprocessing import Metadata, NumericEncoder, OneHotEncoder, StringEncoder, Metadata, FactorEncoder
from coba.simulations import (
    State, Action, Reward, Interaction, Simulation, 
    ClassificationSimulation, MemorySimulation, 
    LambdaSimulation, ShuffleSimulation, LazySimulation
)

class Simulation_Interface_Tests(ABC):

    @abstractmethod
    def _make_simulation(self) -> Tuple[Simulation, Sequence[Interaction], Sequence[Sequence[Tuple[State,Action,Reward]]]]:
        ...

    def test_interactions_is_correct(self) -> None:

        simulation, expected_inters, expected_rwds = self._make_simulation()

        actual_inters = simulation.interactions

        cast(unittest.TestCase, self).assertEqual(len(actual_inters), len(expected_inters))

        for actual_inter, expected_inter, expected_rwd in zip(actual_inters, expected_inters, expected_rwds):

            actual_reward = simulation.rewards(actual_inter.choices)

            cast(unittest.TestCase, self).assertEqual(actual_inter.state, expected_inter.state)
            cast(unittest.TestCase, self).assertCountEqual(actual_inter.actions, expected_inter.actions)
            cast(unittest.TestCase, self).assertCountEqual(actual_reward, expected_rwd)

    def test_interactions_is_reiterable(self) -> None:

        simulation = self._make_simulation()[0]

        for interaction1,interaction2 in zip(simulation.interactions, simulation.interactions):

            interaction1_rewards = simulation.rewards(interaction1.choices)
            interaction2_rewards = simulation.rewards(interaction2.choices)

            cast(unittest.TestCase, self).assertEqual(interaction1.state, interaction2.state)
            cast(unittest.TestCase, self).assertSequenceEqual(interaction1.actions, interaction2.actions)
            cast(unittest.TestCase, self).assertSequenceEqual(interaction1_rewards, interaction2_rewards)

class Simulation_Tests(unittest.TestCase):

    def test_from_json(self):
        json_val = ''' {
            "seed": 1283,
            "type": "classification",
            "from": {
                "format"          : "table",
                "table"           : [["a","b","c"], ["s1","2","3"], ["s2","5","6"]],
                "has_header"      : true,
                "column_default"  : { "ignore":false, "label":false, "encoding":"onehot" },
                "column_overrides": { "b": { "label":true, "encoding":"string" } }
            }
        } '''

        simulation = Simulation.from_json(json_val)

        self.assertEqual(len(simulation.interactions), 2)

class ClassificationSimulation_Tests(Simulation_Interface_Tests, unittest.TestCase):

    def _make_simulation(self) -> Tuple[Simulation, Sequence[Interaction], Sequence[Sequence[Tuple[State,Action,Reward]]]]:
        states      =  [1,2]
        action_sets = [[1,2], [1,2]]
        reward_sets = [[0,1], [1,0]]

        states_actions_rewards = zip(states, action_sets, reward_sets)

        expected_interactions  = list(map(Interaction[int,int], states, action_sets))
        expected_rewards = [ [(s,a,r) for a,r in zip(A,R)] for s, A, R in states_actions_rewards]

        return ClassificationSimulation([1,2], [2,1]), expected_interactions, expected_rewards

    def assert_simulation_for_data(self, simulation, features, labels) -> None:

        self.assertEqual(len(simulation.interactions), len(features))

        #first we make sure that all the labels are included 
        #in the first interactions actions without any concern for order
        self.assertCountEqual(simulation.interactions[0].actions, tuple(set(labels)))

        #then we set our expected actions to the first interaction
        #to make sure that every interaction has the exact same actions
        #with the exact same order
        expected_actions = simulation.interactions[0].actions

        for f,l,r in zip(features, labels, simulation.interactions):

            expected_state   = f
            expected_rewards = [ (f, a, int(a == l)) for a in r.actions]

            actual_state   = r.state
            actual_actions = r.actions
            
            actual_reward  = simulation.rewards(r.choices)

            self.assertEqual(actual_state, expected_state)            
            self.assertSequenceEqual(actual_actions, expected_actions)
            self.assertSequenceEqual(actual_reward, expected_rewards)

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

    def test_from_table_inferred_numeric(self) -> None:
        label_column = 'b'
        table        = [['a','b','c'],
                        ['1','2','3'],
                        ['4','5','6']]

        simulation = ClassificationSimulation.from_table(table,label_column)

        self.assert_simulation_for_data(simulation, [(1,3),(4,6)],[2,5])

    def test_from_table_inferred_onehot(self) -> None:
        label_column = 'b'
        table        = [['a' ,'b','c'],
                        ['s1','2','3'],
                        ['s2','5','6']]

        simulation = ClassificationSimulation.from_table(table, label_column)

        self.assert_simulation_for_data(simulation, [((1,),3),((0,),6)], [2,5])

    def test_from_table_explicit_onehot(self) -> None:
        default_meta = Metadata(False, False, OneHotEncoder())
        defined_meta = {'b': Metadata(None, True, StringEncoder()) }
        table        = [['a' ,'b','c'],
                        ['s1','2','3'],
                        ['s2','5','6']]

        simulation = ClassificationSimulation.from_table(table, default_meta=default_meta, defined_meta=defined_meta)

        self.assert_simulation_for_data(simulation, [(1,1),(0,0)], ['2','5'])

    def test_simple_from_openml(self) -> None:
        #this test requires interet acess to download the data

        ExecutionContext.FileCache = NoneCache()

        simulation = ClassificationSimulation.from_openml(1116)
        #simulation = ClassificationSimulation.from_openml(273)

        self.assertEqual(len(simulation.interactions), 6598)

        for rnd in simulation.interactions:

            hash(rnd.state)      #make sure these are hashable
            hash(rnd.actions[0]) #make sure these are hashable
            hash(rnd.actions[1]) #make sure these are hashable

            self.assertEqual(len(cast(Tuple,rnd.state)), 167)
            self.assertIn(1, rnd.actions)
            self.assertIn(2, rnd.actions)
            self.assertEqual(len(rnd.actions),2)
            
            actual_rewards  = [ rwd[2] for rwd in simulation.rewards(rnd.choices) ]

            self.assertIn(1, actual_rewards)
            self.assertIn(0, actual_rewards)

    @unittest.skip("much of what makes this openml set slow is now tested locally in `test_large_from_table`")
    def test_large_from_openml(self) -> None:
        #this test requires interet acess to download the data

        ExecutionContext.FileCache = NoneCache()

        time = min(timeit.repeat(lambda:ClassificationSimulation.from_openml(154), repeat=1, number=1))

        #print(time)

        #was approximately 18 at best performance
        self.assertLess(time, 30)

    def test_large_from_table(self) -> None:

        table        = [["1","0"]*15]*100000
        label_col    = 0
        default_meta = Metadata(False,False, FactorEncoder(['1','0']))
        defined_meta = { 2:Metadata(None,None,NumericEncoder()), 5:Metadata(None,None,NumericEncoder()) }

        from_table = lambda:ClassificationSimulation.from_table(table, label_col, False, default_meta, defined_meta)

        time = min(timeit.repeat(from_table, repeat=2, number=1))

        print(time)

        #was approximately 0.6 at best performance
        self.assertLess(time, 3)

    def test_simple_from_csv(self) -> None:
        #this test requires interet acess to download the data

        ExecutionContext.FileCache = NoneCache()

        location     = "http://www.openml.org/data/v1/get_csv/53999"
        default_meta = Metadata(False, False, NumericEncoder())
        defined_meta: Dict[str,Metadata] = {
            "class"            : Metadata(None, True, FactorEncoder()), 
            "molecule_name"    : Metadata(None, None, OneHotEncoder()),
            "ID"               : Metadata(True, None, None),
            "conformation_name": Metadata(True, None, None)
        }
        md5_checksum = "4fbb00ba35dd05a29be1f52b7e0faeb6"

        simulation = ClassificationSimulation.from_csv(location, md5_checksum=md5_checksum, default_meta=default_meta, defined_meta=defined_meta)

        self.assertEqual(len(simulation.interactions), 6598)

        for rnd in simulation.interactions:
            hash(rnd.state)      #make sure these are hashable
            hash(rnd.actions[0]) #make sure these are hashable
            hash(rnd.actions[1]) #make sure these are hashable

            self.assertEqual(len(cast(Tuple,rnd.state)), 268)
            self.assertIn(1, rnd.actions)
            self.assertIn(2, rnd.actions)
            self.assertEqual(len(rnd.actions),2)
            
            actual_rewards = [ rwd[2] for rwd in simulation.rewards(rnd.choices) ]

            self.assertIn(1, actual_rewards)
            self.assertIn(0, actual_rewards)

    def test_from_json_table(self) -> None:
        
        json_val = '''{
            "format"          : "table",
            "table"           : [["a","b","c"], ["s1","2","3"], ["s2","5","6"]],
            "has_header"      : true,
            "column_default"  : { "ignore":false, "label":false, "encoding":"onehot" },
            "column_overrides": { "b": { "label":true, "encoding":"string" } }
        }'''

        simulation = ClassificationSimulation.from_json(json_val)

        self.assert_simulation_for_data(simulation, [(1,1),((0,0))], ['2','5'])

class MemorySimulation_Tests(Simulation_Interface_Tests, unittest.TestCase):

    def _make_simulation(self) -> Tuple[Simulation, Sequence[Interaction], Sequence[Sequence[Tuple[State,Action,Reward]]]]:
        
        states      =  [1,2]
        action_sets = [[1,2,3], [4,5,6]]
        reward_sets = [[0,1,2], [2,3,4]]

        states_actions_rewards = zip(states, action_sets, reward_sets)

        expected_interactions  = list(map(Interaction[int,int],states,action_sets))
        expected_rewards = [ [(s,a,r) for a,r in zip(A,R)] for s, A, R in states_actions_rewards]

        simulation = MemorySimulation(states, action_sets, reward_sets)

        return simulation, expected_interactions, expected_rewards

class LambdaSimulation_Tests(Simulation_Interface_Tests, unittest.TestCase):

    def _make_simulation(self) -> Tuple[Simulation, Sequence[Interaction], Sequence[Sequence[Tuple[State,Action,Reward]]]]:
        
        states      =  [0,1]
        action_sets = [[1,2,3], [4,5,6]]
        reward_sets = [[1,2,3], [3,4,5]]
        
        states_actions_rewards = zip(states, action_sets, reward_sets)

        expected_interactions  = list(map(Interaction[int,int],states,action_sets))
        expected_rewards = [ [(s,a,r) for a,r in zip(A,R)] for s, A, R in states_actions_rewards]

        def S(i:int) -> int:
            return states[i]

        def A(s:int) -> List[int]:
            return action_sets[s]
        
        def R(s:int,a:int) -> int:
            return a-s

        return LambdaSimulation(2,S,A,R), expected_interactions, expected_rewards

    def test_correct_number_of_interactions_created(self):
        def S(i:int) -> int:
            return [1,2][i]

        def A(s:int) -> List[int]:
            return [1,2,3] if s == 1 else [4,5,6]
        
        def R(s:int,a:int) -> int:
            return a-s

        simulation = LambdaSimulation(2,S,A,R)

        self.assertEqual(len(simulation.interactions), 2)

class ShuffleSimulation_Tests(Simulation_Interface_Tests, unittest.TestCase):

    def _make_simulation(self) -> Tuple[Simulation, Sequence[Interaction], Sequence[Sequence[Tuple[State,Action,Reward]]]]:

        states      = [1,2]
        action_sets = [[1,2,3],[0,1,2]]
        reward_sets = [[0,1,2],[2,3,4]]

        states_actions_rewards = zip(states, action_sets, reward_sets)

        expected_interactions  = list(map(Interaction[int,int],states,action_sets))
        expected_rewards = [ [(s,a,r) for a,r in zip(A,R)] for s, A, R in states_actions_rewards]

        simulation = MemorySimulation(states, action_sets, reward_sets)

        #with the seed set this test should always pass, if the test fails then it may mean
        #that randomization changed which would cause old results to no longer be reproducible
        return ShuffleSimulation(simulation, 1), expected_interactions, expected_rewards

    def test_interactions_not_duplicated_in_memory(self):
        
        states      = [1,2]
        action_sets = [[1,2,3], [4,5,6]]
        reward_sets = [[0,1,2], [2,3,4]]

        simulation = ShuffleSimulation(MemorySimulation(states,action_sets,reward_sets))

        self.assertEqual(len(simulation.interactions),2)

        self.assertEqual(sum(1 for r in simulation.interactions if r.state == 1),1)
        self.assertEqual(sum(1 for r in simulation.interactions if r.state == 2),1)

        simulation.interactions[0]._state = 3
        simulation.interactions[1]._state = 3

        self.assertEqual(sum(1 for r in simulation.interactions if r.state == 3),2)

class LazySimulation_Tests(Simulation_Interface_Tests, unittest.TestCase):

    def _make_simulation(self) -> Tuple[Simulation, Sequence[Interaction], Sequence[Sequence[Tuple[State,Action,Reward]]]]:
    
        states      =  [1,2]
        action_sets = [[1,2,3], [4,5,6]]
        reward_sets = [[0,1,2], [2,3,4]]

        states_actions_rewards = zip(states, action_sets, reward_sets)

        expected_interactions  = list(map(Interaction[int,int],states,action_sets))
        expected_rewards = [ [(s,a,r) for a,r in zip(A,R)] for s, A, R in states_actions_rewards]

        simulation = LazySimulation[int,int](lambda: MemorySimulation(states, action_sets, reward_sets)).load()

        return simulation, expected_interactions, expected_rewards    

    def test_unload_removes_simulation(self):

        simulation = cast(LazySimulation, self._make_simulation()[0])

        self.assertIsNotNone(simulation._simulation)
        simulation.unload()
        self.assertIsNone(simulation._simulation)

class Interaction_Tests(unittest.TestCase):

    def test_constructor_no_state(self) -> None:
        Interaction(None, [1, 2])

    def test_constructor_state(self) -> None:
        Interaction((1,2,3,4), [1, 2])

    def test_state_correct_1(self) -> None:
        self.assertEqual(None, Interaction(None, [1, 2]).state)

    def test_actions_correct_1(self) -> None:
        self.assertSequenceEqual([1, 2], Interaction(None, [1, 2]).actions)

    def test_actions_correct_2(self) -> None:
        self.assertSequenceEqual(["A", "B"], Interaction(None, ["A", "B"]).actions)

    def test_actions_correct_3(self) -> None:
        self.assertSequenceEqual([(1,2), (3,4)], Interaction(None, [(1,2), (3,4)]).actions)


if __name__ == '__main__':
    unittest.main()