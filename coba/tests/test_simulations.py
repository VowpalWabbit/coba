import unittest

from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple, cast, Dict

from coba.simulations import State, Action, Reward, Round, Simulation, ClassificationSimulation, MemorySimulation, LambdaSimulation, ShuffleSimulation
from coba.preprocessing import Metadata, NumericEncoder, OneHotEncoder, StringEncoder, Metadata

class Round_Tests(unittest.TestCase):

    def test_constructor_no_state(self) -> None:
        Round(None, [1, 2])

    def test_constructor_state(self) -> None:
        Round((1,2,3,4), [1, 2])

    def test_state_correct_1(self) -> None:
        self.assertEqual(None, Round(None, [1, 2]).state)

    def test_actions_correct_1(self) -> None:
        self.assertSequenceEqual([1, 2], Round(None, [1, 2]).actions)

    def test_actions_correct_2(self) -> None:
        self.assertSequenceEqual(["A", "B"], Round(None, ["A", "B"]).actions)

    def test_actions_correct_3(self) -> None:
        self.assertSequenceEqual([(1,2), (3,4)], Round(None, [(1,2), (3,4)]).actions)

class Simulation_Interface_Tests(ABC):

    @abstractmethod
    def _make_simulation(self) -> Tuple[Simulation, Sequence[Round], Sequence[Sequence[Tuple[State,Action,Reward]]]]:
        ...

    def test_rounds_is_correct(self) -> None:

        simulation, expected_rounds, expected_rewards = self._make_simulation()

        actual_rounds = simulation.rounds

        cast(unittest.TestCase, self).assertEqual(len(actual_rounds), len(expected_rounds))

        for actual_round, expected_round, expected_reward in zip(actual_rounds, expected_rounds, expected_rewards):

            actual_reward = simulation.rewards(actual_round.choices)

            cast(unittest.TestCase, self).assertEqual(actual_round.state, expected_round.state)
            cast(unittest.TestCase, self).assertCountEqual(actual_round.actions, expected_round.actions)
            cast(unittest.TestCase, self).assertCountEqual(actual_reward, expected_reward)

    def test_rounds_is_reiterable(self) -> None:

        simulation = self._make_simulation()[0]

        for round1,round2 in zip(simulation.rounds, simulation.rounds):

            round1_rewards = simulation.rewards(round1.choices)
            round2_rewards = simulation.rewards(round2.choices)

            cast(unittest.TestCase, self).assertEqual(round1.state, round2.state)
            cast(unittest.TestCase, self).assertSequenceEqual(round1.actions, round2.actions)
            cast(unittest.TestCase, self).assertSequenceEqual(round1_rewards, round2_rewards)

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

        self.assertEqual(len(simulation.rounds), 2)

class ClassificationSimulation_Tests(Simulation_Interface_Tests, unittest.TestCase):
    def _make_simulation(self) -> Tuple[Simulation, Sequence[Round], Sequence[Sequence[Tuple[State,Action,Reward]]]]:
        
        states      =  [1,2]
        action_sets = [[1,2], [1,2]]
        reward_sets = [[0,1], [1,0]]

        states_actions_rewards = zip(states, action_sets, reward_sets)

        expected_rounds  = list(map(Round, states, action_sets))
        expected_rewards = [ [(s,a,r) for a,r in zip(A,R)] for s, A, R in states_actions_rewards]

        return ClassificationSimulation([1,2], [2,1]), expected_rounds, expected_rewards

    def assert_simulation_for_data(self, simulation, features, labels) -> None:

        self.assertEqual(len(simulation.rounds), len(features))

        #first we make sure that all the labels are included 
        #in the first rounds actions without any concern for order
        self.assertCountEqual(simulation.rounds[0].actions, tuple(set(labels)))

        #then we set our expected actions to the first round
        #to make sure that every round has the exact same actions
        #with the exact same order
        expected_actions = simulation.rounds[0].actions

        for f,l,r in zip(features, labels, simulation.rounds):

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

        self.assert_simulation_for_data(simulation, [(1,3),(0,6)], [2,5])

    def test_from_table_explicit_onehot(self) -> None:

        default_meta = Metadata(False, False, OneHotEncoder())
        columns_meta = {'b': Metadata(None, True, StringEncoder()) }
        table        = [['a' ,'b','c'],
                        ['s1','2','3'],
                        ['s2','5','6']]
        
        simulation = ClassificationSimulation.from_table(table, default_meta=default_meta, columns_meta=columns_meta)

        self.assert_simulation_for_data(simulation, [(1,1),(0,0)], ['2','5'])

    def test_simple_from_openml(self) -> None:
        #this test requires interet acess to download the data

        simulation = ClassificationSimulation.from_openml(1116)

        self.assertEqual(len(simulation.rounds), 6598)

        for rnd in simulation.rounds:
            
            hash(rnd.state)      #make sure these are hashable
            hash(rnd.actions[0]) #make sure these are hashable
            hash(rnd.actions[1]) #make sure these are hashable
            
            self.assertEqual(len(cast(Tuple,rnd.state)), 268)
            self.assertIn(0, rnd.actions)
            self.assertIn(1, rnd.actions)
            self.assertEqual(len(rnd.actions),2)
            
            actual_rewards  = [ rwd[2] for rwd in simulation.rewards(rnd.choices) ]

            self.assertIn(1, actual_rewards)
            self.assertIn(0, actual_rewards)

    def test_simple_from_csv(self) -> None:
        #this test requires interet acess to download the data

        location     = "http://www.openml.org/data/v1/get_csv/53999"
        default_meta = Metadata(False, False, NumericEncoder())
        columns_meta: Dict[str,Metadata] = {
            "class"            : Metadata(None, True, OneHotEncoder()), 
            "molecule_name"    : Metadata(None, None, OneHotEncoder()),
            "ID"               : Metadata(True, None, None),
            "conformation_name": Metadata(True, None, None)
        }
        md5_checksum = "4fbb00ba35dd05a29be1f52b7e0faeb6"

        simulation = ClassificationSimulation.from_csv(location, md5_checksum=md5_checksum, default_meta=default_meta, columns_meta=columns_meta)

        self.assertEqual(len(simulation.rounds), 6598)

        for rnd in simulation.rounds:
            hash(rnd.state)      #make sure these are hashable
            hash(rnd.actions[0]) #make sure these are hashable
            hash(rnd.actions[1]) #make sure these are hashable

            self.assertEqual(len(cast(Tuple,rnd.state)), 268)
            self.assertIn(0, rnd.actions)
            self.assertIn(1, rnd.actions)
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

        self.assert_simulation_for_data(simulation, [(1,1),(0,0)], ['2','5'])

class MemorySimulation_Tests(Simulation_Interface_Tests, unittest.TestCase):

    def _make_simulation(self) -> Tuple[Simulation, Sequence[Round], Sequence[Sequence[Tuple[State,Action,Reward]]]]:
        
        states      =  [1,2]
        action_sets = [[1,2,3], [4,5,6]]
        reward_sets = [[0,1,2], [2,3,4]]

        states_actions_rewards = zip(states, action_sets, reward_sets)

        expected_rounds  = list(map(Round[int,int],states,action_sets))
        expected_rewards = [ [(s,a,r) for a,r in zip(A,R)] for s, A, R in states_actions_rewards]

        simulation = MemorySimulation(states, action_sets, reward_sets)

        return simulation, expected_rounds, expected_rewards

class LambdaSimulation_Tests(Simulation_Interface_Tests, unittest.TestCase):

    def _make_simulation(self) -> Tuple[Simulation, Sequence[Round], Sequence[Sequence[Tuple[State,Action,Reward]]]]:
        
        states      =  [0,1]
        action_sets = [[1,2,3], [4,5,6]]
        reward_sets = [[1,2,3], [3,4,5]]
        
        states_actions_rewards = zip(states, action_sets, reward_sets)

        expected_rounds  = list(map(Round,states,action_sets))
        expected_rewards = [ [(s,a,r) for a,r in zip(A,R)] for s, A, R in states_actions_rewards]

        def S(i:int) -> int:
            return states[i]

        def A(s:int) -> List[int]:
            return action_sets[s]
        
        def R(s:int,a:int) -> int:
            return a-s

        return LambdaSimulation(2,S,A,R), expected_rounds, expected_rewards

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

    def _make_simulation(self) -> Tuple[Simulation, Sequence[Round], Sequence[Sequence[Tuple[State,Action,Reward]]]]:

        states      = [1,2]
        action_sets = [[1,2,3],[0,1,2]]
        reward_sets = [[0,1,2],[2,3,4]]

        states_actions_rewards = zip(states, action_sets, reward_sets)

        expected_rounds  = list(map(Round,states,action_sets))
        expected_rewards = [ [(s,a,r) for a,r in zip(A,R)] for s, A, R in states_actions_rewards]

        simulation = MemorySimulation(states, action_sets, reward_sets)

        #with the seed set this test should always pass, if the test fails then it may mean
        #that randomization changed which would cause old results to no longer be reproducible
        return ShuffleSimulation(simulation, 1), expected_rounds, expected_rewards

    def test_rounds_not_duplicated_in_memory(self):
        
        states      = [1,2]
        action_sets = [[1,2,3], [4,5,6]]
        reward_sets = [[0,1,2], [2,3,4]]

        simulation = ShuffleSimulation(MemorySimulation(states,action_sets,reward_sets))

        self.assertEqual(len(simulation.rounds),2)

        self.assertEqual(sum(1 for r in simulation.rounds if r.state == 1),1)
        self.assertEqual(sum(1 for r in simulation.rounds if r.state == 2),1)

        simulation.rounds[0]._state = 3
        simulation.rounds[1]._state = 3

        self.assertEqual(sum(1 for r in simulation.rounds if r.state == 3),2)

if __name__ == '__main__':
    unittest.main()