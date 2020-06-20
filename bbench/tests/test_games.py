import unittest
import itertools
import random

from itertools import cycle
from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple
from bbench.games import Round, Game, MemoryGame, LambdaGame, ClassificationGame, State, Action, Reward

class Test_Game_Interface(ABC):

    @abstractmethod
    def _interface_test_setup(self) -> Tuple[Game, List[Round]]:
        ...

    def test_rounds_is_correct(self) -> None:
        #pylint: disable=no-member

        game, expected_rounds = self._interface_test_setup()

        actual_rounds = list(game.rounds)

        self.assertEqual(len(actual_rounds), len(expected_rounds))

        for actual_round, expected_round in zip(actual_rounds, expected_rounds):
            self.assertEqual(actual_round.state  , expected_round.state  ) #type: ignore
            self.assertEqual(actual_round.actions, expected_round.actions) #type: ignore
            self.assertEqual(actual_round.rewards, expected_round.rewards) #type: ignore

    def test_rounds_is_reiterable(self) -> None:
        #pylint: disable=no-member

        game, _ = self._interface_test_setup()

        for round1,round2 in zip(game.rounds, game.rounds):
            self.assertEqual(round1.state  , round2.state  ) #type: ignore
            self.assertEqual(round1.actions, round2.actions) #type: ignore
            self.assertEqual(round1.rewards, round2.rewards) #type: ignore

    def test_rounds_is_readonly(self) -> None:
        #pylint: disable=no-member

        game, _ = self._interface_test_setup()

        def assign_rounds():
            game.rounds = []
        
        self.assertRaises(AttributeError, assign_rounds) #type: ignore

class Test_Round(unittest.TestCase):

    def test_constructor_no_state(self) -> None:
        Round(None, [1, 2], [1, 0])

    def test_constructor_state(self) -> None:
        Round([1,2,3,4], [1, 2], [1, 0])

    def test_constructor_mismatch_actions_rewards_1(self) -> None:
        self.assertRaises(AssertionError, lambda: Round(None, [1, 2, 3], [1, 0]))
   
    def test_constructor_mismatch_actions_rewards_2(self) -> None:
        self.assertRaises(AssertionError, lambda: Round(None, [1, 2], [1, 0, 2]))

    def test_state_correct_1(self) -> None:
        self.assertEqual(None, Round(None, [1, 2], [1, 0]).state)

    def test_actions_correct_1(self) -> None:
        self.assertEqual([1, 2], Round(None, [1, 2], [1, 0]).actions)

    def test_actions_correct_2(self) -> None:
        self.assertEqual(["A", "B"], Round(None, ["A", "B"], [1, 0]).actions)

    def test_actions_correct_3(self) -> None:
        self.assertEqual([[1,2], [3,4]], Round(None, [[1,2], [3,4]], [1, 0]).actions)

    def test_rewards_correct(self) -> None:
        self.assertEqual([1, 0], Round(None, [1, 2], [1, 0]).rewards)
    
    def test_actions_readonly(self) -> None:
        def assign_actions():
           Round(None, [[1],[2],[3]], [1, 0, 1]).actions = [2,0,1]

        self.assertRaises(AttributeError, assign_actions)
    
    def test_rewards_readonly(self) -> None:
        
        def assign_rewards():
            Round(None, [[1],[2],[3]], [1, 0, 1]).rewards = [2,0,1]

        self.assertRaises(AttributeError, assign_rewards)

class Test_MemoryGame(Test_Game_Interface, unittest.TestCase):

    def _interface_test_setup(self) -> Tuple[Game, List[Round]]:
        
        rounds = [Round(1, [1,2,3], [0,1,2]), Round(2, [4,5,6], [2,3,4])]
        return MemoryGame(rounds), rounds

class Test_LambdaGame(Test_Game_Interface, unittest.TestCase):
    
    def _interface_test_setup(self) -> Tuple[Game, List[Round]]:
        rounds = [Round(1, [1,2,3], [0,1,2]), Round(2, [4,5,6], [2,3,4])]
        
        def S(i:int) -> int:
            return [1,2][i]

        def A(s:int) -> List[int]:
            return [1,2,3] if s == 1 else [4,5,6]
        
        def R(s:int,a:int) -> int:
            return a-s

        return LambdaGame(S,A,R,2), rounds

class Test_ClassificationGame(Test_Game_Interface, unittest.TestCase):
    def _interface_test_setup(self) -> Tuple[Game, List[Round]]:
        rounds = [ Round(1, [1,2], [0,1]), Round(2, [1,2], [1,0]) ]
        return ClassificationGame([1,2], [2,1]), rounds

    def assert_game_for_data(self, game, features, labels) -> None:

        self.assertEqual(sum(1 for _ in game.rounds), len(features))

        for f,l,r in zip(features, labels, game.rounds):

            expected_state   = f
            expected_actions = sorted(list(set(labels)))
            expected_rewards = [int(a == l) for a in r.actions]

            self.assertEqual(r.state, expected_state)            
            self.assertEqual(r.actions, expected_actions)
            self.assertEqual(r.rewards, expected_rewards)

    def test_constructor_with_good_features_and_labels1(self) -> None:
        features = [1,2,3,4]
        labels   = [1,1,0,0]
        game     = ClassificationGame(features, labels)

        self.assert_game_for_data(game, features, labels)
    
    def test_constructor_with_good_features_and_labels2(self) -> None:
        features = ["a","b"]
        labels   = ["good","bad"]
        game     = ClassificationGame(features, labels)

        self.assert_game_for_data(game, features, labels)

    def test_constructor_with_good_features_and_labels3(self) -> None:
        features = [[1,2],[3,4]]
        labels   = ["good","bad"]
        game     = ClassificationGame(features, labels)

        self.assert_game_for_data(game, features, labels)
    
    def test_constructor_with_too_few_features(self) -> None:
        self.assertRaises(AssertionError, lambda: ClassificationGame([1], [1,1]))

    def test_constructor_with_too_few_labels(self) -> None:
        self.assertRaises(AssertionError, lambda: ClassificationGame([1,1], [1]))

    def test_simple_from_csv_file(self) -> None:
        game = ClassificationGame.from_csv_file(['a,b,c','1,2,3','4,5,6'],'b')

        self.assert_game_for_data(game, [['1','3'],['4','6']],['2','5'])

    def test_simple_from_csv_file_with_stater(self) -> None:
        
        def stater(row: List[str]) -> State:
            return [row[0] == "s1", row[0] == "s2", int(row[1])]

        game = ClassificationGame.from_csv_file(['a,b,c','s1,2,3','s2,5,6'],'b', csv_stater=stater)

        self.assert_game_for_data(game, [[1,0,3],[0,1,6]],['2','5'])

if __name__ == '__main__':
    unittest.main()