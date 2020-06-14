import unittest
import itertools
import random

from itertools import cycle
from abc import ABC, abstractmethod
from typing import List, Sequence
from bbench.games import Round, Game, MemoryGame, LambdaGame, ClassificationGame, State, Action, Reward

class Test_Game_Interface(ABC):

    @abstractmethod
    def _make_game(self) -> Game:
        ...

    def _expected_rounds(self) -> List[Round]:
        return [
            Round(1, [1,2,3], [0,1,2]),
            Round(2, [4,5,6], [2,3,4]),
            Round(3, [7,8,9], [4,5,6])
        ]

    def setUp(self):
        self._game = self._make_game()

    def test_rounds_is_correct(self) -> None:
        #pylint: disable=no-member

        actual_rounds   = self._game.rounds
        expected_rounds = cycle(self._expected_rounds())

        for actual_round, expected_round in zip(actual_rounds, expected_rounds):
            self.assertEqual(actual_round.state  , expected_round.state  ) #type: ignore
            self.assertEqual(actual_round.actions, expected_round.actions) #type: ignore
            self.assertEqual(actual_round.rewards, expected_round.rewards) #type: ignore

    def test_rounds_is_reiterable(self) -> None:
        #pylint: disable=no-member

        for round1,round2 in zip(self._game.rounds, self._game.rounds):
            self.assertEqual(round1.state  , round2.state  ) #type: ignore
            self.assertEqual(round1.actions, round2.actions) #type: ignore
            self.assertEqual(round1.rewards, round2.rewards) #type: ignore

    def test_rounds_is_readonly(self) -> None:
        #pylint: disable=no-member

        def assign_rounds():
            self._game.rounds = []
        
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

class Test_MemoryGame_Interface(Test_Game_Interface, unittest.TestCase):

    def _make_game(self) -> MemoryGame:
        return MemoryGame(self._expected_rounds())

class Test_LambdaGame_Interface(Test_Game_Interface, unittest.TestCase):
    def _make_game(self) -> LambdaGame:

        def S(i:int) -> int:
            return [1,2,3][i]

        def A(s:int) -> List[int]:
            return [1,2,3] if s == 1 else [4,5,6] if s == 2 else [7,8,9]
        
        def R(s:int,a:int) -> int:
            return a-s

        return LambdaGame(S,A,R,3)

class Test_ClassificationGame_Interface(Test_Game_Interface, unittest.TestCase):
    def _make_game(self) -> ClassificationGame:
        return ClassificationGame([1,2,3], [3,2,1])

    def _expected_rounds(self) -> List[Round]:
        return [
            Round(1, [1,2,3], [0,0,1]),
            Round(2, [1,2,3], [0,1,0]),
            Round(3, [1,2,3], [1,0,0])
        ]

class Test_ClassificationGame(unittest.TestCase):

    def assert_game_for_data(self, game, features, labels) -> None:

        self.assertEqual(sum(1 for _ in game.rounds), len(features))

        for f,l,r in zip(features, labels, game.rounds):

            expected_state   = f
            expected_actions = list(set(labels))
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

if __name__ == '__main__':
    unittest.main()
