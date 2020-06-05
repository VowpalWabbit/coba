import unittest
import itertools
import random

from bbench.games import Round, Game

class Test_Round_Instance(unittest.TestCase):

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

class Test_Game_Instance(unittest.TestCase):
    def test_constructor(self) -> None:
        Game([Round(1, [1,2,3], [1,0,1]), Round(2,[1,2,3], [1, 1, 0])])

    def test_rounds_correct(self) -> None:
        rounds = [Round(1, [1,2,3], [1,0,1]), Round(2,[1,2,3], [1, 1, 0])]
        game   = Game(rounds)
        self.assertIs(rounds, game.rounds)

    def test_rounds_readonly(self) -> None:
        def assign_rounds():
            Game([]).rounds = []
        
        self.assertRaises(AttributeError, assign_rounds)

class Test_Game_Factories(unittest.TestCase):

    def assert_game_from_classifier_data(self,features, labels) -> None:
        game = Game.from_classifier_data(features, labels)

        self.assertEqual(sum(1 for _ in game.rounds), len(features))

        for f,l,r in zip(features, labels, game.rounds):

            expected_state   = f
            expected_actions = list(set(labels))
            expected_rewards = [int(a == l) for a in r.actions]

            self.assertEqual(r.state, expected_state)            
            self.assertEqual(r.actions, expected_actions)
            self.assertEqual(r.rewards, expected_rewards)

    def test_from_classifier_data_with_good_features_and_labels1(self) -> None:
        self.assert_game_from_classifier_data([1,2,3,4], [1,1,0,0])
    
    def test_from_classifier_data_with_good_features_and_labels2(self) -> None:
        self.assert_game_from_classifier_data(["a","b"], ["good", "bad"])

    def test_from_classifier_data_with_good_features_and_labels3(self) -> None:
        self.assert_game_from_classifier_data([[1,2],[3,4]], ["good", "bad"])
    
    def test_from_classifier_data_with_short_features(self) -> None:
        self.assertRaises(AssertionError, lambda: Game.from_classifier_data([1], [1,1]))
    
    def test_from_classifier_data_with_short_labels(self) -> None:
        self.assertRaises(AssertionError, lambda: Game.from_classifier_data([1,1], [1]))

    def test_from_classifier_data_with_list_labels(self) -> None:
        bad_lambda = lambda: Game.from_classifier_data([1], [[1,1]]) #type: ignore
        self.assertRaises(TypeError, bad_lambda)

    def test_from_iterable(self) -> None:
        S = [[1,2,3,4],[5,6,7,8],[2,4,6,8],[1,3,5,7]]
        A = lambda s: list(range(1,s[0]+1))
        R = lambda s,a: s[1]/a
        
        game = Game.from_iterable(S,A,R)

        for s,r in zip(S,game.rounds):
            self.assertEqual(r.state  , s)
            self.assertEqual(r.actions, A(s))
            self.assertEqual(r.rewards, [R(s,a) for a in A(s)])

if __name__ == '__main__':
    unittest.main()
