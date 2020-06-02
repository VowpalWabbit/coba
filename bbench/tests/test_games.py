import unittest
import itertools

import bbench.games as bg

class Test_Round_Instance(unittest.TestCase):

    def setUp(self):
        self.round = bg.Round([[1],[2],[3]], [1, 0, 1])

    def test_setUp(self):
        pass

    def test_action_correct(self):
        self.assertEqual([[1],[2],[3]], self.round.actions)

    def test_action_readonly(self):

        def assign_action():
            self.round.actions = [[4],[5],[6]]

        self.assertRaises(AttributeError, assign_action)

    def test_action_rewards_correct(self):
        self.assertEqual([1, 0, 1], self.round.rewards)

    def test_action_rewards_readonly(self):

        def assign_action_rewards():
            self.round.rewards = [2, 0, 1]

        self.assertRaises(AttributeError, assign_action_rewards)

class Test_ContextRound_Instance(Test_Round_Instance):
    def setUp(self):
        self.round = bg.ContextRound([1, 1, 1], [[1],[2],[3]], [1, 0, 1])

    def test_context_correct(self):
        self.assertEqual([1, 1, 1], self.round.context)

    def test_context_readonly(self):
        def assign_context():
            self.round.context = [2, 0, 1]

        self.assertRaises(AttributeError, assign_context)

class Test_Game_Instance(unittest.TestCase):
    def setUp(self):
        self.rounds = [bg.Round([[1],[2],[3]], [1, 0, 1]), bg.Round([[1],[2],[3]], [1, 0, 1])]
        self.game = bg.Game(self.rounds)

    def test_setUp(self):
        pass

    def test_rounds_correct(self):
        self.assertIs(self.rounds, self.game.rounds)    

class Test_ContextGame_Instance(Test_Game_Instance):
    def setUp(self):
        self.rounds = [bg.ContextRound([1], [[1],[2],[3]], [1, 0, 1])]
        self.game = bg.ContextGame(self.rounds)

class Test_ContextGame_Factories(unittest.TestCase):

    def test_from_classifier_data_with_flat_numeric_features_and_labels(self):

        features = [1,2,3,4]
        labels = [1,1,0,0,1]

        game = bg.ContextGame.from_classifier_data(features, labels)
        
        self.assertEqual(len(game.rounds), len(features))
        
        for i,r in enumerate(game.rounds):

            expected_context = features[i]
            expected_actions = [1,0]
            expected_rewards = [int(a == labels[i]) for a in r.actions]

            self.assertEqual(r.context, expected_context)
            self.assertCountEqual(r.actions, expected_actions)
            self.assertSequenceEqual(r.rewards, expected_rewards)

        

if __name__ == '__main__':
    unittest.main()
