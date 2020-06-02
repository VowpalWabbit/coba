import unittest as ut
from bbench.games import Round, ContextRound, Game, ContextGame

class Test_Round(ut.TestCase):

    def setUp(self):
        self.round = Round([[1],[2],[3]], [1, 0, 1])

    def test_setUp(self):
        pass

    def test_action_features_correct(self):
        self.assertEqual([[1],[2],[3]], self.round.action_features)

    def test_action_features_readonly(self):

        def assign_action_features():
            self.round.action_features = [[4],[5],[6]]

        self.assertRaises(AttributeError, assign_action_features)

    def test_action_rewards_correct(self):
        self.assertEqual([1, 0, 1], self.round.action_rewards)

    def test_action_rewards_readonly(self):

        def assign_action_rewards():
            self.round.action_rewards = [2, 0, 1]

        self.assertRaises(AttributeError, assign_action_rewards)

class Test_ContextRound(Test_Round):
    def setUp(self):
        self.round = ContextRound([1, 1, 1], [[1],[2],[3]], [1, 0, 1])

    def test_context_features_correct(self):
        self.assertEqual([1, 1, 1], self.round.context_features)

    def test_context_features_readonly(self):
        def assign_context_features():
            self.round.context_features = [2, 0, 1]

        self.assertRaises(AttributeError, assign_context_features)

class Test_Game(ut.TestCase):
    def setUp(self):
        self.rounds = [Round([[1],[2],[3]], [1, 0, 1]), Round([[1],[2],[3]], [1, 0, 1])]
        self.game = Game(self.rounds)

    def test_setUp(self):
        pass

    def test_rounds_correct(self):
        self.assertEqual(self.rounds, self.game.rounds())    

if __name__ == '__main__':
    ut.main()
