import unittest as ut
import games as bg

class Test_BanditRound(ut.TestCase):

    def setUp(self):
        self.round = bg.BanditRound([[1],[2],[3]], [1, 0, 1])

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

class Test_ContextualBanditRound(Test_BanditRound):
    def setUp(self):
        self.round = bg.ContextualBanditRound([1, 1, 1], [[1],[2],[3]], [1, 0, 1])

    def test_context_features_correct(self):
        self.assertEqual([1, 1, 1], self.round.context_features)

    def test_context_features_readonly(self):
        def assign_context_features():
            self.round.context_features = [2, 0, 1]

        self.assertRaises(AttributeError, assign_context_features)

if __name__ == '__main__':
    ut.main()
