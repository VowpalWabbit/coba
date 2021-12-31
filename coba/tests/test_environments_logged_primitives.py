import unittest

from coba.environments import LoggedInteraction

class LoggedInteraction_Tests(unittest.TestCase):
    def test_simple_with_actions(self):
        interaction = LoggedInteraction(1, 2, reward=3, probability=.2, actions=[1,2,3])

        self.assertEqual(1, interaction.context)
        self.assertEqual(2, interaction.action)
        self.assertEqual(3, interaction.kwargs["reward"])
        self.assertEqual(.2, interaction.kwargs["probability"])
        self.assertEqual([1,2,3], interaction.kwargs["actions"])

    def test_simple_sans_actions(self):
        interaction = LoggedInteraction(1, 2, reward=3, probability=.2)

        self.assertEqual(1, interaction.context)
        self.assertEqual(2, interaction.action)
        self.assertEqual(3, interaction.kwargs["reward"])
        self.assertEqual(.2, interaction.kwargs["probability"])

if __name__ == '__main__':
    unittest.main()