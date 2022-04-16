import unittest

from coba.environments import LoggedInteraction

class LoggedInteraction_Tests(unittest.TestCase):
    def test_simple_with_actions(self):
        interaction = LoggedInteraction(1, 2, 3, .2, [1,2,3])

        self.assertEqual(1, interaction.context)
        self.assertEqual(2, interaction.action)
        self.assertEqual(3, interaction.reward)
        self.assertEqual(.2, interaction.probability)
        self.assertEqual([1,2,3], interaction.actions)

    def test_simple_sans_actions(self):
        interaction = LoggedInteraction(1, 2, 3, .2)

        self.assertEqual(1, interaction.context)
        self.assertEqual(2, interaction.action)
        self.assertEqual(3, interaction.reward)
        self.assertEqual(.2, interaction.probability)

if __name__ == '__main__':
    unittest.main()