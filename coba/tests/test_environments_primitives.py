import unittest

from coba.environments import SafeEnvironment, Environment
from coba.environments import Interaction, SimulatedInteraction, LoggedInteraction
from coba.exceptions import CobaException
from coba.pipes import Pipes, Shuffle

class DummyEnvironment:

    def read(self):
        return []

class Environment_Tests(unittest.TestCase):

    def test_str_with_params(self):

        class TestEnvironment(Environment):
            def read(self):
                return []
            @property
            def params(self):
                return {'a':1}

        self.assertEqual("{'a': 1}", str(TestEnvironment()))


    def test_str_sans_params(self):

        class TestEnvironment(Environment):
            def read(self):
                return []
            @property
            def params(self):
                return {}

        self.assertEqual("TestEnvironment", str(TestEnvironment()))

class SafeEnvironment_Tests(unittest.TestCase):

    def test_params(self):
        self.assertEqual({'type': 'DummyEnvironment'}, SafeEnvironment(DummyEnvironment()).params)

    def test_read(self):
        self.assertEqual([], SafeEnvironment(DummyEnvironment()).read())

    def test_str(self):
        self.assertEqual('DummyEnvironment(shuffle=1)', str(SafeEnvironment(Pipes.join(DummyEnvironment(), Shuffle(1)))))
        self.assertEqual('DummyEnvironment', str(SafeEnvironment(DummyEnvironment())))

    def test_with_nesting(self):
        self.assertIsInstance(SafeEnvironment(SafeEnvironment(DummyEnvironment()))._environment, DummyEnvironment)

    def test_with_pipes(self):
        self.assertEqual({'type': 'DummyEnvironment', "shuffle":1}, SafeEnvironment(Pipes.join(DummyEnvironment(), Shuffle(1))) .params)

class Interaction_Tests(unittest.TestCase):
    def test_simulated_dict(self):
        mapping = {'context':1,'actions':[1,2],'rewards':[3,4]}
        self.assertEqual(Interaction.from_dict(mapping),mapping)

    def test_logged_dict(self):
        given    = {'context':1,'actions':[1,2],'action':1,'reward':4,'probability':.1}
        expected = {'context':1,'actions':[1,2],'action':1,'reward':4,'probability':.1,'rewards':[40,0]}
        self.assertEqual(Interaction.from_dict(given),expected)

    def test_grounded_dict(self):
        given    = {'context':1,'actions':[1,2],'action':1,'rewards':[3,4],'feedbacks':[5,6]}
        expected = given
        self.assertEqual(Interaction.from_dict(given),expected)

class LoggedInteraction_Tests(unittest.TestCase):
    def test_IPS_sequence(self):
        interaction = LoggedInteraction(1,2,3,probability=1/2,actions=[1,2,3])
        self.assertEqual([0,6,0], interaction['rewards'])

    def test_simple_with_actions(self):
        interaction = LoggedInteraction(1, 2, 3, probability=.2,actions=[1,2,3])

        self.assertEqual(1, interaction['context'])
        self.assertEqual(2, interaction['action'])
        self.assertEqual(3, interaction['reward'])
        self.assertEqual(.2, interaction['probability'])
        self.assertEqual([1,2,3], interaction['actions'])

    def test_simple_sans_actions(self):
        interaction = LoggedInteraction(1, 2, 3, probability=.2)

        self.assertEqual(1, interaction['context'])
        self.assertEqual(2, interaction['action'])
        self.assertEqual(3, interaction['reward'])
        self.assertEqual(.2, interaction['probability'])

class SimulatedInteraction_Tests(unittest.TestCase):
    def test_context_none(self):
        self.assertEqual(None, SimulatedInteraction(None, (1,2,3), (4,5,6))['context'])

    def test_context_str(self):
        self.assertEqual("A", SimulatedInteraction("A", (1,2,3), (4,5,6))['context'])

    def test_context_dense(self):
        self.assertEqual((1,2,3), SimulatedInteraction((1,2,3), (1,2,3), (4,5,6))['context'])

    def test_context_dense_2(self):
        self.assertEqual((1,2,3,(0,0,1)), SimulatedInteraction((1,2,3,(0,0,1)), (1,2,3), (4,5,6))['context'])

    def test_context_sparse_dict(self):
        self.assertEqual({1:0}, SimulatedInteraction({1:0}, (1,2,3), (4,5,6))['context'])

    def test_actions_correct_1(self) -> None:
        self.assertSequenceEqual([1,2], SimulatedInteraction(None, [1,2], [1,2])['actions'])

    def test_actions_correct_2(self) -> None:
        self.assertSequenceEqual(["A","B"], SimulatedInteraction(None, ["A","B"], [1,2])['actions'])

    def test_actions_correct_3(self) -> None:
        self.assertSequenceEqual([(1,2), (3,4)], SimulatedInteraction(None, [(1,2), (3,4)], [1,2])['actions'])

    def test_rewards_correct(self):
        self.assertEqual([4,5,6], SimulatedInteraction((1,2), (1,2,3), [4,5,6])['rewards'])

    def test_rewards_actions_mismatch(self):
        with self.assertRaises(CobaException):
            SimulatedInteraction((1,2), (1,2,3), [4,5])

if __name__ == '__main__':
    unittest.main()