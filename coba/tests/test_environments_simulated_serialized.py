import unittest

from coba.pipes        import ListIO
from coba.contexts     import CobaContext, NullLogger
from coba.environments import SimulatedInteraction, MemorySimulation, SerializedSimulation

CobaContext.logger = NullLogger()

class SerializedSimulation_Tests(unittest.TestCase):

    def test_sim_source(self):
        expected_env = MemorySimulation(params={}, interactions=[SimulatedInteraction(1,[1,2],rewards=[2,3])])
        actual_env = SerializedSimulation(expected_env)

        self.assertEqual(expected_env.params, actual_env.params)
        self.assertEqual(len(list(expected_env.read())), len(list(actual_env.read())))        
        for e_interaction, a_interaction in zip(expected_env.read(), actual_env.read()):
            self.assertEqual(e_interaction.context, a_interaction.context)
            self.assertEqual(e_interaction.actions, a_interaction.actions)
            self.assertEqual(e_interaction.kwargs , a_interaction.kwargs )

    def test_sim_write_read_simple(self):
        expected_env = MemorySimulation(params={}, interactions=[SimulatedInteraction(1,[1,2],rewards=[2,3])])
        serial_IO    = ListIO()
        actual_env   = SerializedSimulation(serial_IO)

        SerializedSimulation(expected_env).write(serial_IO)

        self.assertEqual(expected_env.params, actual_env.params)
        self.assertEqual(len(list(expected_env.read())), len(list(actual_env.read())))        
        for e_interaction, a_interaction in zip(expected_env.read(), actual_env.read()):
            self.assertEqual(e_interaction.context, a_interaction.context)
            self.assertEqual(e_interaction.actions, a_interaction.actions)
            self.assertEqual(e_interaction.kwargs , a_interaction.kwargs )

    def test_sim_write_read_with_params_and_none_context(self):
        expected_env = MemorySimulation(params={'a':1}, interactions=[SimulatedInteraction(None,[1,2],rewards=[2,3])])
        serial_IO    = ListIO()
        actual_env   = SerializedSimulation(serial_IO)

        SerializedSimulation(expected_env).write(serial_IO)

        self.assertEqual(expected_env.params, actual_env.params)
        self.assertEqual(len(list(expected_env.read())), len(list(actual_env.read())))        
        for e_interaction, a_interaction in zip(expected_env.read(), actual_env.read()):
            self.assertEqual(e_interaction.context, a_interaction.context)
            self.assertEqual(e_interaction.actions, a_interaction.actions)
            self.assertEqual(e_interaction.kwargs , a_interaction.kwargs )

    def test_sim_write_read_with_params_and_action_tuple(self):
        expected_env = MemorySimulation(params={'a':1}, interactions=[SimulatedInteraction(None,[(1,0),(0,1)],rewards=[2,3])])
        serial_IO    = ListIO()
        actual_env   = SerializedSimulation(serial_IO)

        SerializedSimulation(expected_env).write(serial_IO)

        self.assertEqual(expected_env.params, actual_env.params)
        self.assertEqual(len(list(expected_env.read())), len(list(actual_env.read())))        
        for e_interaction, a_interaction in zip(expected_env.read(), actual_env.read()):
            self.assertEqual(e_interaction.context, a_interaction.context)
            self.assertEqual(e_interaction.actions, a_interaction.actions)
            self.assertEqual(e_interaction.kwargs , a_interaction.kwargs )

if __name__ == '__main__':
    unittest.main()
