import unittest

from coba.pipes        import ListSink, ListSource, HttpSource, DiskSource
from coba.contexts     import CobaContext, NullLogger
from coba.environments import SimulatedInteraction, MemorySimulation, SerializedSimulation

CobaContext.logger = NullLogger()

class SerializedSimulation_Tests(unittest.TestCase):

    def test_sim_source(self):
        expected_env = MemorySimulation(params={}, interactions=[SimulatedInteraction(1,[1,2],[2,3])])
        actual_env = SerializedSimulation(expected_env)

        self.assertEqual(expected_env.params, actual_env.params)
        self.assertEqual(len(list(expected_env.read())), len(list(actual_env.read())))        
        for e_interaction, a_interaction in zip(expected_env.read(), actual_env.read()):
            self.assertEqual(e_interaction.context, a_interaction.context)
            self.assertEqual(e_interaction.actions, a_interaction.actions)
            self.assertEqual(e_interaction.kwargs , a_interaction.kwargs )

    def test_http_url(self):
        env = SerializedSimulation("https://github.com")
        self.assertIsInstance(env._source._source, HttpSource)
        self.assertEqual("https://github.com", env._source._url)

    def test_filepath(self):
        env = SerializedSimulation("C:/test")
        self.assertIsInstance(env._source._source, DiskSource)
        self.assertEqual("C:/test", env._source._source._filename)

    def test_sim_write_read_simple(self):
        sink = ListSink()

        expected_env = MemorySimulation(params={}, interactions=[SimulatedInteraction(1,[1,2],[2,3])])
        
        SerializedSimulation(expected_env).write(sink)
        actual_env = SerializedSimulation(ListSource(sink.items))

        self.assertEqual(expected_env.params, actual_env.params)
        self.assertEqual(len(list(expected_env.read())), len(list(actual_env.read())))        
        for e_interaction, a_interaction in zip(expected_env.read(), actual_env.read()):
            self.assertEqual(e_interaction.context, a_interaction.context)
            self.assertEqual(e_interaction.actions, a_interaction.actions)
            self.assertEqual(e_interaction.kwargs , a_interaction.kwargs )

    def test_sim_write_read_with_params_and_none_context(self):
        sink = ListSink()

        expected_env = MemorySimulation(params={'a':1}, interactions=[SimulatedInteraction(None,[1,2],[2,3])])
        SerializedSimulation(expected_env).write(sink)
        actual_env = SerializedSimulation(ListSource(sink.items))

        self.assertEqual(expected_env.params, actual_env.params)
        self.assertEqual(len(list(expected_env.read())), len(list(actual_env.read())))        
        for e_interaction, a_interaction in zip(expected_env.read(), actual_env.read()):
            self.assertEqual(e_interaction.context, a_interaction.context)
            self.assertEqual(e_interaction.actions, a_interaction.actions)
            self.assertEqual(e_interaction.kwargs , a_interaction.kwargs )

    def test_sim_write_read_with_params_and_action_tuple(self):
        sink = ListSink()
        
        expected_env = MemorySimulation(params={'a':1}, interactions=[SimulatedInteraction(None,[(1,0),(0,1)],[2,3])])
        SerializedSimulation(expected_env).write(sink)
        actual_env = SerializedSimulation(ListSource(sink.items))

        self.assertEqual(expected_env.params, actual_env.params)
        self.assertEqual(len(list(expected_env.read())), len(list(actual_env.read())))        
        for e_interaction, a_interaction in zip(expected_env.read(), actual_env.read()):
            self.assertEqual(e_interaction.context, a_interaction.context)
            self.assertEqual(e_interaction.actions, a_interaction.actions)
            self.assertEqual(e_interaction.kwargs , a_interaction.kwargs )

if __name__ == '__main__':
    unittest.main()
