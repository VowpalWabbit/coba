import unittest

from coba.pipes import ListSource
from coba.registry import CobaRegistry
from coba.exceptions import CobaException

from coba.environments.definitions import EnvironmentDefinitionFileV1
from coba.environments import OpenmlSimulation, Take

SimulatedEnvironment = OpenmlSimulation

class EnvironmentFileFmtV1_Tests(unittest.TestCase):

    def setUp(self) -> None:
        CobaRegistry.register("OpenmlSimulation", OpenmlSimulation)
        CobaRegistry.register("Take", Take)

    def test_one_environment(self):

        json_txt = """{
            "environments" : [
                { "OpenmlSimulation": 150 }
            ]
        }"""

        environments = EnvironmentDefinitionFileV1(ListSource([json_txt])).read()

        self.assertIsInstance(environments[0], OpenmlSimulation)
        self.assertDictEqual({**environments[0].params, 'openml_data':150}, environments[0].params)

    def test_raw_environment(self):
        json_txt = """{
            "environments" : { "OpenmlSimulation": 150 }
        }"""

        environments = EnvironmentDefinitionFileV1(ListSource([json_txt])).read()

        self.assertIsInstance(environments[0], OpenmlSimulation)
        self.assertDictEqual({**environments[0].params, 'openml_data':150}, environments[0].params)

    def test_one_environment_one_filter(self):
        json_txt = """{
            "environments" : [
                [{ "OpenmlSimulation": 150 }, {"Take":10} ]
            ]
        }"""

        environments = EnvironmentDefinitionFileV1(ListSource([json_txt])).read()
        self.assertDictEqual({**environments[0].params, 'openml_data':150, "take":10}, environments[0].params)

    def test_one_environment_two_filters(self):
        json_txt = """{
            "environments" : [
                [{ "OpenmlSimulation": 150 }, {"Take":[10,20], "method":"foreach"} ]
            ]
        }"""

        environments = EnvironmentDefinitionFileV1(ListSource([json_txt])).read()

        self.assertEqual(2, len(environments))
        self.assertDictEqual({**environments[0].params, 'openml_data':150, "take":10}, environments[0].params)
        self.assertDictEqual({**environments[1].params, 'openml_data':150, "take":20}, environments[1].params)

    def test_two_environments_two_filters(self):
        json_txt = """{
            "environments" : [
                [{ "OpenmlSimulation": [150,151], "method":"foreach" }, { "Take":[10,20], "method":"foreach" }]
            ]
        }"""

        environments = EnvironmentDefinitionFileV1(ListSource([json_txt])).read()

        self.assertEqual(4, len(environments))
        self.assertDictEqual({**environments[0].params, 'openml_data':150, "take":10}, environments[0].params)
        self.assertDictEqual({**environments[1].params, 'openml_data':150, "take":20}, environments[1].params)
        self.assertDictEqual({**environments[2].params, 'openml_data':151, "take":10}, environments[2].params)
        self.assertDictEqual({**environments[3].params, 'openml_data':151, "take":20}, environments[3].params)

    def test_two_singular_environments(self):
        json_txt = """{
            "environments" : [
                {"OpenmlSimulation": 150},
                {"OpenmlSimulation": 151}
            ]
        }"""

        environments = EnvironmentDefinitionFileV1(ListSource([json_txt])).read()

        self.assertIsInstance(environments[0], OpenmlSimulation)
        self.assertIsInstance(environments[1], OpenmlSimulation)
        self.assertDictEqual({**environments[0].params, 'openml_data':150}, environments[0].params)
        self.assertDictEqual({**environments[1].params, 'openml_data':151}, environments[1].params)

    def test_one_foreach_environment(self):
        json_txt = """{
            "environments" : [
                {"OpenmlSimulation": [150,151], "method":"foreach"}
            ]
        }"""

        environments = EnvironmentDefinitionFileV1(ListSource([json_txt])).read()

        self.assertIsInstance(environments[0], OpenmlSimulation)
        self.assertIsInstance(environments[1], OpenmlSimulation)
        self.assertDictEqual({**environments[0].params, 'openml_data':150}, environments[0].params)
        self.assertDictEqual({**environments[1].params, 'openml_data':151}, environments[1].params)

    def test_one_variable(self):
        json_txt = """{
            "variables"    : {"$openml_sims": {"OpenmlSimulation": [150,151], "method":"foreach"} },
            "environments" : [ "$openml_sims" ]
        }"""

        environments = EnvironmentDefinitionFileV1(ListSource([json_txt])).read()

        self.assertIsInstance(environments[0], OpenmlSimulation)
        self.assertIsInstance(environments[1], OpenmlSimulation)
        self.assertDictEqual({**environments[0].params, 'openml_data':150}, environments[0].params)
        self.assertDictEqual({**environments[1].params, 'openml_data':151}, environments[1].params)

    def test_two_variables(self):
        json_txt = """{
            "variables": {
                "$openmls": {"OpenmlSimulation": [150,151], "method":"foreach"},
                "$takes"  : {"Take":[10,20], "method":"foreach"}
            },
            "environments": [
                ["$openmls", "$takes"],
                "$openmls"
            ]
        }"""

        environments = EnvironmentDefinitionFileV1(ListSource([json_txt])).read()

        self.assertEqual(6, len(environments))
        self.assertIsInstance(environments[4], OpenmlSimulation)
        self.assertIsInstance(environments[5], OpenmlSimulation)
        self.assertDictEqual({**environments[0].params, 'openml_data':150, "take":10}, environments[0].params)
        self.assertDictEqual({**environments[1].params, 'openml_data':150, "take":20}, environments[1].params)
        self.assertDictEqual({**environments[2].params, 'openml_data':151, "take":10}, environments[2].params)
        self.assertDictEqual({**environments[3].params, 'openml_data':151, "take":20}, environments[3].params)
        self.assertDictEqual({**environments[4].params, 'openml_data':150           }, environments[4].params)
        self.assertDictEqual({**environments[5].params, 'openml_data':151           }, environments[5].params)

    def test_pipe_list(self):
        json_txt = """{
            "environments" : [
                [ {"OpenmlSimulation":150}, [ {"Take":10}, {"Take":20} ] ]
            ]
        }"""

        environments = EnvironmentDefinitionFileV1(ListSource([json_txt])).read()

        self.assertEqual(2, len(environments))
        self.assertDictEqual({**environments[0].params, 'openml_data':150, "take":10}, environments[0].params)
        self.assertDictEqual({**environments[1].params, 'openml_data':150, "take":20}, environments[1].params)

    def test_pipe_str(self):
        json_txt = """{
            "environments" : [
                [ {"OpenmlSimulation":150}, "Identity" ]
            ]
        }"""

        environments = EnvironmentDefinitionFileV1(ListSource([json_txt])).read()

        self.assertEqual(1, len(environments))
        self.assertDictEqual({**environments[0].params, 'openml_data':150}, environments[0].params)

    def test_bad_pipe_exception(self):
        json_txt = """{
            "environments" : [
                [ {"OpenmlSimulation":150}, null ]
            ]
        }"""

        with self.assertRaises(CobaException) as e:
            EnvironmentDefinitionFileV1(ListSource([json_txt])).read()

        self.assertIn("We were unable to construct",str(e.exception))

if __name__ == '__main__':
    unittest.main()