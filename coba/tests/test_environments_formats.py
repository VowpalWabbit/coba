import json
import unittest

from coba.pipes import Take
from coba.registry import CobaRegistry

from coba.environments.formats import EnvironmentFileFmtV1
from coba.environments.core import SimulatedEnvironment
from coba.environments.openml import OpenmlSimulation

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

        environments = EnvironmentFileFmtV1().filter(json.loads(json_txt))

        self.assertIsInstance(environments[0], SimulatedEnvironment)
        self.assertDictEqual({'openml':150, **environments[0].params}, environments[0].params)

    def test_raw_environment(self):
        json_txt = """{
            "environments" : { "OpenmlSimulation": 150 }
        }"""

        environments = EnvironmentFileFmtV1().filter(json.loads(json_txt))

        self.assertIsInstance(environments[0], SimulatedEnvironment)
        self.assertDictEqual({'openml':150, **environments[0].params}, environments[0].params)

    def test_one_environment_one_filter(self):
        json_txt = """{
            "environments" : [
                [{ "OpenmlSimulation": 150 }, {"Take":10} ]
            ]
        }"""

        environments = EnvironmentFileFmtV1().filter(json.loads(json_txt))

        self.assertIsInstance(environments[0], SimulatedEnvironment)
        self.assertDictEqual({"openml":150, "take":10, **environments[0].params}, environments[0].params)

    def test_one_environment_two_filters(self):
        json_txt = """{
            "environments" : [
                [{ "OpenmlSimulation": 150 }, {"Take":[10,20], "method":"foreach"} ]
            ]
        }"""

        environments = EnvironmentFileFmtV1().filter(json.loads(json_txt))

        self.assertEqual(2, len(environments))
        self.assertIsInstance(environments[0], SimulatedEnvironment)
        self.assertIsInstance(environments[1], SimulatedEnvironment)
        self.assertDictEqual({"openml":150, "take":10, **environments[0].params}, environments[0].params)
        self.assertDictEqual({"openml":150, "take":20, **environments[1].params}, environments[1].params)

    def test_two_environments_two_filters(self):
        json_txt = """{
            "environments" : [
                [{ "OpenmlSimulation": [150,151], "method":"foreach" }, { "Take":[10,20], "method":"foreach" }]
            ]
        }"""

        environments = EnvironmentFileFmtV1().filter(json.loads(json_txt))

        self.assertEqual(4, len(environments))        
        self.assertIsInstance(environments[0], SimulatedEnvironment)
        self.assertIsInstance(environments[1], SimulatedEnvironment)
        self.assertIsInstance(environments[2], SimulatedEnvironment)
        self.assertIsInstance(environments[3], SimulatedEnvironment)
        self.assertDictEqual({"openml":150, "take":10, **environments[0].params}, environments[0].params)
        self.assertDictEqual({"openml":150, "take":20, **environments[1].params}, environments[1].params)
        self.assertDictEqual({"openml":151, "take":10, **environments[2].params}, environments[2].params)
        self.assertDictEqual({"openml":151, "take":20, **environments[3].params}, environments[3].params)

    def test_two_singular_environments(self):
        json_txt = """{
            "environments" : [
                {"OpenmlSimulation": 150},
                {"OpenmlSimulation": 151}
            ]
        }"""

        environments = EnvironmentFileFmtV1().filter(json.loads(json_txt))

        self.assertIsInstance(environments[0], SimulatedEnvironment)
        self.assertIsInstance(environments[1], SimulatedEnvironment)
        self.assertDictEqual({"openml":150, **environments[0].params}, environments[0].params)
        self.assertDictEqual({"openml":151, **environments[1].params}, environments[1].params)

    def test_one_foreach_environment(self):
        json_txt = """{
            "environments" : [
                {"OpenmlSimulation": [150,151], "method":"foreach"}
            ]
        }"""

        environments = EnvironmentFileFmtV1().filter(json.loads(json_txt))

        self.assertIsInstance(environments[0], SimulatedEnvironment)
        self.assertIsInstance(environments[1], SimulatedEnvironment)
        self.assertDictEqual({"openml":150, **environments[0].params}, environments[0].params)
        self.assertDictEqual({"openml":151, **environments[1].params}, environments[1].params)

    def test_one_variable(self):
        json_txt = """{
            "variables"    : {"$openml_sims": {"OpenmlSimulation": [150,151], "method":"foreach"} },
            "environments" : [ "$openml_sims" ]
        }"""

        environments = EnvironmentFileFmtV1().filter(json.loads(json_txt))

        self.assertIsInstance(environments[0], SimulatedEnvironment)
        self.assertIsInstance(environments[1], SimulatedEnvironment)
        self.assertDictEqual({"openml":150, **environments[0].params}, environments[0].params)
        self.assertDictEqual({"openml":151, **environments[1].params}, environments[1].params)

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

        environments = EnvironmentFileFmtV1().filter(json.loads(json_txt))

        self.assertEqual(6, len(environments))
        self.assertIsInstance(environments[0], SimulatedEnvironment)
        self.assertIsInstance(environments[1], SimulatedEnvironment)
        self.assertIsInstance(environments[2], SimulatedEnvironment)
        self.assertIsInstance(environments[3], SimulatedEnvironment)
        self.assertIsInstance(environments[4], SimulatedEnvironment)
        self.assertIsInstance(environments[5], SimulatedEnvironment)
        self.assertDictEqual({"openml":150, "take":10, **environments[0].params}, environments[0].params)
        self.assertDictEqual({"openml":150, "take":20, **environments[1].params}, environments[1].params)
        self.assertDictEqual({"openml":151, "take":10, **environments[2].params}, environments[2].params)
        self.assertDictEqual({"openml":151, "take":20, **environments[3].params}, environments[3].params)
        self.assertDictEqual({"openml":150           , **environments[4].params}, environments[4].params)
        self.assertDictEqual({"openml":151           , **environments[5].params}, environments[5].params)

    def test_pipe_list(self):
        json_txt = """{
            "environments" : [
                [ {"OpenmlSimulation":150}, [ {"Take":10}, {"Take":20} ] ]
            ]
        }"""

        environments = EnvironmentFileFmtV1().filter(json.loads(json_txt))

        self.assertEqual(2, len(environments))
        self.assertIsInstance(environments[0], SimulatedEnvironment)
        self.assertIsInstance(environments[1], SimulatedEnvironment)
        self.assertDictEqual({"openml":150, "take":10, **environments[0].params}, environments[0].params)
        self.assertDictEqual({"openml":150, "take":20, **environments[1].params}, environments[1].params)

if __name__ == '__main__':
    unittest.main()