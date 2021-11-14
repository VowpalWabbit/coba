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

    def test_one_simulation(self):
        
        json_txt = """{
            "simulations" : [
                { "OpenmlSimulation": 150 }
            ]
        }"""        

        simulations = EnvironmentFileFmtV1().filter(json.loads(json_txt))

        self.assertIsInstance(simulations[0], SimulatedEnvironment)
        self.assertEqual({'openml':150}, simulations[0].params)

    def test_raw_simulation(self):
        json_txt = """{
            "simulations" : { "OpenmlSimulation": 150 }
        }"""

        simulations = EnvironmentFileFmtV1().filter(json.loads(json_txt))

        self.assertIsInstance(simulations[0], SimulatedEnvironment)
        self.assertEqual({'openml':150}, simulations[0].params)

    def test_one_simulation_one_filter(self):
        json_txt = """{
            "simulations" : [
                [{ "OpenmlSimulation": 150 }, {"Take":10} ]
            ]
        }"""

        simulations = EnvironmentFileFmtV1().filter(json.loads(json_txt))

        self.assertIsInstance(simulations[0], SimulatedEnvironment)
        self.assertEqual({"openml":150, "take":10}, simulations[0].params)

    def test_one_simulation_two_filters(self):
        json_txt = """{
            "simulations" : [
                [{ "OpenmlSimulation": 150 }, {"Take":[10,20], "method":"foreach"} ]
            ]
        }"""

        simulations = EnvironmentFileFmtV1().filter(json.loads(json_txt))

        self.assertEqual(2, len(simulations))
        self.assertIsInstance(simulations[0], SimulatedEnvironment)
        self.assertIsInstance(simulations[1], SimulatedEnvironment)
        self.assertEqual({"openml":150, "take":10}, simulations[0].params)
        self.assertEqual({"openml":150, "take":20}, simulations[1].params)

    def test_two_simulations_two_filters(self):
        json_txt = """{
            "simulations" : [
                [{ "OpenmlSimulation": [150,151], "method":"foreach" }, { "Take":[10,20], "method":"foreach" }]
            ]
        }"""

        simulations = EnvironmentFileFmtV1().filter(json.loads(json_txt))

        self.assertEqual(4, len(simulations))        
        self.assertIsInstance(simulations[0], SimulatedEnvironment)
        self.assertIsInstance(simulations[1], SimulatedEnvironment)
        self.assertIsInstance(simulations[2], SimulatedEnvironment)
        self.assertIsInstance(simulations[3], SimulatedEnvironment)
        self.assertEqual({"openml":150, "take":10}, simulations[0].params)
        self.assertEqual({"openml":150, "take":20}, simulations[1].params)
        self.assertEqual({"openml":151, "take":10}, simulations[2].params)
        self.assertEqual({"openml":151, "take":20}, simulations[3].params)

    def test_two_singular_simulations(self):
        json_txt = """{
            "simulations" : [
                {"OpenmlSimulation": 150},
                {"OpenmlSimulation": 151}
            ]
        }"""

        simulations = EnvironmentFileFmtV1().filter(json.loads(json_txt))

        self.assertIsInstance(simulations[0], SimulatedEnvironment)
        self.assertIsInstance(simulations[1], SimulatedEnvironment)
        self.assertEqual({"openml":150}, simulations[0].params)
        self.assertEqual({"openml":151}, simulations[1].params)

    def test_one_foreach_simulation(self):
        json_txt = """{
            "simulations" : [
                {"OpenmlSimulation": [150,151], "method":"foreach"}
            ]
        }"""

        simulations = EnvironmentFileFmtV1().filter(json.loads(json_txt))

        self.assertIsInstance(simulations[0], SimulatedEnvironment)
        self.assertIsInstance(simulations[1], SimulatedEnvironment)
        self.assertEqual({"openml":150}, simulations[0].params)
        self.assertEqual({"openml":151}, simulations[1].params)

    def test_one_variable(self):
        json_txt = """{
            "variables": {"$openml_sims": {"OpenmlSimulation": [150,151], "method":"foreach"} },
            "simulations" : [ "$openml_sims" ]
        }"""

        simulations = EnvironmentFileFmtV1().filter(json.loads(json_txt))

        self.assertIsInstance(simulations[0], SimulatedEnvironment)
        self.assertIsInstance(simulations[1], SimulatedEnvironment)
        self.assertEqual({"openml":150}, simulations[0].params)
        self.assertEqual({"openml":151}, simulations[1].params)

    def test_two_variables(self):
        json_txt = """{
            "variables": {
                "$openmls": {"OpenmlSimulation": [150,151], "method":"foreach"},
                "$takes"  : {"Take":[10,20], "method":"foreach"}
            },
            "simulations" : [
                ["$openmls", "$takes"],
                "$openmls"
            ]
        }"""

        simulations = EnvironmentFileFmtV1().filter(json.loads(json_txt))

        self.assertEqual(6, len(simulations))
        self.assertIsInstance(simulations[0], SimulatedEnvironment)
        self.assertIsInstance(simulations[1], SimulatedEnvironment)
        self.assertIsInstance(simulations[2], SimulatedEnvironment)
        self.assertIsInstance(simulations[3], SimulatedEnvironment)
        self.assertIsInstance(simulations[4], SimulatedEnvironment)
        self.assertIsInstance(simulations[5], SimulatedEnvironment)
        self.assertEqual({"openml":150, "take":10}, simulations[0].params)
        self.assertEqual({"openml":150, "take":20}, simulations[1].params)
        self.assertEqual({"openml":151, "take":10}, simulations[2].params)
        self.assertEqual({"openml":151, "take":20}, simulations[3].params)
        self.assertEqual({"openml":150           }, simulations[4].params)
        self.assertEqual({"openml":151           }, simulations[5].params)

    def test_pipe_list(self):
        json_txt = """{
            "simulations" : [
                [ {"OpenmlSimulation":150}, [ {"Take":10}, {"Take":20} ] ]
            ]
        }"""

        simulations = EnvironmentFileFmtV1().filter(json.loads(json_txt))

        self.assertEqual(2, len(simulations))
        self.assertIsInstance(simulations[0], SimulatedEnvironment)
        self.assertIsInstance(simulations[1], SimulatedEnvironment)
        self.assertEqual({"openml":150, "take":10}, simulations[0].params)
        self.assertEqual({"openml":150, "take":20}, simulations[1].params)

if __name__ == '__main__':
    unittest.main()