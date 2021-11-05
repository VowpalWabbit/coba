import json
import unittest

from coba.benchmarks.formats import BenchmarkFileFmtV2
from coba.environments.core import Simulation

class BenchmarkFileFmtV2_Tests(unittest.TestCase):

    def test_one_simulation(self):
        json_txt = """{
            "simulations" : [
                { "OpenmlSimulation": 150 }
            ]
        }"""

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))

        self.assertIsInstance(benchmark._simulations[0], Simulation)
        self.assertEqual({'openml':150}, benchmark._simulations[0].params)

    def test_raw_simulation(self):
        json_txt = """{
            "simulations" : { "OpenmlSimulation": 150 }
        }"""

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))

        self.assertIsInstance(benchmark._simulations[0], Simulation)
        self.assertEqual({'openml':150}, benchmark._simulations[0].params)

    def test_one_simulation_one_filter(self):
        json_txt = """{
            "simulations" : [
                [{ "OpenmlSimulation": 150 }, {"Take":10} ]
            ]
        }"""

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))

        self.assertIsInstance(benchmark._simulations[0], Simulation)
        self.assertEqual({"openml":150, "take":10}, benchmark._simulations[0].params)

    def test_one_simulation_two_filters(self):
        json_txt = """{
            "simulations" : [
                [{ "OpenmlSimulation": 150 }, {"Take":[10,20], "method":"foreach"} ]
            ]
        }"""

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))

        self.assertEqual(2, len(benchmark._simulations))
        self.assertIsInstance(benchmark._simulations[0], Simulation)
        self.assertIsInstance(benchmark._simulations[1], Simulation)
        self.assertEqual({"openml":150, "take":10}, benchmark._simulations[0].params)
        self.assertEqual({"openml":150, "take":20}, benchmark._simulations[1].params)

    def test_two_simulations_two_filters(self):
        json_txt = """{
            "simulations" : [
                [{ "OpenmlSimulation": [150,151], "method":"foreach" }, { "Take":[10,20], "method":"foreach" }]
            ]
        }"""

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))

        self.assertEqual(4, len(benchmark._simulations))        
        self.assertIsInstance(benchmark._simulations[0], Simulation)
        self.assertIsInstance(benchmark._simulations[1], Simulation)
        self.assertIsInstance(benchmark._simulations[2], Simulation)
        self.assertIsInstance(benchmark._simulations[3], Simulation)
        self.assertEqual({"openml":150, "take":10}, benchmark._simulations[0].params)
        self.assertEqual({"openml":150, "take":20}, benchmark._simulations[1].params)
        self.assertEqual({"openml":151, "take":10}, benchmark._simulations[2].params)
        self.assertEqual({"openml":151, "take":20}, benchmark._simulations[3].params)

    def test_two_singular_simulations(self):
        json_txt = """{
            "simulations" : [
                {"OpenmlSimulation": 150},
                {"OpenmlSimulation": 151}
            ]
        }"""

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))

        self.assertIsInstance(benchmark._simulations[0], Simulation)
        self.assertIsInstance(benchmark._simulations[1], Simulation)
        self.assertEqual({"openml":150}, benchmark._simulations[0].params)
        self.assertEqual({"openml":151}, benchmark._simulations[1].params)

    def test_one_foreach_simulation(self):
        json_txt = """{
            "simulations" : [
                {"OpenmlSimulation": [150,151], "method":"foreach"}
            ]
        }"""

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))

        self.assertIsInstance(benchmark._simulations[0], Simulation)
        self.assertIsInstance(benchmark._simulations[1], Simulation)
        self.assertEqual({"openml":150}, benchmark._simulations[0].params)
        self.assertEqual({"openml":151}, benchmark._simulations[1].params)

    def test_one_variable(self):
        json_txt = """{
            "variables": {"$openml_sims": {"OpenmlSimulation": [150,151], "method":"foreach"} },
            "simulations" : [ "$openml_sims" ]
        }"""

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))

        self.assertIsInstance(benchmark._simulations[0], Simulation)
        self.assertIsInstance(benchmark._simulations[1], Simulation)
        self.assertEqual({"openml":150}, benchmark._simulations[0].params)
        self.assertEqual({"openml":151}, benchmark._simulations[1].params)

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

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))

        self.assertEqual(6, len(benchmark._simulations))
        self.assertIsInstance(benchmark._simulations[0], Simulation)
        self.assertIsInstance(benchmark._simulations[1], Simulation)
        self.assertIsInstance(benchmark._simulations[2], Simulation)
        self.assertIsInstance(benchmark._simulations[3], Simulation)
        self.assertIsInstance(benchmark._simulations[4], Simulation)
        self.assertIsInstance(benchmark._simulations[5], Simulation)
        self.assertEqual({"openml":150, "take":10}, benchmark._simulations[0].params)
        self.assertEqual({"openml":150, "take":20}, benchmark._simulations[1].params)
        self.assertEqual({"openml":151, "take":10}, benchmark._simulations[2].params)
        self.assertEqual({"openml":151, "take":20}, benchmark._simulations[3].params)
        self.assertEqual({"openml":150           }, benchmark._simulations[4].params)
        self.assertEqual({"openml":151           }, benchmark._simulations[5].params)

    def test_pipe_list(self):
        json_txt = """{
            "simulations" : [
                [ {"OpenmlSimulation":150}, [ {"Take":10}, {"Take":20} ] ]
            ]
        }"""

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))

        self.assertEqual(2, len(benchmark._simulations))
        self.assertIsInstance(benchmark._simulations[0], Simulation)
        self.assertIsInstance(benchmark._simulations[1], Simulation)
        self.assertEqual({"openml":150, "take":10}, benchmark._simulations[0].params)
        self.assertEqual({"openml":150, "take":20}, benchmark._simulations[1].params)

if __name__ == '__main__':
    unittest.main()