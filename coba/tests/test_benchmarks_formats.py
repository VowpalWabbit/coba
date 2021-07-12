import json
import unittest

from coba.benchmarks.formats import BenchmarkFileFmtV2

class BenchmarkFileFmtV2_Tests(unittest.TestCase):

    def test_one_simulation(self):
        json_txt = """{
            "simulations" : [
                { "OpenmlSimulation": 150 }
            ]
        }"""

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))

        self.assertTrue(hasattr(benchmark._simulations[0], "read"))
        self.assertEqual('[{"OpenmlSimulation":150}]', str(benchmark._simulations))

    def test_raw_simulation(self):
        json_txt = """{
            "simulations" : { "OpenmlSimulation": 150 }
        }"""

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))

        self.assertTrue(hasattr(benchmark._simulations[0], "read"))
        self.assertEqual('[{"OpenmlSimulation":150}]', str(benchmark._simulations))

    def test_one_simulation_one_filter(self):
        json_txt = """{
            "simulations" : [
                [{ "OpenmlSimulation": 150 }, {"Take":10} ]
            ]
        }"""

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))

        self.assertTrue(hasattr(benchmark._simulations[0]._source, "read"))
        self.assertEqual('[{"OpenmlSimulation":150},{"Take":10}]', str(benchmark._simulations))

    def test_one_simulation_two_filters(self):
        json_txt = """{
            "simulations" : [
                [{ "OpenmlSimulation": 150 }, {"Take":[10,20], "method":"foreach"} ]
            ]
        }"""

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))

        self.assertTrue(hasattr(benchmark._simulations[0]._source, "read"))
        self.assertTrue(hasattr(benchmark._simulations[1]._source, "read"))
        self.assertEqual('[{"OpenmlSimulation":150},{"Take":10}, {"OpenmlSimulation":150},{"Take":20}]', str(benchmark._simulations))

    def test_two_simulations_two_filters(self):
        json_txt = """{
            "simulations" : [
                [{ "OpenmlSimulation": [150,151], "method":"foreach" }, { "Take":[10,20], "method":"foreach" }]
            ]
        }"""

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))

        self.assertEqual(4, len(benchmark._simulations))        
        self.assertTrue(hasattr(benchmark._simulations[0]._source, "read"))
        self.assertTrue(hasattr(benchmark._simulations[1]._source, "read"))
        self.assertTrue(hasattr(benchmark._simulations[2]._source, "read"))
        self.assertTrue(hasattr(benchmark._simulations[3]._source, "read"))
        self.assertEqual('{"OpenmlSimulation":150},{"Take":10}', str(benchmark._simulations[0]))
        self.assertEqual('{"OpenmlSimulation":150},{"Take":20}', str(benchmark._simulations[1]))
        self.assertEqual('{"OpenmlSimulation":151},{"Take":10}', str(benchmark._simulations[2]))
        self.assertEqual('{"OpenmlSimulation":151},{"Take":20}', str(benchmark._simulations[3]))

    def test_two_singular_simulations(self):
        json_txt = """{
            "simulations" : [
                {"OpenmlSimulation": 150},
                {"OpenmlSimulation": 151}
            ]
        }"""

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))

        self.assertTrue(hasattr(benchmark._simulations[0], "read"))
        self.assertTrue(hasattr(benchmark._simulations[1], "read"))
        self.assertEqual('[{"OpenmlSimulation":150}, {"OpenmlSimulation":151}]', str(benchmark._simulations))

    def test_one_foreach_simulation(self):
        json_txt = """{
            "simulations" : [
                {"OpenmlSimulation": [150,151], "method":"foreach"}
            ]
        }"""

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))

        self.assertTrue(hasattr(benchmark._simulations[0], "read"))
        self.assertTrue(hasattr(benchmark._simulations[1], "read"))
        self.assertEqual('[{"OpenmlSimulation":150}, {"OpenmlSimulation":151}]', str(benchmark._simulations))

    def test_one_variable(self):
        json_txt = """{
            "variables": {"$openml_sims": {"OpenmlSimulation": [150,151], "method":"foreach"} },
            "simulations" : [ "$openml_sims" ]
        }"""

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))

        self.assertTrue(hasattr(benchmark._simulations[0], "read"))
        self.assertTrue(hasattr(benchmark._simulations[1], "read"))
        self.assertEqual('[{"OpenmlSimulation":150}, {"OpenmlSimulation":151}]', str(benchmark._simulations))

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
        self.assertTrue(hasattr(benchmark._simulations[0]._source, "read"))
        self.assertTrue(hasattr(benchmark._simulations[1]._source, "read"))
        self.assertTrue(hasattr(benchmark._simulations[2]._source, "read"))
        self.assertTrue(hasattr(benchmark._simulations[3]._source, "read"))
        self.assertTrue(hasattr(benchmark._simulations[4], "read"))
        self.assertTrue(hasattr(benchmark._simulations[5], "read"))
        self.assertEqual('{"OpenmlSimulation":150},{"Take":10}', str(benchmark._simulations[0]))
        self.assertEqual('{"OpenmlSimulation":150},{"Take":20}', str(benchmark._simulations[1]))
        self.assertEqual('{"OpenmlSimulation":151},{"Take":10}', str(benchmark._simulations[2]))
        self.assertEqual('{"OpenmlSimulation":151},{"Take":20}', str(benchmark._simulations[3]))
        self.assertEqual('{"OpenmlSimulation":150}'            , str(benchmark._simulations[4]))
        self.assertEqual('{"OpenmlSimulation":151}'            , str(benchmark._simulations[5]))

if __name__ == '__main__':
    unittest.main()