import json
import unittest

from coba.benchmarks.formats import BenchmarkFileFmtV1, BenchmarkFileFmtV2

class BenchmarkFileFmtV1_Tests(unittest.TestCase):
    def test_materialize_templates_sans_template_1(self):        
        self.assertEqual(BenchmarkFileFmtV1().materialize_templates(json.loads("[1,2,3]")), [1,2,3])

    def test_materialize_templates_sans_template_2(self):
        actual = BenchmarkFileFmtV1().materialize_templates(json.loads('{"a":1}'))

        self.assertCountEqual(actual.keys(), ["a"])
        self.assertEqual(actual["a"], 1)

    def test_materialize_template_with_templates(self):
        json_str = """{
            "templates"  : { "shuffled_openml_classification": { "seed":1283, "type":"classification", "from": {"format":"openml", "id":"$id"} } },
            "batches"    : { "count":100 },
            "simulations": [
                {"template":"shuffled_openml_classification", "$id":3},
                {"template":"shuffled_openml_classification", "$id":6}
            ]
        }"""

        actual = BenchmarkFileFmtV1().materialize_templates(json.loads(json_str))

        self.assertCountEqual(actual.keys(), ["batches", "simulations"])
        self.assertCountEqual(actual["batches"], ["count"])
        self.assertEqual(len(actual["simulations"]), 2)

        for simulation in actual["simulations"]:
            self.assertCountEqual(simulation, ["seed", "type", "from"])
            self.assertEqual(simulation["seed"], 1283)
            self.assertEqual(simulation["type"], "classification")
            self.assertCountEqual(simulation["from"], ["format", "id"])
            self.assertEqual(simulation["from"]["format"], "openml")

        self.assertCountEqual([ sim["from"]["id"] for sim in actual["simulations"] ], [3,6])

    def test_parse(self):
        json_txt = """{
            "batches"     : {"count":1},
            "ignore_first": false,
            "shuffle"     : [1283],
            "simulations" : [
                {"type":"classification","from":{"format":"openml","id":1116}}
            ]
        }"""

        benchmark = BenchmarkFileFmtV1().filter(json.loads(json_txt))

        self.assertEqual(1, len(benchmark._simulations))

class BenchmarkFileFmtV2_Tests(unittest.TestCase):

    def test_one_simulation(self):
        json_txt = """{
            "simulations" : [
                { "OpenmlSimulation": 150 }
            ]
        }"""

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))

        self.assertEqual('[{"OpenmlSimulation":150}]', str(benchmark._simulations))

    def test_raw_simulation(self):
        json_txt = """{
            "simulations" : { "OpenmlSimulation": 150 }
        }"""

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))

        self.assertEqual('[{"OpenmlSimulation":150}]', str(benchmark._simulations))

    def test_one_simulation_one_filter(self):
        json_txt = """{
            "simulations" : [
                [{ "OpenmlSimulation": 150 }, {"Take":10} ]
            ]
        }"""

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))

        self.assertEqual('[{"OpenmlSimulation":150},{"Take":10}]', str(benchmark._simulations))

    def test_one_simulation_two_filters(self):
        json_txt = """{
            "simulations" : [
                [{ "OpenmlSimulation": 150 }, {"Take":[10,20], "method":"foreach"} ]
            ]
        }"""

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))

        self.assertEqual('[{"OpenmlSimulation":150},{"Take":10}, {"OpenmlSimulation":150},{"Take":20}]', str(benchmark._simulations))

    def test_two_simulations_two_filters(self):
        json_txt = """{
            "simulations" : [
                [{ "OpenmlSimulation": [150,151], "method":"foreach" }, { "Take":[10,20], "method":"foreach" }]
            ]
        }"""

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))

        self.assertEqual(4, len(benchmark._simulations))
        self.assertEqual('{"OpenmlSimulation":150},{"Take":10}', str(benchmark._simulations[0]))
        self.assertEqual('{"OpenmlSimulation":150},{"Take":20}', str(benchmark._simulations[1]))
        self.assertEqual('{"OpenmlSimulation":151},{"Take":10}', str(benchmark._simulations[2]))
        self.assertEqual('{"OpenmlSimulation":151},{"Take":20}', str(benchmark._simulations[3]))

    def test_two_singular_simulations(self):
        json_txt = """{
            "simulations" : [
                { "OpenmlSimulation": 150},
                { "OpenmlSimulation": 151}
            ]
        }"""

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))

        self.assertEqual('[{"OpenmlSimulation":150}, {"OpenmlSimulation":151}]', str(benchmark._simulations))

    def test_one_foreach_simulation(self):
        json_txt = """{
            "simulations" : [
                {"OpenmlSimulation": [150,151], "method":"foreach"}
            ]
        }"""

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))

        self.assertEqual('[{"OpenmlSimulation":150}, {"OpenmlSimulation":151}]', str(benchmark._simulations))

    def test_one_variable(self):
        json_txt = """{
            "variables": {"$openml_sims": {"OpenmlSimulation": [150,151], "method":"foreach"} },
            "simulations" : [ "$openml_sims" ]
        }"""

        benchmark = BenchmarkFileFmtV2().filter(json.loads(json_txt))
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
        self.assertEqual('{"OpenmlSimulation":150},{"Take":10}', str(benchmark._simulations[0]))
        self.assertEqual('{"OpenmlSimulation":150},{"Take":20}', str(benchmark._simulations[1]))
        self.assertEqual('{"OpenmlSimulation":151},{"Take":10}', str(benchmark._simulations[2]))
        self.assertEqual('{"OpenmlSimulation":151},{"Take":20}', str(benchmark._simulations[3]))
        self.assertEqual('{"OpenmlSimulation":150}'            , str(benchmark._simulations[4]))
        self.assertEqual('{"OpenmlSimulation":151}'            , str(benchmark._simulations[5]))

if __name__ == '__main__':
    unittest.main()