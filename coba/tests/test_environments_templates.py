import unittest

from coba.pipes import IterableSource, ListSink
from coba.registry import CobaRegistry
from coba.exceptions import CobaException
from coba.contexts import CobaContext, IndentLogger

from coba.environments.templates import EnvironmentsTemplateV1, EnvironmentsTemplateV2
from coba.environments import OpenmlSimulation, Take

SimulatedEnvironment = OpenmlSimulation

class EnvironmentsTemplateV1_Tests(unittest.TestCase):

    def setUp(self) -> None:
        CobaRegistry.register("OpenmlSimulation", OpenmlSimulation)
        CobaRegistry.register("Take", Take)

    def test_one_environment(self):

        json_txt = """{
            "environments" : [
                { "OpenmlSimulation": 150 }
            ]
        }"""

        environments = EnvironmentsTemplateV1(IterableSource([json_txt])).read()

        self.assertIsInstance(environments[0], OpenmlSimulation)
        self.assertDictEqual({**environments[0].params, 'openml_data':150}, environments[0].params)

    def test_raw_environment(self):
        json_txt = """{
            "environments" : { "OpenmlSimulation": 150 }
        }"""

        environments = EnvironmentsTemplateV1(IterableSource([json_txt])).read()

        self.assertIsInstance(environments[0], OpenmlSimulation)
        self.assertDictEqual({**environments[0].params, 'openml_data':150}, environments[0].params)

    def test_one_environment_one_filter(self):
        json_txt = """{
            "environments" : [
                [{ "OpenmlSimulation": 150 }, {"Take":10} ]
            ]
        }"""

        environments = EnvironmentsTemplateV1(IterableSource([json_txt])).read()
        self.assertDictEqual({**environments[0].params, 'openml_data':150, "take":10}, environments[0].params)

    def test_one_environment_two_filters(self):
        json_txt = """{
            "environments" : [
                [{ "OpenmlSimulation": 150 }, {"Take":[10,20], "method":"foreach"} ]
            ]
        }"""

        environments = EnvironmentsTemplateV1(IterableSource([json_txt])).read()

        self.assertEqual(2, len(environments))
        self.assertDictEqual({**environments[0].params, 'openml_data':150, "take":10}, environments[0].params)
        self.assertDictEqual({**environments[1].params, 'openml_data':150, "take":20}, environments[1].params)

    def test_two_environments_two_filters(self):
        json_txt = """{
            "environments" : [
                [{ "OpenmlSimulation": [150,151], "method":"foreach" }, { "Take":[10,20], "method":"foreach" }]
            ]
        }"""

        environments = EnvironmentsTemplateV1(IterableSource([json_txt])).read()

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

        environments = EnvironmentsTemplateV1(IterableSource([json_txt])).read()

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

        environments = EnvironmentsTemplateV1(IterableSource([json_txt])).read()

        self.assertIsInstance(environments[0], OpenmlSimulation)
        self.assertIsInstance(environments[1], OpenmlSimulation)
        self.assertDictEqual({**environments[0].params, 'openml_data':150}, environments[0].params)
        self.assertDictEqual({**environments[1].params, 'openml_data':151}, environments[1].params)

    def test_one_variable(self):
        json_txt = """{
            "variables"    : {"$openml_sims": {"OpenmlSimulation": [150,151], "method":"foreach"} },
            "environments" : [ "$openml_sims" ]
        }"""

        environments = EnvironmentsTemplateV1(IterableSource([json_txt])).read()

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

        environments = EnvironmentsTemplateV1(IterableSource([json_txt])).read()

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

        environments = EnvironmentsTemplateV1(IterableSource([json_txt])).read()

        self.assertEqual(2, len(environments))
        self.assertDictEqual({**environments[0].params, 'openml_data':150, "take":10}, environments[0].params)
        self.assertDictEqual({**environments[1].params, 'openml_data':150, "take":20}, environments[1].params)

    def test_pipe_str(self):
        json_txt = """{
            "environments" : [
                [ {"OpenmlSimulation":150}, "Identity" ]
            ]
        }"""

        environments = EnvironmentsTemplateV1(IterableSource([json_txt])).read()

        self.assertEqual(1, len(environments))
        self.assertDictEqual({**environments[0].params, 'openml_data':150}, environments[0].params)

    def test_bad_pipe_exception(self):
        json_txt = """{
            "environments" : [
                [ {"OpenmlSimulation":150}, null ]
            ]
        }"""

        with self.assertRaises(CobaException) as e:
            EnvironmentsTemplateV1(IterableSource([json_txt])).read()

        self.assertIn("We were unable to construct",str(e.exception))

class EnvironmentsTemplateV2_Tests(unittest.TestCase):

    def setUp(self) -> None:
        CobaRegistry.register("OpenmlSimulation", OpenmlSimulation)
        CobaRegistry.register("Take", Take)

    def test_raw_environment(self):
        json_txt = """{
            "environments" : { "OpenmlSimulation": 150 }
        }"""

        environments = EnvironmentsTemplateV2(IterableSource([json_txt])).read()

        self.assertIsInstance(environments[0], OpenmlSimulation)
        self.assertDictEqual({**environments[0].params, 'openml_data':150}, environments[0].params)

    def test_one_environment(self):

        json_txt = """{
            "environments" : [
                { "OpenmlSimulation": 150 }
            ]
        }"""

        environments = EnvironmentsTemplateV2(IterableSource([json_txt])).read()

        self.assertIsInstance(environments[0], OpenmlSimulation)
        self.assertDictEqual({**environments[0].params, 'openml_data':150}, environments[0].params)

    def test_two_environments(self):
        
        json_txt = """{
            "environments" : [
                {"OpenmlSimulation": 150},
                {"OpenmlSimulation": 151}
            ]
        }"""

        environments = EnvironmentsTemplateV2(IterableSource([json_txt])).read()

        self.assertIsInstance(environments[0], OpenmlSimulation)
        self.assertIsInstance(environments[1], OpenmlSimulation)
        self.assertDictEqual({**environments[0].params, 'openml_data':150}, environments[0].params)
        self.assertDictEqual({**environments[1].params, 'openml_data':151}, environments[1].params)

    def test_one_environment_one_filter(self):
        json_txt = """{
            "environments" : [
                [{ "OpenmlSimulation": 150 }, {"Take":10}]
            ]
        }"""

        environments = EnvironmentsTemplateV2(IterableSource([json_txt])).read()
        self.assertDictEqual({**environments[0].params, 'openml_data':150, "take":10}, environments[0].params)

    def test_one_environment_two_filters(self):
        
        json_txt = """{
            "environments" : [
                [{ "OpenmlSimulation": 150 }, [{"Take":10}, {"Take":20}] ]
            ]
        }"""

        environments = EnvironmentsTemplateV2(IterableSource([json_txt])).read()

        self.assertEqual(2, len(environments))
        self.assertDictEqual({**environments[0].params, 'openml_data':150, "take":10}, environments[0].params)
        self.assertDictEqual({**environments[1].params, 'openml_data':150, "take":20}, environments[1].params)

    def test_two_environments_two_filters(self):
        
        json_txt = """{
            "environments" : [
                [ [{"OpenmlSimulation":150},{"OpenmlSimulation":151}], [{"Take":10},{"Take":20}] ]
            ]
        }"""

        environments = EnvironmentsTemplateV2(IterableSource([json_txt])).read()

        self.assertEqual(4, len(environments))
        self.assertDictEqual({**environments[0].params, 'openml_data':150, "take":10}, environments[0].params)
        self.assertDictEqual({**environments[1].params, 'openml_data':150, "take":20}, environments[1].params)
        self.assertDictEqual({**environments[2].params, 'openml_data':151, "take":10}, environments[2].params)
        self.assertDictEqual({**environments[3].params, 'openml_data':151, "take":20}, environments[3].params)

    def test_one_for_environment(self):

        json_txt = """{
            "environments" : [
                {"OpenmlSimulation": "$", "for":[150,151]}
            ]
        }"""

        environments = EnvironmentsTemplateV2(IterableSource([json_txt])).read()

        self.assertIsInstance(environments[0], OpenmlSimulation)
        self.assertIsInstance(environments[1], OpenmlSimulation)
        self.assertDictEqual({**environments[0].params, 'openml_data':150}, environments[0].params)
        self.assertDictEqual({**environments[1].params, 'openml_data':151}, environments[1].params)

    def test_one_variable(self):
        json_txt = """{
            "variables"    : { "$openml_sims": {"OpenmlSimulation":"$", "for":[150,151]} },
            "environments" : [ "$openml_sims" ]
        }"""

        environments = EnvironmentsTemplateV2(IterableSource([json_txt])).read()

        self.assertIsInstance(environments[0], OpenmlSimulation)
        self.assertIsInstance(environments[1], OpenmlSimulation)
        self.assertDictEqual({**environments[0].params, 'openml_data':150}, environments[0].params)
        self.assertDictEqual({**environments[1].params, 'openml_data':151}, environments[1].params)

    def test_two_variables(self):
        json_txt = """{
            "variables": {
                "$openmls": {"OpenmlSimulation":"$", "for":[150,151]},
                "$takes"  : {"Take":"$", "for":[10,20]}
            },
            "environments": [
                ["$openmls", "$takes"],
                "$openmls"
            ]
        }"""

        environments = EnvironmentsTemplateV2(IterableSource([json_txt])).read()

        self.assertEqual(6, len(environments))
        self.assertIsInstance(environments[4], OpenmlSimulation)
        self.assertIsInstance(environments[5], OpenmlSimulation)
        self.assertDictEqual({**environments[0].params, 'openml_data':150, "take":10}, environments[0].params)
        self.assertDictEqual({**environments[1].params, 'openml_data':150, "take":20}, environments[1].params)
        self.assertDictEqual({**environments[2].params, 'openml_data':151, "take":10}, environments[2].params)
        self.assertDictEqual({**environments[3].params, 'openml_data':151, "take":20}, environments[3].params)
        self.assertDictEqual({**environments[4].params, 'openml_data':150           }, environments[4].params)
        self.assertDictEqual({**environments[5].params, 'openml_data':151           }, environments[5].params)

    def test_direct_variable(self):
        json_txt = """{
            "variables"    : { "$dataids": [150,151] },
            "environments" : { "OpenmlSimulation":"$", "for": "$dataids" }
        }"""

        environments = EnvironmentsTemplateV2(IterableSource([json_txt])).read()

        self.assertIsInstance(environments[0], OpenmlSimulation)
        self.assertIsInstance(environments[1], OpenmlSimulation)
        self.assertDictEqual({**environments[0].params, 'openml_data':150}, environments[0].params)
        self.assertDictEqual({**environments[1].params, 'openml_data':151}, environments[1].params)

    def test_recursive_variables(self):
        json_txt = """{
            "variables": {
                "$dataids": [150,151],
                "$openmls": {"OpenmlSimulation":"$", "for":"$dataids"}
            },
            "environments": [
                "$openmls"
            ]
        }"""

        environments = EnvironmentsTemplateV2(IterableSource([json_txt])).read()

        self.assertEqual(2, len(environments))
        self.assertIsInstance(environments[0], OpenmlSimulation)
        self.assertIsInstance(environments[1], OpenmlSimulation)
        self.assertDictEqual({**environments[0].params, 'openml_data':150}, environments[0].params)
        self.assertDictEqual({**environments[1].params, 'openml_data':151}, environments[1].params)

    def test_recursive_infinite(self):
        json_txt = """{
            "variables": {
                "$dataids": [0, "$dataids"]
            }
        }"""

        with self.assertRaises(CobaException) as e:
            EnvironmentsTemplateV2(IterableSource([json_txt])).read()

        self.assertEqual("There appears to be an infinite loop in the template.", str(e.exception))

    def test_undefined_variable(self):
        json_txt = """{
            "environments": [
                "$openmls"
            ]
        }"""

        with self.assertRaises(CobaException) as e:
            EnvironmentsTemplateV2(IterableSource([json_txt])).read()

        self.assertEqual("The following variables were not defined: [$openmls]", str(e.exception))

    def test_user_defined_variable(self):
        json_txt = """{
            "environments": [
                "$openmls"
            ]
        }"""

        environments = EnvironmentsTemplateV2(IterableSource([json_txt]), openmls={"OpenmlSimulation":150}).read()

        self.assertEqual(len(environments), 1)
        self.assertIsInstance(environments[0], OpenmlSimulation)
        self.assertDictEqual({**environments[0].params, 'openml_data':150}, environments[0].params)

    def test_user_defined_variable_overrides(self):
        json_txt = """{
            "variables": {
                "$openmls": {"OpenmlSimulation":151}
            },
            "environments": [
                "$openmls"
            ]
        }"""

        environments = EnvironmentsTemplateV2(IterableSource([json_txt]), openmls={"OpenmlSimulation":150}).read()

        self.assertEqual(len(environments), 1)
        self.assertIsInstance(environments[0], OpenmlSimulation)
        self.assertDictEqual({**environments[0].params, 'openml_data':150}, environments[0].params)

        environments = EnvironmentsTemplateV2(IterableSource([json_txt])).read()

        self.assertEqual(len(environments), 1)
        self.assertIsInstance(environments[0], OpenmlSimulation)
        self.assertDictEqual({**environments[0].params, 'openml_data':151}, environments[0].params)

    def test_unused_user_defined_variable(self):
        json_txt = """{
            "environments": [
                "$openmls"
            ]
        }"""

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        environments = EnvironmentsTemplateV2(IterableSource([json_txt]), openmls={"OpenmlSimulation":150}, O={"OpenmlSimulation":150}).read()

        self.assertEqual(len(environments), 1)
        self.assertIsInstance(environments[0], OpenmlSimulation)
        self.assertDictEqual({**environments[0].params, 'openml_data':150}, environments[0].params)

        expected_log = "The following provided variables were not used by the template:['O']"

        self.assertEqual(expected_log, CobaContext.logger.sink.items[0])

    def test_str_pipe(self):
        json_txt = """{
            "environments" : [
                [ {"OpenmlSimulation":150}, "Identity" ]
            ]
        }"""

        environments = EnvironmentsTemplateV2(IterableSource([json_txt])).read()

        self.assertEqual(1, len(environments))
        self.assertDictEqual({**environments[0].params, 'openml_data':150}, environments[0].params)

    def test_bad_pipe_exception(self):
        json_txt = """{
            "environments" : [
                [ {"OpenmlSimulation":150}, null ]
            ]
        }"""

        with self.assertRaises(CobaException) as e:
            EnvironmentsTemplateV2(IterableSource([json_txt])).read()

        self.assertIn("We were unable to construct",str(e.exception))

if __name__ == '__main__':
    unittest.main()
    