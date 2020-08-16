import unittest
import json

from coba.execution import TemplatingEngine, UniversalLogger

class TemplatingEngine_Tests(unittest.TestCase):
    def test_no_template_string_unchanged_1(self):
        self.assertEqual(TemplatingEngine.parse("[1,2,3]"), [1,2,3])

    def test_no_template_string_unchanged_2(self):
        actual = TemplatingEngine.parse('{"a":1}')

        self.assertCountEqual(actual.keys(), ["a"])
        self.assertEqual(actual["a"], 1)

    def test_no_template_object_unchanged_1(self):
        self.assertEqual(TemplatingEngine.parse(json.loads("[1,2,3]")), [1,2,3])

    def test_no_template_object_unchanged2(self):
        actual = TemplatingEngine.parse(json.loads('{"a":1}'))

        self.assertCountEqual(actual.keys(), ["a"])
        self.assertEqual(actual["a"], 1)

    def test_templated_config(self):
        json_str = """{
            "templates"  : { "shuffled_openml_classification": { "seed":1283, "type":"classification", "from": {"format":"openml", "id":"$id"} } },
            "batches"    : { "count":100 },
            "simulations": [
                {"template":"shuffled_openml_classification", "$id":3},
                {"template":"shuffled_openml_classification", "$id":6}
            ]
        }"""

        actual = TemplatingEngine.parse(json_str)

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

class UniversalLogger_Tests(unittest.TestCase):

    def test_log(self):

        actual_prints = []

        logger = UniversalLogger(print_function = lambda m,e: actual_prints.append((m,e)) )

        logger.log('a', end='b')
        logger.log('c')
        logger.log('d')

        self.assertEqual(actual_prints[0][0][20:], 'a' )
        self.assertEqual(actual_prints[0][1]     , 'b' )
        self.assertEqual(actual_prints[1][0]     , 'c' )
        self.assertEqual(actual_prints[1][1]     , None)
        self.assertEqual(actual_prints[2][0][20:], 'd' )
        self.assertEqual(actual_prints[2][1]     , None)

    def test_log_with_1(self):

        actual_prints = []

        logger = UniversalLogger(print_function = lambda m,e: actual_prints.append((m,e)) )

        with logger.log('a', end='b'):
            logger.log('c')
            logger.log('d')
        logger.log('e')

        self.assertEqual(actual_prints[0][0][20:], 'a' )
        self.assertEqual(actual_prints[0][1]     , 'b' )
        self.assertEqual(actual_prints[1][0]     , 'c' )
        self.assertEqual(actual_prints[1][1]     , None)
        self.assertEqual(actual_prints[2][0][20:], '  * d')
        self.assertEqual(actual_prints[2][1]     , None)
        self.assertEqual(actual_prints[4][0][20:], 'e')
        self.assertEqual(actual_prints[4][1]     , None)

    def test_log_with_2(self):

        actual_prints = []

        logger = UniversalLogger(print_function = lambda m,e: actual_prints.append((m,e)) )

        with logger.log('a', end='b'):
            logger.log('c')
            with logger.log('d'):
                logger.log('e')
            logger.log('f')
        logger.log('g')

        self.assertEqual(actual_prints[0][0][20:], 'a' )
        self.assertEqual(actual_prints[0][1]     , 'b' )
        self.assertEqual(actual_prints[1][0]     , 'c' )
        self.assertEqual(actual_prints[1][1]     , None)
        self.assertEqual(actual_prints[2][0][20:], '  * d')
        self.assertEqual(actual_prints[2][1]     , None)
        self.assertEqual(actual_prints[3][0][20:], '    > e')
        self.assertEqual(actual_prints[3][1]     , None)
        self.assertEqual(actual_prints[5][0][20:], '  * f')
        self.assertEqual(actual_prints[5][1]     , None)
        self.assertEqual(actual_prints[7][0][20:], 'g')
        self.assertEqual(actual_prints[7][1]     , None)

        #self.assertEqual(actual_prints[0], ('a','b'))
        #self.assertEqual(actual_prints[1], ('c',None))
        #self.assertEqual(actual_prints[2], ('  *d',None))
        #self.assertEqual(actual_prints[3], ('    >e',None))
        #self.assertEqual(actual_prints[4], ('  *f',None))
        #self.assertEqual(actual_prints[5], ('g',None))

if __name__ == '__main__':
    unittest.main()