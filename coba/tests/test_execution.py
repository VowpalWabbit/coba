import unittest
import json

from gzip import compress, decompress
from pathlib import Path
from io import BytesIO

from coba.execution import DiskCache, TemplatingEngine, UniversalLogger

class TemplatingEngine_Tests(unittest.TestCase):
    def test_no_template_string_unchanged_1(self):
        self.assertEqual(TemplatingEngine().parse("[1,2,3]"), [1,2,3])

    def test_no_template_string_unchanged_2(self):
        actual = TemplatingEngine().parse('{"a":1}')

        self.assertCountEqual(actual.keys(), ["a"])
        self.assertEqual(actual["a"], 1)

    def test_no_template_object_unchanged_1(self):
        self.assertEqual(TemplatingEngine().parse(json.loads("[1,2,3]")), [1,2,3])

    def test_no_template_object_unchanged2(self):
        actual = TemplatingEngine().parse(json.loads('{"a":1}'))

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

        actual = TemplatingEngine().parse(json_str)

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

    def test_log_exception_1(self):
        actual_prints = []
        exception = Exception("Test Exception")

        logger = UniversalLogger(print_function = lambda m,e: actual_prints.append((m,e)))

        logger.log_exception(exception)

        self.assertTrue(hasattr(exception, '__logged__'))
        self.assertEqual(actual_prints[0][0][20:], "Test Exception")
        self.assertEqual(actual_prints[0][1], None)

        logger.log_exception(exception)

        self.assertEqual(len(actual_prints), 1)

    def test_log_exception_2(self):
        actual_prints = []
        exception = Exception("Test Exception")

        logger = UniversalLogger(print_function = lambda m,e: actual_prints.append((m,e)))

        logger.log('a', end='b')
        logger.log_exception(exception)

        self.assertTrue(hasattr(exception, '__logged__'))
        self.assertEqual(actual_prints[0][0][20:], "a")
        self.assertEqual(actual_prints[0][1]     , "b")
        self.assertEqual(actual_prints[1][0][20:], '')
        self.assertEqual(actual_prints[1][1]     , None)
        self.assertEqual(actual_prints[2][0][20:], "Test Exception")
        self.assertEqual(actual_prints[2][1], None)

        logger.log_exception(exception)

    def test_log_exception_3(self):
        actual_prints = []
        exception = Exception("Test Exception")

        logger = UniversalLogger(print_function = lambda m,e: actual_prints.append((m,e)))

        try:
            with logger.log('a', end='b'):
                raise exception
        except:
            pass

        self.assertTrue(hasattr(exception, '__logged__'))
        self.assertEqual(actual_prints[0][0][20:], "a")
        self.assertEqual(actual_prints[0][1]     , "b")
        self.assertEqual(actual_prints[1][0][20:], '')
        self.assertEqual(actual_prints[1][1]     , None)
        self.assertEqual(actual_prints[2][0][53:], "Test Exception")
        self.assertEqual(actual_prints[2][1], None)

        logger.log_exception(exception)

class DiskCache_Tests(unittest.TestCase):

    def setUp(self):
        if Path(".test/test.csv.gz").exists():
            Path(".test/test.csv.gz").unlink()

    def tearDown(self) -> None:
        if Path(".test/test.csv.gz").exists():
            Path(".test/test.csv.gz").unlink()

    def test_write_csv_to_cache(self):

        cache = DiskCache(".test")

        self.assertFalse("test.csv"    in cache)
        self.assertFalse("test.csv.gz" in cache)

        cache.put("test.csv", BytesIO(b"test"))

        self.assertTrue("test.csv"    in cache)
        self.assertTrue("test.csv.gz" in cache)

        self.assertEqual(           cache.get("test.csv"   ).read() .decode('utf8'), "test")
        self.assertEqual(decompress(cache.get("test.csv.gz").read()).decode('utf8'), "test")

    def test_write_gz_to_cache(self):

        cache = DiskCache(".test")

        self.assertFalse("test.csv"    in cache)
        self.assertFalse("test.csv.gz" in cache)

        cache.put("test.csv.gz", BytesIO(compress(b"test")))

        self.assertTrue("test.csv"    in cache)
        self.assertTrue("test.csv.gz" in cache)

        self.assertEqual(           cache.get("test.csv"   ).read() .decode('utf8'), "test")
        self.assertEqual(decompress(cache.get("test.csv.gz").read()).decode('utf8'), "test")
    
    def test_rmv_csv_from_cache(self):

        cache = DiskCache(".test/")

        self.assertFalse("test.csv"    in cache)
        self.assertFalse("test.csv.gz" in cache)

        cache.put("test.csv", BytesIO(b"test"))

        self.assertTrue("test.csv"    in cache)
        self.assertTrue("test.csv.gz" in cache)

        cache.rmv("test.csv")

        self.assertFalse("test.csv"    in cache)
        self.assertFalse("test.csv.gz" in cache)

if __name__ == '__main__':
    unittest.main()