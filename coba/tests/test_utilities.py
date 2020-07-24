
import unittest
import json
import math
import statistics

from coba.utilities import OnlineMean, OnlineVariance, JsonTemplating, check_matplotlib_support, check_vowpal_support

class check_library_Tests(unittest.TestCase):
    
    def test_check_matplotlib_support(self):
        try:
            check_matplotlib_support("test_check_matplotlib_support")
        except Exception:
            self.fail("check_matplotlib_support raised an exception")

    def test_check_vowpal_support(self):
        try:
            check_vowpal_support("test_check_vowpal_support")
        except Exception:
            self.fail("check_vowpal_support raised an exception")

class JsonTemplating_Tests(unittest.TestCase):
    def test_no_template_string_unchanged_1(self):
        self.assertEqual(JsonTemplating.parse("[1,2,3]"), [1,2,3])

    def test_no_template_string_unchanged_2(self):
        actual = JsonTemplating.parse('{"a":1}')
        
        self.assertCountEqual(actual.keys(), ["a"])
        self.assertEqual(actual["a"], 1)

    def test_no_template_object_unchanged_1(self):
        self.assertEqual(JsonTemplating.parse(json.loads("[1,2,3]")), [1,2,3])

    def test_no_template_object_unchanged2(self):
        actual = JsonTemplating.parse(json.loads('{"a":1}'))

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

        actual = JsonTemplating.parse(json_str)

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

class OnlineVariance_Tests(unittest.TestCase):

    def test_no_updates_variance_nan(self):
        online = OnlineVariance()
        self.assertTrue(math.isnan(online.variance))

    def test_one_update_variance_nan(self):
        online = OnlineVariance()

        online.update(1)

        self.assertTrue(math.isnan(online.variance))

    def test_two_update_variance(self):

        batches = [ [0, 2], [1, 1], [1,2], [-1,1], [10.5,20] ]

        for batch in batches:
            online = OnlineVariance()

            for number in batch:
                online.update(number)

            self.assertEqual(online.variance, statistics.variance(batch))

    def test_three_update_variance(self):

        batches = [ [0, 2, 4], [1, 1, 1], [1,2,3], [-1,1,-1], [10.5,20,29.5] ]

        for batch in batches:
            online = OnlineVariance()

            for number in batch:
                online.update(number)

            #note: this test will fail on the final the batch if `places` > 15
            self.assertAlmostEqual(online.variance, statistics.variance(batch), places = 15)

    def test_100_integers_update_variance(self):

        batch = list(range(0,100))

        online = OnlineVariance()

        for number in batch:
            online.update(number)

        self.assertEqual(online.variance, statistics.variance(batch))

    def test_100_floats_update_variance(self):

        batch = [ i/3 for i in range(0,100) ]

        online = OnlineVariance()

        for number in batch:
            online.update(number)

        #note: this test will fail on the final the batch if `places` > 12
        self.assertAlmostEqual(online.variance, statistics.variance(batch), places=12)

class OnlineMean_Tests(unittest.TestCase):

    def test_no_updates_variance_nan(self):
        
        online = OnlineMean()
        
        self.assertTrue(math.isnan(online.mean))

    def test_one_update_variance_nan(self):
        
        batch = [1]
        
        online = OnlineMean()

        for number in batch:
            online.update(number)

        self.assertEqual(online.mean, statistics.mean(batch))

    def test_two_update_variance(self):

        batches = [ [0, 2], [1, 1], [1,2], [-1,1], [10.5,20] ]

        for batch in batches:
            online = OnlineMean()

            for number in batch:
                online.update(number)

            self.assertEqual(online.mean, statistics.mean(batch))

    def test_three_update_variance(self):

        batches = [ [0, 2, 4], [1, 1, 1], [1,2,3], [-1,1,-1], [10.5,20,29.5] ]

        for batch in batches:
            online = OnlineMean()

            for number in batch:
                online.update(number)

            self.assertEqual(online.mean, statistics.mean(batch))

    def test_100_integers_update_variance(self):

        batch = list(range(0,100))

        online = OnlineMean()

        for number in batch:
            online.update(number)

        self.assertAlmostEqual(online.mean, statistics.mean(batch))

    def test_100_floats_update_variance(self):

        batch = [ i/3 for i in range(0,100) ]

        online = OnlineMean()

        for number in batch:
            online.update(number)

        self.assertAlmostEqual(online.mean, statistics.mean(batch))


if __name__ == '__main__':
    unittest.main()