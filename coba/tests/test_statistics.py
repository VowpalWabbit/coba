import unittest

from statistics import mean, variance
from math import isnan

from coba.statistics import OnlineVariance, OnlineMean, iqr, percentile

class iqr_Tests(unittest.TestCase):
    def test_simple_exclusive(self):
        self.assertEqual(1, iqr([1,2,3]))

class percentile_Tests(unittest.TestCase):
    def test_simple_0_00(self):
        self.assertEqual(1, percentile([3,2,1], 0))

    def test_simple_1_00(self):
        self.assertEqual(3, percentile([3,2,1], 1))

    def test_simple_0_50(self):
        self.assertEqual(2, percentile([3,2,1], .5))

    def test_simple_00_50_01(self):
        self.assertEqual((1,2,3), percentile([3,2,1], [0,.5,1]))

class OnlineVariance_Tests(unittest.TestCase):

    def test_no_updates_variance_nan(self):
        online = OnlineVariance()
        self.assertTrue(isnan(online.variance))

    def test_one_update_variance_nan(self):
        online = OnlineVariance()
        online.update(1)
        self.assertTrue(isnan(online.variance))

    def test_two_update_variance(self):

        test_sets = [ [0, 2], [1, 1], [1,2], [-1,1], [10.5,20] ]

        for test_set in test_sets:
            online = OnlineVariance()

            for number in test_set:
                online.update(number)

            self.assertEqual(online.variance, variance(test_set))

    def test_three_update_variance(self):

        test_sets = [ [0, 2, 4], [1, 1, 1], [1,2,3], [-1,1,-1], [10.5,20,29.5] ]

        for test_set in test_sets:
            online = OnlineVariance()

            for number in test_set:
                online.update(number)

            #note: this test will fail on the final the test_set if `places` > 15
            self.assertAlmostEqual(online.variance, variance(test_set), places = 15)

    def test_100_integers_update_variance(self):

        test_set = list(range(0,100))

        online = OnlineVariance()

        for number in test_set:
            online.update(number)

        self.assertEqual(online.variance, variance(test_set))

    def test_100_floats_update_variance(self):

        test_set = [ i/3 for i in range(0,100) ]

        online = OnlineVariance()

        for number in test_set:
            online.update(number)

        #note: this test will fail on the final the test_set if `places` > 12
        self.assertAlmostEqual(online.variance, variance(test_set), places=12)

class OnlineMean_Tests(unittest.TestCase):

    def test_no_updates_variance_nan(self):
        
        online = OnlineMean()
        
        self.assertTrue(isnan(online.mean))

    def test_one_update_variance_nan(self):
        
        test_set = [1]
        
        online = OnlineMean()

        for number in test_set:
            online.update(number)

        self.assertEqual(online.mean, mean(test_set))

    def test_two_update_variance(self):

        test_sets = [ [0, 2], [1, 1], [1,2], [-1,1], [10.5,20] ]

        for test_set in test_sets:
            online = OnlineMean()

            for number in test_set:
                online.update(number)

            self.assertEqual(online.mean, mean(test_set))

    def test_three_update_variance(self):

        test_sets = [ [0, 2, 4], [1, 1, 1], [1,2,3], [-1,1,-1], [10.5,20,29.5] ]

        for test_set in test_sets:
            online = OnlineMean()

            for number in test_set:
                online.update(number)

            self.assertEqual(online.mean, mean(test_set))

    def test_100_integers_update_variance(self):

        test_set = list(range(0,100))

        online = OnlineMean()

        for number in test_set:
            online.update(number)

        self.assertAlmostEqual(online.mean, mean(test_set))

    def test_100_floats_update_variance(self):

        test_set = [ i/3 for i in range(0,100) ]

        online = OnlineMean()

        for number in test_set:
            online.update(number)

        self.assertAlmostEqual(online.mean, mean(test_set))

if __name__ == '__main__':
    unittest.main()