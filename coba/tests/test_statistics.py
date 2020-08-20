import unittest

from statistics import mean, variance, stdev
from math import isnan, sqrt

from coba.statistics import BatchMeanEstimator, OnlineVariance, OnlineMean

class BatchMeanEstimator_Tests(unittest.TestCase):

    def test_from_observations(self):
        for batch in [[1,1,3,3], [1,1,1,1], [3], [] ]:
            actual_stats = BatchMeanEstimator(batch)

            expected_estimate       = mean(batch)                   if len(batch) > 0 else float('nan')
            expected_standard_error = stdev(batch)/sqrt(len(batch)) if len(batch) > 1 else float('nan')

            if len(batch) > 0:
                self.assertEqual(actual_stats.estimate, expected_estimate)
            else:
                self.assertTrue(isnan(actual_stats.estimate))

            if len(batch) > 1:
                self.assertEqual(actual_stats.standard_error, expected_standard_error)
            else:
                self.assertTrue(isnan(actual_stats.standard_error))

class OnlineVariance_Tests(unittest.TestCase):

    def test_no_updates_variance_nan(self):
        online = OnlineVariance()
        self.assertTrue(isnan(online.variance))

    def test_one_update_variance_nan(self):
        online = OnlineVariance()

        online.update(1)

        self.assertTrue(isnan(online.variance))

    def test_two_update_variance(self):

        batches = [ [0, 2], [1, 1], [1,2], [-1,1], [10.5,20] ]

        for batch in batches:
            online = OnlineVariance()

            for number in batch:
                online.update(number)

            self.assertEqual(online.variance, variance(batch))

    def test_three_update_variance(self):

        batches = [ [0, 2, 4], [1, 1, 1], [1,2,3], [-1,1,-1], [10.5,20,29.5] ]

        for batch in batches:
            online = OnlineVariance()

            for number in batch:
                online.update(number)

            #note: this test will fail on the final the batch if `places` > 15
            self.assertAlmostEqual(online.variance, variance(batch), places = 15)

    def test_100_integers_update_variance(self):

        batch = list(range(0,100))

        online = OnlineVariance()

        for number in batch:
            online.update(number)

        self.assertEqual(online.variance, variance(batch))

    def test_100_floats_update_variance(self):

        batch = [ i/3 for i in range(0,100) ]

        online = OnlineVariance()

        for number in batch:
            online.update(number)

        #note: this test will fail on the final the batch if `places` > 12
        self.assertAlmostEqual(online.variance, variance(batch), places=12)

class OnlineMean_Tests(unittest.TestCase):

    def test_no_updates_variance_nan(self):
        
        online = OnlineMean()
        
        self.assertTrue(isnan(online.mean))

    def test_one_update_variance_nan(self):
        
        batch = [1]
        
        online = OnlineMean()

        for number in batch:
            online.update(number)

        self.assertEqual(online.mean, mean(batch))

    def test_two_update_variance(self):

        batches = [ [0, 2], [1, 1], [1,2], [-1,1], [10.5,20] ]

        for batch in batches:
            online = OnlineMean()

            for number in batch:
                online.update(number)

            self.assertEqual(online.mean, mean(batch))

    def test_three_update_variance(self):

        batches = [ [0, 2, 4], [1, 1, 1], [1,2,3], [-1,1,-1], [10.5,20,29.5] ]

        for batch in batches:
            online = OnlineMean()

            for number in batch:
                online.update(number)

            self.assertEqual(online.mean, mean(batch))

    def test_100_integers_update_variance(self):

        batch = list(range(0,100))

        online = OnlineMean()

        for number in batch:
            online.update(number)

        self.assertAlmostEqual(online.mean, mean(batch))

    def test_100_floats_update_variance(self):

        batch = [ i/3 for i in range(0,100) ]

        online = OnlineMean()

        for number in batch:
            online.update(number)

        self.assertAlmostEqual(online.mean, mean(batch))

if __name__ == '__main__':
    unittest.main()