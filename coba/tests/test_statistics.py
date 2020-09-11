import unittest

from statistics import mean, variance, stdev
from math import isnan, sqrt

from coba.statistics import BatchMeanEstimator, OnlineVariance, OnlineMean, StatisticalEstimate

class StatisticalEstimate_Tests(unittest.TestCase):

    def test_addition_of_estimate(self) -> None:
        a = StatisticalEstimate(1,2)
        b = StatisticalEstimate(1,3)

        actual = a+b
        expected = StatisticalEstimate(2, sqrt(4+9))

        self.assertEqual(actual, expected)

    def test_subtraction_of_estimate(self) -> None:
        a = StatisticalEstimate(1,2)
        b = StatisticalEstimate(1,3)

        actual = a-b
        expected = StatisticalEstimate(0, sqrt(4+9))

        self.assertEqual(actual, expected)

    def test_negation_of_estimate(self) -> None:
        a = StatisticalEstimate(1,2)

        actual = -a
        expected = StatisticalEstimate(-1, 2)

        self.assertEqual(actual, expected)

    def test_addition_of_scalar(self) -> None:
        a = StatisticalEstimate(1,2)
        
        l_actual = 2.2 + a
        r_actual = a + 2.2
        expected = StatisticalEstimate(3.2, 2)

        self.assertEqual(l_actual, expected)
        self.assertEqual(r_actual, expected)

    def test_subtraction_of_scalar(self) -> None:
        a = StatisticalEstimate(1,2)
        
        l_actual = 2.2 - a
        r_actual = a - 2.2
        l_expected = StatisticalEstimate(+1.2, 2)
        r_expected = StatisticalEstimate(-1.2, 2)

        self.assertEqual(l_actual, l_expected)
        self.assertEqual(r_actual, r_expected)

    def test_multiplication_by_scalar(self) -> None:
        a = StatisticalEstimate(1,2)
        
        l_actual = 2 * a
        r_actual = a * 2
        expected = StatisticalEstimate(2,4)

        self.assertEqual(l_actual, expected)
        self.assertEqual(r_actual, expected)

    def test_division_by_scalar(self) -> None:
        a = StatisticalEstimate(1,2)

        r_actual = a / 2
        expected = StatisticalEstimate(1/2,1)

        self.failUnlessRaises(Exception, lambda: 2/a)
        self.assertEqual(r_actual, expected)

    def test_sum_of_estimates(self) -> None:

        a1 = StatisticalEstimate(1,2)
        a2 = StatisticalEstimate(2,3)
        a3 = StatisticalEstimate(3,4)
        a4 = StatisticalEstimate(4,5)

        actual = a1+a2+a3+a4

        self.assertEqual(actual.estimate, sum([1,2,3,4]))
        self.assertEqual(actual.standard_error, sqrt(sum([4,9,16,25])))

    def test_mean_direct_of_estimates(self) -> None:

        a = StatisticalEstimate(1,2)

        actual = (a+a+a+a)/4

        self.assertEqual(actual.estimate, 1)
        self.assertEqual(actual.standard_error, 2/sqrt(4))

    def test_mean_statistics_of_estimates(self) -> None:

        a = StatisticalEstimate(1,2)

        actual = mean([a,a,a,a]) #type: ignore

        self.assertEqual(actual.estimate, 1)
        self.assertEqual(actual.standard_error, 2/sqrt(4))

    def test_mean_weighted_of_estimates(self) -> None:

        a1 = StatisticalEstimate(1,2)
        a2 = StatisticalEstimate(2,3)
        a3 = StatisticalEstimate(3,4)
        a4 = StatisticalEstimate(4,5)

        w1 = 1
        w2 = 2
        w3 = 3
        w4 = 4

        actual = (w1*a1+w2*a2+w3*a3+w4*a4)/(w1+w2+w3+w4)

        expected_estimate       = (1+4+9+16)/10
        expected_standard_error = sqrt((1*4+4*9+9*16+16*25)/100)

        self.assertEqual(actual.estimate, expected_estimate)
        self.assertEqual(actual.standard_error, expected_standard_error)

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