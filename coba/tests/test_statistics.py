import unittest
import importlib.util

from math import isnan

from coba.exceptions import CobaException
from coba.statistics import mean, stdev, var, iqr, percentile, phi
from coba.statistics import OnlineVariance, OnlineMean, StandardErrorOfMean, BootstrapConfidenceInterval, BinomialConfidenceInterval

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

class Mean_Tests(unittest.TestCase):
    def test(self):
        self.assertEqual(2,mean([1,2,3]))

class StandardDeviation_Tests(unittest.TestCase):
    def test(self):
        self.assertAlmostEqual(1.4142,stdev([1,3]),4)

    def test_length_1(self):
        self.assertTrue(isnan(stdev([1])))

class StandardErrorOfMean_Tests(unittest.TestCase):
    def test(self):
        mu,(lo,hi) = StandardErrorOfMean().calculate([1,3])
        self.assertEqual(2,mu)
        self.assertAlmostEqual(1.96,lo)
        self.assertAlmostEqual(1.96,hi)

class BootstrapConfidenceInterval_Tests(unittest.TestCase):
    
    def test1(self):
        mu,(lo,hi) = BootstrapConfidenceInterval(.95,mean).calculate([0,2])
        self.assertEqual(1,mu)
        self.assertEqual(1,lo)
        self.assertEqual(1,hi)

    def test2(self):
        mu,(lo,hi) = BootstrapConfidenceInterval(.1,mean).calculate([0,2])
        self.assertEqual(1,mu)
        self.assertEqual(0,lo)
        self.assertEqual(0,hi)

class BinomialConfidenceInterval_Tests(unittest.TestCase):

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "sklearn is not installed so we must skip this test.")
    def test_copper_pearson(self):
        p_hat, (lo,hi) = BinomialConfidenceInterval('clopper-pearson').calculate([0,0,0,1,1,1])

        self.assertEqual(0.5, p_hat)
        self.assertAlmostEqual(0.118117, p_hat-lo, 6)
        self.assertAlmostEqual(0.881883, p_hat+hi, 6)

    def test_wilson(self):
        p_hat, (lo,hi) = BinomialConfidenceInterval('wilson').calculate([0]*3+[1]*3)

        self.assertEqual(0.5, p_hat)
        self.assertAlmostEqual(0.18761, p_hat-lo, 4)
        self.assertAlmostEqual(0.81238, p_hat+hi, 4)

    def test_bad_data(self):
        with self.assertRaises(CobaException):
            BinomialConfidenceInterval('wilson').calculate([1,2,3])

class Phi_Tests(unittest.TestCase):
    def test(self):
        self.assertAlmostEqual(.975, phi(1.96),3)

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

            self.assertAlmostEqual(online.variance, var(test_set))

    def test_three_update_variance(self):

        test_sets = [ [0, 2, 4], [1, 1, 1], [1,2,3], [-1,1,-1], [10.5,20,29.5] ]

        for test_set in test_sets:
            online = OnlineVariance()

            for number in test_set:
                online.update(number)

            self.assertAlmostEqual(online.variance, var(test_set), places = 8)

    def test_100_integers_update_variance(self):

        test_set = list(range(0,100))

        online = OnlineVariance()

        for number in test_set:
            online.update(number)

        self.assertAlmostEqual(online.variance, var(test_set))

    def test_100_floats_update_variance(self):

        test_set = [ i/3 for i in range(0,100) ]

        online = OnlineVariance()

        for number in test_set:
            online.update(number)

        #note: this test will fail on the final the test_set if `places` > 12
        self.assertAlmostEqual(online.variance, var(test_set), places=12)

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
