import unittest
import unittest.mock

from coba.exceptions import CobaException
from coba.statistics import mean
from coba.utilities import PackageChecker
from coba.results.errors import StdDevCI, StdErrCI, BootstrapCI, BinomialCI

class StdDevCI_Tests(unittest.TestCase):
    def test_two_items(self):
        mu,(lo,hi) = StdDevCI().point_interval([1,3])
        self.assertEqual(2,mu)
        self.assertAlmostEqual(1.41421,lo)
        self.assertAlmostEqual(1.41421,hi)
        mu = StdDevCI().point([1,3])
        self.assertEqual(2,mu)

    def test_one_item(self):
        mu,(lo,hi) = StdDevCI().point_interval([1])
        self.assertEqual(1,mu)
        self.assertAlmostEqual(0,lo)
        self.assertAlmostEqual(0,hi)
        mu = StdDevCI().point([1])
        self.assertEqual(1,mu)

class StdErrCI_Tests(unittest.TestCase):
    def test_two_items(self):
        mu,(lo,hi) = StdErrCI().point_interval([1,3])
        self.assertEqual(2,mu)
        self.assertAlmostEqual(1.96,lo,delta=.001)
        self.assertAlmostEqual(1.96,hi,delta=.001)
        mu = StdErrCI().point([1,3])
        self.assertEqual(2,mu)

    def test_one_item(self):
        mu,(lo,hi) = StdErrCI().point_interval([1])
        self.assertEqual(1,mu)
        self.assertAlmostEqual(0,lo)
        self.assertAlmostEqual(0,hi)
        mu = StdErrCI().point([1])
        self.assertEqual(1,mu)

@unittest.skipUnless(PackageChecker.scipy(strict=False), "this test requires scipy")
class BootstrapCI_Tests(unittest.TestCase):
    def test1(self):
        mu,(lo,hi) = BootstrapCI(.95,mean).point_interval([0,1,2,3,4])
        self.assertEqual(2,mu)
        self.assertAlmostEqual(1.2,lo)
        self.assertAlmostEqual(1.2,hi)
        mu = BootstrapCI(.95,mean).point([0,1,2,3,4])
        self.assertEqual(2,mu)

    def test2(self):
        mu,(lo,hi) = BootstrapCI(.1,mean).point_interval([0,2])
        self.assertEqual(1,mu)
        self.assertEqual(0,lo)
        self.assertEqual(0,hi)
        mu = BootstrapCI(.1,mean).point([0,2])
        self.assertEqual(1,mu)

class BinomialCI_Tests(unittest.TestCase):
    @unittest.skipUnless(PackageChecker.scipy(strict=False), "scipy is not installed so we must skip this test.")
    def test_copper_pearson(self):
        p_hat, (lo,hi) = BinomialCI('clopper-pearson').point_interval([0,0,0,1,1,1])
        self.assertEqual(0.5, p_hat)
        self.assertAlmostEqual(0.118117, p_hat-lo, 4)
        self.assertAlmostEqual(0.881883, p_hat+hi, 4)
        p_hat = BinomialCI('clopper-pearson').point([0,0,0,1,1,1])
        self.assertEqual(0.5, p_hat)

    def test_wilson(self):
        p_hat, (lo,hi) = BinomialCI('wilson').point_interval([0]*3+[1]*3)
        self.assertEqual(0.5, p_hat)
        self.assertAlmostEqual(0.18761, p_hat-lo, 4)
        self.assertAlmostEqual(0.81238, p_hat+hi, 4)
        p_hat = BinomialCI('wilson').point([0]*3+[1]*3)
        self.assertEqual(0.5, p_hat)

    def test_bad_data(self):
        with self.assertRaises(CobaException):
            BinomialCI('wilson').point_interval([1,2,3])

if __name__ == '__main__':
    unittest.main()
