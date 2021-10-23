import unittest
import timeit

from coba.utilities import PackageChecker, HashableDict

class PackageChecker_Tests(unittest.TestCase):
    
    def test_check_matplotlib_support(self):

        try:
            import matplotlib
        except:
            installed=False
        else:
            installed=True

        try:
            PackageChecker.matplotlib("",silent=True)
        except:
            if installed:
                self.fail("PackageChecker raised an exception even though matplotlib is installed")
        else:
            if not installed:
                self.fail("PackageChecker did not raise an exception even though matplotib is not installed")

    def test_check_pandas_support(self):
        
        try:
            import pandas
        except:
            installed=False
        else:
            installed=True

        try:
            PackageChecker.pandas("",silent=True)
        except:
            if installed:
                self.fail("PackageChecker raised an exception even though pandas is installed")
        else:
            if not installed:
                self.fail("PackageChecker did not raise an exception even though pandas is not installed")

    def test_check_numpy_support(self):
        try:
            import numpy
        except:
            installed=False
        else:
            installed=True

        try:
            PackageChecker.numpy("",silent=True)
        except:
            if installed:
                self.fail("PackageChecker raised an exception even though numpy is installed")
        else:
            if not installed:
                self.fail("PackageChecker did not raise an exception even though numpy is not installed")

    def test_check_vowpal_support(self):
        try:
            import vowpalwabbit
        except:
            installed=False
        else:
            installed=True

        try:
            PackageChecker.vowpalwabbit("",silent=True)
        except:
            if installed:
                self.fail("PackageChecker raised an exception even though vowpalwabbit is installed")
        else:
            if not installed:
                self.fail("PackageChecker did not raise an exception even though vowpalwabbit is not installed")

class HashableDict_Tests(unittest.TestCase):

    def test_hash(self):

        hash_dict = HashableDict({'a':1,'b':2})

        self.assertEqual(hash(hash_dict), hash(hash_dict))
        self.assertEqual(hash_dict,hash_dict)

    def test_mutable_fail(self):

        hash_dict = HashableDict({'a':1,'b':2})
        hash_dict["b"] = 3

        with self.assertRaises(AssertionError):
            hash(hash_dict)
 
if __name__ == '__main__':
    unittest.main()