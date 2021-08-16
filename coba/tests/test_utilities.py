import unittest
import timeit

from coba.utilities import PackageChecker, HashableDict

class PackageChecker_Tests(unittest.TestCase):
    
    def test_check_matplotlib_support(self):
        try:
            PackageChecker.matplotlib("test_check_matplotlib_support")
        except Exception:
            self.fail("check_matplotlib_support raised an exception")

    def test_check_pandas_support(self):
        try:
            PackageChecker.pandas("test_check_pandas_support")
        except Exception:
            self.fail("check_pandas_support raised an exception")

    def test_check_numpy_support(self):
        try:
            PackageChecker.numpy("test_check_numpy_support")
        except Exception:
            self.fail("check_numpy_support raised an exception")

    def test_check_vowpal_support(self):
        try:
            PackageChecker.vowpalwabbit("test_check_vowpal_support")
        except Exception:
            self.fail("check_vowpal_support raised an exception")

class HashableDict_Tests(unittest.TestCase):

    def test_hash(self):

        hash_dict = HashableDict({'a':1,'b':2})

        self.assertEqual(hash(hash_dict), hash(hash_dict))
        self.assertEqual(hash_dict,hash_dict)

    def test_performance(self):

        base_dict = dict(enumerate(range(1000)))

        time1 = timeit.timeit(lambda: dict(enumerate(range(1000))), number=1000)
        time2 = timeit.timeit(lambda: HashableDict(base_dict)     , number=1000)

        self.assertLess(abs(time1-time2), 1)

    def test_mutable_fail(self):

        hash_dict = HashableDict({'a':1,'b':2})
        hash_dict["b"] = 3

        with self.assertRaises(AssertionError):
            hash(hash_dict)
 
if __name__ == '__main__':
    unittest.main()