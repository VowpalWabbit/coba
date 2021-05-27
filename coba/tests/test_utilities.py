import unittest

from coba.utilities import PackageChecker

class check_library_Tests(unittest.TestCase):
    
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

if __name__ == '__main__':
    unittest.main()