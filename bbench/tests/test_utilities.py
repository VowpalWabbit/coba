import unittest

from bbench.utilities import check_matplotlib_support, check_vowpal_support, check_sklearn_datasets_support

class Test_Utilities(unittest.TestCase):
    
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

    def test_check_sklearn_datasets_support(self):
        try:
            check_sklearn_datasets_support("test_check_sklearn_datasets_support")
        except Exception:
            self.fail("check_sklearn_datasets_support raised an exception")

if __name__ == '__main__':
    unittest.main()