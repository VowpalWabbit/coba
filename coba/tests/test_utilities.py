import unittest

from coba.utilities import check_matplotlib_support, check_vowpal_support

class Utilities_Tests(unittest.TestCase):
    
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

if __name__ == '__main__':
    unittest.main()