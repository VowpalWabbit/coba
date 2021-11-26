import unittest

from coba.exceptions import CobaException, CobaFatal

class CobaException_Tests(unittest.TestCase):
    def test_no_init_exception(self):
        CobaException("test")

    def test_simple(self):
        self.assertIsNone(CobaException()._render_traceback_())

class CobaFatal_Tests(unittest.TestCase):
    def test_no_init_exception(self):
        CobaFatal("test")

    def test_render_traceback(self):
        self.assertIsNone(CobaFatal()._render_traceback_())

if __name__ == '__main__':
    unittest.main()