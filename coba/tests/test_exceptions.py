import unittest

from coba.exceptions import CobaException, CobaExit

class CobaException_Tests(unittest.TestCase):
    def test_no_init_exception(self):
        CobaException("test")

    def test_simple(self):
        self.assertEqual(["abc"], CobaException("abc")._render_traceback_())

class CobaExit_Tests(unittest.TestCase):
    def test_no_init_exception(self):
        CobaExit("test")

    def test_render_traceback(self):
        self.assertEqual(["abc"], CobaExit("abc")._render_traceback_())

if __name__ == '__main__':
    unittest.main()