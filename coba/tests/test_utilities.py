import sys
import unittest
import unittest.mock

from coba.exceptions import CobaExit, sans_tb_sys_except_hook
from coba.utilities import PackageChecker, HashableDict, KeyDefaultDict, coba_exit

class coba_exit_Tests(unittest.TestCase):
    def test_coba_exit(self):
        
        self.assertEqual(sans_tb_sys_except_hook, sys.excepthook)

        with self.assertRaises(CobaExit):
            coba_exit("abc")

class PackageChecker_sans_package_Tests(unittest.TestCase):
    
    def setUp(self) -> None:
        self.patch = unittest.mock.patch('importlib.import_module', side_effect=ImportError())
        self.patch.start()

    def tearDown(self) -> None:
        self.patch.stop()

    def test_check_matplotlib_support(self):
        with self.assertRaises(CobaExit):
            PackageChecker.matplotlib("")

    def test_check_pandas_support(self):
        with self.assertRaises(CobaExit):
            PackageChecker.pandas("")

    def test_check_numpy_support(self):
        with self.assertRaises(CobaExit):
            PackageChecker.numpy("")

    def test_check_vowpal_support(self):
        with self.assertRaises(CobaExit):
            PackageChecker.vowpalwabbit("")

    def test_check_sklearn_support(self):
        with self.assertRaises(CobaExit):
            PackageChecker.sklearn("")

class PackageChecker_with_package_Tests(unittest.TestCase):

    def setUp(self) -> None:
        self.patch = unittest.mock.patch('importlib.import_module')
        self.patch.start()

    def tearDown(self) -> None:
        self.patch.stop()

    def test_check_matplotlib_support(self):
        PackageChecker.matplotlib("")

    def test_check_pandas_support(self):
        PackageChecker.pandas("")
            
    def test_check_numpy_support(self):
        PackageChecker.numpy("")

    def test_check_vowpal_support(self):
        PackageChecker.vowpalwabbit("")

    def test_check_sklearn_support(self):
        PackageChecker.sklearn("")

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

class KeyDefaultDict_Tests(unittest.TestCase):

    def test_with_factory(self):
        a = KeyDefaultDict(lambda key: str(key))
        self.assertEqual("1",a[1])
        self.assertEqual("2",a[2])
        self.assertEqual(2, len(a))

    def test_sans_factory(self):
        a = KeyDefaultDict(None)
        with self.assertRaises(KeyError):
            a[1]

if __name__ == '__main__':
    unittest.main()