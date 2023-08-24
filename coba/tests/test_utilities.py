import sys
import unittest
import unittest.mock

from coba import CobaRandom
from coba.exceptions import CobaExit, sans_tb_sys_except_hook
from coba.utilities import PackageChecker, KeyDefaultDict, coba_exit, peek_first, sample_actions


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

    def test_check_scipy_support(self):
        with self.assertRaises(CobaExit):
            PackageChecker.scipy("")

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

    def test_check_scipy_support(self):
        PackageChecker.scipy("")

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

class peek_first_Tests(unittest.TestCase):

    def test_simple_empty(self):
        first,items = peek_first([])

        self.assertIsNone(first)
        self.assertEqual(items,[])

        first,items = peek_first([],n=3)

        self.assertIsNone(first)
        self.assertEqual(items,[])

    def test_simple_peek_1(self):
        init_items = iter([1,2,3])

        first,items = peek_first(init_items)

        self.assertEqual(first,1)
        self.assertEqual(list(items),[1,2,3])

    def test_simple_peek_n(self):
        init_items = iter([1,2,3])

        first,items = peek_first(init_items,n=3)

        self.assertEqual(first,[1,2,3])
        self.assertEqual(list(items),[1,2,3])

class sample_actions_Tests(unittest.TestCase):

    def test_sampling(self):
        actions = [1,2,3]
        probs = [0,0,1]
        action, prob = sample_actions(actions, probs)
        self.assertEqual(action, 3)
        self.assertEqual(prob, 1)

    def test_statistics(self):
        actions = [1,2,3]
        probs = [0.1, 0.2, 0.7]
        action, prob = zip(*[sample_actions(actions, probs) for _ in range(10_000)])
        self.assertTrue(action.count(3) > action.count(2) > action.count(1))


    def test_custom_rng(self):
        actions = [1,2,3]
        probs = [0,0,1]
        action, prob = sample_actions(actions, probs, CobaRandom(seed=1.23))
        self.assertEqual(action, 3)
        self.assertEqual(prob, 1)

if __name__ == '__main__':
    unittest.main()