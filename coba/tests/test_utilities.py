import sys
import unittest
import unittest.mock
import operator

from coba.exceptions import CobaExit, sans_tb_sys_except_hook
from coba.utilities import PackageChecker, KeyDefaultDict, coba_exit, peek_first, minimize, grouper

class coba_exit_Tests(unittest.TestCase):
    def test_coba_exit(self):

        self.assertEqual(sans_tb_sys_except_hook, sys.excepthook)

        with self.assertRaises(CobaExit):
            coba_exit("abc")

class PackageChecker_sans_package_Tests(unittest.TestCase):

    def setUp(self) -> None:
        self.patch = unittest.mock.patch('importlib.util.find_spec',return_value=None)
        self.patch.start()

    def tearDown(self) -> None:
        self.patch.stop()

    def test_check_matplotlib_support(self):
        self.assertEqual(False,PackageChecker.matplotlib("",strict=False))
        with self.assertRaises(CobaExit): PackageChecker.matplotlib("")

    def test_check_pandas_support(self):
        self.assertEqual(False,PackageChecker.pandas("",strict=False))
        with self.assertRaises(CobaExit): PackageChecker.pandas("")

    def test_check_numpy_support(self):
        self.assertEqual(False,PackageChecker.numpy("",strict=False))
        with self.assertRaises(CobaExit): PackageChecker.numpy("")

    def test_check_vowpal_support(self):
        self.assertEqual(False,PackageChecker.vowpalwabbit("",strict=False))
        with self.assertRaises(CobaExit): PackageChecker.vowpalwabbit("")

    def test_check_sklearn_support(self):
        self.assertEqual(False,PackageChecker.sklearn("",strict=False))
        with self.assertRaises(CobaExit): PackageChecker.sklearn("")

    def test_check_scipy_support(self):
        self.assertEqual(False,PackageChecker.scipy("",strict=False))
        with self.assertRaises(CobaExit): PackageChecker.scipy("")

    def test_check_torch_support(self):
        self.assertEqual(False,PackageChecker.torch("",strict=False))
        with self.assertRaises(CobaExit): PackageChecker.torch("")

    def test_check_cloudpickle_support(self):
        self.assertEqual(False,PackageChecker.cloudpickle("",strict=False))
        with self.assertRaises(CobaExit): PackageChecker.cloudpickle("")

    def test_submodule_missing(self):
        with unittest.mock.patch('importlib.util.find_spec', side_effect=ModuleNotFoundError()):
            self.assertEqual(False,PackageChecker.matplotlib("",strict=False))
            with self.assertRaises(CobaExit): PackageChecker.matplotlib("")

class PackageChecker_with_package_Tests(unittest.TestCase):

    def setUp(self) -> None:
        self.patch = unittest.mock.patch('importlib.util.find_spec',return_value=True)
        self.patch.start()

    def tearDown(self) -> None:
        self.patch.stop()

    def test_check_matplotlib_support(self):
        self.assertEqual(True,PackageChecker.matplotlib("",strict=True))
        self.assertEqual(True,PackageChecker.matplotlib("",strict=False))

    def test_check_pandas_support(self):
        self.assertEqual(True,PackageChecker.pandas("",strict=True))
        self.assertEqual(True,PackageChecker.pandas("",strict=False))

    def test_check_numpy_support(self):
        self.assertEqual(True,PackageChecker.numpy("",strict=True))
        self.assertEqual(True,PackageChecker.numpy("",strict=False))

    def test_check_vowpal_support(self):
        self.assertEqual(True,PackageChecker.vowpalwabbit("",strict=True))
        self.assertEqual(True,PackageChecker.vowpalwabbit("",strict=False))

    def test_check_sklearn_support(self):
        self.assertEqual(True,PackageChecker.sklearn("",strict=True))
        self.assertEqual(True,PackageChecker.sklearn("",strict=False))

    def test_check_scipy_support(self):
        self.assertEqual(True,PackageChecker.scipy("",strict=True))
        self.assertEqual(True,PackageChecker.scipy("",strict=False))

    def test_check_torch_support(self):
        self.assertEqual(True,PackageChecker.torch("",strict=True))
        self.assertEqual(True,PackageChecker.torch("",strict=False))

    def test_check_cloudpickle_support(self):
        self.assertEqual(True,PackageChecker.cloudpickle("",strict=True))
        self.assertEqual(True,PackageChecker.cloudpickle("",strict=False))

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

class minimize_Tests(unittest.TestCase):
    def test_list(self):
        self.assertEqual(minimize([1,2.0,2.3333384905]),[1,2,2.33334])

    def test_bool(self):
        self.assertEqual(minimize(True),True)

    def test_list_list(self):
        data = [1,[2.,[1,2.123]]]
        self.assertEqual(minimize(data), [1,[2,[1,2.123]]])
        self.assertEqual(data, [1,[2.,[1,2.123]]])

    def test_list_tuple(self):
        data = [(1,0.123456),(0,1)]
        self.assertEqual(minimize(data),[[1,0.12346],[0,1]])
        self.assertEqual(data,[(1,0.123456),(0,1)])

    def test_minize_tuple(self):
        data = (1,2.123456)
        self.assertEqual(minimize(data),[1,2.12346])

    def test_dict(self):
        data = {'a':[1.123456,2],'b':{'c':1.}}
        self.assertEqual(minimize(data), {'a':[1.12346,2],'b':{'c':1}})
        self.assertEqual(data,{'a':[1.123456,2],'b':{'c':1.}})

    def test_inf(self):
        self.assertEqual(minimize(float('inf')),float('inf'))

    def test_nan(self):
        out = minimize(float('nan'))
        self.assertNotEqual(out,out)

    @unittest.skipUnless(PackageChecker.torch(strict=False), "This test requires pytorch.")
    def test_torch_tensor(self):
        import torch
        self.assertEqual(minimize(torch.tensor([[1],[2]])), [[1],[2]])

    @unittest.skipUnless(PackageChecker.torch(strict=False), "This test requires pytorch.")
    def test_torch_number(self):
        import torch
        self.assertEqual(minimize(torch.tensor(1.123456)), 1.12346)

class grouper_Tests(unittest.TestCase):

    def test_int_list_unsorted(self):
        groups = {k:list(g) for k,g in grouper([1,2,1,2])}
        self.assertEqual({1:[1,1],2:[2,2]}, groups)

    def test_int_list_sorted(self):
        groups = {k:list(g) for k,g in grouper([1,1,2,2], sorted_=True)}
        self.assertEqual({1:[1,1],2:[2,2]}, groups)

    def test_str_list_unsorted(self):
        groups = {k:list(g) for k,g in grouper(['1','2','1','2'])}
        self.assertEqual({'1':['1','1'],'2':['2','2']}, groups)

    def test_str_iterable_unsorted(self):
        groups = {k:list(g) for k,g in grouper(map(str,[1,2,1,2]))}
        self.assertEqual({'1':['1','1'],'2':['2','2']}, groups)

    def test_mixed_list_unsorted(self):
        groups = {k:list(g) for k,g in grouper([1,2,'1','2',None,None])}
        self.assertEqual({1:[1],'1':['1'],2:[2],'2':['2'],None:[None,None]}, groups)

    def test_tuples_val_none_unsorted(self):
        key = operator.itemgetter(0)
        groups = {k:list(g) for k,g in grouper([(1,2),(2,2)]*2,key=key)}
        self.assertEqual({1:[(1,2),(1,2)],2:[(2,2),(2,2)]}, groups)

    def test_tuples_val_getter_unsorted(self):
        key = operator.itemgetter(0)
        val = operator.itemgetter(1)
        groups = {k:list(g) for k,g in grouper([(1,2),(1,3),(2,3),(2,4)],key=key,val=val)}
        self.assertEqual({1:[2,3],2:[3,4]}, groups)

if __name__ == '__main__':
    unittest.main()
