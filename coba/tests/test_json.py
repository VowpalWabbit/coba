import unittest

import coba.json
from coba.utilities import PackageChecker
from coba.registry import CobaRegistry

class minimize_Tests(unittest.TestCase):
    def test_list(self):
        self.assertEqual(coba.json.minimize([1,2.0,2.3333384905]),[1,2,2.33334])

    def test_bool(self):
        self.assertEqual(coba.json.minimize(True),True)

    def test_list_list(self):
        data = [1,[2.,[1,2.123]]]
        self.assertEqual(coba.json.minimize(data), [1,[2,[1,2.123]]])
        self.assertEqual(data, [1,[2.,[1,2.123]]])

    def test_list_tuple(self):
        data = [(1,0.123456),(0,1)]
        self.assertEqual(coba.json.minimize(data),[[1,0.12346],[0,1]])
        self.assertEqual(data,[(1,0.123456),(0,1)])

    def test_minize_tuple(self):
        data = (1,2.123456)
        self.assertEqual(coba.json.minimize(data),[1,2.12346])

    def test_dict(self):
        data = {'a':[1.123456,2],'b':{'c':1.}}
        self.assertEqual(coba.json.minimize(data), {'a':[1.12346,2],'b':{'c':1}})
        self.assertEqual(data,{'a':[1.123456,2],'b':{'c':1.}})

    def test_inf(self):
        self.assertEqual(coba.json.minimize(float('inf')),float('inf'))

    def test_nan(self):
        out = coba.json.minimize(float('nan'))
        self.assertNotEqual(out,out)

    @unittest.skipUnless(PackageChecker.torch(strict=False), "This test requires pytorch.")
    def test_torch_tensor(self):
        import torch
        self.assertEqual(coba.json.minimize(torch.tensor([[1],[2]])), [[1],[2]])

    @unittest.skipUnless(PackageChecker.torch(strict=False), "This test requires pytorch.")
    def test_torch_number(self):
        import torch
        self.assertEqual(coba.json.minimize(torch.tensor(1.123456)), 1.12346)

class dumps_Tests(unittest.TestCase):

    def test_dumps(self):
        self.assertEqual(coba.json.dumps([1,2,3]),'[1, 2, 3]')

    def test_not_serializable(self):
        with self.assertRaises(TypeError) as e:
            coba.json.dumps({1,2,3})
        self.assertIn("Object of type set is not JSON serializable", str(e.exception))

        with self.assertRaises(TypeError) as e:
            coba.json.dumps({1,2,3},default=None)
        self.assertIn("Object of type set is not JSON serializable", str(e.exception))

    @unittest.skipUnless(PackageChecker.torch(strict=False), "This test requires pytorch.")
    def test_torch_tensor(self):
        import torch
        self.assertEqual('[1, 2]',coba.json.dumps(torch.tensor([1,2])))

    @unittest.skipUnless(PackageChecker.torch(strict=False), "This test requires pytorch.")
    def test_torch_number(self):
        import torch
        self.assertEqual('1.2',coba.json.dumps(torch.tensor(1.2))[:3])

    def test_registered_class_getstate(self):
        class Test:
            def __init__(self):
                self.a,self.b = 1,2
            def __getstate__(self):
                return (1,2)
            def __setstate__(self,arg):
                pass
        CobaRegistry.clear()
        CobaRegistry.register("T",Test)
        self.assertEqual(coba.json.dumps(Test()), '{"T": [1, 2]}')

    def test_registered_class_nostate(self):
        class Test:
            def __init__(self):
                self.a,self.b = 1,2
        CobaRegistry.clear()
        CobaRegistry.register("T",Test)
        with self.assertRaises(TypeError) as e:
            coba.json.dumps(Test())

class loads_Tests(unittest.TestCase):
    def test_loads(self):
        self.assertEqual(coba.json.loads('[1, 2, 3]'),[1,2,3])
        self.assertEqual(coba.json.loads('{"a":1}'),{"a":1})
        self.assertEqual(coba.json.loads('{"a":1,"b":2}'),{"a":1,"b":2})

if __name__ == '__main__':
    unittest.main()
