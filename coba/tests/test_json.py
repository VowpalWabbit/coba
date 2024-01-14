import unittest

import coba.json
from coba.utilities import PackageChecker
from coba.registry import CobaRegistry

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
