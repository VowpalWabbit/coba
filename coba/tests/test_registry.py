import unittest

from coba.registry import CobaRegistry, coba_registry_class

class TestObject:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

class TestArgObject:
    def __init__(self, arg):
        self.arg = arg

class TestOptionalArgObject:
    def __init__(self, arg=1):
        self.arg = arg

class CobaRegistry_Tests(unittest.TestCase):

    def setUp(self) -> None:
        CobaRegistry.clear() #make sure the registry is fresh each test

    def test_endpoint_loaded(self):
        klass = CobaRegistry.retrieve("Null")

        self.assertEqual("NullIO", klass.__name__)

    def test_endpoint_loaded_after_decorator_register(self):
        
        @coba_registry_class("MyTestObject")
        class MyTestObject(TestObject): pass

        klass = CobaRegistry.retrieve("Null")

        self.assertEqual("NullIO", klass.__name__)

    def test_register_decorator(self):

        @coba_registry_class("MyTestObject")
        class MyTestObject(TestObject): pass

        klass = CobaRegistry.construct("MyTestObject")

        self.assertIsInstance(klass, MyTestObject)
        self.assertEqual(klass.args, ())
        self.assertEqual(klass.kwargs, {})

    def test_registered_create(self):

        CobaRegistry.register("test", TestObject)

        klass = CobaRegistry.construct("test")

        self.assertEqual(klass.args, ())
        self.assertEqual(klass.kwargs, {})

    def test_registered_create_args1(self):

        CobaRegistry.register("test", TestObject)

        klass = CobaRegistry.construct({ "test": [1,2,3] })

        self.assertEqual(klass.args, (1,2,3))
        self.assertEqual(klass.kwargs, {})

    def test_registered_create_args2(self):

        CobaRegistry.register("test", TestObject)

        klass = CobaRegistry.construct({ "test": 1 })

        self.assertEqual(klass.args, (1,))
        self.assertEqual(klass.kwargs, {})

    def test_registered_create_kwargs(self):

        CobaRegistry.register("test", TestObject)

        klass = CobaRegistry.construct({ "test": {"a":1} })

        self.assertEqual(klass.args, ())
        self.assertEqual(klass.kwargs, {"a":1})

    def test_registered_create_args3(self):

        CobaRegistry.register("test", TestObject)

        klass = CobaRegistry.construct({ "test": "abc" })

        self.assertEqual(klass.args, ("abc",))
        self.assertEqual(klass.kwargs, {})

    def test_registered_create_args_kwargs(self):

        CobaRegistry.register("test", TestObject)

        klass = CobaRegistry.construct({ "test": [1,2,3], "kwargs": {"a":1} })

        self.assertEqual(klass.args, (1,2,3))
        self.assertEqual(klass.kwargs, {"a":1})

    def test_registered_create_name_args_kwargs(self):

        CobaRegistry.register("test", TestObject)

        klass = CobaRegistry.construct({ "name": "test", "args": [1,2,3], "kwargs": {"a":1} })

        self.assertEqual(klass.args, (1,2,3))
        self.assertEqual(klass.kwargs, {"a":1})

    def test_registered_create_foreach1(self):

        CobaRegistry.register("test", TestObject)

        recipe = { "test":[[1,2,3]], "kwargs": {"a":1}, "method":"foreach" }

        klasses = CobaRegistry.construct(recipe)

        self.assertEqual(len(klasses), 1)
        self.assertEqual(klasses[0].args, (1,2,3))
        self.assertEqual(klasses[0].kwargs, {"a":1})

    def test_registered_create_foreach2(self):

        CobaRegistry.register("test", TestObject)

        recipe = { "test":[1,2,3], "kwargs": {"a":1}, "method":"foreach" }

        klasses = CobaRegistry.construct(recipe)

        self.assertEqual(len(klasses), 3)
    
        self.assertEqual(klasses[0].args, (1,))
        self.assertEqual(klasses[0].kwargs, {"a":1})

        self.assertEqual(klasses[1].args, (2,))
        self.assertEqual(klasses[1].kwargs, {"a":1})

        self.assertEqual(klasses[2].args, (3,))
        self.assertEqual(klasses[2].kwargs, {"a":1})

    def test_registered_create_foreach3(self):

        CobaRegistry.register("test", TestObject)

        recipe = { "test":[1,2], "kwargs": [{"a":1},{"a":2}], "method":"foreach" }

        klasses = CobaRegistry.construct(recipe)

        self.assertEqual(len(klasses), 2)

        self.assertEqual(klasses[0].args, (1,))
        self.assertEqual(klasses[0].kwargs, {"a":1})

        self.assertEqual(klasses[1].args, (2,))
        self.assertEqual(klasses[1].kwargs, {"a":2})

    def test_registered_create_foreach4(self):

        CobaRegistry.register("test", TestObject)

        recipe = { "test":[[1,2],3], "method":"foreach" }

        klasses = CobaRegistry.construct(recipe)

        self.assertEqual(len(klasses), 2)
    
        self.assertEqual(klasses[0].args, (1,2))
        self.assertEqual(klasses[0].kwargs, {})

        self.assertEqual(klasses[1].args, (3,))
        self.assertEqual(klasses[1].kwargs, {})

    def test_registered_create_recursive1(self):

        CobaRegistry.register("test", TestObject)

        klass = CobaRegistry.construct({ "test": "test" })

        self.assertEqual(1, len(klass.args))
        self.assertEqual(klass.kwargs, {})
        
        self.assertIsInstance(klass.args[0], TestObject)
        self.assertEqual(klass.args[0].args, ())
        self.assertEqual(klass.args[0].kwargs, {})

    def test_registered_create_recursive2(self):

        CobaRegistry.register("test", TestObject)

        klass = CobaRegistry.construct({ "test": {"test":1} })

        self.assertEqual(1, len(klass.args))
        self.assertEqual(klass.kwargs, {})

        self.assertIsInstance(klass.args[0], TestObject)
        self.assertEqual(klass.args[0].args, (1,))
        self.assertEqual(klass.args[0].kwargs, {})

    def test_registered_create_recursive3(self):

        CobaRegistry.register("test", TestObject)

        klass = CobaRegistry.construct({ "test": {"a": "test"} })

        self.assertEqual(klass.args, ())
        self.assertEqual(1, len(klass.kwargs))

        self.assertIsInstance(klass.kwargs["a"], TestObject)
        self.assertEqual(klass.kwargs["a"].args, ())
        self.assertEqual(klass.kwargs["a"].kwargs, {})

    def test_registered_create_array_arg(self):

        CobaRegistry.register("test", TestArgObject)

        klass = CobaRegistry.construct({ "test": [1,2,3] })

        self.assertEqual(klass.arg, [1,2,3])

    def test_registered_create_dict_arg(self):

        CobaRegistry.register("test", TestArgObject)

        with self.assertRaises(Exception):
            klass = CobaRegistry.construct({ "test": {"a":1} })

    def test_not_registered(self):

        CobaRegistry.register("test", TestObject)

        with self.assertRaises(Exception) as cm:
            CobaRegistry.construct("test2")

        self.assertEqual("Unknown recipe test2", str(cm.exception))

    def test_invalid_recipe1(self):

        CobaRegistry.register("test", TestObject)

        recipe = {"test":[1,2,3], "args":[4,5,6] }

        with self.assertRaises(Exception) as cm:
            CobaRegistry.construct(recipe)

        self.assertEqual(f"Invalid recipe {str(recipe)}", str(cm.exception))

    def test_invalid_recipe2(self):

        CobaRegistry.register("test", TestObject)

        recipe = {"test":[1,2,3], "name":"test", "args":[4,5,6]}

        with self.assertRaises(Exception) as cm:
            CobaRegistry.construct(recipe)

        self.assertEqual(f"Invalid recipe {str(recipe)}", str(cm.exception))

    def test_invalid_recipe3(self):

        CobaRegistry.register("test", TestObject)

        recipe = {"test":{"a":1}, "name":"test", "kwargs":{"a":1}}

        with self.assertRaises(Exception) as cm:
            CobaRegistry.construct(recipe)

        self.assertEqual(f"Invalid recipe {str(recipe)}", str(cm.exception))

    def test_registered_create_optionalarray_arg(self):

        CobaRegistry.register("test", TestOptionalArgObject)

        klass = CobaRegistry.construct({ "test": [1,2,3] })

        self.assertEqual(klass.arg, [1,2,3])

if __name__ == '__main__':
    unittest.main()