import unittest

from coba.exceptions import CobaException
from coba.registry import CobaRegistry, coba_registry_class, JsonMakerV1, JsonMakerV2
from coba.environments import OpenmlSimulation

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
        obj = CobaRegistry.registry["Null"]
        self.assertEqual("NullSink", obj.__name__)

    def test_endpoint_loaded_after_decorator_register(self):
        @coba_registry_class("MyTestObject")
        class MyTestObject(TestObject): pass
        obj = CobaRegistry.registry["Null"]
        self.assertEqual("NullSink", obj.__name__)

    def test_register_decorator(self):
        @coba_registry_class("MyTestObject")
        class MyTestObject(TestObject): pass
        "MyTestObject" in CobaRegistry.registry

class JsonMakerV1_Tests(unittest.TestCase):

    def test_make(self):

        obj = JsonMakerV1({"test": TestObject}).make("test")

        self.assertIsInstance(obj, TestObject)
        self.assertEqual(obj.args, ())
        self.assertEqual(obj.kwargs, {})

    def test_make_args1(self):

        obj = JsonMakerV1({"test": TestObject}).make({ "test": [1,2,3] })

        self.assertIsInstance(obj, TestObject)
        self.assertEqual(obj.args, (1,2,3))
        self.assertEqual(obj.kwargs, {})

    def test_make_args2(self):

        obj = JsonMakerV1({"test": TestObject}).make({ "test": 1 })

        self.assertIsInstance(obj, TestObject)
        self.assertEqual(obj.args, (1,))
        self.assertEqual(obj.kwargs, {})

    def test_make_kwargs(self):

        obj = JsonMakerV1({"test": TestObject}).make({ "test": {"a":1} })

        self.assertIsInstance(obj, TestObject)
        self.assertEqual(obj.args, ())
        self.assertEqual(obj.kwargs, {"a":1})

    def test_make_args3(self):

        obj = JsonMakerV1({"test": TestObject}).make({ "test": "abc" })

        self.assertIsInstance(obj, TestObject)
        self.assertEqual(obj.args, ("abc",))
        self.assertEqual(obj.kwargs, {})

    def test_make_args_kwargs(self):

        obj = JsonMakerV1({"test": TestObject}).make({ "test": [1,2,3], "kwargs": {"a":1} })

        self.assertIsInstance(obj, TestObject)
        self.assertEqual(obj.args, (1,2,3))
        self.assertEqual(obj.kwargs, {"a":1})

    def test_make_name_args_kwargs(self):

        obj = JsonMakerV1({"test": TestObject}).make({ "name": "test", "args": [1,2,3], "kwargs": {"a":1} })

        self.assertIsInstance(obj, TestObject)
        self.assertEqual(obj.args, (1,2,3))
        self.assertEqual(obj.kwargs, {"a":1})

    def test_make_foreach1(self):

        recipe = { "test":[[1,2,3]], "kwargs": {"a":1}, "method":"foreach" }
        objs = JsonMakerV1({"test": TestObject}).make(recipe)

        self.assertEqual(len(objs), 1)
        self.assertEqual(objs[0].args, (1,2,3))
        self.assertEqual(objs[0].kwargs, {"a":1})

    def test_make_foreach2(self):

        recipe = { "test":[1,2,3], "kwargs": {"a":1}, "method":"foreach" }
        objs = JsonMakerV1({"test": TestObject}).make(recipe)

        self.assertEqual(len(objs), 3)

        self.assertEqual(objs[0].args, (1,))
        self.assertEqual(objs[0].kwargs, {"a":1})

        self.assertEqual(objs[1].args, (2,))
        self.assertEqual(objs[1].kwargs, {"a":1})

        self.assertEqual(objs[2].args, (3,))
        self.assertEqual(objs[2].kwargs, {"a":1})

    def test_make_foreach3(self):

        recipe = { "test":[1,2], "kwargs": [{"a":1},{"a":2}], "method":"foreach" }
        objs = JsonMakerV1({"test": TestObject}).make(recipe)

        self.assertEqual(len(objs), 2)

        self.assertEqual(objs[0].args, (1,))
        self.assertEqual(objs[0].kwargs, {"a":1})

        self.assertEqual(objs[1].args, (2,))
        self.assertEqual(objs[1].kwargs, {"a":2})

    def test_make_foreach4(self):

        recipe = { "test":[[1,2],3], "method":"foreach" }
        objs = JsonMakerV1({"test": TestObject}).make(recipe)

        self.assertEqual(len(objs), 2)

        self.assertEqual(objs[0].args, (1,2))
        self.assertEqual(objs[0].kwargs, {})

        self.assertEqual(objs[1].args, (3,))
        self.assertEqual(objs[1].kwargs, {})

    def test_make_recursive1(self):

        obj = JsonMakerV1({"test": TestObject}).make({ "test": "test" })

        self.assertEqual(1, len(obj.args))
        self.assertEqual(obj.kwargs, {})

        self.assertIsInstance(obj.args[0], TestObject)
        self.assertEqual(obj.args[0].args, ())
        self.assertEqual(obj.args[0].kwargs, {})

    def test_make_recursive2(self):

        obj = JsonMakerV1({"test": TestObject}).make({ "test": {"test":1} })

        self.assertEqual(1, len(obj.args))
        self.assertEqual(obj.kwargs, {})

        self.assertIsInstance(obj.args[0], TestObject)
        self.assertEqual(obj.args[0].args, (1,))
        self.assertEqual(obj.args[0].kwargs, {})

    def test_make_recursive3(self):

        obj = JsonMakerV1({"test": TestObject}).make({ "test": {"a": "test"} })

        self.assertEqual(obj.args, ())
        self.assertEqual(1, len(obj.kwargs))

        self.assertIsInstance(obj.kwargs["a"], TestObject)
        self.assertEqual(obj.kwargs["a"].args, ())
        self.assertEqual(obj.kwargs["a"].kwargs, {})

    def test_make_array_arg(self):

        obj = JsonMakerV1({"test": TestArgObject}).make({ "test": [1,2,3] })

        self.assertEqual(obj.arg, [1,2,3])

    def test_make_dict_arg(self):

        with self.assertRaises(Exception):
            JsonMakerV1({"test": TestArgObject}).make({ "test": {"a":1} })

    def test_make_optionalarray_arg(self):

        obj = JsonMakerV1({"test": TestOptionalArgObject}).make({ "test": [1,2,3] })

        self.assertEqual(obj.arg, [1,2,3])

    def test_not_registered(self):

        with self.assertRaises(Exception) as cm:
            JsonMakerV1({"test": TestObject}).make("test2")

        self.assertEqual("Unknown recipe test2", str(cm.exception))

    def test_invalid_recipe1(self):

        recipe = {"test":[1,2,3], "args":[4,5,6] }

        with self.assertRaises(Exception) as cm:
            JsonMakerV1({"test": TestObject}).make(recipe)

        self.assertEqual(f"Invalid recipe {str(recipe)}", str(cm.exception))

    def test_invalid_recipe2(self):

        recipe = {"test":[1,2,3], "name":"test", "args":[4,5,6]}
        with self.assertRaises(Exception) as cm:
            JsonMakerV1({"test": TestObject}).make(recipe)

        self.assertEqual(f"Invalid recipe {str(recipe)}", str(cm.exception))

    def test_invalid_recipe3(self):

        recipe = {"test":{"a":1}, "name":"test", "kwargs":{"a":1}}

        with self.assertRaises(Exception) as cm:
            JsonMakerV1({"test": TestObject}).make(recipe)

        self.assertEqual(f"Invalid recipe {str(recipe)}", str(cm.exception))

    def test_invalid_recipe4(self):

        recipe = 1

        with self.assertRaises(Exception) as cm:
            JsonMakerV1({"test": TestObject}).make(recipe)

        self.assertEqual(f"Invalid recipe {str(recipe)}", str(cm.exception))

    def test_make_optionalarray_arg(self):

        obj = JsonMakerV1({"test": TestOptionalArgObject}).make({ "test": [1,2,3] })

        self.assertEqual(obj.arg, [1,2,3])

class JsonMakerV2_Tests(unittest.TestCase):

    def test_registed_make_no_args_no_kwargs(self):
        obj = JsonMakerV2({"test": TestObject}).make("test")
        self.assertIsInstance(obj, TestObject)
        self.assertEqual(obj.args, ())
        self.assertEqual(obj.kwargs, {})

    def test_make_args1(self):
        obj = JsonMakerV2({"test": TestObject}).make({ "test": [1,2,3] })
        self.assertIsInstance(obj, TestObject)
        self.assertEqual(obj.args, (1,2,3))
        self.assertEqual(obj.kwargs, {})

    def test_make_args2(self):
        obj = JsonMakerV2({"test": TestObject}).make({ "test": 1 })
        self.assertIsInstance(obj, TestObject)
        self.assertEqual(obj.args, (1,))
        self.assertEqual(obj.kwargs, {})

    def test_make_args3(self):
        obj = JsonMakerV1({"test": TestObject}).make({ "test": "abc" })
        self.assertIsInstance(obj, TestObject)
        self.assertEqual(obj.args, ("abc",))
        self.assertEqual(obj.kwargs, {})

    def test_make_kwargs(self):
        obj = JsonMakerV2({"test": TestObject}).make({ "test": {"a":1} })
        self.assertIsInstance(obj, TestObject)
        self.assertEqual(obj.args, ())
        self.assertEqual(obj.kwargs, {"a":1})

    def test_make_args_kwargs(self):
        obj = JsonMakerV2({"test": TestObject}).make({ "test": [1,2,3,'**',{"a":1}] })
        self.assertIsInstance(obj, TestObject)
        self.assertEqual(obj.args, (1,2,3))
        self.assertEqual(obj.kwargs, {"a":1})

    def test_make_for_no_args_no_kwargs(self):
        objs = JsonMakerV2({"test": TestObject}).make({ "test":[], "for":[1,2] })
        self.assertEqual(2, len(objs))

        self.assertIsInstance(objs[0], TestObject)
        self.assertEqual(objs[0].args, ())
        self.assertEqual(objs[0].kwargs, {})

        self.assertIsInstance(objs[1], TestObject)
        self.assertEqual(objs[1].args, ())
        self.assertEqual(objs[1].kwargs, {})

    def test_make_array_arg(self):
        obj = JsonMakerV2({"test": TestArgObject}).make({ "test": [[1,2,3]] })
        self.assertEqual(obj.arg, [1,2,3])

    def test_make_dict_arg(self):
        with self.assertRaises(Exception):
            JsonMakerV2({"test": TestArgObject}).make({ "test": {"a":1} })

    def test_make_default_arg1(self):
        obj = JsonMakerV2({"test": TestOptionalArgObject}).make({ "test": [[1,2,3]] })
        self.assertEqual(obj.arg, [1,2,3])

    def test_make_default_arg2(self):
        obj = JsonMakerV2({"test": TestOptionalArgObject}).make({ "test": [] })
        self.assertEqual(obj.arg, 1)

    def test_make_for_arg(self):
        objs = JsonMakerV2({"test": TestObject}).make({ "test":"$", "for":[1,2] })
        self.assertEqual(2, len(objs))

        self.assertIsInstance(objs[0], TestObject)
        self.assertEqual(objs[0].args, (1,))
        self.assertEqual(objs[0].kwargs, {})

        self.assertIsInstance(objs[1], TestObject)
        self.assertEqual(objs[1].args, (2,))
        self.assertEqual(objs[1].kwargs, {})

    def test_make_for_args(self):
        objs = JsonMakerV2({"test": TestObject}).make({ "test":["$",9], "for":[1,2] })
        self.assertEqual(2, len(objs))

        self.assertIsInstance(objs[0], TestObject)
        self.assertEqual(objs[0].args, (1,9))
        self.assertEqual(objs[0].kwargs, {})

        self.assertIsInstance(objs[1], TestObject)
        self.assertEqual(objs[1].args, (2,9))
        self.assertEqual(objs[1].kwargs, {})

    def test_make_for_zip(self):
        objs = JsonMakerV2({"test": TestObject, "zip":zip}).make({ "test":"$", "for":{"zip":[[1,2],[3,4]] }})
        self.assertEqual(2, len(objs))

        self.assertIsInstance(objs[0], TestObject)
        self.assertEqual(objs[0].args, ((1,3),))
        self.assertEqual(objs[0].kwargs, {})

        self.assertIsInstance(objs[1], TestObject)
        self.assertEqual(objs[1].args, ((2,4),))
        self.assertEqual(objs[1].kwargs, {})

    def test_make_for_kwargs(self):
        objs = JsonMakerV2({"test": TestObject}).make({ "test":{"a":"$",'b':3} , "for":[1,2] })
        self.assertEqual(2, len(objs))

        self.assertIsInstance(objs[0], TestObject)
        self.assertEqual(objs[0].args, ())
        self.assertEqual(objs[0].kwargs, {'a':1,'b':3})

        self.assertIsInstance(objs[1], TestObject)
        self.assertEqual(objs[1].args, ())
        self.assertEqual(objs[1].kwargs, {'a':2,'b':3})

    def test_make_recursive1(self):

        obj = JsonMakerV2({"test": TestObject}).make({ "test": "test" })

        self.assertEqual(1, len(obj.args))
        self.assertEqual(obj.kwargs, {})

        self.assertIsInstance(obj.args[0], TestObject)
        self.assertEqual(obj.args[0].args, ())
        self.assertEqual(obj.args[0].kwargs, {})

    def test_make_recursive2(self):

        obj = JsonMakerV2({"test": TestObject}).make({ "test": [{"test":1}] })

        self.assertEqual(1, len(obj.args))
        self.assertEqual(obj.kwargs, {})

        self.assertIsInstance(obj.args[0], TestObject)
        self.assertEqual(obj.args[0].args, (1,))
        self.assertEqual(obj.args[0].kwargs, {})

    def test_make_recursive3(self):

        obj = JsonMakerV2({"test": TestObject}).make({ "test": {"a": "test"} })

        self.assertEqual(obj.args, ())
        self.assertEqual(1, len(obj.kwargs))

        self.assertIsInstance(obj.kwargs["a"], TestObject)
        self.assertEqual(obj.kwargs["a"].args, ())
        self.assertEqual(obj.kwargs["a"].kwargs, {})

    def test_make_unmakeable(self):

        recipe = 1
        with self.assertRaises(CobaException) as e:
            JsonMakerV2({"test": TestObject}).make(recipe)

        self.assertEqual(f"We were unable to make {recipe}.", str(e.exception))


class JsonMakerV2Regression_Tests(unittest.TestCase):

    def test_openmlsimulation_for_interface_consistency(self):

        sim = JsonMakerV2(CobaRegistry.registry).make({"OpenmlSimulation":1})

        self.assertIsInstance(sim, OpenmlSimulation)
        self.assertEqual(sim.params['openml_data'], 1)
        self.assertEqual(sim.params['drop_missing'], True)
        self.assertEqual(sim.params['drop_missing'], True)
        self.assertNotIn('reservoir_take', sim.params)

        sim = JsonMakerV2(CobaRegistry.registry).make({"OpenmlSimulation":[1,True]})

        self.assertIsInstance(sim, OpenmlSimulation)
        self.assertEqual(sim.params['openml_data'], 1)
        self.assertEqual(sim.params['drop_missing'], True)
        self.assertNotIn('reservoir_take', sim.params)

        sim = JsonMakerV2(CobaRegistry.registry).make({"OpenmlSimulation":[1,False]})

        self.assertIsInstance(sim, OpenmlSimulation)
        self.assertEqual(sim.params['openml_data'], 1)
        self.assertEqual(sim.params['drop_missing'], False)
        self.assertNotIn('reservoir_take', sim.params)

        sim = JsonMakerV2(CobaRegistry.registry).make({"OpenmlSimulation":[1,False,100]})

        self.assertIsInstance(sim, OpenmlSimulation)
        self.assertEqual(sim.params['openml_data'], 1)
        self.assertEqual(sim.params['drop_missing'], False)
        self.assertEqual(sim.params['reservoir_count'], 100)

        sim = JsonMakerV2(CobaRegistry.registry).make({"OpenmlSimulation":{"data_id":1,"drop_missing":False,"take":100}})

        self.assertIsInstance(sim, OpenmlSimulation)
        self.assertEqual(sim.params['openml_data'], 1)
        self.assertEqual(sim.params['drop_missing'], False)
        self.assertEqual(sim.params['reservoir_count'], 100)

        sim = JsonMakerV2(CobaRegistry.registry).make({"OpenmlSimulation":{"task_id":1,"drop_missing":False,"take":100}})

        self.assertIsInstance(sim, OpenmlSimulation)
        self.assertEqual(sim.params['openml_task'], 1)
        self.assertEqual(sim.params['drop_missing'], False)
        self.assertEqual(sim.params['reservoir_count'], 100)

if __name__ == '__main__':
    unittest.main()