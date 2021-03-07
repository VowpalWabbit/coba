import time
import shutil
import unittest
import traceback

from pathlib import Path

from coba.data.sinks import MemorySink
from coba.tools import PackageChecker, DiskCacher, IndentLogger, BasicLogger, CobaRegistry, coba_registry_class

class TestObject:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

class TestArgObject:
    def __init__(self, arg):
        self.arg = arg

class check_library_Tests(unittest.TestCase):
    
    def test_check_matplotlib_support(self):
        try:
            PackageChecker.matplotlib("test_check_matplotlib_support")
        except Exception:
            self.fail("check_matplotlib_support raised an exception")

    def test_check_vowpal_support(self):
        try:
            PackageChecker.vowpalwabbit("test_check_vowpal_support")
        except Exception:
            self.fail("check_vowpal_support raised an exception")

class BasicLogger_Tests(unittest.TestCase):

    def test_log(self):

        sink   = MemorySink()
        logger = BasicLogger(sink, with_stamp=False, with_name=False)
        logs   = sink.items

        logger.log('a')
        logger.log('c')
        logger.log('d')

        self.assertEqual(logs[0], 'a' )
        self.assertEqual(logs[1], 'c' )
        self.assertEqual(logs[2], 'd' )

    def test_log_with_1(self):

        #This test is somewhat time dependent.
        #I don't think it should ever fail, but if it does
        #try running it again and see if it works that time.

        sink   = MemorySink()
        logger = BasicLogger(sink,with_stamp=False, with_name=False)
        logs   = sink.items

        with logger.log('a'):
            logger.log('c')
            logger.log('d')
        logger.log('e')

        self.assertEqual(5, len(logs))

        self.assertEqual(logs[0], 'a')
        self.assertEqual(logs[1], 'c')
        self.assertEqual(logs[2], 'd')
        self.assertEqual(logs[3], 'a (finish)')
        self.assertEqual(logs[4], 'e')

    def test_time_with_1(self):

        #This test is somewhat time dependent.
        #I don't think it should ever fail, but if it does
        #try running it again and see if it works that time.

        sink   = MemorySink()
        logger = BasicLogger(sink,with_stamp=False, with_name=False)
        logs   = sink.items

        with logger.time('a'):
            logger.log('c')
            time.sleep(0.15)
            logger.log('d')
        logger.log('e')

        self.assertEqual(5, len(logs))

        self.assertEqual        (logs[0], 'a')
        self.assertEqual        (logs[1], 'c')
        self.assertEqual        (logs[2], 'd')
        self.assertRegexpMatches(logs[3], 'a \\(\\d+\\.\\d+ seconds\\)')
        self.assertEqual        (logs[4], 'e')


        self.assertAlmostEqual(float(logs[3][3:7 ]), 0.15, 1)

    def test_time_with_3(self):

        sink   = MemorySink()
        logger = BasicLogger(sink, with_stamp=False)
        logs   = sink.items

        with logger.time('a'):
            with logger.log('c'):
                time.sleep(0.05)
                with logger.time('d'):
                    logger.log('e')
                    time.sleep(0.05)
                with logger.time('d'):
                    logger.log('e')
                    time.sleep(0.05)
            logger.log('f')
        logger.log('g')

        self.assertEqual(12, len(logs))

        self.assertEqual        (logs[0 ], 'a')
        self.assertEqual        (logs[1 ], 'c')
        self.assertEqual        (logs[2 ], 'd')
        self.assertEqual        (logs[3 ], 'e')
        self.assertRegexpMatches(logs[4 ], 'd \\(\\d+\\.\\d+ seconds\\)')
        self.assertEqual        (logs[5 ], 'd')
        self.assertEqual        (logs[6 ], 'e')
        self.assertRegexpMatches(logs[7 ], 'd \\(\\d+\\.\\d+ seconds\\)')
        self.assertEqual        (logs[8 ], 'c (finish)')
        self.assertEqual        (logs[9 ], 'f')
        self.assertRegexpMatches(logs[10], 'a \\(\\d+\\.\\d+ seconds\\)')
        self.assertEqual        (logs[11], 'g')

        self.assertAlmostEqual(float(logs[4 ][3:7]), 0.05, 1)
        self.assertAlmostEqual(float(logs[7 ][3:7]), 0.05, 1)
        self.assertAlmostEqual(float(logs[10][3:7]), 0.15, 1)

    def test_time_two_separate(self):

        sink   = MemorySink()
        logger = BasicLogger(sink, with_stamp=False)
        logs   = sink.items

        with logger.time('a'):
            time.sleep(0.05)
            with logger.time('d'):
                logger.log('e')
                time.sleep(0.05)
        
        logger.log('g')
        
        with logger.time('a'):
            time.sleep(0.05)
            with logger.time('d'):
                logger.log('e')
                time.sleep(0.05)
        
        self.assertEqual(11, len(logs))

        self.assertEqual        (logs[0 ], 'a')
        self.assertEqual        (logs[1 ], 'd')
        self.assertEqual        (logs[2 ], 'e')
        self.assertRegexpMatches(logs[3 ], 'd \\(\\d+\\.\\d+ seconds\\)')
        self.assertRegexpMatches(logs[4 ], 'a \\(\\d+\\.\\d+ seconds\\)')
        self.assertEqual        (logs[5 ], 'g')
        self.assertEqual        (logs[6 ], 'a')
        self.assertEqual        (logs[7 ], 'd')
        self.assertEqual        (logs[8 ], 'e')
        self.assertRegexpMatches(logs[9 ], 'd \\(\\d+\\.\\d+ seconds\\)')
        self.assertRegexpMatches(logs[10], 'a \\(\\d+\\.\\d+ seconds\\)')

        self.assertAlmostEqual(float(logs[3 ][3:7]), 0.05, 1)
        self.assertAlmostEqual(float(logs[4 ][3:7]), 0.10, 1)
        self.assertAlmostEqual(float(logs[9 ][3:7]), 0.05, 1)
        self.assertAlmostEqual(float(logs[10][3:7]), 0.10, 1)

    def test_log_exception_1(self):
        
        sink   = MemorySink()
        logger = BasicLogger(sink, with_stamp=False)
        logs   = sink.items

        try:
            raise Exception("Test Exception")
        except Exception as ex:
            logger.log_exception('error:',ex)

            tb = ''.join(traceback.format_tb(ex.__traceback__))
            msg = ''.join(traceback.TracebackException.from_exception(ex).format_exception_only())

            expected_msg = f"error:\n\n{tb}\n  {msg}"

            self.assertTrue(ex.__logged__) #type:ignore
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0], expected_msg)

    def test_log_exception_2(self):
        
        sink   = MemorySink()
        logger = BasicLogger(sink, with_stamp=False)
        logs   = sink.items
        exception = Exception("Test Exception")

        logger.log('a')
        logger.log_exception('',exception)

        tb = ''.join(traceback.format_tb(exception.__traceback__))
        msg = ''.join(traceback.TracebackException.from_exception(exception).format_exception_only())

        expected_msg = f"\n\n{tb}\n  {msg}"

        self.assertTrue(exception.__logged__) #type:ignore
        self.assertEqual(logs[0], "a")
        self.assertEqual(logs[1], expected_msg)

class IndentLogger_Tests(unittest.TestCase):

    def test_log(self):

        sink   = MemorySink()
        logger = IndentLogger(sink, with_stamp=False, with_name=False)
        logs   = sink.items

        logger.log('a')
        logger.log('c')
        logger.log('d')

        self.assertEqual(logs[0], 'a' )
        self.assertEqual(logs[1], 'c' )
        self.assertEqual(logs[2], 'd' )

    def test_log_with_1(self):

        #This test is somewhat time dependent.
        #I don't think it should ever fail, but if it does
        #try running it again and see if it works that time.

        sink   = MemorySink()
        logger = IndentLogger(sink,with_stamp=False, with_name=False)
        logs   = sink.items

        with logger.log('a'):
            logger.log('c')
            logger.log('d')
        logger.log('e')

        self.assertEqual(4, len(logs))
        self.assertEqual(logs[0], 'a'    )
        self.assertEqual(logs[1], '  * c')
        self.assertEqual(logs[2], '  * d')
        self.assertEqual(logs[3], 'e'    )

    def test_time_with_1(self):

        #This test is somewhat time dependent.
        #I don't think it should ever fail, but if it does
        #try running it again and see if it works that time.

        sink   = MemorySink()
        logger = IndentLogger(sink,with_stamp=False, with_name=False)
        logs   = sink.items

        with logger.time('a'):
            logger.log('c')
            time.sleep(0.15)
            logger.log('d')
        logger.log('e')

        self.assertEqual(4, len(logs))
        self.assertRegexpMatches(logs[0], 'a \\(\\d+\\.\\d+ seconds\\)')
        self.assertEqual        (logs[1], '  * c')
        self.assertEqual        (logs[2], '  * d')
        self.assertEqual        (logs[3], 'e'    )

        self.assertAlmostEqual(float(logs[0][3:7 ]), 0.15, 1)

    def test_time_with_3(self):

        sink   = MemorySink()
        logger = IndentLogger(sink, with_stamp=False)
        logs   = sink.items

        with logger.time('a'):
            with logger.log('c'):
                time.sleep(0.05)
                with logger.time('d'):
                    logger.log('e')
                    time.sleep(0.05)
                with logger.time('d'):
                    logger.log('e')
                    time.sleep(0.05)
            logger.log('f')
        logger.log('g')

        self.assertEqual(8, len(logs))
        self.assertRegexpMatches(logs[0], 'a \\(\\d+\\.\\d+ seconds\\)')
        self.assertEqual        (logs[1], '  * c')
        self.assertRegexpMatches(logs[2], '    > d \\(\\d+\\.\\d+ seconds\\)')
        self.assertEqual        (logs[3], '      - e')
        self.assertRegexpMatches(logs[4], '    > d \\(\\d+\\.\\d+ seconds\\)')
        self.assertEqual        (logs[5], '      - e')
        self.assertEqual        (logs[6], '  * f')
        self.assertEqual        (logs[7], 'g'    )

        self.assertAlmostEqual(float(logs[0][3:7 ]), 0.15, 1)
        self.assertAlmostEqual(float(logs[2][9:13]), 0.05, 1)
        self.assertAlmostEqual(float(logs[4][9:13]), 0.05, 1)
    
    def test_time_two_separate(self):

        sink   = MemorySink()
        logger = IndentLogger(sink, with_stamp=False)
        logs   = sink.items

        with logger.time('a'):
            time.sleep(0.05)
            with logger.time('d'):
                logger.log('e')
                time.sleep(0.05)
        
        logger.log('g')
        
        with logger.time('a'):
            time.sleep(0.05)
            with logger.time('d'):
                logger.log('e')
                time.sleep(0.05)
        
        self.assertEqual(7, len(logs))
        self.assertRegexpMatches(logs[0 ], 'a \\(\\d+\\.\\d+ seconds\\)')
        self.assertRegexpMatches(logs[1 ], '  \\* d \\(\\d+\\.\\d+ seconds\\)')
        self.assertEqual        (logs[2 ], '    > e')
        self.assertEqual        (logs[3 ], 'g')
        self.assertRegexpMatches(logs[4 ], 'a \\(\\d+\\.\\d+ seconds\\)')
        self.assertRegexpMatches(logs[5 ], '  \\* d \\(\\d+\\.\\d+ seconds\\)')
        self.assertEqual        (logs[6 ], '    > e')

        self.assertAlmostEqual(float(logs[0][3:7 ]), 0.10, 1)
        self.assertAlmostEqual(float(logs[1][7:11]), 0.05, 1)
        self.assertAlmostEqual(float(logs[4][3:7 ]), 0.10, 1)
        self.assertAlmostEqual(float(logs[5][7:11]), 0.05, 1)

    def test_log_exception_1(self):
        
        sink   = MemorySink()
        logger = IndentLogger(sink, with_stamp=False)
        logs   = sink.items

        try:
            raise Exception("Test Exception")
        except Exception as ex:
            logger.log_exception('error:',ex)

            tb = ''.join(traceback.format_tb(ex.__traceback__))
            msg = ''.join(traceback.TracebackException.from_exception(ex).format_exception_only())

            expected_msg = f"error:\n\n{tb}\n  {msg}"

            self.assertTrue(ex.__logged__) #type:ignore
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0], expected_msg)

    def test_log_exception_2(self):
        
        sink   = MemorySink()
        logger = IndentLogger(sink, with_stamp=False)
        logs   = sink.items
        exception = Exception("Test Exception")

        logger.log('a')
        logger.log_exception('',exception)

        tb = ''.join(traceback.format_tb(exception.__traceback__))
        msg = ''.join(traceback.TracebackException.from_exception(exception).format_exception_only())

        expected_msg = f"\n\n{tb}\n  {msg}"

        self.assertTrue(exception.__logged__) #type:ignore
        self.assertEqual(logs[0], "a")
        self.assertEqual(logs[1], expected_msg)

class DiskCache_Tests(unittest.TestCase):
    Cache_Test_Dir = Path("coba/tests/.temp/cache_tests/")
    
    def setUp(self):
        
        if self.Cache_Test_Dir.exists():
            shutil.rmtree(self.Cache_Test_Dir)
        
        self.Cache_Test_Dir.mkdir()

    def tearDown(self) -> None:
        
        if self.Cache_Test_Dir.exists():
            shutil.rmtree(self.Cache_Test_Dir)

    def test_creates_directory(self):
        cache = DiskCacher(self.Cache_Test_Dir / "folder1/folder2")
        cache.put("test.csv", b"test")
        self.assertTrue("test.csv" in cache)
            
    def test_write_csv_to_cache(self):

        cache = DiskCacher(self.Cache_Test_Dir)

        self.assertFalse("test.csv"    in cache)
        cache.put("test.csv", b"test")
        self.assertTrue("test.csv" in cache)

        self.assertEqual(cache.get("test.csv"), b"test")
    
    def test_rmv_csv_from_cache(self):

        cache = DiskCacher(self.Cache_Test_Dir)

        self.assertFalse("test.csv"    in cache)
        
        cache.put("test.csv", b"test")
        
        self.assertTrue("test.csv"    in cache)

        cache.rmv("test.csv")

        self.assertFalse("test.csv"    in cache)

class CobaRegistry_Tests(unittest.TestCase):

    def setUp(self) -> None:
        CobaRegistry.clear() #make sure the registry is fresh each test

    def test_endpoint_loaded(self):
        klass = CobaRegistry.retrieve("NoneSink")

        self.assertEqual("NoneSink", klass.__name__)

    def test_endpoint_loaded_after_decorator_register(self):
        
        @coba_registry_class("MyTestObject")
        class MyTestObject(TestObject): pass

        klass = CobaRegistry.retrieve("NoneSink")

        self.assertEqual("NoneSink", klass.__name__)

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

if __name__ == '__main__':
    unittest.main()