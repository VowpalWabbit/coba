import unittest
import os

from pathlib import Path

from coba.exceptions import CobaExit
from coba.pipes import JsonEncode, ConsoleSink, DiskSink, NullSink
from coba.context import CobaContext, DiskCacher, IndentLogger, NullLogger

class CobaContext_Tests(unittest.TestCase):

    def setUp(self) -> None:
        if Path("coba/tests/.temp/.coba").exists():
            Path("coba/tests/.temp/.coba").unlink()

        CobaContext.search_paths = ["coba/tests/.temp/"]
        CobaContext._cacher = None
        CobaContext._logger = None
        CobaContext._experiment = None
        CobaContext._store = {}
        CobaContext._config_backing = None
        CobaContext._learning_info = None

    def tearDown(self) -> None:
        if Path("coba/tests/.temp/.coba").exists():
            Path("coba/tests/.temp/.coba").unlink()

    def test_default_settings(self):

        CobaContext.search_paths = ["coba/tests/.temp/"]

        self.assertIsInstance(CobaContext.cacher, DiskCacher)
        self.assertIsInstance(CobaContext.logger, IndentLogger)
        self.assertIsInstance(CobaContext.logger.sink, ConsoleSink)

        self.assertEqual(CobaContext.cacher.cache_directory, None)
        self.assertEqual(CobaContext.experiment.processes, 1)
        self.assertEqual(CobaContext.experiment.maxchunksperchild, 0)
        self.assertEqual(CobaContext.experiment.maxtasksperchunk, 0)
        self.assertEqual(CobaContext.experiment.chunk_by, 'source')
        self.assertEqual(CobaContext.api_keys, {})
        self.assertEqual(CobaContext.store, {})
        self.assertEqual(CobaContext.learning_info, {})

    def test_config_file_disk_cacher_home1(self):

        CobaContext.search_paths = ["coba/tests/.temp/"]

        DiskSink("coba/tests/.temp/.coba").write(JsonEncode().filter({"cacher": { "DiskCacher": "~"}}))

        self.assertIsInstance(CobaContext.cacher, DiskCacher)
        self.assertIsInstance(CobaContext.logger, IndentLogger)
        self.assertIsInstance(CobaContext.logger.sink, ConsoleSink)

        self.assertEqual(CobaContext.cacher.cache_directory, str(Path("~").expanduser()))
        self.assertEqual(CobaContext.experiment.processes, 1)
        self.assertEqual(CobaContext.experiment.maxchunksperchild, 0)
        self.assertEqual(CobaContext.experiment.chunk_by, 'source')
        self.assertEqual(CobaContext.api_keys, {})
        self.assertEqual(CobaContext.store, {})

    def test_config_file_disk_cacher_home2(self):

        CobaContext.search_paths = ["coba/tests/.temp/"]

        DiskSink("coba/tests/.temp/.coba").write(JsonEncode().filter({"cacher": { "DiskCacher": "~/"}}))

        self.assertIsInstance(CobaContext.cacher, DiskCacher)
        self.assertIsInstance(CobaContext.logger, IndentLogger)
        self.assertIsInstance(CobaContext.logger.sink, ConsoleSink)

        self.assertEqual(CobaContext.cacher.cache_directory, str(Path("~/").expanduser()))
        self.assertEqual(CobaContext.experiment.processes, 1)
        self.assertEqual(CobaContext.experiment.maxchunksperchild, 0)
        self.assertEqual(CobaContext.experiment.chunk_by, 'source')
        self.assertEqual(CobaContext.api_keys, {})
        self.assertEqual(CobaContext.store, {})

    def test_config_file_disk_cacher_current_directory(self):

        CobaContext.search_paths = ["coba/tests/.temp/"]

        DiskSink("coba/tests/.temp/.coba").write(JsonEncode().filter({"cacher": { "DiskCacher": "./"}}))

        self.assertIsInstance(CobaContext.cacher, DiskCacher)
        self.assertIsInstance(CobaContext.logger, IndentLogger)
        self.assertIsInstance(CobaContext.logger.sink, ConsoleSink)

        self.assertEqual(CobaContext.cacher.cache_directory, str(Path("coba/tests/.temp").resolve()))
        self.assertEqual(CobaContext.experiment.processes, 1)
        self.assertEqual(CobaContext.experiment.maxchunksperchild, 0)
        self.assertEqual(CobaContext.experiment.chunk_by, 'source')
        self.assertEqual(CobaContext.api_keys, {})
        self.assertEqual(CobaContext.store, {})

    def test_config_file_disk_cacher_up_one_directory(self):

        CobaContext.search_paths = ["coba/tests/.temp/"]

        DiskSink("coba/tests/.temp/.coba").write(JsonEncode().filter({"cacher": { "DiskCacher": "../"}}))

        self.assertIsInstance(CobaContext.cacher, DiskCacher)
        self.assertIsInstance(CobaContext.logger, IndentLogger)
        self.assertIsInstance(CobaContext.logger.sink, ConsoleSink)

        self.assertEqual(CobaContext.cacher.cache_directory, str(Path("coba/tests/").resolve()))
        self.assertEqual(CobaContext.experiment.processes, 1)
        self.assertEqual(CobaContext.experiment.maxchunksperchild, 0)
        self.assertEqual(CobaContext.experiment.chunk_by, 'source')
        self.assertEqual(CobaContext.api_keys, {})
        self.assertEqual(CobaContext.store, {})

    def test_config_file_logger(self):

        CobaContext.search_paths = ["coba/tests/.temp/"]

        DiskSink("coba/tests/.temp/.coba").write(JsonEncode().filter({"logger": "NullLogger"}))

        self.assertIsInstance(CobaContext.cacher, DiskCacher)
        self.assertIsInstance(CobaContext.logger, NullLogger)
        self.assertIsInstance(CobaContext.logger.sink, NullSink)

        self.assertEqual(CobaContext.cacher.cache_directory, None)
        self.assertEqual(CobaContext.experiment.processes, 1)
        self.assertEqual(CobaContext.experiment.maxchunksperchild, 0)
        self.assertEqual(CobaContext.experiment.chunk_by, 'source')
        self.assertEqual(CobaContext.api_keys, {})
        self.assertEqual(CobaContext.store, {})

    def test_config_file_experiment_1(self):

        CobaContext.search_paths = ["coba/tests/.temp/"]

        config1 = {"experiment": {"processes":2, "maxtasksperchild":3, "maxtasksperchunk":1, "chunk_by": "task"}}
        DiskSink("coba/tests/.temp/.coba").write(JsonEncode().filter(config1))

        self.assertIsInstance(CobaContext.cacher, DiskCacher)
        self.assertIsInstance(CobaContext.logger, IndentLogger)
        self.assertIsInstance(CobaContext.logger.sink, ConsoleSink)

        self.assertEqual(CobaContext.cacher.cache_directory, None)
        self.assertEqual(CobaContext.experiment.processes, 2)
        self.assertEqual(CobaContext.experiment.maxchunksperchild, 3)
        self.assertEqual(CobaContext.experiment.maxtasksperchunk, 1)
        self.assertEqual(CobaContext.experiment.chunk_by, 'task')
        self.assertEqual(CobaContext.api_keys, {})
        self.assertEqual(CobaContext.store, {})

    def test_config_file_experiment_2(self):
        config = {"experiment": {"processes":2, "maxchunksperchild":3, "maxtasksperchunk":1, "chunk_by": "task"}}
        DiskSink("coba/tests/.temp/.coba").write(JsonEncode().filter(config))

        self.assertIsInstance(CobaContext.cacher, DiskCacher)
        self.assertIsInstance(CobaContext.logger, IndentLogger)
        self.assertIsInstance(CobaContext.logger.sink, ConsoleSink)

        self.assertEqual(CobaContext.cacher.cache_directory, None)
        self.assertEqual(CobaContext.experiment.processes, 2)
        self.assertEqual(CobaContext.experiment.maxchunksperchild, 3)
        self.assertEqual(CobaContext.experiment.maxtasksperchunk, 1)
        self.assertEqual(CobaContext.experiment.chunk_by, 'task')
        self.assertEqual(CobaContext.api_keys, {})
        self.assertEqual(CobaContext.store, {})

    def test_config_directly_set_experiment(self):

        CobaContext.experiment.processes = 3
        CobaContext.experiment.chunk_by = 'task'
        CobaContext.experiment.maxchunksperchild = 10
        CobaContext.experiment.maxtasksperchunk = 8

        self.assertEqual(3, CobaContext.experiment.processes)
        self.assertEqual('task', CobaContext.experiment.chunk_by)
        self.assertEqual(10, CobaContext.experiment.maxchunksperchild)
        self.assertEqual(8, CobaContext.experiment.maxtasksperchunk)

    def test_bad_config_file1(self):
        CobaContext.search_paths = ["coba/tests/.temp/"]

        DiskSink("coba/tests/.temp/.coba").write('{ "cacher": { "DiskCacher": "~"')

        with self.assertRaises(CobaExit) as e:
            CobaContext.cacher

        lines = str(e.exception).splitlines()

        self.assertEqual('', lines[0])
        self.assertIn("ERROR: An error occured while initializing CobaContext", lines[1])
        self.assertIn(f"Expecting ',' delimiter: line 2 column 1 (char 32) ", lines[2])
        self.assertIn(f" in coba{os.sep}tests{os.sep}.temp{os.sep}.coba.", lines[2])
        self.assertTrue(str(e.exception).endswith("\n"))

    def test_bad_config_file2(self):
        CobaContext.search_paths = ["coba/tests/.temp/"]

        DiskSink("coba/tests/.temp/.coba").write('[1,2,3]')

        with self.assertRaises(CobaExit) as e:
            CobaContext.cacher

        lines = str(e.exception).splitlines()

        self.assertEqual('', lines[0])
        self.assertIn("ERROR: An error occured while initializing CobaContext", lines[1])
        self.assertIn(f"Expecting a JSON object (i.e., {{}}) ", lines[2])
        self.assertIn(f" in coba{os.sep}tests{os.sep}.temp{os.sep}.coba.", lines[2])
        self.assertTrue(str(e.exception).endswith("\n"))

    def test_bad_search_path(self):
        CobaContext.search_paths = [None]

        with self.assertRaises(CobaExit) as e:
            CobaContext.cacher

        lines = str(e.exception).splitlines()

        self.assertEqual('', lines[0])
        self.assertIn("ERROR: An error occured while initializing CobaContext", lines[1])
        self.assertIn("File", lines[2])
        self.assertIn("line", lines[2])
        self.assertIn("TypeError: unsupported operand type(s)", lines[-1])
        self.assertTrue(str(e.exception).endswith("\n"))

    def test_learning_info_persistence(self):
        CobaContext.learning_info['a'] = 1
        self.assertEqual({'a':1},CobaContext.learning_info)
        CobaContext.learning_info.clear()
        self.assertEqual({},CobaContext.learning_info)

if __name__ == '__main__':
    unittest.main()
