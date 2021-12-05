import unittest
import os

from pathlib import Path
from coba.config.loggers import NullLogger

from coba.exceptions import CobaExit
from coba.pipes import JsonEncode
from coba.config import CobaConfig, DiskCacher, IndentLogger
from coba.pipes.io import ConsoleIO, DiskIO, NullIO

class CobaConfig_Tests(unittest.TestCase):

    def setUp(self) -> None:
        if Path("coba/tests/.temp/.coba").exists():
            Path("coba/tests/.temp/.coba").unlink()

        CobaConfig.search_paths = ["coba/tests/.temp/"]
        CobaConfig._cacher = None
        CobaConfig._logger = None
        CobaConfig._experiment = None
        CobaConfig._global = {}
        CobaConfig._config_backing = None

    def tearDown(self) -> None:
        if Path("coba/tests/.temp/.coba").exists():
            Path("coba/tests/.temp/.coba").unlink()

    def test_default_settings(self):

        CobaConfig.search_paths = ["coba/tests/.temp/"]

        self.assertIsInstance(CobaConfig.cacher, DiskCacher)
        self.assertIsInstance(CobaConfig.logger, IndentLogger)
        self.assertIsInstance(CobaConfig.logger.sink, ConsoleIO)

        self.assertEqual(CobaConfig.cacher.cache_directory, None)
        self.assertEqual(CobaConfig.experiment.processes, 1)
        self.assertEqual(CobaConfig.experiment.maxtasksperchild, 0)
        self.assertEqual(CobaConfig.experiment.chunk_by, 'source')
        self.assertEqual(CobaConfig.api_keys, {})
        self.assertEqual(CobaConfig.store, {})

    def test_config_file_disk_cacher_home1(self):

        CobaConfig.search_paths = ["coba/tests/.temp/"]

        DiskIO("coba/tests/.temp/.coba").write(JsonEncode().filter({"cacher": { "DiskCacher": "~"}}))

        self.assertIsInstance(CobaConfig.cacher, DiskCacher)
        self.assertIsInstance(CobaConfig.logger, IndentLogger)
        self.assertIsInstance(CobaConfig.logger.sink, ConsoleIO)

        self.assertEqual(CobaConfig.cacher.cache_directory, str(Path("~").expanduser()))
        self.assertEqual(CobaConfig.experiment.processes, 1)
        self.assertEqual(CobaConfig.experiment.maxtasksperchild, 0)
        self.assertEqual(CobaConfig.experiment.chunk_by, 'source')
        self.assertEqual(CobaConfig.api_keys, {})
        self.assertEqual(CobaConfig.store, {})

    def test_config_file_disk_cacher_home2(self):

        CobaConfig.search_paths = ["coba/tests/.temp/"]

        DiskIO("coba/tests/.temp/.coba").write(JsonEncode().filter({"cacher": { "DiskCacher": "~/"}}))

        self.assertIsInstance(CobaConfig.cacher, DiskCacher)
        self.assertIsInstance(CobaConfig.logger, IndentLogger)
        self.assertIsInstance(CobaConfig.logger.sink, ConsoleIO)

        self.assertEqual(CobaConfig.cacher.cache_directory, str(Path("~/").expanduser()))
        self.assertEqual(CobaConfig.experiment.processes, 1)
        self.assertEqual(CobaConfig.experiment.maxtasksperchild, 0)
        self.assertEqual(CobaConfig.experiment.chunk_by, 'source')
        self.assertEqual(CobaConfig.api_keys, {})
        self.assertEqual(CobaConfig.store, {})

    def test_config_file_disk_cacher_current_directory(self):

        CobaConfig.search_paths = ["coba/tests/.temp/"]

        DiskIO("coba/tests/.temp/.coba").write(JsonEncode().filter({"cacher": { "DiskCacher": "./"}}))

        self.assertIsInstance(CobaConfig.cacher, DiskCacher)
        self.assertIsInstance(CobaConfig.logger, IndentLogger)
        self.assertIsInstance(CobaConfig.logger.sink, ConsoleIO)

        self.assertEqual(CobaConfig.cacher.cache_directory, str(Path("coba/tests/.temp").resolve()))
        self.assertEqual(CobaConfig.experiment.processes, 1)
        self.assertEqual(CobaConfig.experiment.maxtasksperchild, 0)
        self.assertEqual(CobaConfig.experiment.chunk_by, 'source')
        self.assertEqual(CobaConfig.api_keys, {})
        self.assertEqual(CobaConfig.store, {})

    def test_config_file_disk_cacher_up_one_directory(self):

        CobaConfig.search_paths = ["coba/tests/.temp/"]

        DiskIO("coba/tests/.temp/.coba").write(JsonEncode().filter({"cacher": { "DiskCacher": "../"}}))

        self.assertIsInstance(CobaConfig.cacher, DiskCacher)
        self.assertIsInstance(CobaConfig.logger, IndentLogger)
        self.assertIsInstance(CobaConfig.logger.sink, ConsoleIO)

        self.assertEqual(CobaConfig.cacher.cache_directory, str(Path("coba/tests/").resolve()))
        self.assertEqual(CobaConfig.experiment.processes, 1)
        self.assertEqual(CobaConfig.experiment.maxtasksperchild, 0)
        self.assertEqual(CobaConfig.experiment.chunk_by, 'source')
        self.assertEqual(CobaConfig.api_keys, {})
        self.assertEqual(CobaConfig.store, {})

    def test_config_file_logger(self):

        CobaConfig.search_paths = ["coba/tests/.temp/"]

        DiskIO("coba/tests/.temp/.coba").write(JsonEncode().filter({"logger": "NullLogger"}))

        self.assertIsInstance(CobaConfig.cacher, DiskCacher)
        self.assertIsInstance(CobaConfig.logger, NullLogger)
        self.assertIsInstance(CobaConfig.logger.sink, NullIO)

        self.assertEqual(CobaConfig.cacher.cache_directory, None)
        self.assertEqual(CobaConfig.experiment.processes, 1)
        self.assertEqual(CobaConfig.experiment.maxtasksperchild, 0)
        self.assertEqual(CobaConfig.experiment.chunk_by, 'source')
        self.assertEqual(CobaConfig.api_keys, {})
        self.assertEqual(CobaConfig.store, {})

    def test_config_file_experiment(self):

        CobaConfig.search_paths = ["coba/tests/.temp/"]

        DiskIO("coba/tests/.temp/.coba").write(JsonEncode().filter({"experiment": {"processes":2, "chunk_by": "task"}}))

        self.assertIsInstance(CobaConfig.cacher, DiskCacher)
        self.assertIsInstance(CobaConfig.logger, IndentLogger)
        self.assertIsInstance(CobaConfig.logger.sink, ConsoleIO)

        self.assertEqual(CobaConfig.cacher.cache_directory, None)
        self.assertEqual(CobaConfig.experiment.processes, 2)
        self.assertEqual(CobaConfig.experiment.maxtasksperchild, 0)
        self.assertEqual(CobaConfig.experiment.chunk_by, 'task')
        self.assertEqual(CobaConfig.api_keys, {})
        self.assertEqual(CobaConfig.store, {})

    def test_config_directly_set_experiment(self):

        CobaConfig.experiment.processes = 3
        CobaConfig.experiment.chunk_by = 'task'
        CobaConfig.experiment.maxtasksperchild = 10

        self.assertEqual(3, CobaConfig.experiment.processes)
        self.assertEqual('task', CobaConfig.experiment.chunk_by)
        self.assertEqual(10, CobaConfig.experiment.maxtasksperchild)

    def test_bad_config_file1(self):
        CobaConfig.search_paths = ["coba/tests/.temp/"]

        DiskIO("coba/tests/.temp/.coba").write('{ "cacher": { "DiskCacher": "~"')

        with self.assertRaises(CobaExit) as e:
            CobaConfig.cacher

        lines = str(e.exception).splitlines()

        self.assertEqual('', lines[0])
        self.assertIn("ERROR: An error occured while initializing CobaConfig", lines[1])
        self.assertIn(f"Expecting ',' delimiter: line 2 column 1 (char 32) ", lines[2])
        self.assertIn(f" in coba{os.sep}tests{os.sep}.temp{os.sep}.coba.", lines[2])
        self.assertTrue(str(e.exception).endswith("\n"))

    def test_bad_config_file2(self):
        CobaConfig.search_paths = ["coba/tests/.temp/"]

        DiskIO("coba/tests/.temp/.coba").write('[1,2,3]')

        with self.assertRaises(CobaExit) as e:
            CobaConfig.cacher
        
        lines = str(e.exception).splitlines()

        self.assertEqual('', lines[0])
        self.assertIn("ERROR: An error occured while initializing CobaConfig", lines[1]) 
        self.assertIn(f"Expecting a JSON object (i.e., {{}}) ", lines[2])
        self.assertIn(f" in coba{os.sep}tests{os.sep}.temp{os.sep}.coba.", lines[2])
        self.assertTrue(str(e.exception).endswith("\n"))

    def test_bad_search_path(self):
        CobaConfig.search_paths = [None]

        with self.assertRaises(CobaExit) as e:
            CobaConfig.cacher

        lines = str(e.exception).splitlines()

        self.assertEqual('', lines[0])
        self.assertIn("ERROR: An error occured while initializing CobaConfig", lines[1])
        self.assertIn("File", lines[2])
        self.assertIn("line", lines[2])
        self.assertIn("TypeError: unsupported operand type(s)", lines[-1])
        self.assertTrue(str(e.exception).endswith("\n"))

if __name__ == '__main__':
    unittest.main()
