import unittest
import unittest.mock

from pathlib import Path
from coba.config.loggers import NullLogger

from coba.pipes import JsonEncode
from coba.config import CobaConfig, DiskCacher, IndentLogger
from coba.pipes.io import ConsoleIO, DiskIO, NullIO

class CobaConfig_Tests(unittest.TestCase):

    def setUp(self) -> None:
        if Path("coba/tests/.temp/.coba").exists():
            Path("coba/tests/.temp/.coba").unlink()

        CobaConfig._cacher = None
        CobaConfig._logger = None
        CobaConfig._experiment = None
        CobaConfig._global = {}
        CobaConfig._config_backing = None

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

    def test_config_file_disk_cacher_settings_home(self):

        CobaConfig.search_paths = ["coba/tests/.temp/"]

        try:
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

        finally:
            if Path("coba/tests/.temp/.coba").exists():
                Path("coba/tests/.temp/.coba").unlink() 

    def test_config_file_disk_cacher_settings_current_directory(self):

        CobaConfig.search_paths = ["coba/tests/.temp/"]

        try:
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

        finally:
            if Path("coba/tests/.temp/.coba").exists():
                Path("coba/tests/.temp/.coba").unlink()

    def test_config_file_disk_cacher_settings_current_directory(self):

        CobaConfig.search_paths = ["coba/tests/.temp/"]

        try:
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

        finally:
            if Path("coba/tests/.temp/.coba").exists():
                Path("coba/tests/.temp/.coba").unlink()

    def test_config_file_disk_cacher_settings_current_directory(self):

        CobaConfig.search_paths = ["coba/tests/.temp/"]

        try:
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

        finally:
            if Path("coba/tests/.temp/.coba").exists():
                Path("coba/tests/.temp/.coba").unlink()

    def test_config_file_experiment_settings(self):

        CobaConfig.search_paths = ["coba/tests/.temp/"]

        try:
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

        finally:
            if Path("coba/tests/.temp/.coba").exists():
                Path("coba/tests/.temp/.coba").unlink()

    @unittest.mock.patch('builtins.print')
    def test_bad_config_file(self,mock_print):
        CobaConfig.search_paths = ["coba/tests/.temp/"]

        try:
            DiskIO("coba/tests/.temp/.coba").write('{ "cacher": { "DiskCacher": "~"')

            try:
                CobaConfig.cacher
            except:
                pass
            
            self.assertIn("An unexpected error occured when initializing CobaConfig", mock_print.call_args_list[0][0][0])
            self.assertIn("The coba configuration file at coba\\tests\\.temp\\.coba", mock_print.call_args_list[1][0][0])
            self.assertIn("error, Expecting ',' delimiter: line 2 column 1 (char 32).", mock_print.call_args_list[1][0][0])

        finally:
            if Path("coba/tests/.temp/.coba").exists():
                Path("coba/tests/.temp/.coba").unlink()

    @unittest.mock.patch('builtins.print')
    def test_bad_search_path(self,mock_print):
        CobaConfig.search_paths = [None]

        try:
            CobaConfig.cacher
        except:
            pass

        self.assertIn("An unexpected error occured when initializing CobaConfig", mock_print.call_args_list[0][0][0])
        self.assertIn("File", mock_print.call_args_list[1][0][0])
        self.assertIn("line", mock_print.call_args_list[1][0][0])
        self.assertIn("TypeError: unsupported operand type(s)", mock_print.call_args_list[2][0][0])

if __name__ == '__main__':
    unittest.main()