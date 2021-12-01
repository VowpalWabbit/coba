import unittest
from pathlib import Path

from coba.pipes import DiskIO
from coba.environments import Environments

class Environments_Tests(unittest.TestCase):
    def test_from_file_path(self):
        if Path("coba/tests/.temp/from_file.env").exists():
            Path("coba/tests/.temp/from_file.env").unlink()

        try:
            Path("coba/tests/.temp/from_file.env").write_text('{ "environments" : { "OpenmlSimulation": 150 } }')

            env = Environments.from_file("coba/tests/.temp/from_file.env")

            self.assertEqual(1    , len(env))
            self.assertEqual(150  , env[0].params['openml'])
            self.assertEqual(False, env[0].params['cat_as_str'])

        finally:
            if Path("coba/tests/.temp/from_file.env").exists():
                Path("coba/tests/.temp/from_file.env").unlink()

    def test_from_file_source(self):
        if Path("coba/tests/.temp/from_file.env").exists():
            Path("coba/tests/.temp/from_file.env").unlink()

        try:
            Path("coba/tests/.temp/from_file.env").write_text('{ "environments" : { "OpenmlSimulation": 150 } }')

            env = Environments.from_file(DiskIO("coba/tests/.temp/from_file.env"))

            self.assertEqual(1    , len(env))
            self.assertEqual(150  , env[0].params['openml'])
            self.assertEqual(False, env[0].params['cat_as_str'])

        finally:
            if Path("coba/tests/.temp/from_file.env").exists():
                Path("coba/tests/.temp/from_file.env").unlink()

    def test_from_debug(self):
        env = Environments.from_debug(100,2,3,3,0,["xa"],2)

        self.assertEqual(1     , len(env))
        self.assertEqual(100   , len(env[0].read()))
        self.assertEqual(2     , env[0].params['|A|'])
        self.assertEqual(3     , env[0].params['|phi(C)|'])
        self.assertEqual(3     , env[0].params['|phi(A)|'])
        self.assertEqual(0     , env[0].params['e_var'])
        self.assertEqual(['xa'], env[0].params['X'])
        self.assertEqual(2     , env[0].params['seed'])

if __name__ == '__main__':
    unittest.main()