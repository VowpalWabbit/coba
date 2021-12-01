import unittest
from pathlib import Path

from coba.pipes import DiskIO, Shuffle, Take
from coba.environments import Environments, Environment

class TestEnvironment(Environment):

    def __init__(self, id) -> None:
        self._id = id

    @property
    def params(self):
        return {'id':self._id}

    def read(self):
        return []

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

    def test_from_openml_single(self):
        env = Environments.from_openml(100,100,True,'regression')

        self.assertEqual(1           , len(env))
        self.assertEqual(100         , env[0].params['openml'])
        self.assertEqual(True        , env[0].params['cat_as_str'])
        self.assertEqual('regression', env[0].params['openml_type'])
        self.assertEqual(100         , env[0].params['openml_take'])

    def test_from_openml_multi(self):
        env = Environments.from_openml([100,200],100,True,'regression')

        self.assertEqual(2           , len(env))
        self.assertEqual(100         , env[0].params['openml'])
        self.assertEqual(True        , env[0].params['cat_as_str'])
        self.assertEqual('regression', env[0].params['openml_type'])
        self.assertEqual(100         , env[0].params['openml_take'])
        self.assertEqual(200         , env[1].params['openml'])
        self.assertEqual(True        , env[1].params['cat_as_str'])
        self.assertEqual('regression', env[1].params['openml_type'])
        self.assertEqual(100         , env[1].params['openml_take'])

    def test_init_args(self):
        env = Environments(TestEnvironment('A'), TestEnvironment('B'))

        self.assertEqual(2  , len(env))
        self.assertEqual('A', env[0].params['id'])
        self.assertEqual('B', env[1].params['id'])

    def test_init_sequence_args(self):
        env = Environments([TestEnvironment('A'), TestEnvironment('B')])

        self.assertEqual(2  , len(env))
        self.assertEqual('A', env[0].params['id'])
        self.assertEqual('B', env[1].params['id'])

    def test_init_empty_args(self):
        env = Environments()
        self.assertEqual(0  , len(env))

    def test_iter(self):
        envs = Environments([TestEnvironment('A'), TestEnvironment('B')])

        for env,id in zip(envs,['A','B']):
            self.assertEqual(id, env.params['id'])

    def test_add(self):
        envs_1 = Environments(TestEnvironment('A'))
        envs_2 = Environments(TestEnvironment('B'))
        envs_3 = envs_1+envs_2

        self.assertEqual(1  , len(envs_1))
        self.assertEqual('A', envs_1[0].params['id'])

        self.assertEqual(1  , len(envs_2))
        self.assertEqual('B', envs_2[0].params['id'])

        self.assertEqual(2  , len(envs_3))
        self.assertEqual('A', envs_3[0].params['id'])
        self.assertEqual('B', envs_3[1].params['id'])

    def test_repr(self):
        envs = Environments(TestEnvironment('A'),TestEnvironment('B'))
        self.assertEqual(str(envs), f'1. {envs[0]}\n2. {envs[1]}')

    def test_binary(self):
        envs = Environments(TestEnvironment('A'),TestEnvironment('B')).binary()

        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(True, envs[0].params['binary'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(True, envs[1].params['binary'])

    def test_shuffle(self):
        envs = Environments(TestEnvironment('A'),TestEnvironment('B')).shuffle([1,2])

        self.assertEqual(4   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(1   , envs[0].params['shuffle'])
        self.assertEqual('A' , envs[1].params['id'])
        self.assertEqual(2   , envs[1].params['shuffle'])
        self.assertEqual('B' , envs[2].params['id'])
        self.assertEqual(1   , envs[2].params['shuffle'])
        self.assertEqual('B' , envs[3].params['id'])
        self.assertEqual(2   , envs[3].params['shuffle'])

    def test_take(self):
        envs = Environments(TestEnvironment('A'),TestEnvironment('B')).take(1,2)

        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(1   , envs[0].params['take'])
        self.assertEqual(2   , envs[0].params['take_seed'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(1   , envs[1].params['take'])
        self.assertEqual(2   , envs[1].params['take_seed'])

    def test_singular_filter(self):
        envs = Environments(TestEnvironment('A'),TestEnvironment('B')).filter(Shuffle(1))

        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(1   , envs[0].params['shuffle'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(1   , envs[1].params['shuffle'])

    def test_sequence_filter(self):
        envs = Environments(TestEnvironment('A'),TestEnvironment('B')).filter([Shuffle(1),Shuffle(2)])

        self.assertEqual(4   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(1   , envs[0].params['shuffle'])
        self.assertEqual('A' , envs[1].params['id'])
        self.assertEqual(2   , envs[1].params['shuffle'])
        self.assertEqual('B' , envs[2].params['id'])
        self.assertEqual(1   , envs[2].params['shuffle'])
        self.assertEqual('B' , envs[3].params['id'])
        self.assertEqual(2   , envs[3].params['shuffle'])

    def test_two_step_filter(self):
        envs = Environments(TestEnvironment('A'),TestEnvironment('B')).filter(Shuffle(1)).filter(Take(1,2))

        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(1   , envs[0].params['shuffle'])
        self.assertEqual(1   , envs[0].params['take'])
        self.assertEqual(2   , envs[0].params['take_seed'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(1   , envs[1].params['shuffle'])
        self.assertEqual(1   , envs[1].params['take'])
        self.assertEqual(2   , envs[1].params['take_seed'])

if __name__ == '__main__':
    unittest.main()