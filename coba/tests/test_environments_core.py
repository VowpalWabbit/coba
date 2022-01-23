import unittest
import unittest.mock
from pathlib import Path

from coba.pipes import DiskIO
from coba.environments import Environments, Environment, Shuffle, Take

class TestEnvironment(Environment):

    def __init__(self, id) -> None:
        self._id = id

    @property
    def params(self):
        return {'id':self._id}

    def read(self):
        return []

class Environments_Tests(unittest.TestCase):
    def test_from_definition_path(self):
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

    def test_from_definition_source(self):
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

    def test_from_linear_synthetic(self):
        env = Environments.from_linear_synthetic(100,2,3,3,0,["xa"],2)

        self.assertEqual(1     , len(env))
        self.assertEqual(100   , len(list(env[0].read())))
        self.assertEqual(2     , env[0].params['n_A'])
        self.assertEqual(3     , env[0].params['n_C_phi'])
        self.assertEqual(3     , env[0].params['n_A_phi'])
        self.assertEqual(0     , env[0].params['r_noise'])
        self.assertEqual(['xa'], env[0].params['X'])
        self.assertEqual(2     , env[0].params['seed'])

    def test_from_local_synthetic(self):
        env = Environments.from_local_synthetic(100,2,1,10,2)

        self.assertEqual(1  , len(env))
        self.assertEqual(100, len(list(env[0].read())))
        self.assertEqual(2  , env[0].params['n_A'])
        self.assertEqual(10 , env[0].params['n_C'])
        self.assertEqual(1  , env[0].params['n_C_phi'])
        self.assertEqual(2  , env[0].params['seed'])

    def test_from_supervised(self):
        X = [1,2]
        Y = [2,3]

        env = Environments.from_supervised([1,2], [2,3], label_type="R", take=2)
        self.assertEqual(1   , len(env))
        self.assertEqual("XY", env[0].params['super_source'])
        self.assertEqual("R" , env[0].params['super_type'])
        self.assertEqual(2   , env[0].params['super_take'])

    def test_from_openml_single(self):
        env = Environments.from_openml(100,100,'R',True)

        self.assertEqual(1   , len(env))
        self.assertEqual(100 , env[0].params['openml'])
        self.assertEqual(True, env[0].params['cat_as_str'])
        self.assertEqual('R' , env[0].params['openml_type'])
        self.assertEqual('R' , env[0].params['super_type'])
        self.assertEqual(100 , env[0].params['super_take'])

    def test_from_openml_multi(self):
        env = Environments.from_openml([100,200],100,'R',True)

        self.assertEqual(2   , len(env))
        self.assertEqual(100 , env[0].params['openml'])
        self.assertEqual(True, env[0].params['cat_as_str'])
        self.assertEqual('R' , env[0].params['openml_type'])
        self.assertEqual('R' , env[0].params['super_type'])
        self.assertEqual(100 , env[0].params['super_take'])
        self.assertEqual(200 , env[1].params['openml'])
        self.assertEqual(True, env[1].params['cat_as_str'])
        self.assertEqual('R' , env[1].params['openml_type'])
        self.assertEqual('R' , env[1].params['super_type'])
        self.assertEqual(100 , env[1].params['super_take'])

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

    def test_str(self):
        envs = Environments(TestEnvironment('A'),TestEnvironment('B'))
        self.assertEqual(str(envs), f'1. {envs[0]}\n2. {envs[1]}')

    def test_binary(self):
        envs = Environments(TestEnvironment('A'),TestEnvironment('B')).binary()

        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(True, envs[0].params['binary'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(True, envs[1].params['binary'])

    def test_cycle(self):
        envs = Environments(TestEnvironment('A'),TestEnvironment('B')).cycle(2)

        self.assertEqual(2  , len(envs))
        self.assertEqual('A', envs[0].params['id'])
        self.assertEqual(2  , envs[0].params['cycle_after'])
        self.assertEqual('B', envs[1].params['id'])
        self.assertEqual(2  , envs[1].params['cycle_after'])

    def test_shuffle_default(self):
        envs = Environments(TestEnvironment('A'),TestEnvironment('B')).shuffle()
        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(1   , envs[0].params['shuffle'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(1   , envs[1].params['shuffle'])

    def test_shuffle_int(self):
        envs = Environments(TestEnvironment('A'),TestEnvironment('B')).shuffle(1)
        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(1   , envs[0].params['shuffle'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(1   , envs[1].params['shuffle'])

    def test_shuffle_sequence(self):
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

    def test_sparse(self):
        envs = Environments(TestEnvironment('A'),TestEnvironment('B')).sparse(False,True)

        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(False, envs[0].params['sparse_C'])
        self.assertEqual(True , envs[0].params['sparse_A'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(False, envs[1].params['sparse_C'])
        self.assertEqual(True, envs[1].params['sparse_A'])

    def test_take(self):
        envs = Environments(TestEnvironment('A'),TestEnvironment('B')).take(1)

        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(1   , envs[0].params['take'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(1   , envs[1].params['take'])

    def test_reservoir(self):
        envs = Environments(TestEnvironment('A'),TestEnvironment('B')).reservoir(1,[2])

        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(1   , envs[0].params['reservoir_count'])
        self.assertEqual(2   , envs[0].params['reservoir_seed'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(1   , envs[1].params['reservoir_count'])
        self.assertEqual(2   , envs[1].params['reservoir_seed'])

    def test_scale(self):
        envs = Environments(TestEnvironment('A')).scale("med", "std", 2)

        self.assertEqual(1  , len(envs))
        self.assertEqual('A'  , envs[0].params['id'])
        self.assertEqual('med', envs[0].params['scale_shift'])
        self.assertEqual('std', envs[0].params['scale_scale'])
        self.assertEqual(2    , envs[0].params['scale_using'])

    def test_impute(self):
        envs = Environments(TestEnvironment('A')).impute('median', 2)

        self.assertEqual(1       , len(envs))
        self.assertEqual('A'     , envs[0].params['id'])
        self.assertEqual('median', envs[0].params['impute_stat'])
        self.assertEqual(2       , envs[0].params['impute_using'])


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
        envs = Environments(TestEnvironment('A'),TestEnvironment('B')).filter(Shuffle(1)).filter(Take(1))

        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(1   , envs[0].params['shuffle'])
        self.assertEqual(1   , envs[0].params['take'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(1   , envs[1].params['shuffle'])
        self.assertEqual(1   , envs[1].params['take'])

    def test_ipython_display(self):
        with unittest.mock.patch("builtins.print") as mock:
            envs = Environments(TestEnvironment('A'),TestEnvironment('B'))
            envs._ipython_display_()
            mock.assert_called_once_with(f'1. {envs[0]}\n2. {envs[1]}')

if __name__ == '__main__':
    unittest.main()