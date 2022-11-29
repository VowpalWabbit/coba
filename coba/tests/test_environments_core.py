import unittest
import unittest.mock
import requests

from pathlib import Path

from coba.contexts import CobaContext, DiskCacher
from coba.pipes import DiskSource, UrlSource
from coba.exceptions import CobaException
from coba.environments import Environments, Environment, Shuffle, Take
from coba.environments import SerializedSimulation, LinearSyntheticSimulation
from coba.environments import NeighborsSyntheticSimulation, KernelSyntheticSimulation, MLPSyntheticSimulation

class TestEnvironment(Environment):

    def __init__(self, id) -> None:
        self._id = id

    @property
    def params(self):
        return {'id':self._id}

    def read(self):
        return []

class MockResponse:
    def __init__(self, status_code, content):
        self.status_code = status_code
        self.content     = content

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass

class Environments_Tests(unittest.TestCase):
    
    def test_cache(self):
        env = Environments.cache('abc').from_linear_synthetic(100)[0]
        self.assertIsInstance(CobaContext.cacher, DiskCacher)
        self.assertEqual("abc", CobaContext.cacher.cache_directory)
        self.assertEqual(len(list(env.read())),100)

    def test_from_definition_path(self):
        if Path("coba/tests/.temp/from_file.env").exists():
            Path("coba/tests/.temp/from_file.env").unlink()

        try:
            Path("coba/tests/.temp/from_file.env").write_text('{ "environments" : { "OpenmlSimulation": 150 } }')

            env = Environments.from_template("coba/tests/.temp/from_file.env")

            self.assertEqual(1    , len(env))
            self.assertEqual(150  , env[0].params['openml_data'])

        finally:
            if Path("coba/tests/.temp/from_file.env").exists():
                Path("coba/tests/.temp/from_file.env").unlink()

    def test_from_definition_source(self):
        if Path("coba/tests/.temp/from_file.env").exists():
            Path("coba/tests/.temp/from_file.env").unlink()

        try:
            Path("coba/tests/.temp/from_file.env").write_text('{ "environments" : { "OpenmlSimulation": 150 } }')

            env = Environments.from_template(DiskSource("coba/tests/.temp/from_file.env"))

            self.assertEqual(1    , len(env))
            self.assertEqual(150  , env[0].params['openml_data'])

        finally:
            if Path("coba/tests/.temp/from_file.env").exists():
                Path("coba/tests/.temp/from_file.env").unlink()

    def test_from_prebuilt_recognized_name(self):

        index_url = "https://github.com/mrucker/coba_prebuilds/blob/main/test/index.json?raw=True"
        simulation_url = "https://github.com/mrucker/coba_prebuilds/blob/main/test/test.json?raw=True"

        def mocked_requests_get(*args, **kwargs):

            if args[0] == index_url:
                return MockResponse(200, b'{ "environments": { "SerializedSimulation": "./test.json" } }')

            return MockResponse(None, 404, [])

        with unittest.mock.patch.object(requests, 'get', side_effect=mocked_requests_get):
            envs = Environments.from_prebuilt("test")

        self.assertEqual(1, len(envs))
        self.assertIsInstance(envs[0], SerializedSimulation)
        self.assertIsInstance(envs[0]._source, UrlSource)
        self.assertEqual(simulation_url, envs[0]._source._url)

    def test_from_prebuilt_unrecognized_name(self):

        root_directory_url = "https://api.github.com/repos/mrucker/coba_prebuilds/contents/"

        def mocked_requests_get(*args, **kwargs):

            if args[0] == root_directory_url:
                return MockResponse(200, b'[{ "name":"test"}]')

            return MockResponse(404, None)

        with unittest.mock.patch.object(requests, 'get', side_effect=mocked_requests_get):
            with self.assertRaises(CobaException) as e:
                envs = Environments.from_prebuilt("nada")

            self.assertIn('nada', str(e.exception) )
            self.assertIn('test', str(e.exception) )

    def test_from_linear_synthetic(self):
        envs = Environments.from_linear_synthetic(100,2,3,4,["xa"],5)
        env  = envs[0]

        self.assertIsInstance(env, LinearSyntheticSimulation)
        interactions = list(env.read())

        self.assertEqual(1     , len(envs))
        self.assertEqual(100   , len(interactions))
        self.assertEqual(2     , len(interactions[0]['actions']))
        self.assertEqual(3     , len(interactions[0]['context']))
        self.assertEqual(4     , len(interactions[0]['actions'][0]))
        self.assertEqual(['xa'], env.params['reward_features'])
        self.assertEqual(5     , env.params['seed'])

    def test_from_neighbors_synthetic(self):
        envs = Environments.from_neighbors_synthetic(100,2,3,4,10,5)
        env  = envs[0]

        self.assertIsInstance(env, NeighborsSyntheticSimulation)
        interactions = list(env.read())

        self.assertEqual(1  , len(envs))
        self.assertEqual(100, len(interactions))
        self.assertEqual(2  , len(interactions[0]['actions']))
        self.assertEqual(3  , len(interactions[0]['context']))
        self.assertEqual(4  , len(interactions[0]['actions'][0]))
        self.assertEqual(10 , env.params['n_neighborhoods'])
        self.assertEqual(5  , env.params['seed'])

    def test_from_kernel_synthetic(self):
        envs = Environments.from_kernel_synthetic(100,2,3,4,5,kernel='polynomial',degree=3,seed=5)
        env  = envs[0]

        self.assertIsInstance(env, KernelSyntheticSimulation)
        interactions = list(env.read())

        self.assertEqual(1           , len(envs))
        self.assertEqual(100         , len(interactions))
        self.assertEqual(2           , len(interactions[0]['actions']))
        self.assertEqual(3           , len(interactions[0]['context']))
        self.assertEqual(4           , len(interactions[0]['actions'][0]))
        self.assertEqual(5           , env.params['n_exemplars'])
        self.assertEqual('polynomial', env.params['kernel'])
        self.assertEqual(3           , env.params['degree'])
        self.assertEqual(5           , env.params['seed'])

    def test_from_mlp_synthetic(self):
        envs = Environments.from_mlp_synthetic(100,2,3,4,5)
        env  = envs[0]

        self.assertIsInstance(env, MLPSyntheticSimulation)
        interactions = list(env.read())

        self.assertEqual(1           , len(envs))
        self.assertEqual(100         , len(interactions))
        self.assertEqual(2           , len(interactions[0]['actions']))
        self.assertEqual(3           , len(interactions[0]['context']))
        self.assertEqual(4           , len(interactions[0]['actions'][0]))
        self.assertEqual(5           , env.params['seed'])

    def test_from_supervised(self):
        env = Environments.from_supervised([1,2], [2,3], label_type="R")
        self.assertEqual(1   , len(env))
        self.assertEqual("[X,Y]", env[0].params['source'])
        self.assertEqual("R" , env[0].params['label_type'])

    def test_from_openml_data_id(self):
        env = Environments.from_openml(100,True,100)
        self.assertEqual(1   , len(env))
        self.assertEqual(100 , env[0].params['openml_data'])
        self.assertEqual(True, env[0].params['drop_missing'])
        self.assertEqual(None, env[0].params['label_type'])
        self.assertEqual(100 , env[0].params['reservoir_count'])

    def test_from_openml_data_ids(self):
        env = Environments.from_openml([100,200],True,100)

        self.assertEqual(2   , len(env))
        self.assertEqual(100 , env[0].params['openml_data'])
        self.assertEqual(True, env[0].params['drop_missing'])
        self.assertEqual(None, env[0].params['label_type'])
        self.assertEqual(100 , env[0].params['reservoir_count'])
        self.assertEqual(200 , env[1].params['openml_data'])
        self.assertEqual(True, env[1].params['drop_missing'])
        self.assertEqual(None, env[1].params['label_type'])
        self.assertEqual(100 , env[1].params['reservoir_count'])

    def test_from_openml_task_id(self):
        env = Environments.from_openml(task_id=100,take=100)
        self.assertEqual(1   , len(env))
        self.assertEqual(100 , env[0].params['openml_task'])
        self.assertEqual(None, env[0].params['label_type'])
        self.assertEqual(100 , env[0].params['reservoir_count'])

    def test_from_openml_task_ids(self):
        env = Environments.from_openml(task_id=[100,200],take=100)

        self.assertEqual(2   , len(env))
        self.assertEqual(100 , env[0].params['openml_task'])
        self.assertEqual(100 , env[0].params['reservoir_count'])
        self.assertEqual(200 , env[1].params['openml_task'])
        self.assertEqual(100 , env[1].params['reservoir_count'])

    def test_from_lambda(self):
        context = lambda index,rng               : [ round(r,2) for r in rng.randoms(5) ]
        actions = lambda index,context,rng       : [rng.randoms(5) for _ in range(3)]
        rewards = lambda index,context,action,rng: sum(c*a for c,a in zip(context,action))

        env = Environments.from_lambda(1, context, actions, rewards, 1)[0]

        self.assertEqual(1, len(list(env.read())))
        self.assertEqual([0.11, 0.8, 0.44, 0.17, 0.42], list(env.read())[0]['context'])
        self.assertEqual(3, len(list(env.read())[0]['actions']))

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

    def test_shuffle_n(self):
        envs = Environments(TestEnvironment('A'),TestEnvironment('B')).shuffle(n=2)

        self.assertEqual(4   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(0   , envs[0].params['shuffle'])
        self.assertEqual('A' , envs[1].params['id'])
        self.assertEqual(1   , envs[1].params['shuffle'])
        self.assertEqual('B' , envs[2].params['id'])
        self.assertEqual(0   , envs[2].params['shuffle'])
        self.assertEqual('B' , envs[3].params['id'])
        self.assertEqual(1   , envs[3].params['shuffle'])

    def test_shuffle_args(self):
        envs = Environments(TestEnvironment('A'),TestEnvironment('B')).shuffle(0,1)

        self.assertEqual(4   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(0   , envs[0].params['shuffle'])
        self.assertEqual('A' , envs[1].params['id'])
        self.assertEqual(1   , envs[1].params['shuffle'])
        self.assertEqual('B' , envs[2].params['id'])
        self.assertEqual(0   , envs[2].params['shuffle'])
        self.assertEqual('B' , envs[3].params['id'])
        self.assertEqual(1   , envs[3].params['shuffle'])

    def test_shuffle_iterable(self):
        envs = Environments(TestEnvironment('A'),TestEnvironment('B')).shuffle(range(2))

        self.assertEqual(4   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(0   , envs[0].params['shuffle'])
        self.assertEqual('A' , envs[1].params['id'])
        self.assertEqual(1   , envs[1].params['shuffle'])
        self.assertEqual('B' , envs[2].params['id'])
        self.assertEqual(0   , envs[2].params['shuffle'])
        self.assertEqual('B' , envs[3].params['id'])
        self.assertEqual(1   , envs[3].params['shuffle'])

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

    def test_reservoir_seed(self):
        envs = Environments(TestEnvironment('A'),TestEnvironment('B')).reservoir(1,2)

        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(1   , envs[0].params['reservoir_count'])
        self.assertEqual(2   , envs[0].params['reservoir_seed'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(1   , envs[1].params['reservoir_count'])
        self.assertEqual(2   , envs[1].params['reservoir_seed'])

    def test_reservoir_seeds(self):
        envs = Environments(TestEnvironment('A'),TestEnvironment('B')).reservoir(1,[2])

        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(1   , envs[0].params['reservoir_count'])
        self.assertEqual(2   , envs[0].params['reservoir_seed'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(1   , envs[1].params['reservoir_count'])
        self.assertEqual(2   , envs[1].params['reservoir_seed'])

    def test_scale(self):
        envs = Environments(TestEnvironment('A')).scale("med", "std", "features", 2)

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

    def test_where(self):
        envs = Environments(TestEnvironment('A'),TestEnvironment('B')).where(n_interactions = (1,2))

        self.assertEqual(2    , len(envs))

        self.assertEqual('A'  , envs[0].params['id'])
        self.assertEqual((1,2), envs[0].params['where_n_interactions'])
        self.assertEqual('B'  , envs[1].params['id'])
        self.assertEqual((1,2), envs[1].params['where_n_interactions'])

    def test_flatten(self):
        envs = Environments(TestEnvironment('A'),TestEnvironment('B')).flat()

        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(True, envs[0].params['flat'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(True, envs[1].params['flat'])

    def test_noise(self):
        envs = Environments(TestEnvironment('A')).noise(lambda x,r: x+1, lambda x,r: x+2, lambda x,r: x+3, lambda x,r: x+4)

        self.assertEqual(1, len(envs))

        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(True, envs[0].params['context_noise'])
        self.assertEqual(True, envs[0].params['action_noise'])
        self.assertEqual(True, envs[0].params['context_noise'])

    def test_riffle(self):
        envs = Environments(TestEnvironment('A')).riffle(2,1)

        self.assertEqual(1, len(envs))

        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(2, envs[0].params['riffle_spacing'])
        self.assertEqual(1, envs[0].params['riffle_seed'])

    def test_sort(self):
        envs = Environments(TestEnvironment('A')).sort()

        self.assertEqual(1, len(envs))

        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual('*', envs[0].params['sort'])

    def test_repr(self):
        envs = Environments(TestEnvironment('A')).repr()

        self.assertEqual(1, len(envs))
        self.assertEqual('onehot' , envs[0].params['cat_actions'])
        self.assertEqual('onehot', envs[0].params['cat_context'])


    def test_materialize(self):
        envs  = Environments.from_linear_synthetic(100,2,3,4,["xa"],5)
        envs += Environments.from_linear_synthetic(10 ,2,3,4,["xa"],6)

        envs = envs.materialize()

        self.assertEqual(100,len(envs[0][-1]._cache))
        self.assertEqual(10 ,len(envs[1][-1]._cache))

        interactions = list(envs[0].read())

        self.assertEqual(2     , len(envs))
        self.assertEqual(100   , len(interactions))
        self.assertEqual(2     , len(interactions[0]['actions']))
        self.assertEqual(3     , len(interactions[0]['context']))
        self.assertEqual(4     , len(interactions[0]['actions'][0]))
        self.assertEqual(['xa'], envs[0].params['reward_features'])
        
        self.assertEqual(5     , envs[0].params['seed'])
        self.assertEqual(6     , envs[1].params['seed'])

    def test_grounded(self):
        envs = Environments.from_linear_synthetic(100,2,3,4,["xa"],5)        
        envs = envs.grounded(10,5,4,2,3)
        env  = envs[0]

        self.assertEqual(10,env.params['n_users'])
        self.assertEqual(5,env.params['n_normal'])
        self.assertEqual(4,env.params['n_words'])
        self.assertEqual(2,env.params['n_good'])
        self.assertEqual(3,env.params['igl_seed'])

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

    def test_params(self):
        envs = Environments(TestEnvironment('A'),TestEnvironment('B')).params({'a':123})

        self.assertEqual(2  , len(envs))
        self.assertEqual('A', envs[0].params['id'])
        self.assertEqual(123, envs[0].params['a'])
        self.assertEqual('B', envs[1].params['id'])
        self.assertEqual(123, envs[1].params['a'])

    def test_batch(self):
        envs = Environments(TestEnvironment('A'),TestEnvironment('B')).batch(3)

        self.assertEqual(2  , len(envs))
        self.assertEqual('A', envs[0].params['id'])
        self.assertEqual(3  , envs[0].params['batched'])
        self.assertEqual('B', envs[1].params['id'])
        self.assertEqual(3  , envs[1].params['batched'])


    def test_filter_new(self):
        envs1 = Environments(TestEnvironment('A'))
        envs2 = envs1.params({'a':123})

        self.assertEqual(1  , len(envs1))
        self.assertEqual(1  , len(envs2))
        self.assertEqual({'id':'A'}, envs1[0].params)
        self.assertEqual({'id':'A','a':123}, envs2[0].params)

    def test_ipython_display(self):
        with unittest.mock.patch("builtins.print") as mock:
            envs = Environments(TestEnvironment('A'),TestEnvironment('B'))
            envs._ipython_display_()
            mock.assert_called_once_with(f'1. {envs[0]}\n2. {envs[1]}')

if __name__ == '__main__':
    unittest.main()
