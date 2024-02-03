import unittest
import unittest.mock

from urllib import request
from pathlib import Path

from coba.utilities    import PackageChecker
from coba.context      import CobaContext, DiskCacher, NullLogger
from coba.pipes        import DiskSource, LazyDense, Cache
from coba.exceptions   import CobaException
from coba.primitives   import Categorical, L1Reward, DiscreteReward
from coba.environments import Environments, Shuffle, Take
from coba.environments import LinearSyntheticSimulation
from coba.environments import NeighborsSyntheticSimulation, KernelSyntheticSimulation, MLPSyntheticSimulation
from coba.learners     import FixedLearner
from coba.results      import Result

class BatchList(list):
    is_batch = True

class TestEnvironment1:
    def __init__(self, id) -> None:
        self._id = id
    @property
    def params(self):
        return {'id':self._id}
    def read(self):
        return []

class TestEnvironment2:
    def __init__(self,n=2):
        self._n = n
    @property
    def params(self,):
        return {'a':1}
    def read(self):
        for n in range(self._n):
            yield {'b':n+1}

class TestEnvironment3:
    def __init__(self):
        self._reads = 0
    @property
    def params(self):
        return {'a':2}
    def read(self):
        assert self._reads == 0
        self._reads += 1
        yield {'b':2}
        yield {'b':3}

class MockResponse:
    def __init__(self, status_code, content):
        self.status_code = status_code
        self.content     = content

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass

class Environments_Tests(unittest.TestCase):

    def setUp(self) -> None:
        CobaContext.logger = NullLogger()
        if Path("coba/tests/.temp/test.zip").exists(): Path("coba/tests/.temp/test.zip").unlink()

    def tearDown(self) -> None:
        if Path("coba/tests/.temp/test.zip").exists(): Path("coba/tests/.temp/test.zip").unlink()

    def test_save_load_one_process(self):
        input_environments  = [TestEnvironment2(), TestEnvironment2()]
        output_environments = Environments(input_environments).save("coba/tests/.temp/test.zip")

        for env_in,env_out in zip(input_environments, output_environments):
            self.assertEqual(env_in.params,env_out.params)
            self.assertEqual(list(env_in.read()), list(env_out.read()))

    def test_save_load_two_process(self):
        input_environments = [TestEnvironment2(), TestEnvironment2()]
        output_environments = Environments(input_environments).save("coba/tests/.temp/test.zip",processes=2)

        for env_in,env_out in zip(input_environments, output_environments):
            self.assertEqual(env_in.params,env_out.params)
            self.assertEqual(list(env_in.read()), list(env_out.read()))

    def test_save_save_no_change(self):
        input_environments = [TestEnvironment2(), TestEnvironment2()]

        Environments(input_environments).save("coba/tests/.temp/test.zip")
        Environments(input_environments).save("coba/tests/.temp/test.zip")

        output_environments = Environments.from_save("coba/tests/.temp/test.zip")

        for env_in,env_out in zip(input_environments, output_environments):
            self.assertEqual(env_in.params,env_out.params)
            self.assertEqual(list(env_in.read()), list(env_out.read()))

    def test_save_save_change_no_overwrite(self):
        input_environments_1 = [TestEnvironment2(), TestEnvironment2()]
        input_environments_2 = [TestEnvironment1(3), TestEnvironment1(2)]

        with self.assertRaises(CobaException):
            Environments(input_environments_1).save("coba/tests/.temp/test.zip")
            Environments(input_environments_2).save("coba/tests/.temp/test.zip")

    def test_save_save_change_overwrite1(self):
        input_environments_1 = [TestEnvironment2(), TestEnvironment2()]
        input_environments_2 = [TestEnvironment1(3), TestEnvironment1(2)]

        Environments(input_environments_1).save("coba/tests/.temp/test.zip")
        Environments(input_environments_2).save("coba/tests/.temp/test.zip",overwrite=True)

        output_environments = Environments.from_save("coba/tests/.temp/test.zip")

        self.assertEqual(len(input_environments_2), len(output_environments))

        for env_in,env_out in zip(input_environments_2, output_environments):
            self.assertEqual(env_in.params,env_out.params)
            self.assertEqual(list(env_in.read()), list(env_out.read()))

    def test_save_save_change_overwrite2(self):
        input_environments_1 = [TestEnvironment1(2)]
        input_environments_2 = [TestEnvironment1(2), TestEnvironment2()]

        Environments(input_environments_1).save("coba/tests/.temp/test.zip")
        Environments(input_environments_2).save("coba/tests/.temp/test.zip",overwrite=True)

        output_environments = Environments.from_save("coba/tests/.temp/test.zip")

        self.assertEqual(len(input_environments_2), len(output_environments))

        for env_in,env_out in zip(input_environments_2, output_environments):
            self.assertEqual(env_in.params,env_out.params)
            self.assertEqual(list(env_in.read()), list(env_out.read()))

    def test_save_continue(self):
        input_environments_1 = Environments([TestEnvironment3()])

        #confirming there is an error if we don't continue
        with self.assertRaises(AssertionError):
            input_environments_1.save("coba/tests/.temp/test.zip")
            Path("coba/tests/.temp/test.zip").unlink()
            input_environments_1.save("coba/tests/.temp/test.zip")

        input_environments_1 = Environments([TestEnvironment3()])
        input_environments_2 = Environments([TestEnvironment2()])

        output_environments1 = input_environments_1.save("coba/tests/.temp/test.zip")
        output_environments2 = (input_environments_1+input_environments_2).save("coba/tests/.temp/test.zip")

        self.assertEqual(len(output_environments1),1)
        self.assertEqual(len(output_environments2),2)

        self.assertEqual(output_environments2[0].params, input_environments_1[0].params)
        self.assertEqual(output_environments2[1].params, input_environments_2[0].params)

    def test_save_badzip_overwrite(self):
        Path("coba/tests/.temp/test.zip").write_text("abc")
        input_environments  = [TestEnvironment2(), TestEnvironment2()]
        output_environments = Environments(input_environments).save("coba/tests/.temp/test.zip",overwrite=True)

        for env_in,env_out in zip(input_environments, output_environments):
            self.assertEqual(env_in.params,env_out.params)
            self.assertEqual(list(env_in.read()), list(env_out.read()))

    def test_save_badzip_no_overwrite(self):
        Path("coba/tests/.temp/test.zip").write_text("abc")
        with self.assertRaises(CobaException):
            Environments([TestEnvironment2(), TestEnvironment2()]).save("coba/tests/.temp/test.zip")

    def test_cache(self):
        Environments.cache_dir('abc')
        self.assertIsInstance(CobaContext.cacher, DiskCacher)
        self.assertEqual("abc", CobaContext.cacher.cache_directory)

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

    @unittest.mock.patch('coba.environments.core.HttpSource')
    def test_from_prebuilt_recognized_name(self, mock):
        index_url = "https://github.com/mrucker/coba_prebuilds/blob/main/test/index.json?raw=True"

        mock.return_value.read.return_value = '{ "environments": { "OpenmlSimulation": 10 } }'

        envs = Environments.from_prebuilt("test")
        self.assertEqual(1, len(envs))
        self.assertEqual(envs[0].params['openml_data'],10)
        mock.assert_called_once_with(index_url)

    @unittest.mock.patch('coba.environments.core.HttpSource')
    def test_from_prebuilt_unrecognized_name(self, mock):
        index_url = "https://github.com/mrucker/coba_prebuilds/blob/main/nada/index.json?raw=True"
        root_url  = "https://api.github.com/repos/mrucker/coba_prebuilds/contents/"

        def func(first_time=[True]):
            if first_time[0]:
                first_time[0] = False
                raise request.HTTPError('',404,'','','')
            else:
                return '[{ "name":"test"}]'

        # init.return_value = None
        mock.return_value.read.side_effect = func

        with self.assertRaises(CobaException) as e:
            Environments.from_prebuilt("nada")

        self.assertIn('nada', str(e.exception) )
        self.assertIn('test', str(e.exception) )
        self.assertEqual(mock.call_args_list[0][0][0], index_url)
        self.assertEqual(mock.call_args_list[1][0][0], root_url)

    @unittest.mock.patch('coba.environments.core.HttpSource')
    def test_from_prebuilt_unexpected_error(self, mock):

        ex = request.HTTPError('',420,'','','')

        mock.return_value.read.side_effect = ex

        with self.assertRaises(request.HTTPError) as r:
            Environments.from_prebuilt("nada")

        self.assertIs(r.exception,ex)

    def test_from_result_path(self):
        if Path("coba/tests/.temp/from_result.log").exists():
            Path("coba/tests/.temp/from_result.log").unlink()

        try:
            Path("coba/tests/.temp/from_result.log").write_text('''
                ["version",4]
                ["E", 0, {"e":1}]
                ["L", 0, {"l":1}]
                ["V", 0, {"v":1}]
                ["I",[0,0,0],{"_packed":{"actions":[[1,2],[1,2]],"rewards":[[0,1],[1,0]]}}]
                ["I",[0,1,2],{"_packed":{"action":[1,2],"reward":[1,0]} }]
                ["L", 1, {"l":2}]
            ''')

            env = Environments.from_result("coba/tests/.temp/from_result.log")

            self.assertEqual(2, len(env))

            actual = list(env[0].read())
            expected = [
                {'actions':[1,2],"rewards":DiscreteReward([1,2],[0,1])},
                {'actions':[1,2],"rewards":DiscreteReward([1,2],[1,0])}
            ]

            self.assertEqual(env[0].params, {'e':1,'l':1,'v':1})
            self.assertEqual(actual, expected)

            actual = list(env[1].read())
            expected = [{'action':1,"reward":1},{'action':2,"reward":0}]

            self.assertEqual(env[1].params, {'e':1,'l':2})
            self.assertEqual(actual,expected)

        finally:
            if Path("coba/tests/.temp/from_result.log").exists():
                Path("coba/tests/.temp/from_result.log").unlink()

    def test_from_result_obj(self):

        if Path("coba/tests/.temp/from_result.log").exists():
            Path("coba/tests/.temp/from_result.log").unlink()

        try:
            Path("coba/tests/.temp/from_result.log").write_text('''
                ["version",4]
                ["E", 0, {"e":1}]
                ["L", 0, {"l":1}]
                ["V", 0, {"v":1}]
                ["I",[0,0,0],{"_packed":{"actions":[[1,2],[1,2]],"rewards":[[0,1],[1,0]]}}]
                ["I",[0,1,2],{"_packed":{"action":[1,2],"reward":[1,0]} }]
                ["L", 1, {"l":2}]
            ''')

            env = Environments.from_result(Result.from_save("coba/tests/.temp/from_result.log"))

            self.assertEqual(2, len(env))

            actual = list(env[0].read())
            expected = [
                {'actions':[1,2],"rewards":DiscreteReward([1,2],[0,1])},
                {'actions':[1,2],"rewards":DiscreteReward([1,2],[1,0])}
            ]

            self.assertEqual(env[0].params, {'e':1,'l':1,'v':1})
            self.assertEqual(actual, expected)

            actual = list(env[1].read())
            expected = [{'action':1,"reward":1},{'action':2,"reward":0}]

            self.assertEqual(env[1].params, {'e':1,'l':2})
            self.assertEqual(actual,expected)

        finally:
            try:
                if Path("coba/tests/.temp/from_result.log").exists():
                    Path("coba/tests/.temp/from_result.log").unlink()
            except:
                pass


    def test_from_linear_synthetic(self):
        envs = Environments.from_linear_synthetic(100,2,3,4,5,["xa"],5)
        env  = envs[0]

        self.assertIsInstance(env[0], LinearSyntheticSimulation)
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

        self.assertIsInstance(env[0], NeighborsSyntheticSimulation)
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

        self.assertIsInstance(env[0], KernelSyntheticSimulation)
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

        self.assertIsInstance(env[0], MLPSyntheticSimulation)
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
        self.assertEqual(None, env[0].params['label_type'])
        self.assertEqual(200 , env[1].params['openml_task'])
        self.assertEqual(100 , env[1].params['reservoir_count'])
        self.assertEqual(None, env[1].params['label_type'])

    def test_from_openml_label_type(self):
        env = Environments.from_openml(task_id=[100,200],label_type='c')

        self.assertEqual(2   , len(env))
        self.assertEqual(100 , env[0].params['openml_task'])
        self.assertEqual('c' , env[0].params['label_type'])
        self.assertEqual(200 , env[1].params['openml_task'])
        self.assertEqual('c' , env[1].params['label_type'])

    def test_from_feurer(self):
        actual_tasks = set([e.params['openml_task'] for e in Environments.from_feurer()])

        expected_tasks = {232,236,241,245,253,254,256,258,260,262,267,271,273,275,279,288,336,340,
            2119,2120,2121,2122,2123,2125,2356,3044,3047,3048,3049,3053,3054,3055,75089,75092,75093,
            75097,75098,75100,75105,75108,75109,75112,75114,75115,75116,75118,75120,75121,75125,75126,
            75127,75129,75131,75133,75134,75136,75139,75141,75142,75143,75146,75147,75148,75149,75153,
            75154,75156,75157,75159,75161,75163,75166,75169,75171,75173,75174,75176,75178,75179,75180,
            75184,75185,75187,75192,75193,75195,75196,75199,75210,75212,75213,75215,75217,75219,75221,
            75223,75225,75232,75233,75234,75235,75236,75237,75239,75250,126021,126024,126025,126026,
            126028,126029,126030,126031,146574,146575,146576,146577,146578,146583,146586,146592,146593,
            146594,146596,146597,146600,146601,146602,146603,146679,166859,166866,166872,166875,166882,
            166897,166905,166906,166913,166915,166931,166932,166944,166950,166951,166953,166956,166957,
            166958,166959,166970,166996,167083,167085,167086,167087,167088,167089,167090,167094,167096,
            167097,167099,167100,167101,167103,167104,167105,167106,167149,167152,167161,167168,167181,
            167184,167185,167190,167200,167201,167202,167203,167204,167205,168785,168791,168792,168793,
            168794,168795,168796,168797,168798,189779,189786,189828,189829,189836,189840,189841,189843,
            189844,189845,189846,189858,189859,189860,189861,189862,189863,189864,189865,361282,189869,
            189870,189871,189872,189873,189874,189875,189878,189880,189881,189882,189883,189884,189887,
            189890,189893,189894,189899,189900,189902,189905,189906,189908,189909,190154,190155,190156,
            190157,190158,190159,211720,211721,211722,211723,211724}

        self.assertEqual(actual_tasks,expected_tasks)

    def test_from_lambda(self):
        context = lambda index,rng               : [ round(r,2) for r in rng.randoms(5) ]
        actions = lambda index,context,rng       : [rng.randoms(5) for _ in range(3)]
        rewards = lambda index,context,action,rng: sum(c*a for c,a in zip(context,action))

        env = Environments.from_lambda(1, context, actions, rewards, 1)[0]

        self.assertEqual(1, len(list(env.read())))
        self.assertEqual([0.11, 0.8, 0.44, 0.17, 0.42], list(env.read())[0]['context'])
        self.assertEqual(3, len(list(env.read())[0]['actions']))

    @unittest.skipUnless(PackageChecker.pandas(strict=False), "pandas is not installed so we must skip pandas tests")
    def test_from_dataframe(self):
        import pandas as pd

        df = pd.DataFrame({'context':[1,2],'actions':[[0,1]]*2,'rewards':[[2,3]]*2})
        expected = [
            {'context':1,'actions':[0,1],'rewards':DiscreteReward([0,1],[2,3])},
            {'context':2,'actions':[0,1],'rewards':DiscreteReward([0,1],[2,3])}
        ]

        env = Environments.from_dataframe(df)

        self.assertEqual(len(env),1)
        self.assertEqual(expected, list(env[0].read()))

    def test_from_custom(self):
        env = Environments.from_custom(TestEnvironment1('A'), TestEnvironment1('B'))

        self.assertEqual(2  , len(env))
        self.assertEqual('A', env[0].params['id'])
        self.assertEqual('B', env[1].params['id'])

    def test_init_args(self):
        env = Environments(TestEnvironment1('A'), TestEnvironment1('B'))

        self.assertEqual(2  , len(env))
        self.assertEqual('A', env[0].params['id'])
        self.assertEqual('B', env[1].params['id'])

    def test_init_sequence_args(self):
        env = Environments([TestEnvironment1('A'), TestEnvironment1('B')])

        self.assertEqual(2  , len(env))
        self.assertEqual('A', env[0].params['id'])
        self.assertEqual('B', env[1].params['id'])

    def test_init_generator_args(self):
        env = Environments(TestEnvironment1(x) for x in ['A','B'])

        self.assertEqual(2  , len(env))
        self.assertEqual('A', env[0].params['id'])
        self.assertEqual('B', env[1].params['id'])

    def test_init_empty_args(self):
        env = Environments()
        self.assertEqual(0  , len(env))

    def test_iter(self):
        envs = Environments([TestEnvironment1('A'), TestEnvironment1('B')])

        for env,id in zip(envs,['A','B']):
            self.assertEqual(id, env.params['id'])

    def test_add(self):
        envs_1 = Environments(TestEnvironment1('A'))
        envs_2 = Environments(TestEnvironment1('B'))
        envs_3 = envs_1+envs_2

        self.assertEqual(1  , len(envs_1))
        self.assertEqual('A', envs_1[0].params['id'])
        self.assertEqual(1  , len(envs_2))
        self.assertEqual('B', envs_2[0].params['id'])
        self.assertEqual(2  , len(envs_3))
        self.assertEqual('A', envs_3[0].params['id'])
        self.assertEqual('B', envs_3[1].params['id'])

    def test_str(self):
        envs = Environments(TestEnvironment1('A'),TestEnvironment1('B'))
        self.assertEqual(str(envs), f'1. {envs[0]}\n2. {envs[1]}')

    def test_binary(self):
        envs = Environments(TestEnvironment1('A'),TestEnvironment1('B')).binary()

        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(True, envs[0].params['binary'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(True, envs[1].params['binary'])

    def test_cycle(self):
        envs = Environments(TestEnvironment1('A'),TestEnvironment1('B')).cycle(2)

        self.assertEqual(2  , len(envs))
        self.assertEqual('A', envs[0].params['id'])
        self.assertEqual(2  , envs[0].params['cycle_after'])
        self.assertEqual('B', envs[1].params['id'])
        self.assertEqual(2  , envs[1].params['cycle_after'])

    def test_shuffle_default(self):
        envs = Environments(TestEnvironment1('A'),TestEnvironment1('B')).shuffle()

        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(1   , envs[0].params['shuffle_seed'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(1   , envs[1].params['shuffle_seed'])

    def test_shuffle_seed(self):
        envs = Environments(TestEnvironment1('A'),TestEnvironment1('B')).shuffle(seed=1)

        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(1   , envs[0].params['shuffle_seed'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(1   , envs[1].params['shuffle_seed'])

        envs = Environments(TestEnvironment1('A'),TestEnvironment1('B')).shuffle(seed=0)

        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(0   , envs[0].params['shuffle_seed'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(0   , envs[1].params['shuffle_seed'])

    def test_shuffle_int(self):
        envs = Environments(TestEnvironment1('A'),TestEnvironment1('B')).shuffle(1)

        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(1   , envs[0].params['shuffle_seed'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(1   , envs[1].params['shuffle_seed'])

    def test_shuffle_seeds(self):
        envs = Environments(TestEnvironment1('A'),TestEnvironment1('B')).shuffle(seeds=[1,2])

        self.assertEqual(4   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(1   , envs[0].params['shuffle_seed'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(1   , envs[1].params['shuffle_seed'])
        self.assertEqual('A' , envs[2].params['id'])
        self.assertEqual(2   , envs[2].params['shuffle_seed'])
        self.assertEqual('B' , envs[3].params['id'])
        self.assertEqual(2   , envs[3].params['shuffle_seed'])

    def test_shuffle_sequence(self):
        envs = Environments(TestEnvironment1('A'),TestEnvironment1('B')).shuffle([1,2])

        self.assertEqual(4   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(1   , envs[0].params['shuffle_seed'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(1   , envs[1].params['shuffle_seed'])
        self.assertEqual('A' , envs[2].params['id'])
        self.assertEqual(2   , envs[2].params['shuffle_seed'])
        self.assertEqual('B' , envs[3].params['id'])
        self.assertEqual(2   , envs[3].params['shuffle_seed'])

    def test_shuffle_n(self):
        envs = Environments(TestEnvironment1('A'),TestEnvironment1('B')).shuffle(n=2)

        self.assertEqual(4   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(0   , envs[0].params['shuffle_seed'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(0   , envs[1].params['shuffle_seed'])
        self.assertEqual('A' , envs[2].params['id'])
        self.assertEqual(1   , envs[2].params['shuffle_seed'])
        self.assertEqual('B' , envs[3].params['id'])
        self.assertEqual(1   , envs[3].params['shuffle_seed'])

    def test_shuffle_args(self):
        envs = Environments(TestEnvironment1('A'),TestEnvironment1('B')).shuffle(0,1)

        self.assertEqual(4   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(0   , envs[0].params['shuffle_seed'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(0   , envs[1].params['shuffle_seed'])
        self.assertEqual('A' , envs[2].params['id'])
        self.assertEqual(1   , envs[2].params['shuffle_seed'])
        self.assertEqual('B' , envs[3].params['id'])
        self.assertEqual(1   , envs[3].params['shuffle_seed'])

    def test_shuffle_iterable(self):
        envs = Environments(TestEnvironment1('A'),TestEnvironment1('B')).shuffle(range(2))

        self.assertEqual(4   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(0   , envs[0].params['shuffle_seed'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(0   , envs[1].params['shuffle_seed'])
        self.assertEqual('A' , envs[2].params['id'])
        self.assertEqual(1   , envs[2].params['shuffle_seed'])
        self.assertEqual('B' , envs[3].params['id'])
        self.assertEqual(1   , envs[3].params['shuffle_seed'])

    def test_sparse(self):
        envs = Environments(TestEnvironment1('A'),TestEnvironment1('B')).sparse(False,True)

        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(False, envs[0].params['sparse_c'])
        self.assertEqual(True , envs[0].params['sparse_a'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(False, envs[1].params['sparse_c'])
        self.assertEqual(True, envs[1].params['sparse_a'])

    def test_dense(self):
        envs = Environments(TestEnvironment1('A'),TestEnvironment1('B')).dense(10,'hashing',False,True)

        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(10, envs[0].params['dense_n'])
        self.assertEqual('hashing', envs[0].params['dense_m'])
        self.assertEqual(False, envs[0].params['dense_c'])
        self.assertEqual(True , envs[0].params['dense_a'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(10, envs[1].params['dense_n'])
        self.assertEqual('hashing', envs[1].params['dense_m'])
        self.assertEqual(False, envs[1].params['dense_c'])
        self.assertEqual(True, envs[1].params['dense_a'])
        self.assertIsNot(envs[0][-1],envs[1][-1])

    def test_take(self):
        envs = Environments(TestEnvironment1('A'),TestEnvironment1('B')).take(1)

        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(1   , envs[0].params['take'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(1   , envs[1].params['take'])

    def test_take_strict(self):
        envs = Environments(TestEnvironment2(1),TestEnvironment2(2)).take(2,strict=True)

        self.assertEqual(2   , len(envs))
        self.assertEqual(2   , envs[0].params['take'])
        self.assertEqual(2   , envs[1].params['take'])
        self.assertFalse(list(envs[0].read()))
        self.assertTrue(list(envs[1].read()))

    def test_slice(self):
        envs = Environments(TestEnvironment1('A'),TestEnvironment1('B')).slice(1,2)

        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(1   , envs[0].params['slice_start'])
        self.assertEqual(2   , envs[0].params['slice_stop'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(1   , envs[1].params['slice_start'])
        self.assertEqual(2   , envs[1].params['slice_stop'])

    def test_reservoir_seed(self):
        envs = Environments(TestEnvironment1('A'),TestEnvironment1('B')).reservoir(1,2)

        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(1   , envs[0].params['reservoir_count'])
        self.assertEqual(2   , envs[0].params['reservoir_seed'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(1   , envs[1].params['reservoir_count'])
        self.assertEqual(2   , envs[1].params['reservoir_seed'])

    def test_reservoir_seeds(self):
        envs = Environments(TestEnvironment1('A'),TestEnvironment1('B')).reservoir(1,[2])

        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(1   , envs[0].params['reservoir_count'])
        self.assertEqual(2   , envs[0].params['reservoir_seed'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(1   , envs[1].params['reservoir_count'])
        self.assertEqual(2   , envs[1].params['reservoir_seed'])

    def test_scale(self):
        envs = Environments(TestEnvironment1('A')).scale("med", "std", "features", 2)

        self.assertEqual(1  , len(envs))
        self.assertEqual('A'  , envs[0].params['id'])
        self.assertEqual('med', envs[0].params['shift'])
        self.assertEqual('std', envs[0].params['scale'])
        self.assertEqual(2    , envs[0].params['scale_using'])

    def test_impute(self):
        envs = Environments(TestEnvironment1('A')).impute('median', False, 2)

        self.assertEqual(1       , len(envs))
        self.assertEqual('A'     , envs[0].params['id'])
        self.assertEqual('median', envs[0].params['impute_stat'])
        self.assertEqual(2       , envs[0].params['impute_using'])
        self.assertEqual(False   , envs[0].params['impute_indicator'])

    def test_where(self):
        envs = Environments(TestEnvironment1('A'),TestEnvironment1('B')).where(n_interactions = (1,2))

        self.assertEqual(2    , len(envs))
        self.assertEqual('A'  , envs[0].params['id'])
        self.assertEqual((1,2), envs[0].params['where_n_interactions'])
        self.assertEqual('B'  , envs[1].params['id'])
        self.assertEqual((1,2), envs[1].params['where_n_interactions'])

    def test_flatten(self):
        envs = Environments(TestEnvironment1('A'),TestEnvironment1('B')).flatten()

        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(True, envs[0].params['flat'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(True, envs[1].params['flat'])

    def test_noise(self):
        envs = Environments(TestEnvironment1('A')).noise(lambda x,r: x+1, lambda x,r: x+2, lambda x,r: x+3)

        self.assertEqual(1, len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(True, envs[0].params['context_noise'])
        self.assertEqual(True, envs[0].params['action_noise'])
        self.assertEqual(True, envs[0].params['context_noise'])

    def test_riffle(self):
        envs = Environments(TestEnvironment1('A')).riffle(2,1)

        self.assertEqual(1, len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(2, envs[0].params['riffle_spacing'])
        self.assertEqual(1, envs[0].params['riffle_seed'])

    def test_sort(self):
        envs = Environments(TestEnvironment1('A')).sort()

        self.assertEqual(1, len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual('*', envs[0].params['sort_keys'])

    def test_repr(self):
        envs = Environments(TestEnvironment1('A')).repr()

        self.assertEqual(1, len(envs))
        self.assertEqual('onehot' , envs[0].params['categoricals_in_actions'])
        self.assertEqual('onehot', envs[0].params['categoricals_in_context'])

    def test_materialize(self):
        envs  = Environments.from_linear_synthetic(100,2,3,4,5,["xa"],5)
        envs += Environments.from_linear_synthetic(10 ,2,3,4,5,["xa"],6)

        envs = envs.materialize()

        self.assertTrue(envs[0][-1]._protected)
        self.assertTrue(envs[1][-1]._protected)

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

    def test_materialize_lazy(self):
        class TestEnv:
            def read(self):
                yield {"context":LazyDense(lambda: [1,2,3]), "actions":["A"]}
        envs = Environments(TestEnv()).materialize()
        ints = envs[0][-1]._cache
        self.assertTrue(envs[0][-1]._protected)
        self.assertEqual(1,len(ints))
        self.assertIsInstance(ints[0]['context'],list)
        self.assertEqual(ints[0]['actions'][0],"A")

    def test_materialize_cached(self):
        class TestEnv:
            def read(self):
                yield {"context":LazyDense(lambda: [1,2,3]), "actions":[Categorical("A",["A","B"])]}
        envs = Environments(TestEnv()).cache().materialize()
        last_cache  = envs[0][-1]._cache
        self.assertEqual(len([e for e in envs[0] if isinstance(e,Cache)]),1)
        self.assertEqual(1,len(last_cache))
        self.assertIsInstance(last_cache[0]['context'],list)
        self.assertEqual(last_cache[0]['context'],[1,2,3])
        self.assertEqual(last_cache[0]['actions'][0],(1,0))

    def test_materialize_after_protected_cached(self):
        class TestEnv:
            def read(self):
                yield {"context":LazyDense(lambda: [1,2,3]), "actions":[Categorical("A",["A"])]}
        envs = Environments(TestEnv()).materialize().materialize()
        last_cache  = envs[0][-1]._cache
        self.assertEqual(len([e for e in envs[0] if isinstance(e,Cache)]),1)
        self.assertEqual(1,len(last_cache))
        self.assertIsInstance(last_cache[0]['context'],list)
        self.assertEqual(last_cache[0]['actions'][0],(1,))

    def test_grounded(self):
        envs = Environments.from_linear_synthetic(100,2,3,4,5,["xa"],5)
        envs = envs.grounded(10,5,4,2,3)
        env  = envs[0]
        self.assertEqual(10,env.params['n_users'])
        self.assertEqual(5,env.params['n_normal'])
        self.assertEqual(4,env.params['n_words'])
        self.assertEqual(2,env.params['n_good'])
        self.assertEqual(3,env.params['igl_seed'])

    def test_singular_filter(self):
        envs = Environments(TestEnvironment1('A'),TestEnvironment1('B')).filter(Shuffle(1))
        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(1   , envs[0].params['shuffle_seed'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(1   , envs[1].params['shuffle_seed'])

    def test_sequence_filter(self):
        envs = Environments(TestEnvironment1('A'),TestEnvironment1('B')).filter([Shuffle(1),Shuffle(2)])
        self.assertEqual(4   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(1   , envs[0].params['shuffle_seed'])
        self.assertEqual('A' , envs[1].params['id'])
        self.assertEqual(2   , envs[1].params['shuffle_seed'])
        self.assertEqual('B' , envs[2].params['id'])
        self.assertEqual(1   , envs[2].params['shuffle_seed'])
        self.assertEqual('B' , envs[3].params['id'])
        self.assertEqual(2   , envs[3].params['shuffle_seed'])

    def test_two_step_filter(self):
        envs = Environments(TestEnvironment1('A'),TestEnvironment1('B')).filter(Shuffle(1)).filter(Take(1))
        self.assertEqual(2   , len(envs))
        self.assertEqual('A' , envs[0].params['id'])
        self.assertEqual(1   , envs[0].params['shuffle_seed'])
        self.assertEqual(1   , envs[0].params['take'])
        self.assertEqual('B' , envs[1].params['id'])
        self.assertEqual(1   , envs[1].params['shuffle_seed'])
        self.assertEqual(1   , envs[1].params['take'])

    def test_params(self):
        envs = Environments(TestEnvironment1('A'),TestEnvironment1('B')).params({'a':123})
        self.assertEqual(2  , len(envs))
        self.assertEqual('A', envs[0].params['id'])
        self.assertEqual(123, envs[0].params['a'])
        self.assertEqual('B', envs[1].params['id'])
        self.assertEqual(123, envs[1].params['a'])

    def test_batch(self):
        envs = Environments(TestEnvironment1('A'),TestEnvironment1('B')).batch(3)
        self.assertEqual(2     , len(envs))
        self.assertEqual('A'   , envs[0].params['id'])
        self.assertEqual(3     , envs[0].params['batch_size'])
        self.assertEqual('list', envs[0].params['batch_type'])
        self.assertEqual('B'   , envs[1].params['id'])
        self.assertEqual(3     , envs[1].params['batch_size'])
        self.assertEqual('list', envs[1].params['batch_type'])

    def test_logged_one(self):
        class TestEnvironment:
            def read(self):
                yield {'context':None, 'actions':[0,1,2], "rewards":L1Reward(1)}
        envs = Environments(TestEnvironment()).logged(FixedLearner([1,0,0]))
        self.assertEqual(len(envs),1)
        self.assertEqual(next(envs[0].read()),{'context':None, 'action':0, "reward":-1, 'probability':1, 'actions':[0,1,2], "rewards":L1Reward(1)})

    def test_logged_two(self):
        class TestEnvironment:
            def read(self):
                yield {'context':None, 'actions':[0,1,2], "rewards":L1Reward(1)}

        envs = Environments(TestEnvironment()).logged([FixedLearner([1,0,0]),FixedLearner([0,1,0])])

        self.assertEqual(len(envs),2)
        self.assertEqual(next(envs[0].read()),{'context':None, 'action':0, "reward":-1, 'probability':1, 'actions':[0,1,2], "rewards":L1Reward(1)})
        self.assertEqual(next(envs[1].read()),{'context':None, 'action':1, "reward": 0, 'probability':1, 'actions':[0,1,2], "rewards":L1Reward(1)})

    def test_unbatch(self):
        class TestEnvironment:
            def read(self):
                yield {'context':BatchList([1,2]), 'actions':BatchList([[1,2],[3,4]]), 'rewards':BatchList([L1Reward(1),L1Reward(2)]) }

        actual = list(Environments(TestEnvironment()).unbatch()[0].read())
        expected = [
            {'context':1, 'actions':[1,2], "rewards":L1Reward(1)},
            {'context':2, 'actions':[3,4], "rewards":L1Reward(2)}
        ]
        self.assertEqual(actual,expected)

    def test_filter_new(self):
        envs1 = Environments(TestEnvironment1('A'))
        envs2 = envs1.params({'a':123})
        self.assertEqual(1  , len(envs1))
        self.assertEqual(1  , len(envs2))
        self.assertEqual({'id':'A'}, envs1[0].params)
        self.assertEqual({'id':'A','a':123}, envs2[0].params)

    def test_ope_rewards(self):
        envs = Environments(TestEnvironment1('A')).ope_rewards("IPS")
        self.assertEqual(1  , len(envs))
        self.assertEqual('IPS', envs[0].params['ope_reward'])

    def test_getitem(self):
        env  = TestEnvironment1('A')
        envs = Environments(env)
        self.assertIs(envs[0][0],env)
        self.assertIsInstance(envs[:1],Environments)
        self.assertIs(envs[:1][0][0],env)

    def test_ipython_display(self):
        with unittest.mock.patch("builtins.print") as mock:
            envs = Environments(TestEnvironment1('A'),TestEnvironment1('B'))
            envs._ipython_display_()
            mock.assert_called_once_with(f'1. {envs[0]}\n2. {envs[1]}')

if __name__ == '__main__':
    unittest.main()
