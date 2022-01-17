import unittest
import pickle

from coba.random import CobaRandom
from coba.exceptions import CobaException
from coba.contexts import CobaContext, NullLogger

from coba.environments import LambdaSimulation, LinearSyntheticSimulation, LocalSyntheticSimulation

CobaContext.logger = NullLogger()

class LambdaSimulation_Tests(unittest.TestCase):

    def test_n_interactions_2_seed_none(self):
        
        def C(i:int):
            return [1,2][i]

        def A(i:int,c:int):
            return [[1,2,3],[4,5,6]][i]

        def R(i:int,c:int,a:int):
            return a-c

        simulation = LambdaSimulation(2,C,A,R)
        interactions = list(simulation.read())

        self.assertEqual(len(interactions), 2)

        self.assertEqual(1      , interactions[0].context)
        self.assertEqual([1,2,3], interactions[0].actions)
        self.assertEqual([0,1,2], interactions[0].kwargs["rewards"])

        self.assertEqual(2      , interactions[1].context)
        self.assertEqual([4,5,6], interactions[1].actions)
        self.assertEqual([2,3,4], interactions[1].kwargs["rewards"])

    def test_n_interactions_none_seed_none(self):
        def C(i:int):
            return [1,2][i]

        def A(i:int,c:int):
            return [[1,2,3],[4,5,6]][i]

        def R(i:int,c:int,a:int):
            return a-c

        simulation   = LambdaSimulation(None,C,A,R)
        interactions = iter(simulation.read())

        interaction = next(interactions)

        self.assertEqual(1      , interaction.context)
        self.assertEqual([1,2,3], interaction.actions)
        self.assertEqual([0,1,2], interaction.kwargs["rewards"])

        interaction = next(interactions)

        self.assertEqual(2      , interaction.context)
        self.assertEqual([4,5,6], interaction.actions)
        self.assertEqual([2,3,4], interaction.kwargs["rewards"])

    def test_n_interactions_2_seed_1(self):
        
        def C(i:int, rng: CobaRandom):
            return [1,2][i]

        def A(i:int,c:int, rng: CobaRandom):
            return [[1,2,3],[4,5,6]][i]

        def R(i:int,c:int,a:int, rng: CobaRandom):
            return a-c

        simulation = LambdaSimulation(2,C,A,R,seed=1)
        interactions = list(simulation.read())

        self.assertEqual(len(interactions), 2)

        self.assertEqual(1      , interactions[0].context)
        self.assertEqual([1,2,3], interactions[0].actions)
        self.assertEqual([0,1,2], interactions[0].kwargs["rewards"])

        self.assertEqual(2      , interactions[1].context)
        self.assertEqual([4,5,6], interactions[1].actions)
        self.assertEqual([2,3,4], interactions[1].kwargs["rewards"])

    def test_params(self):
        def C(i:int):
            return [1,2][i]

        def A(i:int,c:int):
            return [[1,2,3],[4,5,6]][i]

        def R(i:int,c:int,a:int):
            return a-c

        self.assertEqual({}, LambdaSimulation(2,C,A,R).params)

    def test_pickle_n_interactions_2(self):
        def C(i:int):
            return [1,2][i]

        def A(i:int,c:int):
            return [[1,2,3],[4,5,6]][i]
        
        def R(i:int,c:int,a:int):
            return a-c

        simulation = pickle.loads(pickle.dumps(LambdaSimulation(2,C,A,R)))
        interactions = list(simulation.read())

        self.assertEqual("LambdaSimulation",str(simulation))
        self.assertEqual({}, simulation.params)

        self.assertEqual(len(interactions), 2)

        self.assertEqual(1      , interactions[0].context)
        self.assertEqual([1,2,3], interactions[0].actions)
        self.assertEqual([0,1,2], interactions[0].kwargs["rewards"])

        self.assertEqual(2      , interactions[1].context)
        self.assertEqual([4,5,6], interactions[1].actions)
        self.assertEqual([2,3,4], interactions[1].kwargs["rewards"])

    def test_pickle_n_interactions_none(self):
        def C(i:int):
            return [1,2][i]

        def A(i:int,c:int):
            return [[1,2,3],[4,5,6]][i]
        
        def R(i:int,c:int,a:int):
            return a-c

        with self.assertRaises(CobaException) as e:
            pickle.loads(pickle.dumps(LambdaSimulation(None,C,A,R)))
        
        self.assertIn("pickle", str(e.exception))

class LinearSyntheticSimulation_Tests(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(500, len(list(LinearSyntheticSimulation().read())))

    def test_params(self):
        env = LinearSyntheticSimulation(100,2,3,4,0,["xa"],2)

        self.assertEqual(2     , env.params['n_A'])
        self.assertEqual(3     , env.params['n_C_phi'])
        self.assertEqual(4     , env.params['n_A_phi'])
        self.assertEqual(0     , env.params['r_noise'])
        self.assertEqual(['xa'], env.params['X'])
        self.assertEqual(2     , env.params['seed'])

    def test_str(self):
        self.assertEqual("LinearSynth(A=2,c=3,a=4,X=['xa'],seed=2)", str(LinearSyntheticSimulation(100,2,3,4,0,["xa"],2)))

    def test_pickle(self):
        env = pickle.loads(pickle.dumps(LinearSyntheticSimulation(2000,2,3,4,0,["xa"],2)))

        self.assertEqual(2     , env.params['n_A'])
        self.assertEqual(3     , env.params['n_C_phi'])
        self.assertEqual(4     , env.params['n_A_phi'])
        self.assertEqual(0     , env.params['r_noise'])
        self.assertEqual(['xa'], env.params['X'])
        self.assertEqual(2     , env.params['seed'])
        self.assertEqual("LinearSynth(A=2,c=3,a=4,X=['xa'],seed=2)", str(env))
        self.assertEqual(2000, len(list(env.read())))

class LocalSyntheticSimulation_Tests(unittest.TestCase):

    def test_simple(self):
        self.assertEqual(500, len(list(LocalSyntheticSimulation().read())))

    def test_params(self):
        env = LocalSyntheticSimulation(100,100,3,4,2)

        self.assertEqual(4  , env.params['n_A'])
        self.assertEqual(100, env.params['n_C'])
        self.assertEqual(3  , env.params['n_C_phi'])
        self.assertEqual(2  , env.params['seed'])

    def test_str(self):
        self.assertEqual("LocalSynth(A=4,C=100,c=3,seed=2)", str(LocalSyntheticSimulation(200,100,3,4,2)))

    def test_pickle(self):
        env = pickle.loads(pickle.dumps(LocalSyntheticSimulation(2000,2,3,4,5)))
        
        self.assertEqual(2     , env.params['n_C'])
        self.assertEqual(3     , env.params['n_C_phi'])
        self.assertEqual(4     , env.params['n_A'])
        self.assertEqual(5     , env.params['seed'])
        self.assertEqual("LocalSynth(A=4,C=2,c=3,seed=5)", str(env))
        self.assertEqual(2000, len(list(env.read())))

if __name__ == '__main__':
    unittest.main()
