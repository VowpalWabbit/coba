import unittest
import pickle

from coba.random import CobaRandom
from coba.exceptions import CobaException
from coba.context import CobaContext, NullLogger

from coba.environments import LambdaSimulation, LinearSyntheticSimulation, NeighborsSyntheticSimulation
from coba.environments import KernelSyntheticSimulation, MLPSyntheticSimulation

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

        self.assertEqual(1      , interactions[0]['context'])
        self.assertEqual([1,2,3], interactions[0]['actions'])
        self.assertEqual([0,1,2], interactions[0]['rewards'])

        self.assertEqual(2      , interactions[1]['context'])
        self.assertEqual([4,5,6], interactions[1]['actions'])
        self.assertEqual([2,3,4], interactions[1]['rewards'])

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

        self.assertEqual(1      , interaction['context'])
        self.assertEqual([1,2,3], interaction['actions'])
        self.assertEqual([0,1,2], interaction['rewards'])

        interaction = next(interactions)

        self.assertEqual(2      , interaction['context'])
        self.assertEqual([4,5,6], interaction['actions'])
        self.assertEqual([2,3,4], interaction['rewards'])

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

        self.assertEqual(1      , interactions[0]['context'])
        self.assertEqual([1,2,3], interactions[0]['actions'])
        self.assertEqual([0,1,2], interactions[0]['rewards'])

        self.assertEqual(2      , interactions[1]['context'])
        self.assertEqual([4,5,6], interactions[1]['actions'])
        self.assertEqual([2,3,4], interactions[1]['rewards'])

    def test_params(self):
        def C(i:int):
            return [1,2][i]

        def A(i:int,c:int):
            return [[1,2,3],[4,5,6]][i]

        def R(i:int,c:int,a:int):
            return a-c

        self.assertEqual({"env_type": "LambdaSimulation"}, LambdaSimulation(2,C,A,R).params)

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
        self.assertEqual({"env_type": "LambdaSimulation"}, simulation.params)

        self.assertEqual(len(interactions), 2)

        self.assertEqual(1      , interactions[0]['context'])
        self.assertEqual([1,2,3], interactions[0]['actions'])
        self.assertEqual([0,1,2], interactions[0]['rewards'])

        self.assertEqual(2      , interactions[1]['context'])
        self.assertEqual([4,5,6], interactions[1]['actions'])
        self.assertEqual([2,3,4], interactions[1]['rewards'])

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

    def test_bad_features(self):
        sim = LinearSyntheticSimulation(500,n_actions=2,n_context_features=1,n_action_features=0,reward_features="x")
        self.assertEqual(sim._reward_features,['x'])

        with self.assertRaises(CobaException):
            LinearSyntheticSimulation(500,n_actions=2,n_context_features=1,n_action_features=0,reward_features="a")

        with self.assertRaises(CobaException):
            LinearSyntheticSimulation(500,n_actions=2,n_context_features=0,n_action_features=1,reward_features="x")

    def test_single_feature(self):
        simulation = LinearSyntheticSimulation(500,n_actions=2,n_context_features=1,n_action_features=0,reward_features=["x"])
        interactions = list(simulation.read())

        self.assertEqual(500, len(interactions))
        self.assertEqual(2, len(interactions[0]['actions']))
        self.assertEqual(1, len(interactions[0]['context']))
        self.assertEqual(2, len(interactions[0]['actions'][0]))

        rewards = interactions[0]['rewards']
        self.assertAlmostEqual(rewards[0],0.4545,places=3)
        self.assertAlmostEqual(rewards[1],0.5144,places=3)

    def test_simple_context_action_features(self):
        simulation = LinearSyntheticSimulation(500,n_actions=2,n_context_features=3,n_action_features=4,reward_features=["a","xa"])
        interactions = list(simulation.read())

        self.assertEqual(500, len(interactions))
        self.assertEqual(2, len(interactions[0]['actions']))
        self.assertEqual(3, len(interactions[0]['context']))
        self.assertEqual(4, len(interactions[0]['actions'][0]))

        rewards = [ r for i in interactions for r in i['rewards'] ]
        self.assertLess(max(rewards),1.22)
        self.assertGreater(max(rewards),.75)
        self.assertLess(min(rewards),.25)
        self.assertGreater(min(rewards),-.25)
        self.assertGreater(.05, abs(.5-sum(rewards)/len(rewards)))

    def test_simple_context_no_action_features(self):
        simulation = LinearSyntheticSimulation(500,n_actions=2,n_context_features=3,n_action_features=0,reward_features=["a","xa"])
        interactions = list(simulation.read())

        self.assertEqual(500, len(interactions))
        self.assertEqual(2, len(interactions[0]['actions']))
        self.assertEqual(3, len(interactions[0]['context']))
        self.assertEqual(2, len(interactions[0]['actions'][0]))
        self.assertEqual((1,0), interactions[0]['actions'][0])
        self.assertEqual((0,1), interactions[0]['actions'][1])

        rewards = [ r for i in interactions for r in i['rewards'] ]
        self.assertLess(max(rewards),1.2)
        self.assertGreater(max(rewards),.75)
        self.assertLess(min(rewards),.25)
        self.assertGreater(min(rewards),-.2)
        self.assertGreater(.05, abs(.5-sum(rewards)/len(rewards)))

    def test_simple_no_context_action_features(self):
        simulation = LinearSyntheticSimulation(500,n_actions=2,n_context_features=0,n_action_features=4,reward_features=["a","xa"])
        interactions = list(simulation.read())

        self.assertEqual(500, len(interactions))
        self.assertEqual(2, len(interactions[0]['actions']))
        self.assertEqual(None, interactions[0]['context'])
        self.assertEqual(4, len(interactions[0]['actions'][0]))

        rewards = [ r for i in interactions for r in i['rewards'] ]
        self.assertLess(max(rewards),1.2)
        self.assertGreater(max(rewards),.75)
        self.assertLess(min(rewards),.25)
        self.assertGreater(min(rewards),-.2)
        self.assertGreater(.05, abs(.5-sum(rewards)/len(rewards)))

    def test_simple_no_context_and_no_action_features(self):
        simulation = LinearSyntheticSimulation(500, n_actions=1000, n_context_features=0, n_action_features=0, reward_features=["a","xa"])
        interactions = list(simulation.read())

        self.assertEqual(500, len(interactions))
        self.assertEqual(None, interactions[0]['context'])
        self.assertEqual(1000, len(interactions[0]['actions']))

        rewards = [ r for i in interactions for r in i['rewards'] ]
        self.assertLess(max(rewards),1.2)
        self.assertGreater(max(rewards),.75)
        self.assertLess(min(rewards),.25)
        self.assertGreater(min(rewards),-.2)
        self.assertGreater(.05, abs(.5-sum(rewards)/len(rewards)))
        self.assertEqual(interactions[0]['rewards'], interactions[1]['rewards'])

    def test_params(self):
        env = LinearSyntheticSimulation(100,reward_features=["xa"],seed=2)
        self.assertEqual(['xa'], env.params['reward_features'])
        self.assertEqual(2     , env.params['seed'])

    def test_str(self):
        self.assertEqual("LinearSynth(A=2,c=3,a=4,R=['xa'],seed=2)", str(LinearSyntheticSimulation(100,2,3,4,5,["xa"],2)))

    def test_pickle(self):
        simulation = LinearSyntheticSimulation(500,n_actions=2,n_context_features=3,n_action_features=4,reward_features=["a","xa"], seed=2)
        simulation = pickle.loads(pickle.dumps(simulation))

        interactions = list(simulation.read())

        self.assertEqual(500, len(interactions))
        self.assertEqual(2, len(interactions[0]['actions']))
        self.assertEqual(3, len(interactions[0]['context']))
        self.assertEqual(4, len(interactions[0]['actions'][0]))

        self.assertEqual(['a', 'xa'], simulation.params['reward_features'])
        self.assertEqual(2          , simulation.params['seed'])

        self.assertEqual("LinearSynth(A=2,c=3,a=4,R=['a', 'xa'],seed=2)", str(simulation))

class NeighborsSyntheticSimulation_Tests(unittest.TestCase):

    def test_simple_context_action_features(self):

        simulation = NeighborsSyntheticSimulation(20,n_actions=2,n_context_features=3,n_action_features=4,n_neighborhoods=10)
        interactions = list(simulation.read())

        self.assertEqual(20, len(interactions))
        self.assertEqual(2, len(interactions[0]['actions']))
        self.assertEqual(3, len(interactions[0]['context']))
        self.assertEqual(4, len(interactions[0]['actions'][0]))
        self.assertEqual(10,len(set([i['context'] for i in interactions])))

    def test_simple_context_no_action_features(self):

        simulation = NeighborsSyntheticSimulation(20,n_actions=2,n_context_features=3,n_action_features=0,n_neighborhoods=10)
        interactions = list(simulation.read())

        self.assertEqual(20, len(interactions))
        self.assertEqual(2, len(interactions[0]['actions']))
        self.assertEqual(3, len(interactions[0]['context']))
        self.assertEqual(2, len(interactions[0]['actions'][0]))
        self.assertEqual((1,0), interactions[0]['actions'][0])
        self.assertEqual((0,1), interactions[0]['actions'][1])
        self.assertEqual(10,len(set([i['context'] for i in interactions])))

    def test_simple_no_context_action_features(self):

        simulation = NeighborsSyntheticSimulation(20,n_actions=2,n_context_features=0,n_action_features=4,n_neighborhoods=10)
        interactions = list(simulation.read())

        self.assertEqual(20, len(interactions))
        self.assertEqual(2, len(interactions[0]['actions']))
        self.assertEqual(None,  interactions[0]['context'])
        self.assertEqual(4, len(interactions[0]['actions'][0]))
        self.assertEqual(1,len(set([i['context'] for i in interactions])))

    def test_simple_no_context_no_action_features(self):

        simulation = NeighborsSyntheticSimulation(20,n_actions=2,n_context_features=0,n_action_features=0,n_neighborhoods=10)
        interactions = list(simulation.read())

        self.assertEqual(20, len(interactions))
        self.assertEqual(2, len(interactions[0]['actions']))
        self.assertEqual(None,  interactions[0]['context'])
        self.assertEqual(2, len(interactions[0]['actions'][0]))
        self.assertEqual(1,len(set([i['context'] for i in interactions])))

    def test_params(self):
        env = NeighborsSyntheticSimulation(20,n_neighborhoods=10,seed=2)

        self.assertEqual(10, env.params['n_neighborhoods'])
        self.assertEqual(2 , env.params['seed'])

    def test_str(self):
        self.assertEqual("NeighborsSynth(A=2,c=3,a=4,N=5,seed=6)", str(NeighborsSyntheticSimulation(200,2,3,4,5,6)))

    def test_bad_args(self):
        with self.assertRaises(CobaException):
            LinearSyntheticSimulation(200,0,3,4,seed=6)

    def test_pickle(self):

        simulation = NeighborsSyntheticSimulation(20,n_actions=2,n_context_features=3,n_action_features=0,n_neighborhoods=10)
        simulation = pickle.loads(pickle.dumps(simulation))
        interactions = list(simulation.read())

        self.assertEqual(20, len(interactions))
        self.assertEqual(2, len(interactions[0]['actions']))
        self.assertEqual(3, len(interactions[0]['context']))
        self.assertEqual(2, len(interactions[0]['actions'][0]))
        self.assertEqual((1,0), interactions[0]['actions'][0])
        self.assertEqual((0,1), interactions[0]['actions'][1])
        self.assertEqual(10,len(set([i['context'] for i in interactions])))

class KernelSyntheticSimulation_Tests(unittest.TestCase):

    def test_single_linear_feature(self):

        simulation = KernelSyntheticSimulation(500,n_actions=2,n_context_features=1,n_action_features=0,kernel="linear")
        interactions = list(simulation.read())

        self.assertEqual(500, len(interactions))
        self.assertEqual(2, len(interactions[0]['actions']))
        self.assertEqual(1, len(interactions[0]['context']))
        self.assertEqual(2, len(interactions[0]['actions'][0]))

        rewards = interactions[0]['rewards']
        self.assertAlmostEqual(rewards[0],0.6582,places=3)
        self.assertAlmostEqual(rewards[1],0.8972,places=3)

    def test_single_polynomial_degree1_feature(self):

        simulation = KernelSyntheticSimulation(500,n_actions=2,n_context_features=1,n_action_features=0,kernel="polynomial",n_exemplars=1,degree=1)
        interactions = list(simulation.read())

        self.assertEqual(500, len(interactions))
        self.assertEqual(2, len(interactions[0]['actions']))
        self.assertEqual(1, len(interactions[0]['context']))
        self.assertEqual(2, len(interactions[0]['actions'][0]))

        rewards = interactions[0]['rewards']
        self.assertAlmostEqual(rewards[0],0.7827,places=3)
        self.assertAlmostEqual(rewards[1],0.1050,places=3)

    def test_simple_context_action_features(self):

        for kernel in ['linear','polynomial','exponential','gaussian']:
            simulation = KernelSyntheticSimulation(500,n_actions=2,n_context_features=3,n_action_features=4,n_exemplars=10,kernel=kernel,seed=2)
            interactions = list(simulation.read())

            self.assertEqual(500, len(interactions))
            self.assertEqual(2, len(interactions[0]['actions']))
            self.assertEqual(3, len(interactions[0]['context']))
            self.assertEqual(4, len(interactions[0]['actions'][0]))

            rewards = [ r for i in interactions for r in i['rewards'] ]
            self.assertLess(max(rewards),1.25)
            self.assertGreater(max(rewards),.75)
            self.assertLess(min(rewards),.25)
            self.assertGreater(min(rewards),-.2)
            self.assertGreater(.05, abs(.5-sum(rewards)/len(rewards)))

    def test_simple_no_action_features(self):

        for kernel in ['linear','polynomial','exponential','gaussian']:
            simulation = KernelSyntheticSimulation(500,n_actions=2,n_context_features=3,n_action_features=0,n_exemplars=10,kernel=kernel,seed=2)
            interactions = list(simulation.read())

            self.assertEqual(500, len(interactions))
            self.assertEqual(2, len(interactions[0]['actions']))
            self.assertEqual(3, len(interactions[0]['context']))
            self.assertEqual(2, len(interactions[0]['actions'][0]))
            self.assertEqual((1,0), interactions[0]['actions'][0])
            self.assertEqual((0,1), interactions[0]['actions'][1])

            rewards = [ r for i in interactions for r in i['rewards'] ]
            self.assertLess(max(rewards),1.25)
            self.assertGreater(max(rewards),.75)
            self.assertLess(min(rewards),.25)
            self.assertGreater(min(rewards),-.4)
            self.assertGreater(.05, abs(.5-sum(rewards)/len(rewards)))

    def test_simple_no_context_action_features(self):

        for kernel in ['linear','polynomial','exponential','gaussian']:
            simulation = KernelSyntheticSimulation(500,n_actions=2,n_context_features=0,n_action_features=4,n_exemplars=10,kernel=kernel,seed=3)
            interactions = list(simulation.read())

            self.assertEqual(500, len(interactions))
            self.assertEqual(2, len(interactions[0]['actions']))
            self.assertEqual(None, interactions[0]['context'])
            self.assertEqual(4, len(interactions[0]['actions'][0]))

            rewards = [ r for i in interactions for r in i['rewards'] ]
            self.assertLess(max(rewards),1.75)
            self.assertGreater(max(rewards),.75)
            self.assertLess(min(rewards),.25)
            self.assertGreater(min(rewards),-.2)
            self.assertGreater(.05, abs(.5-sum(rewards)/len(rewards)))

    def test_simple_no_context_no_action_features(self):

        for kernel in ['linear','polynomial','exponential','gaussian']:

            simulation = KernelSyntheticSimulation(500,n_actions=2,n_context_features=0,n_action_features=0,n_exemplars=10,kernel=kernel)
            interactions = list(simulation.read())

            self.assertEqual(500, len(interactions))
            self.assertEqual(2, len(interactions[0]['actions']))
            self.assertEqual(None, interactions[0]['context'])
            self.assertEqual(2, len(interactions[0]['actions'][0]))

            rewards = [ r for i in interactions for r in i['rewards'] ]
            self.assertLess(max(rewards),1.2)
            self.assertGreater(max(rewards),.75)
            self.assertLess(min(rewards),.25)
            self.assertGreater(min(rewards),-.2)
            self.assertGreater(.05, abs(.5-sum(rewards)/len(rewards)))

    def test_exponential_params(self):

        env = KernelSyntheticSimulation(20,n_exemplars=10,kernel='exponential',seed=2)

        expected = {
            'env_type'   : "KernelSynthetic",
            'seed'       : 2,
            'n_exemplars': 10,
            'kernel'     : 'exponential',
            'gamma'      : 1
        }

        self.assertDictEqual(expected, env.params)


    def test_gaussian_params(self):

        env = KernelSyntheticSimulation(20,n_exemplars=10,kernel='gaussian',seed=2)

        expected = {
            'env_type'   : "KernelSynthetic",
            'seed'       : 2,
            'n_exemplars': 10,
            'kernel'     : 'gaussian',
            'gamma'      : 1
        }

        self.assertDictEqual(expected, env.params)

    def test_polynomial_params(self):

        env = KernelSyntheticSimulation(20,n_exemplars=10,seed=2,kernel='polynomial',degree=3)

        expected = {
            'env_type'   : "KernelSynthetic",
            'seed'       : 2,
            'n_exemplars': 10,
            'kernel'     : 'polynomial',
            'degree'     : 3
        }

        self.assertDictEqual(expected, env.params)

    def test_linear_params(self):

        env = KernelSyntheticSimulation(20,n_exemplars=10,seed=2,kernel='linear')

        expected = {
            'env_type'   : "KernelSynthetic",
            'seed'       : 2,
            'n_exemplars': 10,
            'kernel'     : 'linear',
        }

        self.assertDictEqual(expected, env.params)

    def test_bad_args(self):
        with self.assertRaises(CobaException):
            KernelSyntheticSimulation(200,0,3,4,5,kernel='exponential',seed=6)
        with self.assertRaises(CobaException):
            KernelSyntheticSimulation(200,2,3,4,0,kernel='exponential',seed=6)

    def test_str(self):
        self.assertEqual("KernelSynth(A=2,c=3,a=4,E=5,K=exponential,seed=6)", str(KernelSyntheticSimulation(200,2,3,4,5,kernel='exponential',seed=6)))

    def test_pickle(self):
        simulation = KernelSyntheticSimulation(20,n_actions=2,n_context_features=3,n_action_features=0,n_exemplars=10)
        simulation = pickle.loads(pickle.dumps(simulation))
        interactions = list(simulation.read())

        self.assertEqual(20, len(interactions))
        self.assertEqual(2, len(interactions[0]['actions']))
        self.assertEqual(3, len(interactions[0]['context']))
        self.assertEqual(2, len(interactions[0]['actions'][0]))
        self.assertEqual((1,0), interactions[0]['actions'][0])
        self.assertEqual((0,1), interactions[0]['actions'][1])

class MLPSyntheticSimulation_Tests(unittest.TestCase):

    def test_simple_context_action_features(self):

        simulation = MLPSyntheticSimulation(500,n_actions=2,n_context_features=3,n_action_features=4)
        interactions = list(simulation.read())

        self.assertEqual(500, len(interactions))
        self.assertEqual(2, len(interactions[0]['actions']))
        self.assertEqual(3, len(interactions[0]['context']))
        self.assertEqual(4, len(interactions[0]['actions'][0]))

        rewards = [ r for i in interactions for r in i['rewards'] ]
        self.assertLess(max(rewards),1.2)
        self.assertGreater(max(rewards),.75)
        self.assertLess(min(rewards),.25)
        self.assertGreater(min(rewards),-.2)
        self.assertGreater(.05, abs(.5-sum(rewards)/len(rewards)))

    def test_simple_context_no_action_features(self):
        simulation = MLPSyntheticSimulation(500,n_actions=2,n_context_features=3,n_action_features=0,seed=2)
        interactions = list(simulation.read())

        self.assertEqual(500, len(interactions))
        self.assertEqual(2, len(interactions[0]['actions']))
        self.assertEqual(3, len(interactions[0]['context']))
        self.assertEqual(2, len(interactions[0]['actions'][0]))
        self.assertEqual((1,0), interactions[0]['actions'][0])
        self.assertEqual((0,1), interactions[0]['actions'][1])

        rewards = [ r for i in interactions for r in i['rewards'] ]
        self.assertLess(max(rewards),1.2)
        self.assertGreater(max(rewards),.75)
        self.assertLess(min(rewards),.25)
        self.assertGreater(min(rewards),-.2)
        self.assertGreater(.05, abs(.5-sum(rewards)/len(rewards)))

    def test_simple_no_context_action_features(self):

        simulation = MLPSyntheticSimulation(500,n_actions=2,n_context_features=0,n_action_features=4)
        interactions = list(simulation.read())

        self.assertEqual(500 , len(interactions))
        self.assertEqual(2   , len(interactions[0]['actions']))
        self.assertEqual(None, interactions[0]['context'])
        self.assertEqual(4   , len(interactions[0]['actions'][0]))

        rewards = [ r for i in interactions for r in i['rewards'] ]
        self.assertLess(max(rewards),1.2)
        self.assertGreater(max(rewards),.75)
        self.assertLess(min(rewards),.25)
        self.assertGreater(min(rewards),-.2)
        self.assertGreater(.05, abs(.5-sum(rewards)/len(rewards)))

    def test_simple_no_context_no_action_features(self):

        simulation = MLPSyntheticSimulation(500,n_actions=100,n_context_features=0,n_action_features=0)
        interactions = list(simulation.read())

        self.assertEqual(500 , len(interactions))
        self.assertEqual(100 , len(interactions[0]['actions']))
        self.assertEqual(None, interactions[0]['context'])
        self.assertEqual(100 , len(interactions[0]['actions'][0]))

        rewards = [ r for i in interactions for r in i['rewards'] ]
        self.assertLess(max(rewards),1.2)
        self.assertGreater(max(rewards),.75)
        self.assertLess(min(rewards),.25)
        self.assertGreater(min(rewards),-.2)
        self.assertGreater(.05, abs(.5-sum(rewards)/len(rewards)))

    def test_params(self):
        env = MLPSyntheticSimulation(20,seed=2)
        self.assertEqual(2 , env.params['seed'])

    def test_str(self):
        self.assertEqual("MLPSynth(A=2,c=3,a=4,seed=6)", str(MLPSyntheticSimulation(200,2,3,4,6)))

    def test_bad_args(self):
        with self.assertRaises(CobaException):
            MLPSyntheticSimulation(200,0,3,4,seed=6)

    def test_pickle(self):

        simulation = MLPSyntheticSimulation(20,n_actions=2,n_context_features=3,n_action_features=0)
        simulation = pickle.loads(pickle.dumps(simulation))
        interactions = list(simulation.read())

        self.assertEqual(20, len(interactions))
        self.assertEqual(2, len(interactions[0]['actions']))
        self.assertEqual(3, len(interactions[0]['context']))
        self.assertEqual(2, len(interactions[0]['actions'][0]))
        self.assertEqual((1,0), interactions[0]['actions'][0])
        self.assertEqual((0,1), interactions[0]['actions'][1])

if __name__ == '__main__':
    unittest.main()
