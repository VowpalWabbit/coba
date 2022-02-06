import unittest
import pickle

from coba.random import CobaRandom
from coba.exceptions import CobaException
from coba.contexts import CobaContext, NullLogger

from coba.environments import LambdaSimulation, LinearSyntheticSimulation, NeighborsSyntheticSimulation

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
    def test_simple_action_features(self):
        
        simulation = LinearSyntheticSimulation(500,n_actions=2,n_context_features=3,n_action_features=4,reward_features=["a","xa"])
        interactions = list(simulation.read())

        self.assertEqual(500, len(interactions))
        self.assertEqual(2, len(interactions[0].actions))
        self.assertEqual(3, len(interactions[0].context))
        self.assertEqual(4, len(interactions[0].actions[0]))
        self.assertEqual(1, len(simulation._action_weights))
        self.assertEqual(16, len(simulation._action_weights[0]))

    def test_simple_no_action_features(self):
        
        simulation = LinearSyntheticSimulation(500,n_actions=2,n_context_features=3,n_action_features=0,reward_features=["a","xa"])
        interactions = list(simulation.read())

        self.assertEqual(500, len(interactions))
        self.assertEqual(2, len(interactions[0].actions))
        self.assertEqual(3, len(interactions[0].context))
        self.assertEqual(2, len(interactions[0].actions[0]))
        self.assertEqual((1,0), interactions[0].actions[0])
        self.assertEqual((0,1), interactions[0].actions[1])
        self.assertEqual(2, len(simulation._action_weights))
        self.assertEqual(4, len(simulation._action_weights[0]))
        self.assertEqual(4, len(simulation._action_weights[1]))

    def test_simple_no_context_and_no_action_features(self):
        
        simulation = LinearSyntheticSimulation(500,n_actions=2,n_context_features=0,n_action_features=0,reward_features=["a","xa"])
        interactions = list(simulation.read())

        self.assertEqual(500, len(interactions))
        self.assertEqual(None, interactions[0].context)
        self.assertEqual(2, len(interactions[0].actions))        
        self.assertEqual((1,0), interactions[0].actions[0])
        self.assertEqual((0,1), interactions[0].actions[1])
        self.assertEqual(2, len(simulation._action_weights))
        self.assertEqual(1, len(simulation._action_weights[0]))
        self.assertEqual(1, len(simulation._action_weights[1]))



    def test_params(self):
        env = LinearSyntheticSimulation(100,reward_features=["xa"],seed=2)
        self.assertEqual(['xa'], env.params['reward_features'])
        self.assertEqual(2     , env.params['seed'])

    def test_str(self):
        self.assertEqual("LinearSynth(A=2,c=3,a=4,R=['xa'],seed=2)", str(LinearSyntheticSimulation(100,2,3,4,["xa"],2)))

    def test_pickle(self):
        simulation = LinearSyntheticSimulation(500,n_actions=2,n_context_features=3,n_action_features=4,reward_features=["a","xa"], seed=2)
        simulation = pickle.loads(pickle.dumps(simulation))
        
        interactions = list(simulation.read())

        self.assertEqual(500, len(interactions))
        self.assertEqual(2, len(interactions[0].actions))
        self.assertEqual(3, len(interactions[0].context))
        self.assertEqual(4, len(interactions[0].actions[0]))

        self.assertEqual(['a', 'xa'], simulation.params['reward_features'])
        self.assertEqual(2          , simulation.params['seed'])

        self.assertEqual("LinearSynth(A=2,c=3,a=4,R=['a', 'xa'],seed=2)", str(simulation))

class NeighborsSyntheticSimulation_Tests(unittest.TestCase):

    def test_simple_action_features(self):

        simulation = NeighborsSyntheticSimulation(20,n_actions=2,n_context_features=3,n_action_features=4,n_neighborhoods=10)
        interactions = list(simulation.read())

        self.assertEqual(20, len(interactions))
        self.assertEqual(2, len(interactions[0].actions))
        self.assertEqual(3, len(interactions[0].context))
        self.assertEqual(4, len(interactions[0].actions[0]))
        self.assertEqual(10,len(set([i.context for i in interactions])))

    def test_simple_no_action_features(self):

        simulation = NeighborsSyntheticSimulation(20,n_actions=2,n_context_features=3,n_action_features=0,n_neighborhoods=10)
        interactions = list(simulation.read())

        self.assertEqual(20, len(interactions))
        self.assertEqual(2, len(interactions[0].actions))
        self.assertEqual(3, len(interactions[0].context))
        self.assertEqual(2, len(interactions[0].actions[0]))
        self.assertEqual((1,0), interactions[0].actions[0])
        self.assertEqual((0,1), interactions[0].actions[1])
        self.assertEqual(10,len(set([i.context for i in interactions])))

    def test_params(self):
        env = NeighborsSyntheticSimulation(20,n_neighborhoods=10,seed=2)

        self.assertEqual(10, env.params['n_neighborhoods'])
        self.assertEqual(2 , env.params['seed'])

    def test_str(self):
        self.assertEqual("NeighborsSynth(A=2,c=3,a=4,N=5,seed=6)", str(NeighborsSyntheticSimulation(200,2,3,4,5,6)))

    def test_pickle(self):
        
        simulation = NeighborsSyntheticSimulation(20,n_actions=2,n_context_features=3,n_action_features=0,n_neighborhoods=10)
        simulation = pickle.loads(pickle.dumps(simulation))
        interactions = list(simulation.read())

        self.assertEqual(20, len(interactions))
        self.assertEqual(2, len(interactions[0].actions))
        self.assertEqual(3, len(interactions[0].context))
        self.assertEqual(2, len(interactions[0].actions[0]))
        self.assertEqual((1,0), interactions[0].actions[0])
        self.assertEqual((0,1), interactions[0].actions[1])
        self.assertEqual(10,len(set([i.context for i in interactions])))

if __name__ == '__main__':
    unittest.main()
