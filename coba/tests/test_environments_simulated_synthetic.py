import unittest
import pickle

from pathlib import Path

from coba.contexts import CobaContext, NullLogger

from coba.environments import LinearSyntheticSimulation, LocalSyntheticSimulation

CobaContext.logger = NullLogger()

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
        env = pickle.loads(pickle.dumps(LinearSyntheticSimulation(100,2,3,4,0,["xa"],2)))
        
        self.assertEqual(2     , env.params['n_A'])
        self.assertEqual(3     , env.params['n_C_phi'])
        self.assertEqual(4     , env.params['n_A_phi'])
        self.assertEqual(0     , env.params['r_noise'])
        self.assertEqual(['xa'], env.params['X'])
        self.assertEqual(2     , env.params['seed'])
        self.assertEqual("LinearSynth(A=2,c=3,a=4,X=['xa'],seed=2)", str(env))
        self.assertEqual(100, len(list(env.read())))

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

if __name__ == '__main__':
    unittest.main()
