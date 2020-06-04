import unittest
import itertools
import random

from bbench.solvers import Solver

class Test_Solver_Instance(unittest.TestCase):

    def setUp(self):
        self._solver = Solver()
        
        # many of these tests assume the solver's choices are non-deterministic
        # we can never guarantee via test that a non-deterministic result will
        # never return a bad value. Therefore we merely show that the maximum 
        # likelihood of solver returns being wrong is less than below
        self._maximum_likelihood_incorrect = 1/1000

    def rand_state(self):
        return random.sample([None, 1, [1,2,3], "ABC", ["A","B","C"]],1)
    
    def rand_actions(self):
        return random.sample([[1,2], [1,2,3,4], [[1,2],[3,4]], ["A","B","C"]],1)
    
    def test_choose_in_range(self):
        for _ in range(int(1/self._maximum_likelihood_incorrect)):
            state   = self.rand_state()
            actions = self.rand_actions()
            self.assertIn(self._solver.choose(state, actions), range(len(actions)))

    def test_learn_no_exceptions(self):

        for _ in range(int(1/self._maximum_likelihood_incorrect)):
            state   = self.rand_state()
            actions = self.rand_actions()
            reward  = random.uniform(-2,2)
            self._solver.learn(state, actions, reward)

        self._solver.learn(1,[1,2,3],4)
        self._solver.learn(None,[1,2,3],-1)
        self._solver.learn(["A",1,3],[1,2,3],0)

if __name__ == '__main__':
    unittest.main()
