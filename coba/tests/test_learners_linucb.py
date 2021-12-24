import unittest
import importlib.util

from coba.exceptions import CobaException
from coba.learners import LinUCBLearner

@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is not installed so we must skip these tests")
class LinUCBLearner_Tests(unittest.TestCase):

    def test_matrix_vector_sizes(self):

        learner = LinUCBLearner()
        probs   = learner.predict([1,2,3], [1,1,1])

        self.assertEqual(probs, [1/3,1/3,1/3])
        self.assertEqual(learner._theta.shape, (4,))
        self.assertEqual(learner._A_inv.shape, (4,4))

    def test_exploration_bound(self):

        learner = LinUCBLearner(alpha=0.2)
        probs   = learner.predict([1,2,3], [1,2,3])
        self.assertEqual(probs, [0,0,1])

        learner = LinUCBLearner(alpha=0)
        probs   = learner.predict([1,2,3], [1,2,3])
        self.assertEqual(probs, [1/3,1/3,1/3])

    def test_learn_something(self):

        learner = LinUCBLearner(alpha=0.2)
        learner.learn([1,2,3], 1, 1, 1/3, None)

        self.assertEqual(learner._theta.shape, (4,))
        self.assertEqual(learner._A_inv.shape, (4,4))
        self.assertTrue(not (learner._theta == 0).any())
        self.assertTrue(not (learner._A_inv == 0).any())

    def test_sparse_exception(self):
        learner = LinUCBLearner(alpha=0.2)

        with self.assertRaises(CobaException):
            learner.learn({}, 1, 1, 1/3, None)
        
        with self.assertRaises(CobaException):
            learner.learn(None, {}, 1, 1/3, None)

        with self.assertRaises(CobaException):
            learner.predict({}, [1,2,3])

        with self.assertRaises(CobaException):
            learner.predict(None, [{},{},{}])

    def test_params(self):
        actual = LinUCBLearner(alpha=0.2,X=['a','xa']).params
        expected = {'family':'LinUCB', 'alpha':0.2, 'X':['a','xa']}
        self.assertEqual(actual,expected)

if __name__ == '__main__':
    unittest.main()