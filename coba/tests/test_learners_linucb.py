import unittest
import importlib.util

from coba.exceptions import CobaException
from coba.learners import LinUCBLearner

@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is not installed so we must skip these tests")
class LinUCBLearner_Tests(unittest.TestCase):

    def test_action_array_no_exception(self):
        learner = LinUCBLearner()
        learner.predict([1,2,3], [[1,2,3,4],[5,6,7,8]])
        learner.learn([1,2,3], [[1,2,3,4],[5,6,7,8]], [5,6,7,8], 1, 1)

    def test_matrix_vector_sizes(self):
        learner = LinUCBLearner()
        probs   = learner.predict([1,2,3], [1,1,1])

        self.assertEqual(probs, [1/3,1/3,1/3])
        self.assertEqual(learner._theta.shape, (5,))
        self.assertEqual(learner._A_inv.shape, (5,5))

    def test_request(self):
        learner = LinUCBLearner()
        probs   = learner.request(None, [1,1,1], [1,1,1])
        learner.learn(None, [1,1,1], 1, 1, .5)

        self.assertEqual(probs, [1/3,1/3,1/3])
        self.assertEqual(learner._theta.shape, (2,))
        self.assertEqual(learner._A_inv.shape, (2,2))

    def test_none_context(self):
        learner = LinUCBLearner()
        probs   = learner.predict(None, [1,1,1])
        learner.learn(None, [1,1,1], 1, 1, .5)

        self.assertEqual(probs, [1/3,1/3,1/3])
        self.assertEqual(learner._theta.shape, (2,))
        self.assertEqual(learner._A_inv.shape, (2,2))

    def test_exploration_bound(self):
        learner = LinUCBLearner(alpha=0.2)
        probs   = learner.predict([1,2,3], [1,2,3])
        self.assertEqual(probs, [0,0,1])

        learner = LinUCBLearner(alpha=0)
        probs   = learner.predict([1,2,3], [1,2,3])
        self.assertEqual(probs, [1/3,1/3,1/3])

    def test_learn_something(self):
        learner = LinUCBLearner(alpha=0.2, features='a')

        for _ in range(30):
            learner.learn([1,2,3], [(1,0,0),(0,1,0),(0,0,1)], (1,0,0), 1/4, 1/3)
            learner.learn([1,2,3], [(1,0,0),(0,1,0),(0,0,1)], (0,1,0), 4/4, 1/3)
            learner.learn([1,2,3], [(1,0,0),(0,1,0),(0,0,1)], (0,0,1), 3/4, 1/3)

        self.assertEqual(learner._theta.shape, (3,))
        self.assertEqual(learner._A_inv.shape, (3,3))

        self.assertAlmostEqual(learner._theta[0], 1/4, places=1)
        self.assertAlmostEqual(learner._theta[1], 4/4, places=1)
        self.assertAlmostEqual(learner._theta[2], 3/4, places=1)

    def test_sparse_exception(self):
        learner = LinUCBLearner(alpha=0.2)

        with self.assertRaises(CobaException):
            learner.learn({}, [1,1], 1, 1, 1/3)

        with self.assertRaises(CobaException):
            learner.learn(None, [{},{}], {}, 1, 1/3)

        with self.assertRaises(CobaException):
            learner.predict({}, [1,2,3])

        with self.assertRaises(CobaException):
            learner.predict(None, [{},{},{}])

    def test_params(self):
        actual = LinUCBLearner(alpha=0.2,features=['a','xa']).params
        expected = {'family':'LinUCB', 'alpha':0.2, 'features':['a','xa']}
        self.assertEqual(actual,expected)

if __name__ == '__main__':
    unittest.main()