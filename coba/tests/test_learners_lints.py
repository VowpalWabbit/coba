import unittest

from coba.utilities import PackageChecker
from coba.exceptions import CobaException
from coba.learners import LinTSLearner

@unittest.skipUnless(PackageChecker.numpy(strict=False), "numpy is not installed so we must skip these tests")
class LinTSLearner_Tests(unittest.TestCase):

    def test_none_context_value_actions(self):
        learner = LinTSLearner()
        probs = learner.predict(None, [1,1,1])
        learner.learn(None, 1, 1, .5)
        self.assertEqual(probs, [1/3,1/3,1/3])
        self.assertEqual(learner._mu_hat.shape, (2,))
        self.assertEqual(learner._B_inv.shape, (2,2))

    def test_value_context_value_actions(self):
        learner = LinTSLearner()
        probs = learner.predict(1, [1,1,1])
        learner.learn(1, 1, 1, 1)
        self.assertEqual(probs, [1/3,1/3,1/3])
        self.assertEqual(learner._mu_hat.shape, (3,))
        self.assertEqual(learner._B_inv.shape, (3,3))

    def test_dense_context_dense_actions(self):
        learner = LinTSLearner()
        learner.predict([1,2,3], [[1,2,3,4],[5,6,7,8]])
        learner.learn([1,2,3], [5,6,7,8], 1, 1)
        self.assertEqual(learner._mu_hat.shape, (17,))
        self.assertEqual(learner._B_inv.shape, (17,17))

    def test_dense_context_value_actions(self):
        learner = LinTSLearner()
        probs = learner.predict([1,2,3], [1,1,1])
        self.assertEqual(probs, [1/3,1/3,1/3])
        self.assertEqual(learner._mu_hat.shape, (5,))
        self.assertEqual(learner._B_inv.shape, (5,5))

    def test_score(self):
        learner = LinTSLearner()
        learner.learn(None, 1, 1, .5)
        self.assertEqual(1/3,learner.score(None, [1,1,1],1))
        self.assertEqual(learner._mu_hat.shape, (2,))
        self.assertEqual(learner._B_inv.shape, (2,2))

    def test_exploration_bound(self):
        learner = LinTSLearner(v=0.2)
        probs   = learner.predict([1,2,3], [1,2,3])
        self.assertEqual(probs, [0,0,1])
        learner = LinTSLearner(v=0)
        probs   = learner.predict([1,2,3], [1,2,3])
        self.assertEqual(probs, [1/3,1/3,1/3])

    def test_learn_something(self):
        learner = LinTSLearner(features='a')

        for _ in range(30):
            learner.learn([1,2,3], (1,0,0), 1/4, 1/3)
            learner.learn([1,2,3], (0,1,0), 4/4, 1/3)
            learner.learn([1,2,3], (0,0,1), 3/4, 1/3)

        self.assertAlmostEqual(learner._mu_hat[0], 1/4, places=1)
        self.assertAlmostEqual(learner._mu_hat[1], 4/4, places=1)
        self.assertAlmostEqual(learner._mu_hat[2], 3/4, places=1)

    def test_sparse_exception(self):
        learner = LinTSLearner()

        with self.assertRaises(CobaException):
            learner.learn({}, 1, 1, 1/3)

        with self.assertRaises(CobaException):
            learner.learn(None, {}, 1, 1/3)

        with self.assertRaises(CobaException):
            learner.predict({}, [1,2,3])

        with self.assertRaises(CobaException):
            learner.predict(None, [{},{},{}])

    def test_params(self):
        actual = LinTSLearner(v=0.2,features=['a','xa']).params
        expected = {'family':'LinTS', 'v':0.2, 'features':['a','xa']}
        self.assertEqual(actual,expected)

if __name__ == '__main__':
    unittest.main()