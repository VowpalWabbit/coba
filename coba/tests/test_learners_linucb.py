import unittest

from collections import Counter

from coba.utilities import PackageChecker
from coba.exceptions import CobaException
from coba.learners import LinUCBLearner

@unittest.skipUnless(PackageChecker.numpy(strict=False), "numpy is not installed so we must skip these tests")
class LinUCBLearner_Tests(unittest.TestCase):

    def test_feature_arrays(self):
        learner = LinUCBLearner()
        learner.predict([1,2,3], [[1,2,3,4],[5,6,7,8]])
        learner.learn([1,2,3], [5,6,7,8], 1, 1)

    def test_predict(self):
        learner = LinUCBLearner()
        actions = [float(1),float(1),float(1)]
        preds   = [learner.predict([1,2,3], actions) for _ in range(1000)]
        counts  = Counter([(id(a),round(p,2)) for a,p in preds])
        self.assertEqual(len(counts),3)
        self.assertAlmostEqual(counts[(id(actions[0]),.33)]/sum(counts.values()),.33, delta=.05)
        self.assertAlmostEqual(counts[(id(actions[1]),.33)]/sum(counts.values()),.33, delta=.05)
        self.assertAlmostEqual(counts[(id(actions[2]),.33)]/sum(counts.values()),.33, delta=.05)
        self.assertEqual(learner._theta.shape, (5,))
        self.assertEqual(learner._A_inv.shape, (5,5))

    def test_score(self):
        learner = LinUCBLearner()
        learner.learn(None, 1, 1, .5)
        self.assertEqual(1/3,learner.score(None, [1,1,1],1))
        self.assertEqual(learner._theta.shape, (2,))
        self.assertEqual(learner._A_inv.shape, (2,2))

    def test_none_context(self):
        learner = LinUCBLearner()
        learner.learn(None, 1, 1, .5)
        actions = [float(1),float(1),float(1)]
        preds   = [learner.predict([1,2,3], actions) for _ in range(1000)]
        counts  = Counter([(id(a),round(p,2)) for a,p in preds])
        self.assertEqual(len(counts),3)
        self.assertAlmostEqual(counts[(id(actions[0]),.33)]/sum(counts.values()),.33, delta=.05)
        self.assertAlmostEqual(counts[(id(actions[1]),.33)]/sum(counts.values()),.33, delta=.05)
        self.assertAlmostEqual(counts[(id(actions[2]),.33)]/sum(counts.values()),.33, delta=.05)
        self.assertEqual(learner._theta.shape, (2,))
        self.assertEqual(learner._A_inv.shape, (2,2))

    def test_exploration_bound_greed(self):
        learner = LinUCBLearner(alpha=0.2)
        preds   = [learner.predict(None, [1,2,3]) for _ in range(1000)]
        counts  = Counter([(a,round(p,2)) for a,p in preds])
        self.assertEqual(len(counts),1)
        self.assertAlmostEqual(counts[(3,1.0)]/sum(counts.values()),1, delta=.05)

    def test_exploration_bound_explore(self):
        learner = LinUCBLearner(alpha=0)
        preds   = [learner.predict(None, [1,2,3]) for _ in range(1000)]
        counts  = Counter([(a,round(p,2)) for a,p in preds])
        self.assertEqual(len(counts),3)
        self.assertAlmostEqual(counts[(1,.33)]/sum(counts.values()),.33, delta=.05)
        self.assertAlmostEqual(counts[(2,.33)]/sum(counts.values()),.33, delta=.05)
        self.assertAlmostEqual(counts[(3,.33)]/sum(counts.values()),.33, delta=.05)

    def test_learn_something(self):
        learner = LinUCBLearner(alpha=0.2, features='a')
        for _ in range(30):
            learner.learn([1,2,3], (1,0,0), 1/4, 1/3)
            learner.learn([1,2,3], (0,1,0), 4/4, 1/3)
            learner.learn([1,2,3], (0,0,1), 3/4, 1/3)
        self.assertEqual(learner._theta.shape, (3,))
        self.assertEqual(learner._A_inv.shape, (3,3))
        self.assertAlmostEqual(learner._theta[0], 1/4, delta=.05)
        self.assertAlmostEqual(learner._theta[1], 4/4, delta=.05)
        self.assertAlmostEqual(learner._theta[2], 3/4, delta=.05)

    def test_sparse_exception(self):
        learner = LinUCBLearner(alpha=0.2)
        with self.assertRaises(CobaException):
            learner.learn({}, 1, 1, 1/3)
        with self.assertRaises(CobaException):
            learner.learn(None, {}, 1, 1/3)
        with self.assertRaises(CobaException):
            learner.predict({}, [1,2,3])
        with self.assertRaises(CobaException):
            learner.predict(None, [{},{},{}])

    def test_params(self):
        actual = LinUCBLearner(alpha=0.2,features=['a','xa']).params
        expected = {'family':'LinUCB', 'alpha':0.2, 'features':['a','xa'], 'seed':1}
        self.assertEqual(actual,expected)

if __name__ == '__main__':
    unittest.main()