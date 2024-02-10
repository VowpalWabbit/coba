import math
import unittest
from collections import Counter

from coba.learners import BanditEpsilonLearner, BanditUCBLearner, RandomLearner, FixedLearner

class BanditEpsilonLearner_Tests(unittest.TestCase):
    def test_params(self):
        self.assertEqual({"family":"BanditEpsilon", "epsilon":0.05, "seed": 1}, BanditEpsilonLearner().params)

    def test_score_no_learn(self):
        learner = BanditEpsilonLearner(epsilon=0.5)
        self.assertEqual([.25,.25,.25,.25],[learner.score(None, [1,2,3,4],a) for a in [1,2,3,4]])

    def test_predict_no_learn(self):
        learner = BanditEpsilonLearner(epsilon=0.5)
        self.assertEqual((1,0.25),learner.predict(None, [1,2,3,4]))
        self.assertEqual((4,0.25),learner.predict(None, [1,2,3,4]))

    def test_predict_lots_of_actions(self):
        learner = BanditEpsilonLearner(epsilon=0.5)
        self.assertAlmostEqual(learner.predict(None, list(range(993)))[1] * 993, 1, delta=.001)

    def test_learn_predict_no_epsilon(self):
        learner = BanditEpsilonLearner(epsilon=0)
        learner.learn(None, 2, 1, None)
        learner.learn(None, 1, 2, None)
        learner.learn(None, 3, 3, None)
        self.assertEqual((3,1.0),learner.predict(None, [1,2,3]))

    def test_learn_predict_epsilon(self):
        learner = BanditEpsilonLearner(epsilon=0.1)
        learner.learn(None, 2, 1, None)
        learner.learn(None, 1, 2, None)
        learner.learn(None, 2, 1, None)
        preds  = [learner.predict(None, [1,2]) for _ in range(1000)]
        counts = Counter([(a,round(p,2)) for a,p in preds])
        self.assertEqual(len(counts),2)
        self.assertAlmostEqual(counts[(1,.95)]/sum(counts.values()), .95, delta=0.05)
        self.assertAlmostEqual(counts[(2,.05)]/sum(counts.values()), .05, delta=0.05)

    def test_learn_score_epsilon(self):
        learner = BanditEpsilonLearner(epsilon=0.1)
        learner.learn(None, 2, 1, None)
        learner.learn(None, 1, 2, None)
        learner.learn(None, 2, 1, None)
        self.assertAlmostEqual(.95,learner.score(None, [1,2], 1))
        self.assertAlmostEqual(.05,learner.score(None, [1,2], 2))

    def test_learn_predict_epsilon_unhashables(self):
        learner = BanditEpsilonLearner(epsilon=0.1)
        learner.learn(None, [2], 1, None)
        learner.learn(None, [1], 2, None)
        learner.learn(None, [2], 1, None)
        preds  = [learner.predict(None, [[1],[2]]) for _ in range(1000)]
        counts = Counter([( tuple(a),round(p,2)) for a,p in preds])
        self.assertEqual(len(counts),2)
        self.assertEqual(counts.most_common()[0][0],((1,),.95))
        self.assertEqual(counts.most_common()[1][0],((2,),.05))
        self.assertAlmostEqual(counts.most_common()[0][1]/sum(counts.values()), .95, delta=0.05)
        self.assertAlmostEqual(counts.most_common()[1][1]/sum(counts.values()), .05, delta=0.05)

    def test_learn_predict_epsilon_all_equal(self):
        learner = BanditEpsilonLearner(epsilon=0.1)
        learner.learn(None, 2, 1, None)
        learner.learn(None, 1, 2, None)
        learner.learn(None, 2, 3, None)
        self.assertEqual(.5, learner.score(None,[1,2],1))
        self.assertEqual(.5, learner.score(None,[1,2],2))

class BanditUCBLearner_Tests(unittest.TestCase):
    def test_params(self):
        self.assertEqual({'family': 'BanditUCB', 'seed': 1 }, BanditUCBLearner().params)

    def test_predict_all_actions_first(self):
        learner,actions = BanditUCBLearner(),[1,2,3]
        self.assertEqual((1,1/3),learner.predict(None, actions))
        learner.learn(None, 1, 0, 0)
        self.assertEqual((3,1/2),learner.predict(None, actions))
        learner.learn(None, 3, 0, 0)
        self.assertEqual((2,1  ),learner.predict(None, actions))
        learner.learn(None, 2, 0, 0)
        #the last time all actions have the same value so we pick randomly
        self.assertEqual([1/3, 1/3, 1/3],[learner.score(None, actions,a) for a in actions])

    def test_score_all_actions_first(self):
        learner,actions = BanditUCBLearner(),[1,2,3]
        self.assertEqual([1/3,1/3,1/3],[learner.score(None, actions, a) for a in actions])
        learner.learn(None, 1, 0, 0)
        self.assertEqual([0,1/2,1/2],[learner.score(None, actions, a) for a in actions])
        learner.learn(None, 2, 0, 0)
        self.assertEqual([0,0,1],[learner.score(None, actions, a) for a in actions])
        learner.learn(None, 3, 0, 0)
        #the last time all actions have the same value so we pick randomly
        self.assertEqual([1/3,1/3,1/3],[learner.score(None, actions, a) for a in actions])

    def test_learn_predict_best1(self):
        learner,actions = BanditUCBLearner(),[1,2,3,4]
        learner.learn(None, 1, 1, None)
        learner.learn(None, 2, 1, None)
        learner.learn(None, 3, 1, None)
        learner.learn(None, 4, 1, None)
        preds  = [learner.predict(None, actions) for _ in range(1000)]
        counts = Counter([(a,round(p,2)) for a,p in preds])
        self.assertEqual(len(counts),4)
        self.assertAlmostEqual(counts[(1,1/4)]/sum(counts.values()), .25, delta=0.05)
        self.assertAlmostEqual(counts[(2,1/4)]/sum(counts.values()), .25, delta=0.05)
        self.assertAlmostEqual(counts[(3,1/4)]/sum(counts.values()), .25, delta=0.05)
        self.assertAlmostEqual(counts[(4,1/4)]/sum(counts.values()), .25, delta=0.05)

    def test_learn_predict_best2(self):
        learner,actions = BanditUCBLearner(),[1,2,3,4]
        learner.learn(None, 1, 0, None)
        learner.learn(None, 2, 0, None)
        learner.learn(None, 3, 0, None)
        learner.learn(None, 4, 1, None)
        preds  = [learner.predict(None, actions) for _ in range(1000)]
        counts = Counter([(a,round(p,2)) for a,p in preds])
        self.assertEqual(len(counts),1)
        self.assertIn((4,1),counts)

    def test_learn_predict_best3(self):
        learner,actions = BanditUCBLearner(),[1,2,3,4]
        learner.learn(None, 1, 0, None)
        learner.learn(None, 2, 0, None)
        learner.learn(None, 3, 0, None)
        learner.learn(None, 4, 1, None)
        learner.learn(None, 1, 0, None)
        learner.learn(None, 2, 0, None)
        learner.learn(None, 3, 0, None)
        learner.learn(None, 4, 1, None)
        preds  = [learner.predict(None, actions) for _ in range(1000)]
        counts = Counter([(a,round(p,2)) for a,p in preds])
        self.assertEqual(len(counts),1)
        self.assertIn((4,1),counts)

    def test_learn_score_best2(self):
        learner,actions = BanditUCBLearner(),[1,2,3,4]
        learner.learn(None, 1, 0, None)
        learner.learn(None, 2, 0, None)
        learner.learn(None, 3, 0, None)
        learner.learn(None, 4, 1, None)
        self.assertEqual(0 , learner.score(None, actions, 1))
        self.assertEqual(0 , learner.score(None, actions, 2))
        self.assertEqual(0 , learner.score(None, actions, 3))
        self.assertEqual(1 , learner.score(None, actions, 4))

class FixedLearner_Tests(unittest.TestCase):
    def test_params(self):
        self.assertEqual({'family':'fixed','seed':1}, FixedLearner([1/2,1/2]).params)

    def test_create_errors(self):
        with self.assertRaises(AssertionError):
            FixedLearner([1/3, 1/2])
        with self.assertRaises(AssertionError):
            FixedLearner([-1, 2])

    def test_score(self):
        learner = FixedLearner([1/3,1/6,3/6])
        self.assertEqual(1/3 , learner.score(None, [1,2,3],1))
        self.assertEqual(1/6 , learner.score(None, [1,2,3],2))
        self.assertEqual(3/6 , learner.score(None, [1,2,3],3))

    def test_predict(self):
        learner = FixedLearner([1/3,1/3,1/3])
        preds  = [learner.predict(None, [1,2,3]) for _ in range(1000)]
        counts = Counter([(a,round(p,2)) for a,p in preds])
        self.assertEqual(len(counts),3)
        self.assertAlmostEqual(counts[(1,.33)]/sum(counts.values()), .33, delta=0.05)
        self.assertAlmostEqual(counts[(2,.33)]/sum(counts.values()), .33, delta=0.05)
        self.assertAlmostEqual(counts[(3,.33)]/sum(counts.values()), .33, delta=0.05)

    def test_learn(self):
        FixedLearner([1/3,1/3,1/3]).learn(None, 1, .5, None)

class RandomLearner_Tests(unittest.TestCase):

    def test_params(self):
        self.assertEqual({'family':'Random','seed':1}, RandomLearner().params)

    def test_score(self):
        learner = RandomLearner()
        self.assertEqual(1/2, learner.score(None, [1,2  ], 2))
        self.assertEqual(1/3, learner.score(None, [1,2,3], 3))

    def test_predict(self):
        learner = RandomLearner()
        preds  = [learner.predict(None, [1,2,3]) for _ in range(1000)]
        counts = Counter([(a,round(p,2)) for a,p in preds])
        self.assertEqual(len(counts),3)
        self.assertAlmostEqual(counts[(1,.33)]/sum(counts.values()), .33, delta=0.05)
        self.assertAlmostEqual(counts[(2,.33)]/sum(counts.values()), .33, delta=0.05)
        self.assertAlmostEqual(counts[(3,.33)]/sum(counts.values()), .33, delta=0.05)

    def test_learn(self):
        learner = RandomLearner()
        learner.learn(2, 1, 1, 1)

if __name__ == '__main__':
    unittest.main()
