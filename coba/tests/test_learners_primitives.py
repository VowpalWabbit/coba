import unittest

from coba.exceptions import CobaException
from coba.learners import Learner

class Learner_Tests(unittest.TestCase):

    def test_params_empty(self):
        class MyLearner(Learner):
            pass

        self.assertEqual(MyLearner().params,{})

    def test_score_not_implemented(self):
        class MyLearner(Learner):
            pass

        with self.assertRaises(CobaException) as ex:
            MyLearner().score(None,[],[])

        self.assertIn("`score`", str(ex.exception))

    def test_predict_not_implemented(self):
        class MyLearner(Learner):
            pass

        with self.assertRaises(CobaException) as ex:
            MyLearner().predict(None,[])

        self.assertIn("`predict`", str(ex.exception))

    def test_learn_not_implemented(self):
        class MyLearner(Learner):
            pass

        with self.assertRaises(CobaException) as ex:
            MyLearner().learn(None,None,None,None)

        self.assertIn("`learn`", str(ex.exception))

if __name__ == '__main__':
    unittest.main()
