import unittest

from coba.exceptions import CobaException
from coba.learners import Learner, ActionProb, PMF

class Learner_Tests(unittest.TestCase):

    def test_request_not_implemented(self):
        class MyLearner(Learner):
            def learn(self, *args, **kwargs) -> None:
                pass

        with self.assertRaises(CobaException) as ex:
            MyLearner().request(None,[],[])

        self.assertIn("`request`", str(ex.exception))

    def test_predict_not_implemented(self):
        class MyLearner(Learner):
            def learn(self, *args, **kwargs) -> None:
                pass

        with self.assertRaises(CobaException) as ex:
            MyLearner().predict(None,[])

        self.assertIn("`predict`", str(ex.exception))

class ActionProb_Tests(unittest.TestCase):

    def test_simple(self):
        self.assertEqual((1,.5), ActionProb(1,.5))
        self.assertIsInstance(ActionProb(1,2), ActionProb)

class Probs_Tests(unittest.TestCase):

    def test_simple(self):
        self.assertEqual([1/4,1/2,1/4], PMF([1/4,1/2,1/4]))
        self.assertIsInstance(PMF([1/4,1/2,1/4]), PMF)

if __name__ == '__main__':
    unittest.main()
