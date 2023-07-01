import unittest

from coba.learners import MisguidedLearner

class MisguidedLearner_Tests(unittest.TestCase):

    def test_params(self):
        class MyLearner:
            params = {'a':1}

        self.assertEqual(MisguidedLearner(MyLearner(),0,0).params,{'a':1,'misguided':(0,0)})

    def test_request(self):
        request_args = []
        class MyLearner:
            def request(self,*args):
                request_args.extend(args)

        MisguidedLearner(MyLearner(),0,0).request(1,[1,2],[1])

        self.assertEqual(request_args,[1,[1,2],[1]])

    def test_predict(self):
        predict_args = []
        class MyLearner:
            def predict(self,*args):
                predict_args.extend(args)

        MisguidedLearner(MyLearner(),0,0).predict(1,[1,2])

        self.assertEqual(predict_args,[1,[1,2]])

    def test_learn(self):
        learn_args = []
        class MyLearner:
            def learn(self,*args):
                learn_args.extend(args)

        MisguidedLearner(MyLearner(),1,-1).learn(1,[3,4], 3, 1, 0)

        self.assertEqual(learn_args,[1,[3,4], 3, 0, 0])

if __name__ == '__main__':
    unittest.main()
