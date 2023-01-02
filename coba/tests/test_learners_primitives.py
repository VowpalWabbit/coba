import unittest

from coba.exceptions import CobaException
from coba.primitives import Batch
from coba.learners import SafeLearner, FixedLearner, ActionScore, Probs

class ParamsLearner:
    def __init__(self, params):
        self._params = params

    @property
    def params(self):
        return self._params

    def predict(self, context, actions):
        pass

    def learn(self, context, action, reward, probability, info):
        pass

class ParamsNotPropertyLearner:
    def __init__(self, params):
        self._params = params

    def params(self):
        return self._params

    def predict(self, context, actions):
        pass

    def learn(self, context, action, reward, probability, info):
        pass

class NoParamsLearner:
    def predict(self, context, actions):
        pass

    def learn(self, context, action, reward, probability, info):
        pass

class LearnerType1:
    def predict(self, context, actions):
        pass

    def learn(self, context, action, reward, probability):
        pass

class LearnerType2:
    def predict(self, context, actions):
        pass

    def learn(self, context, action, reward, probability, info):
        pass

class LearnerType3:
    def predict(self, context, actions):
        pass

    def learn(self, context, actions, action, reward, probability, **kwargs):
        pass

class BrokenLearnSignature:
    def predict(self, context, actions):
        pass

    def learn(self, context):
        pass

class UnsafeFixedLearner:
    def __init__(self, pmf, info):
        self._pmf = pmf
        self._info = info

    def predict(self, context, actions):
        if self._info is None:
            return self._pmf
        else:
            return self._pmf, self._info

class AmbiguousPredictionLearner:

    def predict(self, context, actions):
        return [1,0]

    def learn(self, context, actions, action, reward, probability) -> None:
        pass

class SafeLearner_Tests(unittest.TestCase):

    def test_no_params(self):
        learner = SafeLearner(NoParamsLearner())
        self.assertEqual("NoParamsLearner", learner.params["family"])

    def test_params_not_property(self):
        learner = SafeLearner(ParamsNotPropertyLearner({'a':"A"}))
        self.assertDictEqual({"family":"ParamsNotPropertyLearner", "a":"A"}, learner.params)

    def test_params_no_family(self):
        learner = SafeLearner(ParamsLearner({'a':"A"}))
        self.assertDictEqual({"family":"ParamsLearner", "a":"A"}, learner.params)

    def test_params_family(self):
        learner = SafeLearner(ParamsLearner({'a':"A", "family":"B"}))
        self.assertDictEqual({"family":"B", "a":"A"}, learner.params)

    def test_params_fullname_0_params(self):
        learner = SafeLearner(ParamsLearner({"family":"B"}))

        self.assertEqual("B", learner.full_name)

    def test_params_fullname_1_param(self):
        learner = SafeLearner(ParamsLearner({'a':"A", "family":"B"}))

        self.assertEqual("B(a=A)", learner.full_name)

    def test_params_fullname_2_params(self):
        learner = SafeLearner(ParamsLearner({'a':"A","b":"B", "family":"B"}))

        self.assertEqual("B(a=A,b=B)", learner.full_name)

    def test_no_sum_one_no_info_action_match_predict(self):
        learner = SafeLearner(UnsafeFixedLearner([1/3,1/2], None))

        with self.assertRaises(CobaException):
            learner.predict(None, [1,2])

    def test_no_sum_one_info_action_match_predict(self):
        learner = SafeLearner(UnsafeFixedLearner([1/3,1/2], 1))

        with self.assertRaises(AssertionError):
            learner.predict(None, [1,2])

    def test_sum_one_no_info_action_mismatch_predict(self):
        learner = SafeLearner(UnsafeFixedLearner([1/2,1/2], None))

        with self.assertRaises(CobaException):
            learner.predict(None, [1,2,3])

    def test_sum_one_info_action_mismatch_predict(self):
        learner = SafeLearner(UnsafeFixedLearner([1/2,1/2], 1))

        with self.assertRaises(AssertionError):
            learner.predict(None, [1,2,3])

    def test_sum_one_no_info_action_match_predict(self):
        learner = SafeLearner(UnsafeFixedLearner([1/2,1/2], None))
        predict = learner.predict(None, [1,2])

        self.assertEqual(0  , predict[0])
        self.assertEqual(1/2, predict[1])
        self.assertEqual({} , predict[2])

    def test_sum_one_info_action_match_predict(self):
        learner = SafeLearner(UnsafeFixedLearner([1/2,1/2], 1))

        predict = learner.predict(None, [1,2])

        self.assertEqual(0      , predict[0])
        self.assertEqual(1/2    , predict[1])
        self.assertEqual({'_':1}, predict[2])

    def test_no_exception_with_learner_type1(self):
        learner = SafeLearner(LearnerType1())
        learner.learn(1,[1,2],2,3,4,info={})
        learner.learn(1,[1,2],2,3,4,info={})

    def test_no_exception_with_learner_type2(self):
        learner = SafeLearner(LearnerType2())
        learner.learn(1,[1,2],2,3,4,info=1)
        learner.learn(1,[1,2],2,3,4,info=1)

    def test_no_exception_with_learner_type3(self):
        learner = SafeLearner(LearnerType3())
        learner.learn(1,[1,2],2,3,4,info={})
        learner.learn(1,[1,2],2,3,4,info={})

    def test_exception_with_broken_learn_signature(self):
        learner = SafeLearner(BrokenLearnSignature())
        with self.assertRaises(Exception) as e:
            learner.learn(1,[1,2],2,3,4,**{})

        self.assertIn("takes 2 positional arguments but 6 were given", str(e.exception))

    def test_type2_prediction_sans_info(self):
        class MyLearner:
            def predict(self,context,actions):
                return [0,0,1]

        self.assertEqual(SafeLearner(MyLearner()).predict(None,[1,2,3]), (2,1,{}))

    def test_batched_type2_prediction_sans_info(self):
        class MyLearner:
            def predict(self,context,actions):
                return [[0,0,1],[0,1,0],[1,0,0]][:len(context)]

        self.assertEqual(SafeLearner(MyLearner()).predict(Batch([None]*3), Batch([[1,2,3]]*3)), ([2,1,0],[1,1,1],{}))

    def test_type3_prediction_sans_info(self):
        class MyLearner:
            def predict(self,context,actions):
                return 0,.5

        self.assertEqual(SafeLearner(MyLearner()).predict(None,[1,2,3]), (0,.5,{}))

    def test_batched_type3_prediction_sans_info(self):
        class MyLearner:
            def predict(self,context,actions):
                return [(0,1,2),(1,.5,1)]

        self.assertEqual(SafeLearner(MyLearner()).predict(Batch([None]*3), Batch([[1,2,3]]*3)), ([0,1,2],[1,.5,1],{}))

    def test_batched_type3_prediction_with_info(self):
        class MyLearner:
            def predict(self,context,actions):
                if len(context) == 1:
                    return [(0,),(1,),{'a':[1,]}]
                else:
                    return [(0,1,2),(1,.5,1),{'a':[1,2,3]}]

        safe_learner = SafeLearner(MyLearner())

        #test initial call
        self.assertEqual(safe_learner.predict(Batch([None]*3), Batch([[1,2,3]]*3)), ([0,1,2],[1,.5,1],{'a':[1,2,3]}))
        #test shortcut logic after initial
        self.assertEqual(safe_learner.predict(Batch([None]*3), Batch([[1,2,3]]*3)), ([0,1,2],[1,.5,1],{'a':[1,2,3]}))

    def test_type3_not_batched_prediction_exception_with_info(self):

        class MyLearner:
            def predict(self,context,actions):
                if isinstance(context,Batch): raise Exception()
                preds = [(0,1,{'a':1}),(1,.5,{'a':2}),(2,1,{'a':3})]
                return preds[context]

        safe_learner = SafeLearner(MyLearner())

        #test initial call
        self.assertEqual(safe_learner.predict(Batch([0,1,2]), Batch([[1,2,3]]*3)), ([0,1,2],[1,.5,1],{'a':[1,2,3]}))
        #test shortcut logic after initial
        self.assertEqual(safe_learner.predict(Batch([0,1,2]), Batch([[1,2,3]]*3)), ([0,1,2],[1,.5,1],{'a':[1,2,3]}))

    def test_type3_not_batched_prediction_with_info(self):

        class MyLearner:
            def predict(self,context,actions):
                preds = [(0,1,{'a':1}),(1,.5,{'a':2}),(2,1,{'a':3})]
                return [] if isinstance(context,Batch) else preds[context]

        safe_learner = SafeLearner(MyLearner())

        #test initial call
        self.assertEqual(safe_learner.predict(Batch([0,1,2]), Batch([[1,2,3]]*3)), ([0,1,2],[1,.5,1],{'a':[1,2,3]}))
        #test shortcut logic after initial
        self.assertEqual(safe_learner.predict(Batch([0,1,2]), Batch([[1,2,3]]*3)), ([0,1,2],[1,.5,1],{'a':[1,2,3]}))

    def test_type3_not_batched_learn_exception_with_info(self):

        class MyLearner:
            calls = []
            def learn(self,*args,**kwargs):
                if isinstance(args[0],Batch): raise Exception()
                self.calls.append((args,kwargs))

        learner = MyLearner()
        safe_learner = SafeLearner(learner)

        learn_args   = (Batch([0,1]), Batch([[1,2]]*2), [0,1], [1,2], [1,1])
        learn_kwargs = {'a':[1,2]}

        excpected_calls = [ ((0, [1,2], 0, 1, 1),{'a':1}), ((1, [1,2], 1, 2, 1),{'a':2}) ]

        #test initial call        
        safe_learner.learn(*learn_args, **learn_kwargs)
        self.assertEqual(learner.calls, excpected_calls)
        learner.calls.clear()
        
        #test shortcut logic after initial
        safe_learner.learn(*learn_args, **learn_kwargs)
        self.assertEqual(learner.calls, excpected_calls)

    def test_type3_not_batched_prediction_exception_throws_first(self):

        class MyLearner:
            calls = []
            def predict(self,*args,**kwargs):
                if isinstance(args[0],Batch): 
                    raise Exception("1")
                else:
                    raise Exception("2")

        learner = MyLearner()
        safe_learner = SafeLearner(learner)

        with self.assertRaises(Exception) as e:
            safe_learner.predict(Batch([1,2,3]), Batch([1,2,3]))

        self.assertEqual(str(e.exception),"1")

    @unittest.skip("Skipped because it is a rare use case and we don't want to support it for now. Can be added later.")
    def test_pdf_prediction(self):
        with self.assertRaises(CobaException) as e:
            action,score,kwargs = SafeLearner(FixedLearner(lambda a: 1 if a == 0 else 0)).predict(None,[0,1])
        
        self.assertEqual(str(e.exception), "PDFs predictions are currently not supported.")

    def test_infer_types(self):
        learner = SafeLearner(None)

        with self.assertRaises(CobaException):
            #can be either 2 or 3 because (0,1) is either a valid PMF or an action index of 0 with score of 1
            self.assertEqual(learner.get_inferred_type((0,1),[1,2]),2)

        #can't be a pmf because pred_0 does not add to 1
        self.assertEqual(learner.get_inferred_type((5,1),[]),3)

    def test_definite_types(self):
        learner = SafeLearner(None)

        #this is definitely a pdf because it is callable
        #self.assertEqual(learner.get_type(lambda a: 1, True),1)

        #this is definitely a pmf because of how long it is
        self.assertEqual(learner._determine_pred_type((0,1,0,0,0,0), True),2)

        #this is definitely a pmf because it is explicitly typed
        self.assertEqual(learner._determine_pred_type(Probs([0,1]), True),2)

        #this is definitely an action-score pair because it is explicitly typed
        self.assertEqual(learner._determine_pred_type(ActionScore(0,1), False),3)

        #this can't be determined
        self.assertEqual(learner._determine_pred_type((0,1), True),None)

class ActionScore_Tests(unittest.TestCase):

    def test_simple(self):
        self.assertEqual((1,.5), ActionScore(1,.5))
        self.assertIsInstance(ActionScore(1,2), ActionScore)

class Probs_Tests(unittest.TestCase):

    def test_simple(self):
        self.assertEqual([1/4,1/2,1/4], Probs([1/4,1/2,1/4]))
        self.assertIsInstance(Probs([1/4,1/2,1/4]), Probs)

if __name__ == '__main__':
    unittest.main()
