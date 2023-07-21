import unittest

from coba.exceptions import CobaException
from coba.primitives import Batch
from coba.learners import ActionProb, PMF
from coba.learners.safety import SafeLearner, pred_format, batch_order, possible_action, possible_pmf

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

class batch_order_Tests(unittest.TestCase):

    def test_PM_not(self):
        self.assertEqual(batch_order(None, (0,1), None, None),'not')

    def test_PM_not_kw(self):
        self.assertEqual(batch_order(None, ([0,1],{}), None, None),'not')

    def test_AP_not(self):
        self.assertEqual(batch_order(None, (0,1), None, None),'not')

    def test_AP_not_kw(self):
        self.assertEqual(batch_order(None, [(0,1),{}], None, None),'not')

    def test_AP_not_kw_flat(self):
        self.assertEqual(batch_order(None, (0,1,{}), None, None),'not')

    def test_PM_row(self):
        self.assertEqual(batch_order(None, [(1/2,1/2)], Batch([1]), Batch([[1,2]])),'row')

    def test_PM_row_kw(self):
        self.assertEqual(batch_order(None, [((1/2,1/2),{})], Batch([1]), Batch([[1,2]])),'row')

    def test_AP_row(self):
        self.assertEqual(batch_order(None, [(1,1)], Batch([1]), Batch([1])),'row')

    def test_AP_row_kw(self):
        self.assertEqual(batch_order(None, [([1,1],{})], Batch([1]), Batch([1])),'row')

    def test_AP_row_kw_flat(self):
        self.assertEqual(batch_order(None, [(1,1,{})], Batch([1]), Batch([1])),'row')

    def test_AP_col(self):
        self.assertEqual(batch_order(None, [[0],[1]], Batch([1]), Batch([1])), 'col')

    def test_AP_col_kw(self):
        self.assertEqual(batch_order(None, [[0],[1],{}], Batch([1]), Batch([1])), 'col')

    def test_PM_col(self):
        self.assertEqual(batch_order(None, [[0],[1],[0]], Batch([1]), Batch([[1,2,3]])), 'col')

    def test_PM_col_kw(self):
        self.assertEqual(batch_order(None, [[0],[1],[0],{}], Batch([1]), Batch([[1,2,3]])), 'col')

    def test_PM_matching_dims_col(self):
        self.assertEqual(batch_order(lambda x,a:[[0],[1]], [[0,0],[1,1]], Batch([1,1]), Batch([[1,2],[1,2]])), 'col')

    def test_PM_matching_dims_row(self):
        self.assertEqual(batch_order(lambda x,a:[[0,1]], [[0,0],[1,1]], Batch([1,1]), Batch([[1,2],[1,2]])), 'row')

class pred_format_Tests(unittest.TestCase):

    def test_PM3_explicit(self):
        self.assertEqual(pred_format(PMF([1,0,0]),'not',False,['a','b','c']),'PM')

    def test_PM3_batchnot(self):
        self.assertEqual(pred_format((1,0,0),'not',False,['a','b','c']),'PM')

    def test_PM3_batchnot_kw(self):
        self.assertEqual(pred_format(((1,0,0),{}),'not',True,['a','b','c']),'PM')

    def test_PM3_batchnot_and_pmf_in_actions(self):
        self.assertEqual(pred_format((1,0,0),'not',False,[(1,0,0),(0,1,0),(0,0,1)]),'PM')

    def test_PM2_batchnot_and_pmf_in_actions(self):
        with self.assertRaises(CobaException):
            pred_format((1,0),'not',False,[0,1])

    def test_PM2_batchnot_and_pmf_not_in_actions(self):
        self.assertEqual(pred_format((1,0),'not',False,['a','b']),'PM')

    def test_AP_explicit(self):
        self.assertEqual(pred_format(ActionProb('a',1),'not',False,['a','b','c']),'AP')

    def test_AP_batchnot(self):
        self.assertEqual(pred_format(('a',1),'not',False,['a','b','c']),'AP')

    def test_AP_batchnot_kw(self):
        self.assertEqual(pred_format((('a',1),{}),'not',True,['a','b','c']),'AP')

    def test_AP_batchnot_sum_to_1(self):
        self.assertEqual(pred_format((2,-1),'not',False,[2,1]),'AP')

    def test_AP_batchnot_kw_flat(self):
        self.assertEqual(pred_format(('a',1,{}),'not',True,['a','b','c']),'AP')

    def test_AP_batchnot_and_action_not_in_actions(self):
        with self.assertRaises(CobaException):
            pred_format(('a',0),'not',False,[0,1])

    def test_PM_batchrow(self):
        self.assertEqual(pred_format([(1,0,0)],'row',False,[['a','b','c']]),'PM')

    def test_PM_batchcol(self):
        self.assertEqual(pred_format(([1],[0],[0]),'col',False,[['a','b','c']]),'PM')

    def test_AP_batchrow(self):
        self.assertEqual(pred_format([('a',0)],'row',False,[['a','b','c']]),'AP')

    def test_AP_batchcol(self):
        self.assertEqual(pred_format((['a'],[0]),'col',False,[['a','b','c']]),'AP')

class possible_action_Tests(unittest.TestCase):

    def test_possible_action(self):
        self.assertTrue(possible_action(1,[1,2,3]))

    def test_not_possible_action(self):
        self.assertFalse(possible_action(0,[1,2,3]))

    def test_not_possible_exception(self):
        self.assertFalse(possible_action(0,1))

class possible_pmf_Tests(unittest.TestCase):

    def test_possible_pmf(self):
        self.assertTrue(possible_pmf([1/2,1/2],[1,2]))

    def test_not_possible_pmf_sum(self):
        self.assertFalse(possible_pmf([1/2,1/3],[1,2]))

    def test_not_possible_pmf_len(self):
        self.assertFalse(possible_pmf([1/2,1/2],[1,2,3]))

    def test_not_possible_pmf_exception(self):
        self.assertFalse(possible_pmf([1/2,1/2],1))

class SafeLearner_Tests(unittest.TestCase):

    def test_no_params(self):
        learner = SafeLearner(NoParamsLearner())
        self.assertEqual("NoParamsLearner", learner.params["family"])

    def test_params_not_dict(self):
        learner = SafeLearner(ParamsLearner([]))
        self.assertDictEqual({"params":"[]", "family":"ParamsLearner"}, learner.params)

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

    def test_predict_PM2_batchnot_no_sum_one(self):
        learner = SafeLearner(UnsafeFixedLearner([1/3,1/2], None))

        with self.assertRaises(CobaException):
            learner.predict(None, [1,2])

    def test_predict_PM2_batchnot_no_sum_one_kw(self):
        learner = SafeLearner(UnsafeFixedLearner([1/3,1/2], {'_':1}))

        with self.assertRaises(CobaException):
            learner.predict(None, [1,2])

    def test_predict_PM2_batchnot_action_mismatch(self):
        learner = SafeLearner(UnsafeFixedLearner([1/2,1/2], None))

        with self.assertRaises(CobaException):
            learner.predict(None, [1,2,3])

    def test_predict_PM2_batchnot_action_mismatch_kw(self):
        learner = SafeLearner(UnsafeFixedLearner([1/2,1/2], {'_':1}))

        with self.assertRaises(CobaException):
            learner.predict(None, [1,2,3])

    def test_predict_PM2_batchnot(self):
        learner = SafeLearner(UnsafeFixedLearner([1/2,1/2], None))
        predict = learner.predict(None, [1,2])

        self.assertEqual(1  , predict[0])
        self.assertEqual(1/2, predict[1])
        self.assertEqual({} , predict[2])

    def test_predict_PM3_batchnot(self):
        class MyLearner:
            def predict(self,context,actions):
                return [0,0,1]

        self.assertEqual(SafeLearner(MyLearner()).predict(None,[1,2,3]), (3,1,{}))

    def test_predict_PM2_batchnot_kw(self):
        learner = SafeLearner(UnsafeFixedLearner([1/2,1/2], {'_':1} ))
        predict = learner.predict(None, [1,2])
        self.assertEqual(1      , predict[0])
        self.assertEqual(1/2    , predict[1])
        self.assertEqual({'_':1}, predict[2])

    def test_predict_PM_batchrow(self):
        class MyLearner:
            def predict(self,context,actions):
                return [[0,0,1],[0,1,0],[1,0,0]][:len(context)]

        self.assertEqual(SafeLearner(MyLearner()).predict(Batch([None]*3), Batch([[1,2,3]]*3)), ([3,2,1],[1,1,1],{}))

    def test_predict_PM_batchcol(self):
        class MyLearner:
            def predict(self,context,actions):
                return list(zip(*[[0,0,1],[0,1,0],[1,0,0]][:len(context)]))

        self.assertEqual(SafeLearner(MyLearner()).predict(Batch([None]*3), Batch([[1,2,3]]*3)), ([3,2,1],[1,1,1],{}))

    def test_predict_AP_batchnot(self):
        class MyLearner:
            def predict(self,context,actions):
                return 1,.5

        self.assertEqual(SafeLearner(MyLearner()).predict(None,[1,2,3]), (1,.5,{}))

    def test_predict_AP_batchnot_kw(self):
        class MyLearner:
            def predict(self,context,actions):
                return 1,.5,{'a':1}

        self.assertEqual(SafeLearner(MyLearner()).predict(None,[1,2,3]), (1,.5,{'a':1}))

    def test_predict_AP_batchcol(self):
        class MyLearner:
            def predict(self,context,actions):
                return [(3,1,2),(1,.5,1)]

        self.assertEqual(SafeLearner(MyLearner()).predict(Batch([None]*3), Batch([[1,2,3]]*3)), ((3,1,2),(1,.5,1),{}))

    def test_predict_AP_batchrow(self):
        class MyLearner:
            def predict(self,context,actions):
                return [((3,1),{}),((1,.5),{}),((2,1),{})]

        self.assertEqual(SafeLearner(MyLearner()).predict(Batch([None]*3), Batch([[1,2,3]]*3)), ([3,1,2],[1,.5,1],{}))

    def test_predict_AP_batchrow_flat(self):
        class MyLearner:
            def predict(self,context,actions):
                return [(3,1),(1,.5),(2,1)]

        self.assertEqual(SafeLearner(MyLearner()).predict(Batch([None]*3), Batch([[1,2,3]]*3)), ([3,1,2],[1,.5,1],{}))

    def test_predict_throws_inner_exception_when_str_context_and_empty_action(self):
        class MyException(Exception):
            pass

        class MyLearner:
            def predict(self,context,actions):
                raise MyException()

        with self.assertRaises(MyException):
            SafeLearner(MyLearner()).predict('abc',[])

    def test_predict_AP_batchcol_kw_dimension_check_then_shortcut(self):
        class MyLearner:
            def __init__(self) -> None:
                self._calls = [
                    [(3,1,2),(1,.5,1),{'a':[1,2,3]}], # first pred call
                    [(0,),(1,),{'a':[1,]}],           # dimension check call
                    [(3,1,2),(1,.5,1),{'a':[1,2,3]}]  # second pred call
                ]
            def predict(self,context,actions):
                return self._calls.pop()

        safe_learner = SafeLearner(MyLearner())

        #test initial call
        self.assertEqual(safe_learner.predict(Batch([None]*3), Batch([[1,2,3]]*3)), ((3,1,2),(1,.5,1),{'a':[1,2,3]}))
        #test shortcut logic after initial
        self.assertEqual(safe_learner.predict(Batch([None]*3), Batch([[1,2,3]]*3)), ((3,1,2),(1,.5,1),{'a':[1,2,3]}))

    def test_predict_AP_batchrow_kw_batch_exception_fallback(self):

        class MyLearner:
            def predict(self,context,actions):
                if isinstance(context,Batch): raise Exception()
                return [(3,1,{'a':1}),(1,.5,{'a':2}),(2,1,{'a':3})][context]

        safe_learner = SafeLearner(MyLearner())

        #test initial call
        self.assertEqual(safe_learner.predict(Batch([0,1,2]), Batch([[1,2,3]]*3)), ([3,1,2],[1,.5,1],{'a':[1,2,3]}))
        #test shortcut logic after initial
        self.assertEqual(safe_learner.predict(Batch([0,1,2]), Batch([[1,2,3]]*3)), ([3,1,2],[1,.5,1],{'a':[1,2,3]}))

    def test_predict_not_batched_learner_first_exception_thrown(self):

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

    def test_AP_not_batched_learn_exception_with_info(self):

        calls = []
        class MyLearner:
            def learn(self,*args,**kwargs):
                if isinstance(args[0],Batch): raise Exception()
                calls.append((args,kwargs))

        safe_learner = SafeLearner(MyLearner())

        learn_args   = (Batch([0,1]), Batch([[1,2]]*2), [0,1], [1,2], [1,1])
        learn_kwargs = {'a':[1,2]}

        excpected_calls = [ ((0, [1,2], 0, 1, 1),{'a':1}), ((1, [1,2], 1, 2, 1),{'a':2}) ]

        #test initial call
        safe_learner.learn(*learn_args, **learn_kwargs)
        self.assertEqual(calls, excpected_calls)
        calls.clear()

        #test shortcut logic after initial
        safe_learner.learn(*learn_args, **learn_kwargs)
        self.assertEqual(calls, excpected_calls)

    def test_learn_learner(self):
        calls = []
        class TestLearner:
            def learn(self, context, actions, action, reward, probability):
                calls.append((context, actions, action, reward, probability))

        SafeLearner(TestLearner()).learn(1,[1,2],2,3,4)
        self.assertEqual(calls[0],(1,[1,2],2,3,4))

    def test_learn_kw_learner(self):
        class TestLearner:
            def learn(self, context, actions, action, reward, probability):
                pass
    
        with self.assertRaises(CobaException):
            SafeLearner(TestLearner()).learn(1,[1,2],2,3,4,a=2)

    def test_learn_learner_kw(self):
        class TestLearner:
            def learn(self, context, actions, action, reward, probability, a):
                pass
        with self.assertRaises(CobaException):
            SafeLearner(TestLearner()).learn(1,[1,2],2,3,4)

    def test_learn_kw_learner_kw(self):
        calls = []
        class TestLearner:
            def learn(self, context, actions, action, reward, probability,a):
                calls.append((context, actions, action, reward, probability,a))

        SafeLearner(TestLearner()).learn(1,[1,2],2,3,4,a=1)
        self.assertEqual(calls[0],(1,[1,2],2,3,4,1))

    def test_learn_exception(self):
        class BrokenLearnSignature:
            def learn(self, context):
                pass

        with self.assertRaises(Exception) as e:
            SafeLearner(BrokenLearnSignature()).learn(1,[1,2],2,3,4,**{})

        self.assertIn("takes 2 positional arguments but 6 were given", str(e.exception))

    def test_learn_batch(self):
        calls = []
        class TestLearner:
            def learn(self, context, actions, action, reward, probability,a):
                if isinstance(context,Batch): raise Exception()
                calls.append((context, actions, action, reward, probability,a))

        context = Batch([1,2])
        actions = Batch([[3,4],[5,6]])
        action  = Batch([3,5])
        reward  = Batch([1,0])
        probs   = Batch([.1,.9])
        a       = Batch([8,9])

        SafeLearner(TestLearner()).learn(context,actions,action,reward,probs,a=a)

        self.assertEqual(calls[0],(1,[3,4],3,1,.1,8))
        self.assertEqual(calls[1],(2,[5,6],5,0,.9,9))

    def test_request(self):
        class MyLearner:
            def request(self,context,actions,request):
                if context is None and actions == [1,2] and request == 2:
                    return 1

        self.assertEqual(SafeLearner(MyLearner()).request(None,[1,2],2), 1)

    def test_request_batch(self):
        calls = []
        class TestLearner:
            def request(self, context, actions, request):
                if isinstance(context,Batch): raise Exception()
                calls.append((context, actions, request))

        context = Batch([1,2])
        actions = Batch([[3,4],[5,6]])
        request = Batch([[3,4],[5,6]])

        SafeLearner(TestLearner()).request(context,actions,request)

        self.assertEqual(calls[0],(1,[3,4],[3,4]))
        self.assertEqual(calls[1],(2,[5,6],[5,6]))

    def test_request_not_implemented(self):
        class MyLearner:
            pass

        with self.assertRaises(CobaException) as ex:
            SafeLearner(MyLearner()).request(None,[],[])

        self.assertIn("`request`", str(ex.exception))

    def test_request_exception(self):
        class MyLearner:
            def request(self,context,actions,request):
                raise AttributeError("TEST")

        with self.assertRaises(AttributeError) as ex:
            SafeLearner(MyLearner()).request(None,[],[])

        self.assertIn("TEST", str(ex.exception))

if __name__ == '__main__':
    unittest.main()
