import unittest

from coba.exceptions import CobaException
from coba.primitives import is_batch
from coba.learners.safety import SafeLearner
from coba.learners.safety import has_kwargs, first_row, pred_format, batch_order, possible_action, possible_pmf

class Batch(list):
    is_batch=True

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
    def __init__(self, pred, pred_kw):
        self._pred = pred
        self._pred_kw = pred_kw

    def predict(self, context, actions):
        if self._pred_kw is None:
            return self._pred
        else:
            return self._pred, self._pred_kw

class AmbiguousPredictionLearner:

    def predict(self, context, actions):
        return [1,0]

    def learn(self, context, actions, action, reward, probability) -> None:
        pass

class batch_order_Tests(unittest.TestCase):

    def test_explicit_PM_not(self):
        self.assertEqual(batch_order(None, {'pmf':1}, None, None), 'not')

    def test_explicit_PM_not_kw(self):
        self.assertEqual(batch_order(None, [{'pmf':1},{}], None, None), 'not')

    def test_PM_not(self):
        self.assertEqual(batch_order(None, (0,1), None, None), 'not')

    def test_PM_not_kw(self):
        self.assertEqual(batch_order(None, ([0,1],{}), None, None), 'not')

    def test_explicit_AX_not(self):
        self.assertEqual(batch_order(None, {'action':1}, None, None), 'not')

    def test_explicit_AX_not_kw(self):
        self.assertEqual(batch_order(None, [{'action':1},{}], None, None), 'not')

    def test_AX_not(self):
        self.assertEqual(batch_order(None, 'a', None, None), 'not')

    def test_AX_not_kw(self):
        self.assertEqual(batch_order(None, ('a',{}), None, None), 'not')

    def test_explicit_AP_not(self):
        self.assertEqual(batch_order(None, {'action_prob':1}, None, None), 'not')

    def test_explicit_AP_not_kw(self):
        self.assertEqual(batch_order(None, [{'action_prob':1},{}], None, None), 'not')

    def test_AP_not(self):
        self.assertEqual(batch_order(None, (0,1), None, None), 'not')

    def test_AP_not_kw(self):
        self.assertEqual(batch_order(None, (0,1,{}), None, None), 'not')

    def test_explicit_PM_row(self):
        self.assertEqual(batch_order(None, [{'pmf':(1/2,1/4,1/4)}], Batch([1]), Batch([[1,2,3]])), 'row')

    def test_explicit_PM_row_kw(self):
        self.assertEqual(batch_order(None, [[{'pmf':(1/2,1/4,1/4)},{}]], Batch([1]), Batch([[1,2,3]])), 'row')

    def test_PM_row(self):
        self.assertEqual(batch_order(None, [(1/2,1/2)], Batch([1]), Batch([[1,2]])), 'row')

    def test_PM_row_kw(self):
        self.assertEqual(batch_order(None, [((1/2,1/4,1/4),{})], Batch([1]), Batch([[1,2,3]])), 'row')

    def test_explicit_AX_row(self):
        self.assertEqual(batch_order(None, [{'action':1}], Batch([1]), Batch([[1,2,3]])), 'row')

    def test_explicit_AX_row_kw(self):
        self.assertEqual(batch_order(None, [[{'action':1},{}]],Batch([1]), Batch([[1,2,3]])), 'row')

    def test_AX_row(self):
        self.assertEqual(batch_order(None, [0], Batch([1]), Batch([[1,2,3]])), 'row')

    def test_AX_row_kw(self):
        self.assertEqual(batch_order(None, [(0,{})], Batch([1]), Batch([[1,2,3]])), 'row')

    def test_explicit_AP_row(self):
        self.assertEqual(batch_order(None, [{'action_prob':1}], Batch([1]), Batch([[1,2,3]])), 'row')

    def test_explicit_AP_row_kw(self):
        self.assertEqual(batch_order(None, [[{'action_prob':1},{}]], Batch([1]), Batch([[1,2,3]])), 'row')

    def test_AP_row(self):
        self.assertEqual(batch_order(None, [(1,1)], Batch([1]), Batch([1])), 'row')

    def test_AP_row_kw(self):
        self.assertEqual(batch_order(None, [(1,1,{})], Batch([1]), Batch([1])), 'row')

    def test_explicit_PM_col(self):
        self.assertEqual(batch_order(None, {'pmf':[1]}, Batch([1]), Batch([[1,2,3]])), 'col')

    def test_explicit_PM_col_kw(self):
        self.assertEqual(batch_order(None, [{'pmf':[1]},{}], Batch([1]), Batch([[1,2,3]])), 'col')

    def test_PM_col(self):
        self.assertEqual(batch_order(None, [[0],[1],[0]], Batch([1]), Batch([[1,2,3]])), 'col')

    def test_PM_col_kw(self):
        self.assertEqual(batch_order(None, [[0],[1],[0],{}], Batch([1]), Batch([[1,2,3]])), 'col')

    def test_explicit_AX_col(self):
        self.assertEqual(batch_order(None, {'action':[1]}, Batch([1]), Batch([[1,2,3]])), 'col')

    def test_explicit_AX_col_kw(self):
        self.assertEqual(batch_order(None, [{'action':[1]},{}], Batch([1]), Batch([[1,2,3]])), 'col')

    #def test_AX_col(self): # this is identical to row so we process it as a row
    #    self.assertEqual(batch_order(None, [0], Batch([1]), Batch([[1,2,3]])), 'col')

    def test_AX_col_kw(self):
        self.assertEqual(batch_order(None, ([0],{}), Batch([1]), Batch([[1,2,3]])), 'col')

    def test_explicit_AP_col(self):
        self.assertEqual(batch_order(None, {'action_prob':[1]}, Batch([1]), Batch([[1,2,3]])), 'col')

    def test_explicit_AP_col_kw(self):
        self.assertEqual(batch_order(None, [{'action_prob':[1]},{}], Batch([1]), Batch([[1,2,3]])), 'col')

    def test_AP_col(self):
        self.assertEqual(batch_order(None, [[0],[1]], Batch([1]), Batch([1])), 'col')

    def test_AP_col_kw(self):
        self.assertEqual(batch_order(None, [[0],[1],{}], Batch([1]), Batch([1])), 'col')

    def test_PM_matching_dims_col(self):
        self.assertEqual(batch_order(lambda x,a:[[0],[1]], [[0,0],[1,1]], Batch([1,1]), Batch([[1,2],[1,2]])), 'col')

    def test_PM_matching_dims_row(self):
        self.assertEqual(batch_order(lambda x,a:[[0,1]], [[0,0],[1,1]], Batch([1,1]), Batch([[1,2],[1,2]])), 'row')

    def test_simple(self):
        pred = {'action_prob': [(1,2),(3,4)]}
        self.assertEqual(batch_order(None, pred, Batch([1,1]), [[],[]]), 'col')

class has_kwargs_Tests(unittest.TestCase):
    def test_batch_not(self):
        self.assertTrue(has_kwargs((0,1,{}), 'not'))
        self.assertTrue(has_kwargs((0,{}), 'not'))
        self.assertTrue(has_kwargs(([.25,.25,.5],{}), 'not'))
        self.assertTrue(has_kwargs(({'action':1},{}), 'not'))
        self.assertFalse(has_kwargs((0,1),'not'))
        self.assertFalse(has_kwargs(0,'not'))
        self.assertFalse(has_kwargs([.25,.25,.5],'not'))
        self.assertFalse(has_kwargs({'action':1}, 'not'))

    def test_batch_row(self):
        self.assertTrue(has_kwargs([(0,1,{})], 'row'))
        self.assertTrue(has_kwargs([(0,{})], 'row'))
        self.assertTrue(has_kwargs([([.25,.25,.5],{})], 'row'))
        self.assertTrue(has_kwargs([({'action':1},{})], 'row'))
        self.assertFalse(has_kwargs([(0,1)],'row'))
        self.assertFalse(has_kwargs([0],'row'))
        self.assertFalse(has_kwargs([[.25,.25,.5]],'row'))
        self.assertFalse(has_kwargs([{'action':1}], 'row'))

    def test_batch_col(self):
        self.assertTrue(has_kwargs(([0],[1],{}), 'col'))
        self.assertTrue(has_kwargs(([0],{}), 'col'))
        self.assertTrue(has_kwargs(([[.25,.25,.5]],{}), 'col'))
        self.assertTrue(has_kwargs([{'action':[1]},{}], 'col'))
        self.assertFalse(has_kwargs(([0],[1]),'col'))
        self.assertFalse(has_kwargs([0],'col'))
        self.assertFalse(has_kwargs([[.25,.25,.5]],'col'))
        self.assertFalse(has_kwargs({'action':[1]},'col'))

class first_row_Tests(unittest.TestCase):

    def test_no_batch_no_kwargs(self):
        self.assertEqual(0      , first_row(0, 'not', False))
        self.assertEqual((0,1)  , first_row((0,1), 'not', False))
        self.assertEqual([0,1,0], first_row([0,1,0], 'not', False))
        self.assertEqual({'b':1}, first_row({'b':1}, 'not', False))

    def test_no_batch_kwargs(self):
        self.assertEqual( 0      , first_row((0,{'a':1}), 'not', True))
        self.assertEqual( (0,1)  , first_row((0,1,{'a':1}), 'not', True))
        self.assertEqual( [0,1,0], first_row(([0,1,0],{'a':1}), 'not', True))
        self.assertEqual( {'b':1}, first_row(({'b':1},{'a':1}), 'not', True))

    def test_row_batch_no_kwargs(self):
        self.assertEqual( 0                , first_row([0,1], 'row', False))
        self.assertEqual( [2,0]            , first_row(([2,0],[1,1]), 'row', False))
        self.assertEqual( {'action':1}     , first_row([{'action':1},{'action':2}], 'row', False))
        self.assertEqual( {'pmf':1}        , first_row([{'pmf':1},{'pmf':2}], 'row', False))
        self.assertEqual( {'action_prob':1}, first_row([{'action_prob':1},{'action_prob':2}], 'row', False))

    def test_row_batch_kwargs(self):
        self.assertEqual( 0                , first_row([(0,{}),(1,{})], 'row', True))
        self.assertEqual( [2,0]            , first_row(([2,0,{}],[1,1,{}]), 'row', True))
        self.assertEqual( [1,0,1]          , first_row([[[1,0,1],{}]], 'row', True))
        self.assertEqual( {'action':1}     , first_row([({'action':1},{}),({'action':2},{})], 'row', True))
        self.assertEqual( {'pmf':1}        , first_row([({'pmf':1},{}),({'pmf':2},{})], 'row', True))
        self.assertEqual( {'action_prob':1}, first_row([({'action_prob':1},{}),({'action_prob':2},{})], 'row', True))

    def test_col_batch_no_kwargs(self):
        #the top test would be marked as 'row' so we don't need to handle it
        #self.assertEqual( 0        ,first_row([0,1], 'col', False))
        self.assertEqual( [2,1]            , first_row(([2,0],[1,1]), 'col', False))
        self.assertEqual( {'action':2}     , first_row({'action':[2,1]}, 'col', False))
        self.assertEqual( {'pmf':1}        , first_row({'pmf':[1,2]}, 'col', False))
        self.assertEqual( {'action_prob':1}, first_row({'action_prob':[1,2]}, 'col', False))

    def test_col_batch_kwargs(self):
        self.assertEqual(  0   , first_row([(0,1),{}]      , 'col', True))
        self.assertEqual( [2,0], first_row(([2,1],[0,1],{}), 'col', True))
        self.assertEqual( {'action':2}     , first_row([{'action':[2,1]},{}], 'col', True))
        self.assertEqual( {'pmf':1}        , first_row([{'pmf':[1,2]},{}], 'col', True))
        self.assertEqual( {'action_prob':1}, first_row([{'action_prob':[1,2]},{}], 'col', True))

class pred_format_Tests(unittest.TestCase):

    def test_PM_size3_explicit(self):
        actual = pred_format({'pmf':[1,0,0]},['a','b','c'])
        expected = 'PM*'
        self.assertEqual(actual,expected)

    def test_PM_size3_list_and_pmf_not_in_actions(self):
        actual = pred_format([1,0,0],['a','b','c'])
        expected = 'PM'
        self.assertEqual(actual,expected)

    def test_PM_size3_tuple_and_pmf_not_in_actions(self):
        actual = pred_format((1,0,0),['a','b','c'])
        expected = 'PM'
        self.assertEqual(actual,expected)

    def test_PM_size3_tuple_and_pmf_in_actions(self):
        pmf = tuple([1,0,0]) # see, https://stackoverflow.com/a/34147516/1066291 for why
        actual = pred_format(pmf,[(1,0,0),(0,1,0),(0,0,1)])
        expected = 'PM'
        self.assertEqual(actual,expected)

    def test_PM_size2_and_pmf0_in_actions(self):
        actual = pred_format((1,0),[0.,1.])
        expected = 'PM'
        self.assertEqual(actual,expected)

    def test_PM_size2_and_pmf_in_actions(self):
        pmf = tuple([1,0]) # see, https://stackoverflow.com/a/34147516/1066291 for why
        actual = pred_format(pmf,[(1,0),(0,1)])
        expected = 'PM'
        self.assertEqual(actual,expected)

    def test_PM_size2_and_pmf_not_in_actions(self):
        actual = pred_format((1,0),['a','b'])
        expected = 'PM'
        self.assertEqual(actual,expected)

    def test_AP_and_AP_valid_PMF(self):
        actual = pred_format((0.,1),[0.,1])
        expected = 'AP'
        self.assertEqual(actual,expected)

    def test_AP_explicit(self):
        actual = pred_format({'action_prob':('a',1)},['a','b','c'])
        expected = 'AP*'
        self.assertEqual(actual,expected)

    def test_AP(self):
        actual = pred_format(('a',1),['a','b','c'])
        expected = 'AP'
        self.assertEqual(actual,expected)

    def test_AP_sum_to_1(self):
        actual = pred_format((2,-1),[2,1])
        expected = 'AP'
        self.assertEqual(actual,expected)

    def test_AX_explicit(self):
        actual = pred_format({'action':'a'},['a','b','c'])
        expected = 'AX*'
        self.assertEqual(actual,expected)

    def test_AX_character(self):
        actual = pred_format('a',['a','b','c'])
        expected = 'AX'
        self.assertEqual(actual,expected)

    def test_AX_string(self):
        actual = pred_format('ab',['ab','ac','ad'])
        expected = 'AX'
        self.assertEqual(actual,expected)

    def test_AX_numeric(self):
        actual = pred_format(8,[6,7,8])
        expected = 'AX'
        self.assertEqual(actual,expected)

    def test_AX_empty_actions(self):
        actual = pred_format(8,[])
        expected = 'AX'
        self.assertEqual(actual,expected)

    def test_AX_none_actions(self):
        actual = pred_format(8,None)
        expected = 'AX'
        self.assertEqual(actual,expected)

    def test_AP_and_action_not_in_actions(self):
        with self.assertRaises(CobaException):
            pred_format(('a',0),[0,1])

    def test_uncertain_empty_actions(self):
        with self.assertRaises(CobaException):
            pred_format([8,2],[])

    def test_uncertain_none_actions(self):
        with self.assertRaises(CobaException):
            pred_format([8,2],None)

    def test_bad_explicit_action_prob_len(self):
        with self.assertRaises(CobaException):
            pred_format({'action_prob':1},None)
        with self.assertRaises(CobaException):
            pred_format({'action_prob':(1,2,3)},None)

    def test_bad_explicit_pmf_len(self):
        with self.assertRaises(CobaException):
            pred_format({'pmf':[1,2,3]},[1,2])
        with self.assertRaises(CobaException):
            pred_format({'pmf':1},[1,2])

    def test_bad_explicit_pmf_actions(self):
        with self.assertRaises(CobaException):
            pred_format({'pmf':[1,2,3]},None)
        with self.assertRaises(CobaException):
            pred_format({'pmf':[1,2,3]},[])

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

    def test_predict_AX_batchnot_no_kw(self):
        actions = [1.,2]
        learner = SafeLearner(UnsafeFixedLearner(actions[0], None))
        predict = learner.predict(None, actions)

        self.assertEqual(1   , predict[0])
        self.assertEqual(None, predict[1])
        self.assertEqual({}  , predict[2])

    def test_predict_AX_batchnot_kw(self):
        actions = [1.,2]
        learner = SafeLearner(UnsafeFixedLearner(actions[0], {'a':1}))
        predict = learner.predict(None, actions)

        self.assertEqual(1      , predict[0])
        self.assertEqual(None   , predict[1])
        self.assertEqual({'a':1}, predict[2])

    def test_predict_AX_batchnot_no_is(self):
        actions = [3,2]
        learner = SafeLearner(UnsafeFixedLearner(3., {'a':1}))
        predict = learner.predict(None, actions)

        self.assertEqual(3      , predict[0])
        self.assertEqual(None   , predict[1])
        self.assertEqual({'a':1}, predict[2])

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
                return actions[0],.5

        self.assertEqual(SafeLearner(MyLearner()).predict(None,[1,2,3]), (1,.5,{}))

    def test_predict_AP_batchnot_kw(self):
        class MyLearner:
            def predict(self,context,actions):
                return actions[0],.5,{'a':1}

        self.assertEqual(SafeLearner(MyLearner()).predict(None,[1,2,3]), (1,.5,{'a':1}))

    def test_predict_AP_batchcol(self):
        class MyLearner:
            def predict(self,context,actions):
                return [(actions[0][2],actions[1][0],actions[2][1]),(1,.5,1)]

        self.assertEqual(SafeLearner(MyLearner()).predict(Batch([None]*3), Batch([[1,2,3]]*3)), ((3,1,2),(1,.5,1),{}))

    def test_predict_AP_batchrow_kw(self):
        class MyLearner:
            def predict(self,context,actions):
                if len(actions) == 1:
                    return [(actions[0][2],1,{})]
                else:
                    return [(actions[0][2],1,{}),(actions[1][0],.5,{}),(actions[2][1],1,{})]

        self.assertEqual(SafeLearner(MyLearner()).predict(Batch([None]*3), Batch([[1,2,3]]*3)), ((3,1,2),(1,.5,1),{}))

    def test_predict_AP_batchrow(self):
        class MyLearner:
            def predict(self,context,actions):
                return [(3,1),(1,.5),(2,1)]

        self.assertEqual(SafeLearner(MyLearner()).predict(Batch([None]*3), Batch([[1,2,3]]*3)), ((3,1,2),(1,.5,1),{}))

    def test_predict_AX_batchrow_no_kw(self):
        class MyLearner:
            def predict(self,context,actions):
                return [actions[0][2],actions[1][0],actions[2][1]]

        self.assertEqual(SafeLearner(MyLearner()).predict(Batch([None]*3), Batch([[1,2,3]]*3)), ([3,1,2],[None,None,None],{}) )

    def test_fallback_safety_call_not_performed_when_no_batch(self):
        class MyException(Exception):
            pass

        class MyLearner:
            def predict(self,context,actions):
                raise MyException()

        with self.assertRaises(MyException):
            SafeLearner(MyLearner()).predict('abc',[])

    def test_predict_throws_coba_exception_when_str_context(self):
        class MyException(Exception):
            pass

        class MyLearner:
            def predict(self,context,actions):
                raise MyException()

        with self.assertRaises(CobaException) as e:
            SafeLearner(MyLearner()).predict('abc',Batch([]))

        self.assertIsInstance(e.exception.__cause__,MyException)

    def test_predict_throws_coba_exception_when_empty_actions(self):
        class MyException(Exception):
            pass

        class MyLearner:
            def predict(self,context,actions):
                raise MyException()

        with self.assertRaises(CobaException) as e:
            SafeLearner(MyLearner()).predict(['abc'],Batch([]))

        self.assertIsInstance(e.exception.__cause__,MyException)

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
                if is_batch(context):
                    raise Exception()
                return [(3,1,{'a':1}),(1,.5,{'a':2}),(2,1,{'a':3})][context]

        safe_learner = SafeLearner(MyLearner())

        #test initial call
        self.assertEqual(safe_learner.predict(Batch([0,1,2]), Batch([[1,2,3]]*3)), ((3,1,2),(1,.5,1),{'a':[1,2,3]}))
        #test shortcut logic after initial
        self.assertEqual(safe_learner.predict(Batch([0,1,2]), Batch([[1,2,3]]*3)), ((3,1,2),(1,.5,1),{'a':[1,2,3]}))

    def test_predict_not_batched_learner_first_exception_thrown(self):

        class MyLearner:
            calls = []
            def predict(self,*args,**kwargs):
                if is_batch(args[0]):
                    raise Exception("1")
                else:
                    raise Exception("2")

        learner = MyLearner()
        safe_learner = SafeLearner(learner)

        with self.assertRaises(Exception) as e:
            safe_learner.predict(Batch([1,2,3]), Batch([1,2,3]))

        self.assertEqual(str(e.exception),"2")
        self.assertEqual(str(e.exception.__cause__),"1")

    def test_AP_not_batched_learn_exception_with_info(self):

        calls = []
        class MyLearner:
            def learn(self,*args,**kwargs):
                if is_batch(args[0]):
                    raise Exception()
                calls.append((args,kwargs))

        safe_learner = SafeLearner(MyLearner())

        learn_args   = (Batch([0,1]), [0,1], [1,2], [1,1])
        learn_kwargs = {'a':[1,2]}

        excpected_calls = [ ((0, 0, 1, 1),{'a':1}), ((1, 1, 2, 1),{'a':2}) ]

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
            def learn(self, context, action, reward, probability):
                calls.append((context, action, reward, probability))

        SafeLearner(TestLearner()).learn(1,2,3,4)
        self.assertEqual(calls[0],(1,2,3,4))

    def test_learn_kw_learner(self):
        class TestLearner:
            def learn(self, context, action, reward, probability):
                pass

        with self.assertRaises(CobaException):
            SafeLearner(TestLearner()).learn(1,2,3,4,a=2)

    def test_learn_learner_kw(self):
        class TestLearner:
            def learn(self, context, action, reward, probability, a):
                pass
        with self.assertRaises(CobaException):
            SafeLearner(TestLearner()).learn(1,2,3,4)

    def test_learn_kw_learner_kw(self):
        calls = []
        class TestLearner:
            def learn(self, context, action, reward, probability,a):
                calls.append((context, action, reward, probability,a))

        SafeLearner(TestLearner()).learn(1,2,3,4,a=1)
        self.assertEqual(calls[0],(1,2,3,4,1))

    def test_learn_exception(self):
        class BrokenLearnSignature:
            def learn(self, context):
                pass

        with self.assertRaises(Exception) as e:
            SafeLearner(BrokenLearnSignature()).learn(1,2,3,4,**{})

        self.assertIn("takes 2 positional arguments but 5 were given", str(e.exception))

    def test_learn_batch(self):
        calls = []
        class TestLearner:
            def learn(self, context, action, reward, probability,a):
                if is_batch(context):
                    raise Exception()
                calls.append((context, action, reward, probability,a))

        context = Batch([1,2])
        action  = Batch([3,5])
        reward  = Batch([1,0])
        probs   = Batch([.1,.9])
        a       = Batch([8,9])

        SafeLearner(TestLearner()).learn(context,action,reward,probs,a=a)

        self.assertEqual(calls[0],(1,3,1,.1,8))
        self.assertEqual(calls[1],(2,5,0,.9,9))

    def test_score(self):
        class MyLearner:
            def score(self,context,actions,action):
                if context is None and actions == [1,2] and action == 2:
                    return 1

        self.assertEqual(SafeLearner(MyLearner()).score(None,[1,2],2), 1)

    def test_score_batch(self):
        calls = []
        class TestLearner:
            def score(self, context, actions, action):
                if is_batch(context): raise Exception()
                calls.append((context, actions, action))
                return .1 if action == 3 else .5 if action == 5 else None

        context = Batch([1,2])
        actions = Batch([[3,4],[5,6]])
        action  = Batch([3,5])

        calls.clear()
        out = SafeLearner(TestLearner()).score(context,actions,action)
        self.assertEqual(calls[0],(1,[3,4],3))
        self.assertEqual(calls[1],(2,[5,6],5))
        self.assertEqual(out,[.1,.5])

        calls.clear()
        out = SafeLearner(TestLearner()).score(context,actions)
        self.assertEqual(calls[0],(1,[3,4],None))
        self.assertEqual(calls[1],(2,[5,6],None))

    def test_score_not_implemented(self):
        class MyLearner:
            pass

        with self.assertRaises(CobaException) as ex:
            SafeLearner(MyLearner()).score(None,[],[])

        self.assertIn("`score`", str(ex.exception))

    def test_score_exception(self):
        class MyLearner:
            def score(self,context,actions,action):
                raise AttributeError("TEST")

        with self.assertRaises(AttributeError) as ex:
            SafeLearner(MyLearner()).score(None,[],[])

        self.assertIn("TEST", str(ex.exception))

    def test_predict_explicit_AX_batchcol(self):
        #this breaks the interface patter but is provided
        #anyway as a convenience to users... We assume this
        #will be the default behavior.
        class MyLearner:
            def predict(self,context,actions):
                return {'action':[(3,1),(1,.5),(2,1)]}

        self.assertEqual(SafeLearner(MyLearner()).predict(Batch([None]*3), [[],[],[]]), ([(3,1),(1,.5),(2,1)],[None,None,None],{}))

    def test_predict_explicit_AP_batchcol(self):
        #this breaks the interface patter but is provided
        #anyway as a convenience to users... We assume this
        #will be the default behavior.
        class MyLearner:
            def predict(self,context,actions):
                return {'action_prob':[(3,1),(1,.5),(2,1)]}

        self.assertEqual(SafeLearner(MyLearner()).predict(Batch([None]*3), [[],[],[]]), ((3,1,2),(1,.5,1),{}))

    def test_predict_explicit_AP_batchrow(self):
        #this breaks the interface patter but is provided
        #anyway as a convenience to users... We assume this
        #will be the default behavior.
        class MyLearner:
            def predict(self,context,actions):
                return {'action_prob':(3,1)},{'action_prob':(1,.5)},{'action_prob':(2,1)}

        self.assertEqual(SafeLearner(MyLearner()).predict(Batch([None]*3), [[],[],[]]), ((3,1,2),(1,.5,1),{}))

    def test_predict_explicit_AP_batchnot(self):
        #this breaks the interface patter but is provided
        #anyway as a convenience to users... We assume this
        #will be the default behavior.
        class MyLearner:
            def predict(self,context,actions):
                return {'action_prob':(3,1)}

        self.assertEqual(SafeLearner(MyLearner()).predict(None,[]), (3,1,{}))


if __name__ == '__main__':
    unittest.main()
