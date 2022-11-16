import unittest
import timeit
import importlib.util

from itertools import count
from typing import Callable,Any

import coba.pipes
import coba.random

from coba.learners import VowpalMediator, SafeLearner
from coba.utilities import HashableDict
from coba.environments import SimulatedInteraction, LinearSyntheticSimulation, ScaleReward, L1Reward
from coba.environments import Scale, Flatten, Grounded
from coba.encodings import NumericEncoder, OneHotEncoder, InteractionsEncoder
from coba.pipes import Reservoir, JsonEncode, Encode, ArffReader, Structure
from coba.pipes.rows import EncodeRow, DenseRow, SparseRow, DenseRow2, EncoderRow, HeaderRow, SelectRow
from coba.experiments.results import Result, moving_average
from coba.experiments import SimpleEvaluation

Timeable = Callable[[],Any]
Scalable = Callable[[int],Timeable]

sensitivity_scaling = 10

class Performance_Tests(unittest.TestCase):

    def test_numeric_encode_performance(self):
        encoder = NumericEncoder()
        items   = ["1"]*5

        self._assert_scale_time(items, encoder.encodes, .0017, False, number=1000)

    def test_onehot_fit_performance(self):
        encoder = OneHotEncoder()
        items   = list(range(10))

        self._assert_scale_time(items, encoder.fit, .01, False, number=1000)

    def test_onehot_encode_performance(self):
        encoder = OneHotEncoder(list(range(1000)), err_if_unknown=False)
        items   = [100,200,300,400,-1]*10
        self._assert_scale_time(items, encoder.encodes, .0025, False, number=1000)

    def test_encode_performance_row_scale(self):
        encoder = Encode(dict(enumerate([NumericEncoder()]*5)))
        row     = ['1.23']*5
        items   = [row]*5
        self._assert_scale_time(items, lambda x: list(encoder.filter(x)), .020, False, number=1000)

    def test_encode_performance_col_scale(self):
        encoder = Encode(dict(enumerate([NumericEncoder()]*5)))
        items   = ['1.23']*1000
        self._assert_scale_time(items, lambda x: list(encoder.filter([x])), .01, False, number=1000)

    def test_dense_interaction_x_encode_performance(self):
        encoder = InteractionsEncoder(["x"])
        x       = list(range(25))
        self._assert_scale_time(x, lambda x: encoder.encode(x=x), .045, False, number=1000)

    def test_dense_interaction_xx_encode_performance(self):
        encoder = InteractionsEncoder(["xx"])
        x       = list(range(10))
        self._assert_call_time(lambda: encoder.encode(x=x), .035, False, number=1000)

    def test_sparse_interaction_xx_encode_performance(self):
        encoder = InteractionsEncoder(["xx"])
        x       = dict(zip(map(str,range(10)), count()))
        self._assert_call_time(lambda: encoder.encode(x=x), .047, False, number=1000)

    def test_sparse_interaction_xxa_encode_performance(self):
        encoder = InteractionsEncoder(["xx"])
        x       = dict(zip(map(str,range(10)), count()))
        a       = [1,2,3]
        self._assert_call_time(lambda: encoder.encode(x=x,a=a), .049, False, number=1000)

    def test_sparse_interaction_abc_encode_performance(self):
        encoder = InteractionsEncoder(["aabc"])
        a       = dict(zip(map(str,range(5)), count()))
        b       = [1,2]
        c       = [2,3]
        self._assert_scale_time(c, lambda x: encoder.encode(a=a,b=b,c=x), .075, False, number=1000)

    def test_interaction_performance1(self):
        i = SimulatedInteraction(1, [1,2], 3)
        self._assert_call_time(lambda: SimulatedInteraction(1,[1,2],[0,1]), .013, False, number=10000)
        self._assert_call_time(lambda: SimulatedInteraction(i.context,i.actions,i.rewards), .009, False, number=10000)

    def test_hashable_dict_performance(self):
        items = list(enumerate(range(100)))
        self._assert_scale_time(items, HashableDict, .007, False, number=1000)

    def test_shuffle_performance(self):
        items = list(range(50))
        self._assert_scale_time(items, coba.random.shuffle, .023, False, number=1000)

    def test_randoms_performance(self):
        self._assert_scale_time(50, coba.random.randoms, .008, False, number=1000)

    def test_choice_performance(self):
        self._assert_scale_time([1]*50, coba.random.choice, .009, False, number=1000)

    def test_choice_performance_weights(self):
        items = [1]+[0]*49
        weights = [1]+[0]*49
        self._assert_call_time(lambda: coba.random.choice(items,weights), .009, False, number=1000)

    def test_gausses_performance(self):
        self._assert_scale_time(50, coba.random.gausses, .04, False, number=1000)

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW not installed.")
    def test_vowpal_mediator_make_example_sequence_str_performance(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --quiet",4)
        x = [ str(i) for i in range(100) ]

        self._assert_scale_time(x, lambda x:vw.make_example({'x':x}, None), .04, False, number=1000)

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW not installed.")
    def test_vowpal_mediator_make_example_highly_sparse_performance(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --quiet",4)

        ns = { 'x': [1]+[0]*1000  }
        self._assert_call_time(lambda:vw.make_example(ns, None), .03, False, number=1000)

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW not installed.")
    def test_vowpal_mediator_make_example_sequence_int_performance(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --quiet",4)
        x = list(range(100))

        self._assert_scale_time(x, lambda x:vw.make_example({'x':x}, None), .03, False, number=1000)

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW not installed.")
    def test_vowpal_mediator_make_example_sequence_dict_performance(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --quiet",4)

        ns = { 'x': { str(i):i for i in range(500)} }
        self._assert_call_time(lambda:vw.make_example(ns, None), .025, False, number=1000)

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW not installed.")
    def test_vowpal_mediator_make_examples_sequence_int_performance(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --quiet",4)

        shared   = { 'a': list(range(100))}
        separate = [{ 'x': list(range(25)) }, { 'x': list(range(25)) }]
        self._assert_call_time(lambda:vw.make_examples(shared, separate, None), .06, False, number=1000)

    def test_reservoir_performance(self):
        res = Reservoir(2,seed=1)
        x = list(range(500))

        self._assert_scale_time(x, lambda x:list(res.filter(x)), .04, False, number=1000)

    def test_jsonencode_performance(self):
        enc = JsonEncode()
        x = [[1.2,1.2],[1.2,1.2],{'a':1.,'b':1.}]*5
        self._assert_scale_time(x, enc.filter, .06, False, number=1000)

    def test_arffreader_performance(self):

        attributes = [f"@attribute {i} {{1,2}}" for i in range(3)]
        data_lines = [",".join(["1"]*3)]*50
        arff       = attributes+["@data"]

        reader = ArffReader()
        self._assert_scale_time(data_lines, lambda x:list(reader.filter(arff+x)), .035, False, number=100)

    def test_structure_performance(self):
        structure = Structure([None,2])        
        self._assert_call_time(lambda: list(structure.filter([[0,0,0] for _ in range(50)])), .05, False, number=1000)

    def test_linear_synthetic(self):
        self._assert_call_time(lambda:list(LinearSyntheticSimulation(10).read()), .075, False, number=1)

    def test_scale_target_features(self):
        items = [SimulatedInteraction((3193.0, 151.0, '0', '0', '0'),[1,2,3],[4,5,6])]*10
        scale = Scale("min","minmax",target="features")
        self._assert_scale_time(items, lambda x:list(scale.filter(x)), .05, False, number=1000)

    def test_scale_target_rewards(self):
        items = [SimulatedInteraction((3193.0, 151.0),[1,2,3],[4,5,6])]*10
        scale = Scale("min","minmax",target="rewards")
        self._assert_scale_time(items, lambda x:list(scale.filter(x)), .12, False, number=1000)

    def test_environments_flat_tuple(self):
        items = [SimulatedInteraction([1,2,3,4]+[(0,1)]*3,[1,2,3],[4,5,6])]*10
        flat  = Flatten()
        self._assert_scale_time(items, lambda x:list(flat.filter(x)), .08, False, number=1000)

    def test_pipes_flat_tuple(self):
        items = [tuple([1,2,3]+[(0,1)]*5)]*10
        flat  = coba.pipes.Flatten()
        self._assert_scale_time(items, lambda x:list(flat.filter(x)), .02, False, number=1000)

    def test_pipes_flat_dict(self):
        items = [dict(enumerate([1,2,3]+[(0,1)]*5))]*10
        flat  = coba.pipes.Flatten()
        self._assert_scale_time(items, lambda x:list(flat.filter(x)), .035, False, number=1000)

    def test_result_filter_env(self):
        envs = { k:{'mod': k%100} for k in range(5) }
        lrns = { 1:{}, 2:{}, 3:{}}
        ints = { (e,l):{} for e in envs.keys() for l in lrns.keys() }
        res  = Result(envs, lrns, ints)
        self._assert_call_time(lambda:res.filter_env(mod=3), .05, False, number=1000)

    def test_moving_average_sliding_window(self):
        items = [1,0]*100
        self._assert_scale_time(items, lambda x:list(moving_average(x,span=2)), .025, False, number=1000)

    def test_moving_average_rolling_window(self):
        items = [1,0]*300
        self._assert_scale_time(items, lambda x:list(moving_average(x)), .07, False, number=1000)

    def test_encoder_row(self):
        R = next(EncodeRow([int]*100).filter([DenseRow(loaded=['1']*100)]))
        self._assert_call_time(lambda:R[1], .0009, True, number=1000)
        
        r2 = EncoderRow(DenseRow2(loaded=['1']*100),[int]*100)
        self._assert_call_time(lambda:r2[1], .0009, True, number=1000)

    def test_dense_row_to_builtin(self):
        r = next(EncodeRow([int]*100).filter([DenseRow(loaded=['1']*100)]))
        self._assert_call_time(lambda:r.to_builtin(), .023, True, number=1000)

        r2 = EncoderRow(DenseRow2(loaded=['1']*100),[int]*100)
        self._assert_call_time(lambda:list(r2), .023, True, number=1000)

    def test_headers(self):
        headers = list(map(str,range(100)))
        r2 = HeaderRow(DenseRow2(loaded=['1']*100),dict(zip(headers,count())))
        r3 = EncoderRow(r2,[int]*100)
        r4 = EncoderRow(r3,[int]*100)

        self._assert_call_time(lambda:hasattr(r2,'headers'), .023, True, number=1000)
        self._assert_call_time(lambda:hasattr(r3,'headers'), .023, True, number=1000)
        self._assert_call_time(lambda:hasattr(r4,'headers'), .023, True, number=1000)

    def test_del(self):
        r1 = next(EncodeRow([int]*100).filter([DenseRow(loaded=['1']*100)]))
        r2 = EncoderRow(DenseRow2(loaded=['1']*100), [int]*100)

        del r1[99]
        r2 = SelectRow(r2,tuple(range(99)))

        self._assert_call_time(lambda:r1[3], .023, True, number=1000)
        self._assert_call_time(lambda:r2[3], .023, True, number=1000)

    def test_sparse_row_to_builtin(self):
        r = SparseRow(loaded=dict(enumerate(['1']*100)))
        r.encoders = dict(enumerate([int]*100))
        self._assert_call_time(lambda:r.to_builtin(), .04, False, number=1000)

    def test_to_grounded_interaction(self):
        items    = [SimulatedInteraction(1, [1,2,3,4], [0,0,0,1])]*10
        grounded = Grounded(10,5,10,5,1)

        self._assert_scale_time(items,lambda x:list(grounded.filter(x)), .11, False, number=1000)

    def test_simple_evaluation(self):

        class DummyLearner:
            def predict(*args):
                return [1,0,0]
            def learn(*args):
                pass

        items = [SimulatedInteraction(1,[1,2,3],[1,2,3])]*100
        eval  = SimpleEvaluation()
        learn = DummyLearner()
        self._assert_scale_time(items,lambda x:list(eval.process(learn, x)), .06, False, number=100)

    def test_safe_learner_predict(self):
        
        class DummyLearner:
            def predict(*args):return [1,0,0]
            def learn(*args): pass

        learn = SafeLearner(DummyLearner())        
        self._assert_call_time(lambda:learn.predict(1,[1,2,3]), .0035, False, number=1000)        

    def test_safe_learner_learn(self):
        
        class DummyLearner:
            def predict(*args):return [1,0,0]
            def learn(*args): pass

        learn = SafeLearner(DummyLearner())        
        self._assert_call_time(lambda:learn.learn(1,[1,2,3], [1], 1, .5, {}), .0006, False, number=1000)        

    def test_scale_reward(self):
        reward = ScaleReward(L1Reward(1), 1, 2, "argmax")
        self._assert_call_time(lambda: reward.eval(4), .04, False, number=100000)

    def _assert_call_time(self, timeable: Timeable, expected:float, print_time:bool, *, number:int=1000, setup="pass") -> None:
        if print_time: print()
        self._z_assert_less(timeit.repeat(timeable, setup=setup, number=1, repeat=number), expected, print_time)
        if print_time: print()

    def _assert_scale_time(self, items: list, func, expected:float, print_time:bool, *, number:int=1000, setup="pass") -> None:
        if print_time: print()
        items_1 = items*1
        items_2 = items*2
        self._z_assert_less(timeit.repeat(lambda: func(items_1), setup=setup, number=1, repeat=number), 1.0*expected, print_time)
        self._z_assert_less(timeit.repeat(lambda: func(items_2), setup=setup, number=1, repeat=number), 2.2*expected, print_time)
        if print_time: print()

    def _z_assert_less(self, samples, expected, print_it):
        from statistics import stdev
        from math import sqrt

        if not print_it:
            expected = sensitivity_scaling*expected

        if len(samples) > 1:
            stderr = sqrt(len(samples))*stdev(samples) #this is the stderr of the total not the avg
        else:
            stderr = 1

        actual = round(sum(samples),5)
        zscore = round((actual-expected)/stderr,5)    
        if print_it:
            print(f"{actual:5.5f}     {expected:5.5f}     {zscore:5.5f}")
        else: 
            self.assertLess(zscore, 1.645)

if __name__ == '__main__':
    unittest.main()
