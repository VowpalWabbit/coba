import unittest
import timeit
import importlib.util

from itertools import count
from typing import Callable, Any

import coba.pipes
import coba.random

from coba.learners import VowpalMediator, SafeLearner
from coba.environments import SimulatedInteraction, LinearSyntheticSimulation
from coba.environments import Scale, Flatten, Grounded, Chunk
from coba.encodings import NumericEncoder, OneHotEncoder, InteractionsEncoder
from coba.pipes import Reservoir, JsonEncode, Encode, ArffReader, Structure
from coba.pipes.rows import LazyDense, LazySparse, EncodeDense, KeepDense, HeadDense, LabelDense, EncodeCatRows
from coba.experiments.results import Result, moving_average
from coba.experiments import SimpleEvaluation
from coba.primitives import Categorical, HashableSparse, ScaleReward, L1Reward

Timeable = Callable[[],Any]
Scalable = Callable[[int],Timeable]

sensitivity_scaling = 10
print_time = False

class Performance_Tests(unittest.TestCase):

    def test_numeric_encode_performance(self):
        encoder = NumericEncoder()
        items   = ["1"]*5

        self._assert_scale_time(items, encoder.encodes, .0017, print_time, number=1000)

    def test_onehot_fit_performance(self):
        encoder = OneHotEncoder()
        items   = list(range(10))

        self._assert_scale_time(items, encoder.fit, .01, print_time, number=1000)

    def test_onehot_encode_performance(self):
        encoder = OneHotEncoder(list(range(1000)), err_if_unknown=False)
        items   = [100,200,300,400,-1]*10
        self._assert_scale_time(items, encoder.encodes, .0025, print_time, number=1000)

    def test_encode_performance_row_scale(self):
        encoder = Encode(dict(enumerate([NumericEncoder()]*5)))
        row     = ['1.23']*5
        items   = [row]*5
        self._assert_scale_time(items, lambda x: list(encoder.filter(x)), .020, print_time, number=1000)

    def test_encode_performance_col_scale(self):
        encoder = Encode(dict(enumerate([NumericEncoder()]*5)))
        items   = ['1.23']*1000
        self._assert_scale_time(items, lambda x: list(encoder.filter([x])), .01, print_time, number=1000)

    def test_dense_interaction_x_encode_performance(self):
        encoder = InteractionsEncoder(["x"])
        x       = list(range(25))
        self._assert_scale_time(x, lambda x: encoder.encode(x=x), .045, print_time, number=1000)

    def test_dense_interaction_xx_encode_performance(self):
        encoder = InteractionsEncoder(["xx"])
        x       = list(range(10))
        self._assert_call_time(lambda: encoder.encode(x=x), .035, print_time, number=1000)

    def test_sparse_interaction_xx_encode_performance(self):
        encoder = InteractionsEncoder(["xx"])
        x       = dict(zip(map(str,range(10)), count()))
        self._assert_call_time(lambda: encoder.encode(x=x), .047, print_time, number=1000)

    def test_sparse_interaction_xxa_encode_performance(self):
        encoder = InteractionsEncoder(["xx"])
        x       = dict(zip(map(str,range(10)), count()))
        a       = [1,2,3]
        self._assert_call_time(lambda: encoder.encode(x=x,a=a), .049, print_time, number=1000)

    def test_sparse_interaction_abc_encode_performance(self):
        encoder = InteractionsEncoder(["aabc"])
        a       = dict(zip(map(str,range(5)), count()))
        b       = [1,2]
        c       = [2,3]
        self._assert_scale_time(c, lambda x: encoder.encode(a=a,b=b,c=x), .075, print_time, number=1000)

    def test_hashable_dict_performance(self):
        items = list(enumerate(range(100)))
        self._assert_scale_time(items, HashableSparse, .0004, print_time, number=1000)

    def test_shuffle_performance(self):
        items = list(range(50))
        self._assert_scale_time(items, coba.random.shuffle, .023, print_time, number=1000)

    def test_randoms_performance(self):
        self._assert_scale_time(50, coba.random.randoms, .008, print_time, number=1000)

    def test_choice_performance_uniform(self):
        self._assert_scale_time([1]*50, coba.random.choice, .0009, print_time, number=1000)

    def test_choice_performance_not_uniform(self):
        A = [1]*50
        B = [1/50]*50
        self._assert_call_time(lambda:coba.random.choice(A,B), .0029, print_time, number=1000)

    def test_choice_performance_weights(self):
        items = [1]+[0]*49
        weights = [1]+[0]*49
        self._assert_call_time(lambda: coba.random.choice(items,weights), .002, print_time, number=1000)

    def test_gausses_performance(self):
        self._assert_scale_time(50, coba.random.gausses, .03, print_time, number=1000)

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW not installed.")
    def test_vowpal_mediator_make_example_sequence_str_performance(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --quiet",4)
        x = [ str(i) for i in range(100) ]

        self._assert_scale_time(x, lambda x:vw.make_example({'x':x}, None), .04, print_time, number=1000)

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW not installed.")
    def test_vowpal_mediator_make_example_highly_sparse_performance(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --quiet",4)

        ns = { 'x': [1]+[0]*1000  }
        self._assert_call_time(lambda:vw.make_example(ns, None), .03, print_time, number=1000)

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW not installed.")
    def test_vowpal_mediator_make_example_sequence_int_performance(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --quiet",4)
        x = list(range(100))

        self._assert_scale_time(x, lambda x:vw.make_example({'x':x}, None), .03, print_time, number=1000)

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW not installed.")
    def test_vowpal_mediator_make_example_sequence_dict_performance(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --quiet",4)

        ns = { 'x': { str(i):i for i in range(500)} }
        self._assert_call_time(lambda:vw.make_example(ns, None), .025, print_time, number=1000)

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW not installed.")
    def test_vowpal_mediator_make_examples_sequence_int_performance(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --quiet",4)

        shared   = { 'a': list(range(100))}
        separate = [{ 'x': list(range(25)) }, { 'x': list(range(25)) }]
        self._assert_call_time(lambda:vw.make_examples(shared, separate, None), .06, print_time, number=1000)

    def test_reservoir_performance(self):
        res = Reservoir(2,seed=1)
        x = list(range(500))

        self._assert_scale_time(x, lambda x:list(res.filter(x)), .04, print_time, number=1000)

    def test_jsonencode_performance(self):
        enc = JsonEncode()
        x = [[1.2,1.2],[1.2,1.2],{'a':1.,'b':1.}]*5
        self._assert_scale_time(x, enc.filter, .045, print_time, number=1000)

    def test_arffreader_performance(self):

        attributes = [f"@attribute {i} {{1,2}}" for i in range(3)]
        data_lines = [",".join(["1"]*3)]*50
        arff       = attributes+["@data"]

        reader = ArffReader()
        self._assert_scale_time(data_lines, lambda x:list(reader.filter(arff+x)), .01, print_time, number=100)

    def test_arffreader_dense_performance(self):

        data_lines = [",".join(["1"]*3)]*50

        reader = ArffReader()
        self._assert_scale_time(data_lines, lambda x:list(reader._dense(x,[],[])), .0034, print_time, number=100)

    def test_arffreader_parse_performance(self):

        attributes = [f"@attribute {i} {{1,2}}" for i in range(3)]
        data_lines = [",".join(["1"]*3)]*50
        arff       = attributes+["@data"]

        reader = ArffReader()
        self._assert_scale_time(data_lines, lambda x: [list(l) for l in reader.filter(arff+x)], .034, print_time, number=100)

    def test_structure_performance(self):
        structure = Structure([None,2])
        self._assert_call_time(lambda: list(structure.filter([[0,0,0] for _ in range(50)])), .07, print_time, number=1000)

    def test_linear_synthetic(self):
        self._assert_call_time(lambda:list(LinearSyntheticSimulation(10).read()), .07, print_time, number=1)

    def test_scale_target_features(self):
        items = [SimulatedInteraction((3193.0, 151.0, '0', '0', '0'),[1,2,3],[4,5,6])]*10
        scale = Scale("min","minmax",target="context")
        self._assert_scale_time(items, lambda x:list(scale.filter(x)), .021, print_time, number=1000)

    def test_scale_target_rewards(self):
        items = [SimulatedInteraction((3193.0, 151.0),[1,2,3],[4,5,6])]*10
        scale = Scale("min","minmax",target="rewards")
        self._assert_scale_time(items, lambda x:list(scale.filter(x)), .04, print_time, number=1000)

    def test_environments_flat_tuple(self):
        items = [SimulatedInteraction([1,2,3,4]+[(0,1)]*3,[1,2,3],[4,5,6])]*10
        flat  = Flatten()
        self._assert_scale_time(items, lambda x:list(flat.filter(x)), .04, print_time, number=1000)

    def test_pipes_flat_tuple(self):
        items = [tuple([1,2,3]+[(0,1)]*5)]*10
        flat  = coba.pipes.Flatten()
        self._assert_scale_time(items, lambda x:list(flat.filter(x)), .025, print_time, number=1000)

    def test_pipes_flat_dict(self):
        items = [dict(enumerate([1,2,3]+[(0,1)]*5))]*10
        flat  = coba.pipes.Flatten()
        self._assert_scale_time(items, lambda x:list(flat.filter(x)), .04, print_time, number=1000)

    def test_result_filter_env(self):
        envs = { k:{'mod': k%100} for k in range(5) }
        lrns = { 1:{}, 2:{}, 3:{}}
        ints = { (e,l):{} for e in envs.keys() for l in lrns.keys() }
        res  = Result(envs, lrns, ints)
        self._assert_call_time(lambda:res.filter_env(mod=3), .05, print_time, number=1000)

    def test_moving_average_sliding_window(self):
        items = [1,0]*100
        self._assert_scale_time(items, lambda x:list(moving_average(x,span=2)), .025, print_time, number=1000)

    def test_moving_average_rolling_window(self):
        items = [1,0]*300
        self._assert_scale_time(items, lambda x:list(moving_average(x)), .07, print_time, number=1000)

    def test_lazy_dense_init_get(self):
        I = ['1']*100
        self._assert_call_time(lambda:LazyDense(lambda:I)[2], .12, print_time, number=100000)

    def test_encode_dense_init_get(self):
        I = ['1']*100
        E = [int]*100
        self._assert_call_time(lambda:EncodeDense(I,E)[1], .092, print_time, number=100000)

    def test_encode_dense_init_iter(self):
        I = ['1']*1
        E = [int]*1
        enc = EncodeDense(I,E)
        self._assert_call_time(lambda:[next(iter(enc)) for _ in range(500000)], .017, print_time, number=1)

    def test_label_dense_init_labeled(self):
        vals  = ['1']*50
        self._assert_call_time(lambda:LabelDense(vals, 49, 'c').labeled, .13, print_time, number=100000)

    def test_dense_row_headers(self):
        headers = list(map(str,range(100)))
        r2 = HeadDense(LazyDense(['1']*100),dict(zip(headers,count())))
        r3 = EncodeDense(r2,[int]*100)
        r4 = EncodeDense(r3,[int]*100)

        self._assert_call_time(lambda:r2.headers, .0003, print_time, number=1000)
        self._assert_call_time(lambda:r3.headers, .0009, print_time, number=1000)
        self._assert_call_time(lambda:r4.headers, .0014, print_time, number=1000)

    def test_dense_row_drop(self):
        r1 = KeepDense(['1']*100,dict(enumerate(range(99))), [True]*99+[False], 99, None)
        self._assert_call_time(lambda:r1[3], .0006, print_time, number=1000)

    def test_dense_row_to_builtin(self):
        r = LazyDense(['1']*100)
        self._assert_call_time(lambda:list(r), .002, print_time, number=1000)

    def test_sparse_row_to_builtin(self):
        d = dict(enumerate(['1']*100))
        r = LazySparse(d)

        #self._assert_call_time(lambda:dict(d), .04, print_time, number=1000)
        self._assert_call_time(lambda:dict(r.items()), .005, print_time, number=1000)

    def test_to_grounded_interaction(self):
        items    = [SimulatedInteraction(1, [1,2,3,4], [0,0,0,1])]*10
        grounded = Grounded(10,5,10,5,1)

        self._assert_scale_time(items,lambda x:list(grounded.filter(x)), .03, print_time, number=1000)

    def test_simple_evaluation(self):

        class DummyLearner:
            def predict(*args):
                return [1,0,0]
            def learn(*args):
                pass

        items = [SimulatedInteraction(1,[1,2,3],[1,2,3])]*100
        eval  = SimpleEvaluation()
        learn = DummyLearner()
        self._assert_scale_time(items,lambda x:list(eval.process(learn, x)), .05, print_time, number=100)

    def test_safe_learner_predict(self):

        class DummyLearner:
            def predict(*args):return [1,0,0]
            def learn(*args): pass

        learn = SafeLearner(DummyLearner())
        self._assert_call_time(lambda:learn.predict(1,[1,2,3]), .0035, print_time, number=1000)

    def test_safe_learner_learn(self):

        class DummyLearner:
            def predict(*args):return [1,0,0]
            def learn(*args): pass

        learn = SafeLearner(DummyLearner())
        self._assert_call_time(lambda:learn.learn(1,[1,2,3], [1], 1, .5), .0008, print_time, number=1000)

    def test_scale_reward(self):
        reward = ScaleReward(L1Reward(1), 1, 2, "argmax")
        self._assert_call_time(lambda: reward.eval(4), .04, print_time, number=100000)

    def test_simulated_interaction_init(self):
        self._assert_call_time(lambda: SimulatedInteraction(1,[1,2],[0,1]), .012, print_time, number=10000)
        #self._assert_call_time(lambda: Interaction2(1,[1,2],[0,1],d), .011, print_time, number=1000000)

    def test_simulated_interaction_copy(self):
        si1 = SimulatedInteraction(1,[1,2],[3,4])
        self._assert_call_time(lambda: si1.copy(), .003, print_time, number=10000)
        #si2 = Interaction2(1,[1,2],[3,4],{})
        #self._assert_call_time(lambda: si2.copy(), .004, print_time, number=1000000)

    def test_simulated_interaction_get_context(self):
        si1 = SimulatedInteraction(1,[1,2],[3,4])
        self._assert_call_time(lambda: si1['actions'], .0022, print_time, number=10000)
        #si2 = Interaction2(1,[1,2],[3,4])
        #self._assert_call_time(lambda: si2.actions, .0022, print_time, number=1000000)

    def test_simulated_interaction_get_context(self):
        si1 = SimulatedInteraction(1,[1,2],[3,4])
        self._assert_call_time(lambda: si1['actions'], .0022, print_time, number=10000)
        #si2 = Interaction2(1,[1,2],[3,4])
        #self._assert_call_time(lambda: si2.actions, .0022, print_time, number=1000000)

    @unittest.skip("An interesting and revealing test but not good to run regularly")
    def test_integration_performance(self):
        import coba as cb

        #here we create an IGL problem from the covertype dataset one time. 
        #It takes about 1 minute to load due to the number of features and examples.
        covertype_id = 150
        ndata        = 500_000
        n_users      = 100
        n_words      = 100

        environment = cb.Environments.cache_dir('./.coba_cache')                               #(1) set a cache directory
        environment = environment.from_openml(data_id=covertype_id, take=ndata)            #(2) begin with covertype
        environment = environment.scale(shift='min',scale='minmax')                        #(3) scale features to [0,1]
        environment = environment.grounded(n_users, n_users/2, n_words, n_words/2, seed=1) #(4) turn into an igl problem

        self._assert_call_time(lambda: environment.materialize(), 48, print_time, number=1)

    def test_encode_cat_rows(self):
        rows = [[Categorical('1',list(map(str,range(20))))]*5]*5
        enc  = EncodeCatRows("onehot")
        self._assert_call_time(lambda: list(enc.filter(rows)), .04, print_time, number=1000)

    def test_chunk(self):
        self._assert_call_time(lambda: list(Chunk().filter(range(100))), .01, print_time, number=1000)

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
