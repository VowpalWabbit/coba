import unittest
import timeit

from itertools import count
from typing import Callable, Any

import coba.json
import coba.pipes
import coba.random

from coba.utilities import PackageChecker
from coba.statistics import mean,var,percentile
from coba.learners import VowpalMediator, SafeLearner
from coba.environments import SimulatedInteraction, LinearSyntheticSimulation
from coba.environments import Scale, Flatten, Grounded, Chunk, Impute, Repr, OpeRewards
from coba.encodings import NumericEncoder, OneHotEncoder, InteractionsEncoder
from coba.primitives import BinaryReward, HammingReward

from coba.pipes import Reservoir, Encode, ArffReader, Structure, Pipes

from coba.pipes.rows import LazyDense, LazySparse, EncodeDense, KeepDense, HeadDense, LabelDense, EncodeCatRows
from coba.pipes.readers import ArffLineReader, ArffDataReader, ArffAttrReader

from coba.experiments.results import Result, moving_average, Table, TransactionResult
from coba.evaluators import SequentialCB
from coba.primitives import Categorical, HashableSparse

Timeable = Callable[[],Any]
Scalable = Callable[[list],Timeable]

sensitivity_scaling = 10 # when deploying to Github workflows we reduce our timing sensitivity by a factor of 10
print_time = False # a global variable that determines whether the below tests print or assert

#################################################
# This file has all of the runtime based unittests.
# All of the benchmark times below were done on Mark Rucker's laptop. They may not be the same for you.
# For individual tests of interest turn the specific test's print_time to True to get the correct baseline.
# Look at the doc args below for _assert_call_time, _assert_scale_time and _z_assert_less for more information.
#################################################

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

    @unittest.skipUnless(PackageChecker.vowpalwabbit(strict=False), "VW not installed.")
    def test_vowpal_mediator_make_example_sequence_str_performance(self):
        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --quiet",4)
        x = [ str(i) for i in range(100) ]

        self._assert_call_time(lambda:vw.make_example({'x':x}, None), .04, print_time, number=1000)

    @unittest.skipUnless(PackageChecker.vowpalwabbit(strict=False), "VW not installed.")
    def test_vowpal_mediator_make_example_highly_sparse_performance(self):
        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --quiet",4)

        ns = { 'x': [1]+[0]*1000 }
        self._assert_call_time(lambda:vw.make_example(ns, None), .03, print_time, number=1000)

    @unittest.skipUnless(PackageChecker.vowpalwabbit(strict=False), "VW not installed.")
    def test_vowpal_mediator_make_example_sequence_int_performance(self):
        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --quiet",4)
        x = list(range(100))

        self._assert_call_time(lambda: vw.make_example({'x':x}, None), .03, print_time, number=1000)

    @unittest.skipUnless(PackageChecker.vowpalwabbit(strict=False), "VW not installed.")
    def test_vowpal_mediator_make_example_sequence_mixed_performance(self):
        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --quiet",4)
        x = [ float(i) if i % 2 == 0 else str(i) for i in range(100) ]

        self._assert_call_time(lambda: vw.make_example({'x':x}, None), .03, print_time, number=1000)

    @unittest.skipUnless(PackageChecker.vowpalwabbit(strict=False), "VW not installed.")
    def test_vowpal_mediator_make_example_sequence_dict_performance(self):
        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --quiet",4)

        ns = { 'x': { str(i):i for i in range(500)} }
        self._assert_call_time(lambda:vw.make_example(ns, None), .05, print_time, number=1000)

    @unittest.skipUnless(PackageChecker.vowpalwabbit(strict=False), "VW not installed.")
    def test_vowpal_mediator_make_examples_sequence_int_performance(self):
        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --quiet",4)

        shared   = { 'a': list(range(100))}
        separate = [{ 'x': list(range(25)) }, { 'x': list(range(25)) }]
        self._assert_call_time(lambda:vw.make_examples(shared, separate, None), .06, print_time, number=1000)

    def test_reservoir_performance(self):
        res = Reservoir(2,seed=1)
        x = list(range(500))

        self._assert_scale_time(x, lambda x:list(res.filter(x)), .03, print_time, number=1000)

    def test_jsonminimize_performance(self):
        m = coba.json.minimize
        x = [[1.2,1.2],[1.2,1.2],{'a':1.,'b':1.}]*20
        self._assert_scale_time(x, lambda x: m(x) , .06, print_time, number=1000)

    def test_arffreader_lazy_performance(self):
        attrs = [f"@attribute {i} {{1,2}}" for i in range(3)]
        data  = ["@data"]
        lines = [",".join(["1"]*3)]*50
        reader = ArffReader()
        self._assert_scale_time(lines, lambda x:list(reader.filter(attrs+data+x)), .0095, print_time, number=100)

    def test_arffreader_full_performance(self):
        attrs = [f"@attribute {i} numeric" for i in range(3)]
        data  = ["@data"]
        lines = [",".join(["1"]*3)]*50
        reader = ArffReader()
        self._assert_scale_time(lines, lambda x:[list(l) for l in reader.filter(attrs+data+x)], .028, print_time, number=100)

    def test_arffattrreader_dense_performance(self):
        reader = ArffAttrReader(True)
        attrs = [f"@attribute {i} {{1,2}}" for i in range(3)]

        self._assert_call_time(lambda:list(reader.filter(attrs)), .004, print_time, number=100)

    def test_arffdatareader_dense_performance(self):
        data_lines = [",".join(["1"]*3)]*50
        reader = ArffDataReader(True)

        self._assert_scale_time(data_lines, lambda x:list(reader.filter(x)), .0007, print_time, number=100)

    def test_arfflinereader_dense_performance(self):
        reader = ArffLineReader(True,3)
        line = ",".join(['1']*3)

        self._assert_call_time(lambda:reader.filter(line), .0019, print_time, number=1000)

    def test_structure_performance(self):
        structure = Structure([None,2])
        self._assert_call_time(lambda: list(structure.filter([[0,0,0] for _ in range(50)])), .07, print_time, number=1000)

    def test_linear_synthetic(self):
        self._assert_call_time(lambda:list(LinearSyntheticSimulation(10).read()), .07, print_time, number=1)

    @unittest.skipUnless(PackageChecker.vowpalwabbit(strict=False), "VW not installed.")
    def test_ope_rewards(self):
        I = [{'context':[1,2,3], 'actions':['a','b','c','d'], 'action':'a', 'probability':.5, 'reward':2}]*50
        ope = OpeRewards('DM')
        self._assert_call_time(lambda: list(ope.filter(I)), .16, print_time, number=10)

    def test_scale_dense_target_features(self):
        items = [SimulatedInteraction((3193.0, 151.0, '0', '0', '0'),[1,2,3],[4,5,6])]*10
        scale = Scale("min","minmax",target="context")
        self._assert_scale_time(items, lambda x:list(scale.filter(x)), .024, print_time, number=1000)

    def test_scale_sparse_target_features(self):
        items = [SimulatedInteraction({1:3193.0, 2:151.0, 3:'0', 4:'0', 5:'0'},[1,2,3],[4,5,6])]*10
        scale = Scale(0,"minmax",target="context")
        self._assert_scale_time(items, lambda x:list(scale.filter(x)), .035, print_time, number=1000)

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

    def test_table_index(self):
        coba.random.seed(1)
        table = Table(columns=['environment_id','learner_id','index','reward'])

        N=4000
        reward = coba.random.randoms(N)
        for environment_id in reversed(range(10)):
            for learner_id in range(5):
                table.insert({"environment_id":[environment_id]*N,"learner_id":[learner_id]*N,"index":list(range(1,N+1)),"reward":reward})

        self._assert_call_time(lambda:table.index('environment_id','learner_id','index'), .15, print_time, number=10)

    def test_table_insert(self):

        table = Table(columns=['environment_id','learner_id','index','reward'])
        insert_cols = {"environment_id":[0],"learner_id":[0],"index":[1],"reward":[1]}

        table._columns = set(table._columns)

        self._assert_call_time(lambda:table.insert(insert_cols), .025, print_time, number=10_000)

    def test_table_where_no_index(self):
        coba.random.seed(1)
        table = Table(columns=['environment_id','learner_id','index','reward'])

        N=4000
        reward = coba.random.randoms(N)
        for environment_id in range(10):
            for learner_id in range(10):
                table.insert({"environment_id":[environment_id]*N,"learner_id":[learner_id]*N,"index":list(range(1,N+1)),"reward":reward})

        self._assert_call_time(lambda:table.where(index=10,comparison='<='), .2, print_time, number=10)

    def test_table_where_index(self):
        coba.random.seed(1)
        table = Table(columns=['environment_id','learner_id','index','reward'])

        N=4000
        reward = coba.random.randoms(N)
        for environment_id in range(10):
            for learner_id in range(10):
                table.insert({"environment_id":[environment_id]*N,"learner_id":[learner_id]*N,"index":list(range(1,N+1)),"reward":reward})

        table.index("index")

        self._assert_call_time(lambda:table.where(index=10,comparison='<='), .15, print_time, number=10)

    def test_table_where_number(self):
        table = Table(columns=['environment_id']).insert([ [k] for k in range(1000)])
        self._assert_call_time(lambda:table.where(environment_id=1), .04, print_time, number=500)

    def test_table_groupby(self):
        coba.random.seed(1)
        table = Table(columns=['environment_id','learner_id','index','reward'])

        N=4
        reward = coba.random.randoms(N)
        for environment_id in range(200*50):
            for learner_id in range(5):
                table.insert({"environment_id":[environment_id]*N,"learner_id":[learner_id]*N, "evaluator_id":[0]*N, "index":list(range(1,N+1)),"reward":reward})

        table = table.index('environment_id','learner_id','evaluator_id','index')

        self._assert_call_time(lambda:list(table.groupby(3,select=None)), .15, print_time, number=10)

    def test_table_to_dicts(self):
        table = Table(columns=['environment_id','learner_id','index'])
        N=4
        for environment_id in range(10):
            for learner_id in range(5):
                table.insert({"environment_id":[environment_id]*N,"learner_id":[learner_id]*N, "evaluator_id":[0]*N, "index":list(range(1,N+1))})
        table = table.index('environment_id','learner_id','evaluator_id','index')
        self._assert_call_time(lambda:list(table.to_dicts()), .055, print_time, number=1000)

    @unittest.skipUnless(PackageChecker.pandas(strict=False), "pandas is not installed so we must skip pandas tests")
    def test_table_to_pandas(self):
        table = Table(columns=['environment_id']).insert([ [k] for k in range(1000)])
        self._assert_call_time(lambda:table.to_pandas(), .4, print_time, number=100)

    def test_transaction_result_filter(self):
        transactions = [
            ["version",4],
            ["I",(0,2),{"_packed":{"reward":[1]*5_000}}],
            ["I",(0,1),{"_packed":{"reward":[1]*5_000}}]
        ]
        self._assert_call_time(lambda:TransactionResult().filter(transactions), .025, print_time, number=100)

    def test_result_filter_fin(self):
        envs = Table(columns=['environment_id','mod']).insert([[k,k%100] for k in range(5)])
        lrns = Table(columns=['learner_id']).insert([[0],[1],[2]])
        vals = Table(columns=['evaluator_id']).insert([[0]])
        ints = Table(columns=['environment_id','learner_id','evaluator_id','index']).insert([[e,l,0,0] for e in range(3) for l in range(2)])

        res  = Result(envs, lrns, vals, ints)
        self._assert_call_time(lambda:res.filter_fin(l='learner_id',p='environment_id'), .06, print_time, number=1000)

    def test_result_where(self):
        envs = Table(columns=['environment_id','mod']).insert([[k,k%100] for k in range(5)])
        lrns = Table(columns=['learner_id']).insert([[0],[1],[2]])
        vals = Table(columns=['evaluator_id']).insert([[0]])
        ints = Table(columns=['environment_id','learner_id','evaluator_id']).insert([[e,l,0] for e in range(3) for l in range(5)])

        res  = Result(envs, lrns, vals, ints)
        self._assert_call_time(lambda:res.where(mod=3), .06, print_time, number=1000)

    def test_result_indexed_gs(self):
        envs = Table(columns=['environment_id','mod']).insert([[k,k%100] for k in range(5)])
        lrns = Table(columns=['learner_id']).insert([[0],[1],[2],[3],[4]])
        vals = Table(columns=['evaluator_id']).insert([[0]])
        ints = Table(columns=['environment_id','learner_id','evaluator_id','index']).insert([[e,l,0,0] for e in range(3) for l in range(5)])

        res  = Result(envs, lrns, vals, ints)
        self._assert_call_time(lambda:list(res._indexed_gs('environment_id','learner_id')), .02, print_time, number=1000)

    def test_result_indexed_ys(self):
        envs = Table(columns=['environment_id','mod']).insert([[k,k%100] for k in range(2)])
        lrns = Table(columns=['learner_id']).insert([[0],[1],[2]])
        vals = Table(columns=['evaluator_id']).insert([[0]])
        ints = Table(columns=['environment_id','learner_id','evaluator_id','index','reward']).insert([[e,l,0,i,1] for i in range(2) for e in range(2) for l in range(3)])

        res  = Result(envs, lrns, vals, ints)
        self._assert_call_time(lambda:list(res._indexed_ys('environment_id','learner_id','index',y='reward',span=None)), .023, print_time, number=1000)

    def test_result_copy(self):
        envs = Table(columns=['environment_id','mod']).insert([[k,k%100] for k in range(5)])
        lrns = Table(columns=['learner_id']).insert([[0],[1],[2]])
        vals = Table(columns=['evaluator_id']).insert([[0]])
        ints = Table(columns=['environment_id','learner_id','evaluator_id','index']).insert([[e,l,0,0] for e in range(3) for l in range(5)])

        res  = Result(envs, lrns, vals, ints)
        self._assert_call_time(lambda:res.copy(), .025, print_time, number=1000)

    def test_moving_average_sliding_window(self):
        items = [1,0]*100
        self._assert_scale_time(items, lambda x:list(moving_average(x,span=2)), .025, print_time, number=1000)

    def test_moving_average_rolling_window(self):
        items = [1,0]*300
        self._assert_scale_time(items, lambda x:list(moving_average(x)), .03, print_time, number=1000)

    def test_mean(self):
        items = [1,0]*3000
        self._assert_scale_time(items, lambda x:mean(x), .04, print_time, number=1000)

    def test_percentile_no_sort(self):
        items = [1,1]*3000
        self._assert_scale_time(items, lambda x:percentile(x,[0,.5,1],sort=False), .005, print_time, number=1000)

    def test_percentile_sort(self):
        items = [1,1]*3000
        self._assert_scale_time(items, lambda x:percentile(x,[0,.5,1],sort=True), .048, print_time, number=1000)

    def test_percentile_no_sort_weights(self):
        items = [1,1]*3000
        self._assert_scale_time(items, lambda x:percentile(x,[0,.5,1],weights=[1]*len(x),sort=False), .008, print_time, number=10)

    def test_percentile_sort_weights(self):
        items = [1,1]*3000
        self._assert_scale_time(items, lambda x:percentile(x,[0,.5,1],weights=[1]*len(x),sort=True), .015, print_time, number=10)

    def test_var(self):
        items = [1,0]*300
        self._assert_scale_time(items, lambda x:var(x), .075, print_time, number=1000)

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

        self._assert_call_time(lambda:dict(r.items()), .007, print_time, number=1000)

    def test_to_grounded_interaction(self):
        items    = [SimulatedInteraction(1, [1,2,3,4], [0,0,0,1])]*10
        grounded = Grounded(10,5,10,5,1)
        self._assert_scale_time(items,lambda x:list(grounded.filter(x)), .04, print_time, number=1000)

    def test_simple_evaluation(self):

        class DummyLearner:
            def predict(*args):
                return [1,0,0]
            def learn(*args):
                pass

        class DummEnv:
            __slots__ = ('read',)
            def __init__(self,interactions):
                self.read = lambda: interactions

        items = [SimulatedInteraction(1,[1,2,3],[1,2,3])]*20
        eval  = SequentialCB()
        learn = DummyLearner()

        #most of this time is being spent in SafeLearner.predict...
        self._assert_scale_time(items,lambda x:list(eval.evaluate(DummEnv(x),learn)), .12, print_time, number=1000)

    def test_safe_learner_predict(self):

        class DummyLearner:
            def predict(*args):return [1,0,0]
            def learn(*args): pass

        learn = SafeLearner(DummyLearner())
        self._assert_call_time(lambda:learn.predict(1,[1,2,3]), .005, print_time, number=1000)

    def test_safe_learner_learn(self):

        class DummyLearner:
            def predict(*args):return [1,0,0]
            def learn(*args): pass

        learn = SafeLearner(DummyLearner())
        self._assert_call_time(lambda:learn.learn(1, [1], 1, .5), .0007, print_time, number=1000)

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
        self._assert_call_time(lambda: si1['context'], .0022, print_time, number=10000)

    @unittest.skip("An interesting and revealing test but not good to run regularly")
    def test_integration_performance(self):
        import coba as cb

        #here we create an IGL problem from the covertype dataset one time.
        #It takes about 1 minute to load due to the number of features and examples.
        covertype_id = 150
        ndata        = 500_000
        n_users      = 100
        n_words      = 100

        environment = cb.Environments.cache_dir('./.coba_cache')                           #(1) set a cache directory
        environment = environment.from_openml(data_id=covertype_id, take=ndata)            #(2) begin with covertype
        environment = environment.scale(shift='min',scale='minmax')                        #(3) scale features to [0,1]
        environment = environment.grounded(n_users, n_users/2, n_words, n_words/2, seed=1) #(4) turn into an igl problem

        self._assert_call_time(lambda: environment.materialize(), 48, print_time, number=1)

    def test_repr_not_repeat(self):
        levels1 = list(map(str,range(10)))
        levels2 = list(map(str,range(1,11)))

        interaction1 = { 'actions': [Categorical(l,levels1) for l in levels1] }
        interaction2 = { 'actions': [Categorical(l,levels2) for l in levels2] }

        interactions = [interaction1,interaction2]*25
        repr         = Repr(categorical_actions='onehot_tuple')
        filterer     = lambda x: list(repr.filter(x))

        self._assert_scale_time(interactions, filterer, .06, print_time, number=1000)

    def test_repr_repeat(self):
        levels = list(map(str,range(10)))

        interaction = { 'actions': [Categorical(l,levels) for l in levels] }

        interactions = [interaction]*100
        repr         = Repr(categorical_actions='onehot_tuple')
        filterer     = lambda x: list(repr.filter(x))

        self._assert_scale_time(interactions, filterer, .04, print_time, number=1000)

    def test_categorical_equality(self):
        cat1 = Categorical('1',list(map(str,range(20))))
        cat2 = Categorical('1',list(map(str,range(20))))

        self._assert_call_time(lambda: cat1==cat2, .02, print_time, number=100000)

    def test_encode_cat_rows(self):
        rows = [[Categorical('1',list(map(str,range(20))))]*5]*5
        enc  = EncodeCatRows("onehot")
        self._assert_call_time(lambda: list(enc.filter(rows)), .04, print_time, number=1000)

    def test_chunk(self):
        self._assert_call_time(lambda: list(Chunk().filter(range(100))), .01, print_time, number=1000)

    def test_impute(self):
        interactions = [
            SimulatedInteraction((7   , 2   , "A" ), [1], [1]),
            SimulatedInteraction((7   , 2   , "A" ), [1], [1]),
            SimulatedInteraction((8   , 3   , "A" ), [1], [1])
        ] * 10
        impute = Impute("mode")
        self._assert_scale_time(interactions, lambda x:list(impute.filter(x)), .06, print_time, number=1000)

    def test_hamming_reward_with_builtin_action(self):
        rwd      = HammingReward([1,2,3])
        action   = [1,2,4]
        self._assert_call_time(lambda: rwd(action), .007, print_time, number=10000)

    def test_binary_reward_with_builtin_action(self):
        rwd      = BinaryReward([1],2)
        action   = [2]
        self._assert_call_time(lambda: rwd(action), .0042, print_time, number=10000)

    @unittest.skipUnless(PackageChecker.torch(strict=False), "Requires pytorch")
    def test_binary_reward_with_number_argmax_and_tensor_action(self):
        import torch
        rwd      = BinaryReward(1,2)
        action   = torch.tensor([2])
        self._assert_call_time(lambda: rwd(action), .024, print_time, number=10000)

    @unittest.skipUnless(PackageChecker.torch(strict=False), "Requires pytorch")
    def test_binary_reward_with_list_argmax_and_tensor_action(self):
        import torch
        rwd      = BinaryReward([1],2)
        action   = torch.tensor([2])
        self._assert_call_time(lambda: rwd(action), .032, print_time, number=10000)

    @unittest.skip("Just for testing. There's not much we can do to speed up process creation.")
    def test_async_pipe(self):
        #this takes about 2.5 with number=10
        def run_async1():
            from multiprocessing import Process
            p = Process(target=coba.pipes.Identity)
            p.start()
            p.join()

        #this takes about 2.5 with number=10
        pipeline = Pipes.join(coba.pipes.IterableSource([1,2,3]), coba.pipes.Identity(), coba.pipes.ListSink())
        def run_async2():
            proc = pipeline.run_async(lambda ex: None)
            proc.join()

        self._assert_call_time(run_async2, 2.5, print_time, number=10)

    def _assert_call_time(self, func: Timeable, expected:float, print_time:bool, *, number:int=1000, setup="pass") -> None:
        """ Test that the given func scales linearly with the number of items.

            Args:
                func: The function we want to test its run time
                expected: The expected runtime in seconds
                print_time: whether to print the run time or assert the run time (if true the run time will be written)
                number: the number of times to pass items to func for timing
                setup: passed through to timeit.repeat.
        """
        if print_time: print()
        self._z_assert_less(timeit.repeat(func, setup=setup, number=1, repeat=number), expected, print_time)
        if print_time: print()

    def _assert_scale_time(self, items: list, func: Scalable, expected:float, print_time:bool, *, number:int=1000, setup="pass") -> None:
        """ Test that the given func scales linearly with the number of items.

            Args:
                items: The items to pass to func
                func: The function we want to test its run time
                expected: The expected runtime in seconds
                print_time: whether to print the run time or assert the run time (if true the run time will be written)
                number: the number of times to pass items to func for timing
                setup: passed through to timeit.repeat.
        """
        if print_time: print()
        items_1 = items*1
        items_2 = items*2
        self._z_assert_less(timeit.repeat(lambda: func(items_1), setup=setup, number=1, repeat=number), 1.0*expected, print_time)
        self._z_assert_less(timeit.repeat(lambda: func(items_2), setup=setup, number=1, repeat=number), 2.0*expected, print_time)
        if print_time: print()

    def _z_assert_less(self, samples, expected, print_it):
        """Perform a Z-Test and only report that a test has failed with greater than 95% CI."""
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
