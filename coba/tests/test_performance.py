import unittest
import timeit
import statistics
import importlib.util

import coba.random

from coba.learners import VowpalMediator
from coba.utilities import HashableDict
from coba.environments import SimulatedInteraction, LinearSyntheticSimulation, Scale
from coba.encodings import NumericEncoder, OneHotEncoder, InteractionsEncoder
from coba.pipes import Reservoir, JsonEncode, Encode, ArffReader, Structure, ListSource
from coba.experiments.results import Result, moving_average

print_time = False

class Performance_Tests(unittest.TestCase):

    def test_numeric_encode_performance_small(self):

        encoder   = NumericEncoder()
        many_ones = ["1"]*500

        time = min(timeit.repeat(lambda:encoder.encodes(many_ones), repeat=1000, number=4))

        #was approximately .0003
        if print_time: print(time)
        self.assertLess(time, .003)

    def test_numeric_encode_performance_large(self):

        encoder   = NumericEncoder()
        many_ones = ["1.2"]*100000

        time = min(timeit.repeat(lambda:encoder.encodes(many_ones), repeat=25, number=1))

        #was approximately .018
        if print_time: print(time)
        self.assertLess(time, .18)

    def test_onehot_fit_performance(self):

        fit_values = list(range(1000))

        time = min(timeit.repeat(lambda:OneHotEncoder(fit_values), repeat=25, number = 1))

        #was approximately 0.017
        if print_time: print(time)
        self.assertLess(time, .17)

    def test_onehot_encode_performance(self):

        encoder = OneHotEncoder(list(range(1000)), err_if_unknown=False )
        to_encode = [100,200,300,400,-1]*100000

        time = min(timeit.repeat(lambda:encoder.encodes(to_encode), repeat=25, number = 1))

        #best observed 0.027
        if print_time: print(time)
        self.assertLess(time, .27)

    def test_encode_performance(self):

        encoder   = Encode(dict(zip(range(50),[NumericEncoder()]*50)))
        to_encode = [['1.23']*50]*6000

        time = min(timeit.repeat(lambda:list(encoder.filter(to_encode)), number = 1))

        #best observed 0.06
        if print_time: print(time)
        self.assertLess(time, .6)

    def test_dense_interaction_xx_encode_performance(self):
        encoder = InteractionsEncoder(["xx"])

        x = list(range(100))

        time = timeit.timeit(lambda: encoder.encode(x=x), number=100)

        #best observed was 0.03
        if print_time: print(time)
        self.assertLess(time, 0.3)

    def test_sparse_interaction_xx_encode_performance(self):
        encoder = InteractionsEncoder(["xx"])

        x = dict(zip(map(str,range(100)), range(100)))

        time = timeit.timeit(lambda: encoder.encode(x=x), number=100)

        #best observed was 0.09
        if print_time: print(time)
        self.assertLess(time, 0.9)

    def test_sparse_interaction_xxa_encode_performance(self):
        encoder = InteractionsEncoder(["xxa"])

        x = dict(zip(map(str,range(100)), range(100)))
        a = [1,2,3]

        time = timeit.timeit(lambda: encoder.encode(x=x, a=a), number=50)

        #best observed was 0.20
        if print_time: print(time)
        self.assertLess(time, 2.0)

    def test_sparse_interaction_abc_encode_performance(self):
        encoder = InteractionsEncoder(["aabc"])

        a = dict(zip(map(str,range(100)), range(100)))
        b = [1,2]
        c = [2,3]

        time = timeit.timeit(lambda: encoder.encode(a=a, b=b, c=c), number=25)

        #best observed was 0.17
        if print_time: print(time)
        self.assertLess(time, 1.7)

    def test_interaction_context_performance(self):

        interaction = SimulatedInteraction([1,2,3]*100, (1,2,3), (4,5,6))

        time = timeit.timeit(lambda: interaction.context, number=10000)

        #best observed was 0.0075
        if print_time: print(time)
        self.assertLess(time, .075)

    def test_hashable_dict_performance(self):

        base_dict = dict(enumerate(range(1000)))

        time1 = timeit.timeit(lambda: dict(enumerate(range(1000))), number=1000)
        time2 = timeit.timeit(lambda: HashableDict(base_dict)     , number=1000)

        self.assertLess(abs(time1-time2), 1)

    def test_shuffle_performance(self):

        to_shuffle = list(range(5000))

        time = min(timeit.repeat(lambda:coba.random.shuffle(to_shuffle), repeat=10, number=3))

        #best observed 0.01
        if print_time: print(time)
        self.assertLess(time,.1)

    def test_randoms_performance(self):

        time = min(timeit.repeat(lambda:coba.random.randoms(5000), repeat=100, number=1))

        #best observed 0.0017
        if print_time: print(time)
        self.assertLess(time,.017)

    def test_gausses_performance(self):

        time = min(timeit.repeat(lambda:coba.random.gausses(5000,0,1), repeat=10, number=3))

        #best observed 0.011
        if print_time: print(time)
        self.assertLess(time,.11)

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW not installed.")
    def test_vowpal_mediator_make_example_sequence_str_performance(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --quiet",4)

        ns = { 'x': [ str(i) for i in range(1000) ] }
        time = statistics.mean(timeit.repeat(lambda:vw.make_example(ns, None), repeat=10, number=100))

        #.028 was my final average time
        if print_time: print(time)
        self.assertLess(time, .28)

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW not installed.")
    def test_vowpal_mediator_make_example_highly_sparse_performance(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --quiet",4)

        ns = { 'x': [1]+[0]*1000  }
        time = statistics.mean(timeit.repeat(lambda:vw.make_example(ns, None), repeat=10, number=100))

        #.004 was my final average time
        if print_time: print(time)
        self.assertLess(time, .04)

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW not installed.")
    def test_vowpal_mediator_make_example_sequence_int_performance(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --quiet",4)

        ns = { 'x': list(range(1000)) }
        time = statistics.mean(timeit.repeat(lambda:vw.make_example(ns, None), repeat=30, number=100))

        #.014 was my final average time
        if print_time: print(time)
        self.assertLess(time, .14)

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW not installed.")
    def test_vowpal_mediator_make_example_sequence_dict_performance(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --quiet",4)

        ns = { 'x': { str(i):i for i in range(1000)} }
        time = statistics.mean(timeit.repeat(lambda:vw.make_example(ns, None), repeat=10, number=100))

        #.0027 was my final average time
        if print_time: print(time)
        self.assertLess(time, .027)

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW not installed.")
    def test_vowpal_mediator_make_examples_sequence_int_performance(self):

        vw = VowpalMediator()
        vw.init_learner("--cb_explore_adf --quiet",4)

        shared    = { 'a': list(range(500))}
        distincts = [{ 'x': list(range(500)) }, { 'x': list(range(500)) }]
        time = statistics.mean(timeit.repeat(lambda:vw.make_examples(shared, distincts, None), repeat=10, number=100))

        #.022 was my final average time
        if print_time: print(time)
        self.assertLess(time, .22)

    def test_reservoir_performance(self):

        x = list(range(10000))

        time = statistics.mean(timeit.repeat(lambda:list(Reservoir(2,seed=1).filter(x)), repeat=10, number=100))

        #0.010 was my final average time.
        if print_time: print(time)
        self.assertLess(time, .10)

    def test_jsonencode_performance(self):

        x = [[1.2,1.2],[1.2,1.2],{'a':1.,'b':1.}]*300
        encoder = JsonEncode()

        time = statistics.mean(timeit.repeat(lambda:encoder.filter(x), repeat=5, number=100))

        #0.15 was my final average time.
        if print_time: print(time)
        self.assertLess(time, 1.5)

    def test_arffreader_performance(self):

        attributes = "\n".join([f"@attribute {i} {{1,2}}" for i in range(1000)])
        data_line  = ",".join(["1"]*1000)
        data_lines = "\n".join([data_line]*700)

        arff = f"{attributes}\n@data\n{data_lines}".split('\n')

        reader = ArffReader()
        time = timeit.timeit(lambda:list(reader.filter(arff)), number = 1)

        #.045 was my final time
        if print_time: print(time)
        self.assertLess(time, 0.45)

    def test_structure_performance(self):

        structure = Structure([None,2])
        time = timeit.timeit(lambda:list(structure.filter([[0,0,0] for _ in range(100)])), number=1000)

        #.092 was my final time
        if print_time: print(time)
        self.assertLess(time, 0.92)

    def test_linear_synthetic(self):

        time = timeit.timeit(lambda:list(LinearSyntheticSimulation(100).read()), number=1)

        #.22 was my final time
        if print_time: print(time)
        self.assertLess(time, 2.2)

    def test_scale(self):

        env = ListSource([SimulatedInteraction((3193.0, 151.0, 17.0, 484.0, -137.0, 319.0, 239.0, 238.0, 121.0, 3322.0, 0.0, 1.0, 0.0, 0.0, '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'),[1,2,3],[4,5,6])]*1000)
        time = timeit.timeit(lambda:list(Scale().filter(env.read())), number=1)

        #.11 was my final 
        if print_time: print(time)
        self.assertLess(time, 1.1)

    def test_result_filter_env(self):
        envs = { k:{ 'mod': k%100 } for k in range(1000) }
        lrns = { 1:{}, 2:{}, 3:{}}
        ints = { (e,l):{} for e in envs.keys() for l in lrns.keys() }
        time = timeit.timeit(lambda: Result(envs, lrns, ints).filter_env(mod=5), number=4)

        #.07 was my final time
        if print_time: print(time)
        self.assertLess(time, .7)

    def test_moving_average_sliding_window(self):
        rwds = [1,0]*1000
        time = timeit.timeit(lambda: moving_average(rwds,span=2), number=250)

        #.056 was my final time
        if print_time: print(time)
        self.assertLess(time, .56)

    def test_moving_average_rolling_window(self):
        rwds = [1,0]*1000
        time = timeit.timeit(lambda: moving_average(rwds), number=500)

        #between .069 and .075 was my final time
        if print_time: print(time)
        self.assertLess(time, .75)

if __name__ == '__main__':
    unittest.main()
