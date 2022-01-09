import unittest
import timeit
import statistics
import importlib.util

import coba.random

from coba.learners import VowpalMediator
from coba.utilities import HashableDict
from coba.environments import SimulatedInteraction
from coba.encodings import NumericEncoder, OneHotEncoder, InteractionsEncoder, StringEncoder
from coba.pipes import Reservoir, JsonEncode, Encode

class Performance_Tests(unittest.TestCase):
    
    def test_numeric_encode_performance_small(self):

        encoder   = NumericEncoder()
        many_ones = ["1"]*100
        
        time = min(timeit.repeat(lambda:encoder.encodes(many_ones), repeat=1000, number=4))
        
        #was approximately .000122
        self.assertLess(time, .0012)

    def test_numeric_encode_performance_large(self):

        encoder   = NumericEncoder()
        many_ones = ["1.2"]*100000
        
        time = min(timeit.repeat(lambda:encoder.encodes(many_ones), repeat=25, number=1))
        
        #was approximately .019
        self.assertLess(time, .19)

    def test_onehot_fit_performance(self):

        fit_values = list(range(1000))

        time = min(timeit.repeat(lambda:OneHotEncoder(fit_values), repeat=25, number = 1))

        #was approximately 0.017
        self.assertLess(time, .17)

    def test_onehot_encode_performance(self):

        encoder = OneHotEncoder(list(range(1000)), err_if_unknown=False )
        to_encode = [100,200,300,400,-1]*100000

        time = min(timeit.repeat(lambda:encoder.encodes(to_encode), repeat=25, number = 1))

        #best observed 0.027
        self.assertLess(time, .27)

    def test_encode_performance(self):

        encoder   = Encode(dict(zip(range(50),[NumericEncoder()]*50)))
        to_encode = [['1.23']*50]*6000

        time = min(timeit.repeat(lambda:list(encoder.filter(to_encode)), number = 1))

        #best observed 0.06
        self.assertLess(time, .6)

    def test_dense_interaction_xx_encode_performance(self):
        encoder = InteractionsEncoder(["xx"])

        x = list(range(100))
        
        time = timeit.timeit(lambda: encoder.encode(x=x), number=100)
        
        #best observed was 0.025
        self.assertLess(time, 0.25)

    def test_sparse_interaction_xx_encode_performance(self):
        encoder = InteractionsEncoder(["xx"])

        x = dict(zip(map(str,range(100)), range(100)))
        
        time = timeit.timeit(lambda: encoder.encode(x=x), number=100)
        
        #best observed was 0.09
        self.assertLess(time, 0.9)

    def test_sparse_interaction_xxa_encode_performance(self):
        encoder = InteractionsEncoder(["xxa"])

        x = dict(zip(map(str,range(100)), range(100)))
        a = [1,2,3]
        
        time = timeit.timeit(lambda: encoder.encode(x=x, a=a), number=50)
        
        #best observed was 0.20
        self.assertLess(time, 2.0)

    def test_sparse_interaction_abc_encode_performance(self):
        encoder = InteractionsEncoder(["aabc"])

        a = dict(zip(map(str,range(100)), range(100)))
        b = [1,2]
        c = [2,3]

        time = timeit.timeit(lambda: encoder.encode(a=a, b=b, c=c), number=25)
        
        #best observed was 0.15
        self.assertLess(time, 1.5)

    def test_interaction_context_performance(self):

        interaction = SimulatedInteraction([1,2,3]*100, (1,2,3), rewards=(4,5,6))

        time = timeit.timeit(lambda: interaction.context, number=10000)

        #best observed was 0.0015
        self.assertLess(time, .015)

    def test_hashable_dict_performance(self):

        base_dict = dict(enumerate(range(1000)))

        time1 = timeit.timeit(lambda: dict(enumerate(range(1000))), number=1000)
        time2 = timeit.timeit(lambda: HashableDict(base_dict)     , number=1000)

        self.assertLess(abs(time1-time2), 1)

    def test_shuffle_performance(self):

        to_shuffle = list(range(5000))

        time = min(timeit.repeat(lambda:coba.random.shuffle(to_shuffle), repeat=10, number=3))
        
        #best observed 0.01
        self.assertLess(time,.1)

    def test_randoms_performance(self):
        
        time = min(timeit.repeat(lambda:coba.random.randoms(5000), repeat=100, number=1))

        #best observed 0.0025
        self.assertLess(time,.025)

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW not installed")
    def test_vowpal_mediator_make_example_performance(self):

        from vowpalwabbit import pyvw

        vw = pyvw.vw("--cb_explore_adf 10 --epsilon 0.1 --interactions xxa --interactions xa --ignore_linear x --quiet")

        ns = { 'x': [ (str(i),v) for i,v in enumerate(range(1000)) ], 'a': [ (str(i),v) for i,v in enumerate(range(20)) ] }
        time = statistics.mean(timeit.repeat(lambda:VowpalMediator.make_example(vw, ns, None, 4), repeat=10, number=100))            

        #.014 was my final average time
        self.assertLess(time, .14)
    
    def test_vowpal_mediator_prep_features_tuple_sequence_performance(self):

        x    = [ (str(i),v) for i,v in enumerate(range(1000)) ]
        time = statistics.mean(timeit.repeat(lambda:VowpalMediator.prep_features(x), repeat=10, number=100))

        #0.026 was my final average time (this time can be cut in half but prep_features becomes much less flexible)
        self.assertLess(time,.26)

    def test_vowpal_mediator_prep_features_dict_performance(self):

        x    = dict(zip(map(str,range(1000)), range(1000)))
        time = statistics.mean(timeit.repeat(lambda:VowpalMediator.prep_features(x), repeat=10, number=100))

        #0.016 was my final average time
        self.assertLess(time,.16)

    def test_vowpal_mediator_prep_features_values_sequence(self):

        x    = list(range(1000))
        time = statistics.mean(timeit.repeat(lambda:VowpalMediator.prep_features(x), repeat=10, number=100))            

        #0.019 was my final average time.
        self.assertLess(time,.19)

    @unittest.skipUnless(importlib.util.find_spec("vowpalwabbit"), "VW not installed")
    def test_vowpal_mediator_prep_and_make_performance(self):

        from vowpalwabbit import pyvw

        vw = pyvw.vw("--cb_explore_adf 10 --epsilon 0.1 --interactions xx --ignore_linear x --quiet")
        x  = [ (str(i),round(coba.random.random(),5)) for i in range(200) ]

        time1 = statistics.mean(timeit.repeat(lambda:VowpalMediator.make_example(vw, {'x': VowpalMediator.prep_features(x) }, None, 4), repeat=5, number=100))
        time2 = statistics.mean(timeit.repeat(lambda:vw.parse("|x " + " ".join(f"{i}:{v}" for i,v in x))                              , repeat=5, number=100))

        self.assertLess(time1/time2,1.5)

    def test_reservoir_performance(self):

        x = list(range(10000))
        
        time = statistics.mean(timeit.repeat(lambda:list(Reservoir(2,seed=1).filter(x)), repeat=10, number=100))

        #0.010 was my final average time.
        self.assertLess(time, .10)

    def test_jsonencode_performance(self):
        
        x = [[1.2,1.2],[1.2,1.2],{'a':1.,'b':1.}]*300
        encoder = JsonEncode()

        time = statistics.mean(timeit.repeat(lambda:encoder.filter(x), repeat=5, number=100))

        #0.11 was my final average time.
        self.assertLess(time, 1.1)

if __name__ == '__main__':
    unittest.main()
