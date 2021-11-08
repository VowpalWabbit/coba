import unittest
import timeit
import statistics

import coba.random

from coba.learners import VowpalMediator
from coba.utilities import HashableDict
from coba.environments import SimulatedInteraction
from coba.encodings import NumericEncoder, OneHotEncoder, InteractionTermsEncoder

class Performance_Tests(unittest.TestCase):
    
    def test_numeric_encode_performance_small(self):

        encoder   = NumericEncoder()
        many_ones = ["1"]*100
        
        time = min(timeit.repeat(lambda:encoder.encode(many_ones), repeat=1000, number=4))
        
        #was approximately .000122
        self.assertLess(time, .0002)

    def test_numeric_encode_performance_large(self):

        encoder   = NumericEncoder()
        many_ones = ["1"]*100000
        
        time = min(timeit.repeat(lambda:encoder.encode(many_ones), repeat=100, number=1))
        
        #was approximately .0301
        self.assertLess(time, .05)

    def test_onehot_fit_performance(self):

        fit_values = list(range(1000))

        time = min(timeit.repeat(lambda:OneHotEncoder(fit_values), repeat=100, number = 1))

        #was approximately 0.017
        self.assertLess(time, .03)

    def test_onehot_encode_performance(self):

        encoder = OneHotEncoder(list(range(1000)), error_if_unknown=False )
        to_encode = [100,200,300,400,-1]*100000

        time = min(timeit.repeat(lambda:encoder.encode(to_encode), repeat=50, number = 1))

        #was approximately 0.040
        self.assertLess(time, 1)

    def test_interaction_encode_performance(self):
        encoder = InteractionTermsEncoder(["xxa"])

        x = dict(zip(map(str,range(100)), range(100)))
        a = [1,2,3]
        
        time = timeit.timeit(lambda: encoder.encode(x=x, a=a), number=100)
        
        print(time)
        #best observed was 0.62 without interning
        #best observed was 0.87 with interning
        #performance time could be reduced to around .47 by using numpy and prime factorization of feature names 
        self.assertLess(time, 1.0)

    def test_interaction_context_performance(self):

        interaction = SimulatedInteraction([1,2,3]*100, (1,2,3), rewards=(4,5,6))

        time = timeit.timeit(lambda: interaction.context, number=10000)

        print(time)
        self.assertLess(time, 1.5)

    def test_hashable_dict_performance(self):

        base_dict = dict(enumerate(range(1000)))

        time1 = timeit.timeit(lambda: dict(enumerate(range(1000))), number=1000)
        time2 = timeit.timeit(lambda: HashableDict(base_dict)     , number=1000)

        self.assertLess(abs(time1-time2), 1)

    def test_shuffle_performance(self):

        to_shuffle = list(range(50000))

        time = min(timeit.repeat(lambda:coba.random.shuffle(to_shuffle), repeat=20, number=1))
        
        #was approximately 0.057
        self.assertLess(time,1)

    def test_randoms_performance(self):
        
        time = min(timeit.repeat(lambda:coba.random.randoms(5000), repeat=200, number=1))

        #was approximately 0.0025
        self.assertLess(time,.005)

    def test_vowpal_mediator_make_example_performance(self):

        try:
            from vowpalwabbit import pyvw

            vw = pyvw.vw("--cb_explore_adf 10 --epsilon 0.1 --interactions xxa --interactions xa --ignore_linear x --quiet")

            ns = { 'x': [ (str(i),v) for i,v in enumerate(range(1000)) ], 'a': [ (str(i),v) for i,v in enumerate(range(20)) ] }
            time = statistics.mean(timeit.repeat(lambda:VowpalMediator.make_example(vw, ns, None, 4), repeat=10, number=1000))            

            print(time)

        except ImportError:
            unittest.skip("VW not installed. Skip this Test")
    
    def test_vowpal_mediator_prep_features_tuple_sequence_performance(self):

        x    = [ (str(i),v) for i,v in enumerate(range(1000)) ]
        time = statistics.mean(timeit.repeat(lambda:VowpalMediator.prep_features(x), repeat=10, number=1000))

        #0.13 was my final average time
        self.assertLess(time,.2)

    def test_vowpal_mediator_prep_features_dict_performance(self):

        x    = dict(zip(map(str,range(1000)), range(1000)))
        time = statistics.mean(timeit.repeat(lambda:VowpalMediator.prep_features(x), repeat=10, number=1000))

        #0.16 was my final average time
        self.assertLess(time,.25)

    def test_vowpal_mediator_prep_features_values_sequence(self):

        x    = list(range(1000))
        time = statistics.mean(timeit.repeat(lambda:VowpalMediator.prep_features(x), repeat=10, number=1000))            

        #0.19 was my final average time.
        self.assertLess(time,.3)

    def test_vowpal_mediator_prep_and_make_performance(self):

        try:
            from vowpalwabbit import pyvw

            vw = pyvw.vw("--cb_explore_adf 10 --epsilon 0.1 --interactions xx --ignore_linear x --quiet")
            x  = [ (str(i),round(coba.random.random(),5)) for i in range(200) ]

            time1 = statistics.mean(timeit.repeat(lambda:VowpalMediator.make_example(vw, {'x': VowpalMediator.prep_features(x) }, None, 4), repeat=10, number=1000))
            time2 = statistics.mean(timeit.repeat(lambda:vw.parse("|x " + " ".join(f"{i}:{v}" for i,v in x))                              , repeat=10, number=1000))

            self.assertLess(time1,time2)

        except ImportError:
            unittest.skip("VW not installed. Skip this Test")

if __name__ == '__main__':
    unittest.main()