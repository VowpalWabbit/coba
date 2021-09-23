import unittest
import timeit
import math

from abc import ABC, abstractmethod
from typing import Sequence, Tuple, cast, Any

from coba.encodings import Encoder,StringEncoder, NumericEncoder, OneHotEncoder, FactorEncoder, InteractionTermsEncoder

class Encoder_Interface_Tests(ABC):

    @abstractmethod
    def _make_unfit_encoder(self) -> Tuple[Encoder, Sequence[str], Sequence[str], Sequence[Any]]:
        ...

    def test_is_fit_initially_false(self):
        unfit_encoder,_,_,_ = self._make_unfit_encoder()

        cast(unittest.TestCase, self).assertTrue(not unfit_encoder.is_fit)

    def test_is_fit_becomes_true_after_fit(self):
        unfit_encoder,train,_,_ = self._make_unfit_encoder()

        fit_encoder = unfit_encoder.fit(train)

        cast(unittest.TestCase, self).assertTrue(fit_encoder.is_fit)
        cast(unittest.TestCase, self).assertFalse(unfit_encoder.is_fit)

    def test_fit_encoder_throws_exception_on_fit(self):
        unfit_encoder,train,_,_ = self._make_unfit_encoder()

        with cast(unittest.TestCase, self).assertRaises(Exception):
            unfit_encoder.fit(train).fit(train)

    def test_unfit_encoder_throws_exception_on_encode(self):
        unfit_encoder,_,test,_ = self._make_unfit_encoder()

        with cast(unittest.TestCase, self).assertRaises(Exception):
            unfit_encoder.encode(test)

    def test_correctly_returns_new_encoder_after_fitting(self):
        unfit_encoder,train,_,_ = self._make_unfit_encoder()

        fit_encoder = unfit_encoder.fit(train)

        cast(unittest.TestCase, self).assertNotEqual(fit_encoder, unfit_encoder)

    def test_correctly_encodes_after_fitting(self):
        unfit_encoder,train,test,expected = self._make_unfit_encoder()

        actual = unfit_encoder.fit(train).encode(test)

        cast(unittest.TestCase, self).assertEqual(actual, expected)

class StringEncoder_Tests(Encoder_Interface_Tests, unittest.TestCase):

    def _make_unfit_encoder(self) -> Tuple[Encoder, Sequence[str], Sequence[str], Sequence[Any]]:
        return StringEncoder(is_fit=False), ["1","2","3"], ["1.23"], ["1.23"]

    def test_is_fit_marks_as_fitted(self):

        encoder = StringEncoder()

        self.assertTrue(encoder.is_fit)

class NumericEncoder_Tests(Encoder_Interface_Tests, unittest.TestCase):

    def _make_unfit_encoder(self) -> Tuple[Encoder, Sequence[str], Sequence[str], Sequence[Any]]:
        return NumericEncoder(is_fit=False), ["1","2","3"], ["1.23"], [1.23]

    def test_is_fit_marks_as_fitted(self):

        encoder = NumericEncoder()

        self.assertTrue(encoder.is_fit)

    def test_performance_small_list(self):

        encoder   = NumericEncoder()
        many_ones = ["1"]*100
        
        time = min(timeit.repeat(lambda:encoder.encode(many_ones), repeat=1000, number=4))
        
        #was approximately .000122
        self.assertLess(time, .0002)

    def test_performance_large_list(self):

        encoder   = NumericEncoder()
        many_ones = ["1"]*100000
        
        time = min(timeit.repeat(lambda:encoder.encode(many_ones), repeat=100, number=1))
        
        #was approximately .0301
        self.assertLess(time, .05)

    def test_not_numeric_string(self):

        actual = NumericEncoder().encode(["5 1"])[0]

        self.assertTrue(math.isnan(actual))

class OneHotEncoder_Tests(Encoder_Interface_Tests, unittest.TestCase):

    def _make_unfit_encoder(self) -> Tuple[Encoder, Sequence[str], Sequence[str], Sequence[Any]]:
        return OneHotEncoder(), ["d","a","b","b","b","d"], ["a"], [(0, 1, 0)]

    def test_singular_if_binary(self):
        encoder = OneHotEncoder(singular_if_binary=True).fit(["1","1","1","0","0"])

        self.assertEqual(encoder.encode(["0"]), [(0,)])
        self.assertEqual(encoder.encode(["1"]), [(1,)])

    def test_error_if_unkonwn_true(self):
        encoder = OneHotEncoder(error_if_unknown=True).fit(["1","1","1","0","0"])

        with self.assertRaises(Exception):
            self.assertEqual(encoder.encode(["2"]), [(0)])

    def test_error_if_unkonwn_false(self):
        encoder = OneHotEncoder(error_if_unknown=False).fit(["0","1","2"])

        try:
            actual = encoder.encode(["5"])
        except:
            self.fail("An exception was raised when it shouldn't have been")

        self.assertEqual(1, len(actual))
        self.assertTrue(math.isnan(actual[0]))

    def test_instantiated_fit_values(self):
        encoder = OneHotEncoder(fit_values=["0","1","2"])

        expected = [(1,0,0),(0,1,0),(0,0,1),(0,1,0)]

        actual = encoder.encode(["0","1","2","1"])

        self.assertEqual(actual, expected)

    def test_performance_fit_values(self):

        fit_values = list(range(1000))

        time = min(timeit.repeat(lambda:OneHotEncoder(fit_values), repeat=100, number = 1))

        #was approximately 0.017
        self.assertLess(time, .03)

    def test_performance_encode(self):

        encoder = OneHotEncoder(list(range(1000)), error_if_unknown=False )
        to_encode = [100,200,300,400,-1]*100000

        time = min(timeit.repeat(lambda:encoder.encode(to_encode), repeat=50, number = 1))

        #was approximately 0.040
        self.assertLess(time, 1)

class FactorEncoder_Tests(Encoder_Interface_Tests, unittest.TestCase):
    def _make_unfit_encoder(self) -> Tuple[Encoder, Sequence[str], Sequence[str], Sequence[Any]]:
        return FactorEncoder(), ["a","z","a","z","1"], ["1","a","z"], [1,2,3]

class InteractionTermsEncoder_Tests(unittest.TestCase):

    def test_dense_a(self):
        encoder = InteractionTermsEncoder(["a"])

        interactions = encoder.encode(x=[1,2,3], a=[1,2])

        self.assertEqual([1,2], interactions)

    def test_dense_x(self):
        encoder = InteractionTermsEncoder(["x"])

        interactions = encoder.encode(x=[1,2,3], a=[1,2])

        self.assertEqual([1,2,3], interactions)

    def test_dense_x_a(self):
        encoder = InteractionTermsEncoder(["x", "a"])

        interactions = encoder.encode(x=[1,2,3], a=[1,2])

        self.assertEqual([1,2,3,1,2], interactions)

    def test_dense_x_a_xa_xxa(self):
        encoder = InteractionTermsEncoder(["x","a","xa","xxa"])

        interactions1 = encoder.encode(x=[1,2,3], a=[1,2])
        interactions2 = encoder.encode(x=[1,2,3], a=[1,2])

        self.assertCountEqual([1,2,3,1,2,1,2,3,2,4,6,1,2,3,4,6,9,2,4,6,8,12,18], interactions1)
        self.assertEqual(interactions1,interactions2)

    def test_sparse_x_a(self):
        encoder = InteractionTermsEncoder(["x","a"])

        interactions = encoder.encode(x={"1":1,"2":2}, a={"1":3,"2":4})

        self.assertEqual([("x1",1), ("x2",2), ("a1",3), ("a2",4)], interactions)

    def test_sparse_xa(self):
        encoder = InteractionTermsEncoder(["xa"])

        interactions = encoder.encode(x={"1":1,"2":2}, a={"1":3,"2":4})

        self.assertEqual([("x1a1",3), ("x1a2",4), ("x2a1",6), ("x2a2",8)], interactions)

    def test_sparse_xxa(self):
        encoder = InteractionTermsEncoder(["xxa"])

        interactions = encoder.encode(x={"1":1,"2":2}, a={"1":3,"2":4})

        self.assertEqual([("x1x1a1",3), ("x1x1a2",4), ("x2x1a1",6), ("x2x1a2",8), ("x2x2a1",12), ("x2x2a2",16)], interactions)

    def test_string_a(self):
        encoder = InteractionTermsEncoder(["a"])

        interactions = encoder.encode(x=["a","b","c"], a=["d","e"])

        self.assertEqual([("a0d",1), ("a1e",1)], interactions)
    
    def test_string_x(self):
        encoder = InteractionTermsEncoder(["x"])

        interactions = encoder.encode(x=["a","b","c"], a=["d","e"])

        self.assertEqual([("x0a",1), ("x1b",1), ("x2c",1)], interactions)
    
    def test_string_xa(self):
        encoder = InteractionTermsEncoder(["xa"])

        interactions = encoder.encode(x=["a"], a=["d","e"])

        self.assertEqual([("x0aa0d",1), ("x0aa1e",1)], interactions)

    def test_string_numeric_xa(self):
        encoder = InteractionTermsEncoder(["xa"])

        interactions = encoder.encode(x=[2], a=["d","e"])

        self.assertEqual([("x0a0d",2), ("x0a1e",2)], interactions)

    def test_singular_string_a(self):
        encoder = InteractionTermsEncoder(["a"])

        interactions = encoder.encode(x=["a"], a="d")

        self.assertEqual([("ad",1)], interactions)

    def test_singular_string_xa(self):
        encoder = InteractionTermsEncoder(["xa"])

        interactions = encoder.encode(x="abc", a="dbc")

        self.assertEqual([("xabcadbc",1)], interactions)

    def test_singular_numeric_xa(self):
        encoder = InteractionTermsEncoder(["xa"])

        interactions1 = encoder.encode(x=(1,2,3), a=2)
        interactions2 = encoder.encode(x=(1,2,3), a=2)

        self.assertCountEqual([2,4,6], interactions1)
        self.assertEqual(interactions1,interactions2)

    def test_string_tuple(self):
        encoder = InteractionTermsEncoder(["xa"])

        interactions1 = encoder.encode(x=('1',2), a=2)
        interactions2 = encoder.encode(x=('1',2), a=2)

        self.assertCountEqual([('x01a0', 2), ('x1a0',4) ], interactions1)
        self.assertEqual(interactions1,interactions2)


    def test_performance(self):
        encoder = InteractionTermsEncoder(["xxa"])

        x = dict(zip(map(str,range(100)), range(100)))
        a = [1,2,3]
        
        time = timeit.timeit(lambda: encoder.encode(x=x, a=a), number=100)
        
        #best observed was 0.62
        #performance time could be reduced to around .47 by using numpy and prime factorization of feature names 
        self.assertLess(time, 0.8)

if __name__ == '__main__':
    unittest.main()
