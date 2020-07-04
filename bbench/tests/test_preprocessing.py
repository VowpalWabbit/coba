import unittest

from abc import ABC, abstractmethod
from typing import Sequence, Tuple

from bbench.preprocessing import Encoder, NumericEncoder, OneHotEncoder, InferredEncoder

class Encoder_Interface_Tests(ABC):

    @abstractmethod
    def _make_encoder(self) -> Tuple[Encoder, Sequence[str], str, Sequence[float]]:
        ...

    def test_is_fit_initially_false(self):
        encoder,_,_,_ = self._make_encoder()

        self.assertTrue(not encoder.is_fit) # type: ignore

    def test_is_fit_becomes_true_after_fit(self):
        encoder,train,_,_ = self._make_encoder()

        encoder.fit(train)

        self.assertTrue(encoder.is_fit) # type: ignore

    def test_throws_exception_on_second_fit_call(self):
        encoder,train,_,_ = self._make_encoder()

        with self.assertRaises(Exception): # type: ignore
            encoder.fit(train)
            encoder.fit(train)

    def test_throws_exception_on_encode_without_fit(self):
        encoder,_,test,_ = self._make_encoder()

        with self.assertRaises(Exception): # type: ignore
            encoder.encode(test)

    def test_correctly_returns_self_after_fitting(self):
        encoder,train,_,_ = self._make_encoder()

        actual = encoder.fit(train)

        self.assertEqual(actual, encoder) #type: ignore

    def test_correctly_encodes_after_fitting(self):
        encoder,train,test,expected = self._make_encoder()

        actual = encoder.fit(train).encode(test)

        self.assertEqual(actual, expected) #type: ignore

class NumericEncoder_Tests(Encoder_Interface_Tests, unittest.TestCase):

    def _make_encoder(self) -> Tuple[Encoder, Sequence[str], str, Sequence[float]]:
        return NumericEncoder(auto_fit=False), ["1","2","3"], "1.23", [1.23]

    def test_auto_fit_marks_as_fitted(self):

        encoder = NumericEncoder()

        self.assertTrue(encoder.is_fit)

class OneHotEncoder_Tests(Encoder_Interface_Tests, unittest.TestCase):

    def _make_encoder(self) -> Tuple[Encoder, Sequence[str], str, Sequence[float]]:
        return OneHotEncoder(), ["d", "a","b","b","b","d"], "a", [1, 0, 0]

    def test_singular_if_binary(self):
        encoder = OneHotEncoder(singular_if_binary=True).fit(["1","1","1","0","0"])

        self.assertEqual(encoder.encode("0"), [1])
        self.assertEqual(encoder.encode("1"), [0])

    def test_error_if_unkonwn_true(self):
        encoder = OneHotEncoder(error_if_unknown=True).fit(["1","1","1","0","0"])

        with self.assertRaises(Exception):
            self.assertEqual(encoder.encode("2"), [1])

    def test_error_if_unkonwn_false(self):
        encoder = OneHotEncoder(error_if_unknown=False).fit(["0","1","2"])

        try:
            actual = encoder.encode("5")
        except:
            self.fail("An exception was raised when it shouldn't have been")

        self.assertEqual(actual, [0,0,0])

    def test_instantiated_fit_values(self):
        encoder = OneHotEncoder(fit_values=["0","1","2"])

        self.assertEqual(encoder.encode("0"), [1,0,0])
        self.assertEqual(encoder.encode("1"), [0,1,0])
        self.assertEqual(encoder.encode("2"), [0,0,1])

class InferredNumeric_Tests(Encoder_Interface_Tests, unittest.TestCase):

    def _make_encoder(self) -> Tuple[Encoder, Sequence[str], str, Sequence[float]]:
        return InferredEncoder(), ["0","1","2","3"], "2", [2]

class InferredOneHot_Tests(Encoder_Interface_Tests, unittest.TestCase):

    def _make_encoder(self) -> Tuple[Encoder, Sequence[str], str, Sequence[float]]:
        return InferredEncoder(), ["a","1","2","3"], "2", [0, 1, 0, 0]

if __name__ == '__main__':
    unittest.main()
