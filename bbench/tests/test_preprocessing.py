import unittest

from abc import ABC, abstractmethod
from typing import Sequence, Tuple, cast, Any

from bbench.preprocessing import Metadata, Encoder, StringEncoder, NumericEncoder, OneHotEncoder, InferredEncoder

class Encoder_Interface_Tests(ABC):

    @abstractmethod
    def _make_unfit_encoder(self) -> Tuple[Encoder, Sequence[str], str, Sequence[Any]]:
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

    def _make_unfit_encoder(self) -> Tuple[Encoder, Sequence[str], str, Sequence]:
        return StringEncoder(is_fit=False), ["1","2","3"], "1.23", ["1.23"]

    def test_is_fit_marks_as_fitted(self):

        encoder = StringEncoder()

        self.assertTrue(encoder.is_fit)

class NumericEncoder_Tests(Encoder_Interface_Tests, unittest.TestCase):

    def _make_unfit_encoder(self) -> Tuple[Encoder, Sequence[str], str, Sequence[Any]]:
        return NumericEncoder(is_fit=False), ["1","2","3"], "1.23", [1.23]

    def test_is_fit_marks_as_fitted(self):

        encoder = NumericEncoder()

        self.assertTrue(encoder.is_fit)

class OneHotEncoder_Tests(Encoder_Interface_Tests, unittest.TestCase):

    def _make_unfit_encoder(self) -> Tuple[Encoder, Sequence[str], str, Sequence[Any]]:
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

    def _make_unfit_encoder(self) -> Tuple[Encoder, Sequence[str], str, Sequence[Any]]:
        return InferredEncoder(), ["0","1","2","3"], "2", [2]

class InferredOneHot_Tests(Encoder_Interface_Tests, unittest.TestCase):

    def _make_unfit_encoder(self) -> Tuple[Encoder, Sequence[str], str, Sequence[Any]]:
        return InferredEncoder(), ["a","1","2","3"], "2", [0, 1, 0, 0]

class Metadata_Tests(unittest.TestCase):

    def test_init_correctly_instantiates_1(self):
        expected_ignore  = True
        expected_label   = None
        expected_encoder = None

        actual_meta = Metadata(expected_ignore, expected_label, expected_encoder)

        self.assertEqual(actual_meta.ignore , expected_ignore )
        self.assertEqual(actual_meta.label  , expected_label  ) 
        self.assertEqual(actual_meta.encoder, expected_encoder)

    def test_init_correctly_instantiates_2(self):
        expected_ignore  = None
        expected_label   = True
        expected_encoder = None

        actual_meta = Metadata(expected_ignore,expected_label,expected_encoder)

        self.assertEqual(actual_meta.ignore , expected_ignore )
        self.assertEqual(actual_meta.label  , expected_label  ) 
        self.assertEqual(actual_meta.encoder, expected_encoder)

    def test_init_correctly_instantiates_3(self):
        expected_ignore  = None
        expected_label   = None
        expected_encoder = NumericEncoder()

        actual_meta = Metadata(expected_ignore, expected_label, expected_encoder)

        self.assertEqual(actual_meta.ignore , expected_ignore )
        self.assertEqual(actual_meta.label  , expected_label  ) 
        self.assertEqual(actual_meta.encoder, expected_encoder)

    def test_init_correctly_instantiates_4(self):
        expected_ignore  = True
        expected_label   = True
        expected_encoder = NumericEncoder()

        actual_meta = Metadata(expected_ignore, expected_label, expected_encoder)

        self.assertEqual(actual_meta.ignore , expected_ignore )
        self.assertEqual(actual_meta.label  , expected_label  ) 
        self.assertEqual(actual_meta.encoder, expected_encoder)

    def test_clone_correctly_clones(self):
        expected_ignore  = True
        expected_label   = True
        expected_encoder = NumericEncoder()

        original_meta = Metadata(expected_ignore, expected_label, expected_encoder)
        clone_meta    = original_meta.clone()

        self.assertEqual(clone_meta.ignore , expected_ignore )
        self.assertEqual(clone_meta.label  , expected_label  ) 
        self.assertEqual(clone_meta.encoder, expected_encoder)

        self.assertNotEqual(clone_meta, original_meta)

    def test_apply_works(self):
        expected_ignore  = False
        expected_label   = False
        expected_encoder = NumericEncoder()

        original_meta = Metadata(True,True,NumericEncoder())
        
        applied_meta = original_meta.override(Metadata(expected_ignore, None, None))
        self.assertEqual(applied_meta.ignore , expected_ignore)

        applied_meta = original_meta.override(Metadata(None, expected_label, None))
        self.assertEqual(applied_meta.label  , expected_label) 

        applied_meta = original_meta.override(Metadata(None, None, expected_encoder))
        self.assertEqual(applied_meta.encoder  , expected_encoder)

    def test_from_json_works(self):
        expected_ignore  = False
        expected_label   = False
        expected_encoder = NumericEncoder

        actual_meta = Metadata.from_json('{ "ignore":false, "label":false, "encoder":"numeric" }')

        self.assertEqual(actual_meta.ignore , expected_ignore )
        self.assertEqual(actual_meta.label  , expected_label  ) 
        self.assertIsInstance(actual_meta.encoder, expected_encoder)

if __name__ == '__main__':
    unittest.main()
