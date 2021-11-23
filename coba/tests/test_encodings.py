import unittest
import math

from coba.encodings import IdentityEncoder, StringEncoder, NumericEncoder, OneHotEncoder, FactorEncoder, InteractionsEncoder
from coba.exceptions import CobaException

class IdentityEncoder_Tests(unittest.TestCase):

    def test_fit_no_except(self):
        encoder = IdentityEncoder()
        encoder.fit([1,2,3])

    def test_is_fit_marks_as_fitted(self):
        encoder = IdentityEncoder()
        self.assertTrue(encoder.is_fit)

    def test_encode(self):
        self.assertEqual([1, "2", 3.23, None], IdentityEncoder().encode([1, "2", 3.23, None]))

    def test_fit_encode(self):
        self.assertEqual([1, "2", 3.23, None], IdentityEncoder().fit_encode([1, "2", 3.23, None]))

class StringEncoder_Tests(unittest.TestCase):

    def test_fit_no_except(self):
        encoder = StringEncoder()
        encoder.fit([1,2,3])

    def test_is_fit_marks_as_fitted(self):
        encoder = StringEncoder()
        self.assertTrue(encoder.is_fit)

    def test_encode(self):
        self.assertEqual(["1","2","3.23","None"], StringEncoder().encode([1, "2", 3.23, None]))

    def test_fit_encode(self):
        self.assertEqual(["1","2","3.23","None"], StringEncoder().fit_encode([1, "2", 3.23, None]))

class StringEncoder_Tests(unittest.TestCase):

    def test_fit_no_except(self):
        encoder = StringEncoder()
        encoder.fit([1,2,3])

    def test_is_fit_marks_as_fitted(self):
        encoder = StringEncoder()
        self.assertTrue(encoder.is_fit)

    def test_encode(self):
        self.assertEqual(["1","2","3.23","None"], StringEncoder().encode([1, "2", 3.23, None]))

    def test_fit_encode(self):
        self.assertEqual(["1","2","3.23","None"], StringEncoder().fit_encode([1, "2", 3.23, None]))

class NumericEncoder_Tests(unittest.TestCase):

    def test_fit_no_except(self):
        NumericEncoder().fit([1,2,3])

    def test_is_fit_marks_as_fitted(self):
        self.assertTrue(NumericEncoder().is_fit)

    def test_encode(self):
        self.assertEqual([1,2,3.23], NumericEncoder().encode(["1", "2", "3.23"]))

    def test_not_numeric_string(self):
        self.assertTrue(math.isnan(NumericEncoder().encode(["5 1"])[0]))

    def test_fit_encode(self):
        self.assertEqual([1,2,3.23], NumericEncoder().fit_encode(["1", "2", "3.23"]))

class OneHotEncoder_Tests(unittest.TestCase):

    def test_err_if_unkonwn_true(self):
        encoder = OneHotEncoder(err_if_unknown=True).fit(["1","1","1","0","0"])

        with self.assertRaises(Exception):
            self.assertEqual(encoder.encode(["2"]), [(0)])

    def test_err_if_unkonwn_false(self):
        encoder = OneHotEncoder().fit(["0","1","2"])
        self.assertEqual([(0,0,0)],encoder.encode(["5"]))

    def test_fit(self):
        encoder = OneHotEncoder()
        fit_encoder = encoder.fit(["0","1","2"])

        self.assertEqual(False, encoder.is_fit)
        self.assertEqual(True, fit_encoder.is_fit)
        self.assertEqual([(1,0,0),(0,1,0),(0,0,1),(0,1,0)], fit_encoder.encode(["0","1","2","1"]))

    def test_encode_sans_fit_exception(self):
        with self.assertRaises(CobaException):
            OneHotEncoder().encode(["0","1","2"])

    def test_init_values(self):
        encoder = OneHotEncoder(values=["0","1","2"])
        self.assertEqual(True, encoder.is_fit)
        self.assertEqual([(1,0,0),(0,1,0),(0,0,1),(0,1,0)], encoder.encode(["0","1","2","1"]))

    def test_fit_encode(self):
        self.assertEqual([(1,0,0),(0,1,0),(0,0,1),(0,1,0)], OneHotEncoder().fit_encode(["0","1","2","1"]))

class FactorEncoder_Tests(unittest.TestCase):
    def test_err_if_unkonwn_true(self):
        encoder = FactorEncoder(err_if_unknown=True).fit(["1","1","1","0","0"])

        with self.assertRaises(Exception):
            self.assertEqual(encoder.encode(["2"]), [(0)])

    def test_err_if_unkonwn_false(self):
        encoder = FactorEncoder().fit(["0","1","2"])
        self.assertTrue([float('nan')],encoder.encode(["5"]))

    def test_err_if_unkonwn_false(self):
        encoder = FactorEncoder().fit(["0","1","2"])
        self.assertTrue(math.isnan(encoder.encode(["5"])[0]))

    def test_fit(self):
        encoder     = FactorEncoder()
        fit_encoder = encoder.fit(["0","1","2"])

        self.assertEqual(False, encoder.is_fit)
        self.assertEqual(True, fit_encoder.is_fit)
        self.assertEqual([1,2,3,2], fit_encoder.encode(["0","1","2","1"]))

    def test_encode_sans_fit_exception(self):
        with self.assertRaises(CobaException):
            FactorEncoder().encode(["0","1","2"])

    def test_init_values(self):
        encoder = FactorEncoder(values=["0","1","2"])
        self.assertEqual(True, encoder.is_fit)
        self.assertEqual([1,2,3,2], encoder.encode(["0","1","2","1"]))

    def test_fit_encode(self):
        self.assertEqual([1,2,3,2], FactorEncoder().fit_encode(["0","1","2","1"]))

class InteractionsEncoder_Tests(unittest.TestCase):

    def test_dense_a(self):
        encoder = InteractionsEncoder(["a"])

        interactions = encoder.encode(x=[1,2,3], a=[1,2])

        self.assertEqual([1,2], interactions)

    def test_dense_x(self):
        encoder = InteractionsEncoder(["x"])

        interactions = encoder.encode(x=[1,2,3], a=[1,2])

        self.assertEqual([1,2,3], interactions)

    def test_dense_xxx(self):
        encoder = InteractionsEncoder(["xxx"])
        interactions = encoder.encode(x=[1,2,3], a=[1,2])
        self.assertEqual([1,2,3,4,6,9,8,12,18,27], interactions)

    def test_dense_x_a(self):
        encoder = InteractionsEncoder(["x", "a"])

        interactions = encoder.encode(x=[1,2,3], a=[1,2])

        self.assertEqual([1,2,3,1,2], interactions)

    def test_dense_x_a_xa_xxa(self):
        encoder = InteractionsEncoder(["x","a","xa","xxa"])

        interactions1 = encoder.encode(x=[1,2,3], a=[1,2])
        interactions2 = encoder.encode(x=[1,2,3], a=[1,2])

        self.assertCountEqual([1,2,3,1,2,1,2,3,2,4,6,1,2,3,4,6,9,2,4,6,8,12,18], interactions1)
        self.assertEqual(interactions1,interactions2)

    def test_sparse_x_a(self):
        encoder = InteractionsEncoder(["x","a"])

        interactions = encoder.encode(x={"1":1,"2":2}, a={"1":3,"2":4})

        self.assertEqual([("x1",1), ("x2",2), ("a1",3), ("a2",4)], interactions)

    def test_sparse_xa(self):
        encoder = InteractionsEncoder(["xa"])

        interactions = encoder.encode(x={"1":1,"2":2}, a={"1":3,"2":4})

        self.assertEqual([("x1a1",3), ("x1a2",4), ("x2a1",6), ("x2a2",8)], interactions)

    def test_sparse_xa_is_string(self):
        encoder = InteractionsEncoder(["xa"])

        interactions = encoder.encode(x={"1":1,"2":2}, a="a")

        self.assertEqual([("x1a0a",1), ("x2a0a",2)], interactions)


    def test_sparse_xa_with_strings(self):
        encoder = InteractionsEncoder(["xa"])

        interactions = encoder.encode(x={"1":"z","2":2}, a={"1":3,"2":4})

        self.assertEqual([("x1za1",3), ("x1za2",4), ("x2a1",6), ("x2a2",8)], interactions)

    def test_sparse_xxa(self):
        encoder = InteractionsEncoder(["xxa"])

        interactions = encoder.encode(x={"1":1,"2":2}, a={"1":3,"2":4})

        self.assertEqual([("x1x1a1",3), ("x1x1a2",4), ("x1x2a1",6), ("x1x2a2",8), ("x2x2a1",12), ("x2x2a2",16)], interactions)

    def test_string_a(self):
        encoder = InteractionsEncoder(["a"])

        interactions = encoder.encode(x=["a","b","c"], a=["d","e"])

        self.assertEqual([("a0d",1), ("a1e",1)], interactions)
    
    def test_string_x(self):
        encoder = InteractionsEncoder(["x"])

        interactions = encoder.encode(x=["a","b","c"], a=["d","e"])

        self.assertEqual([("x0a",1), ("x1b",1), ("x2c",1)], interactions)
    
    def test_string_xa(self):
        encoder = InteractionsEncoder(["xa"])

        interactions = encoder.encode(x=["a"], a=["d","e"])

        self.assertEqual([("x0aa0d",1), ("x0aa1e",1)], interactions)

    def test_string_numeric_xa(self):
        encoder = InteractionsEncoder(["xa"])

        interactions = encoder.encode(x=[2], a=["d","e"])

        self.assertEqual([("x0a0d",2), ("x0a1e",2)], interactions)

    def test_singular_string_a(self):
        encoder = InteractionsEncoder(["a"])

        interactions = encoder.encode(x=["a"], a="d")

        self.assertEqual([("a0d",1)], interactions)

    def test_singular_string_xa(self):
        encoder = InteractionsEncoder(["xa"])

        interactions = encoder.encode(x="abc", a="dbc")

        self.assertEqual([("x0abca0dbc",1)], interactions)

    def test_singular_numeric_xa(self):
        encoder = InteractionsEncoder(["xa"])

        interactions1 = encoder.encode(x=(1,2,3), a=2)
        interactions2 = encoder.encode(x=(1,2,3), a=2)

        self.assertCountEqual([2,4,6], interactions1)
        self.assertEqual(interactions1,interactions2)

    def test_string_tuple(self):
        encoder = InteractionsEncoder(["xa"])

        interactions1 = encoder.encode(x=('d',2), a=2)
        interactions2 = encoder.encode(x=('d',2), a=2)

        self.assertCountEqual([('x0da0', 2), ('x1a0',4) ], interactions1)
        self.assertEqual(interactions1,interactions2)

    def test_singular_string_abc(self):
        encoder = InteractionsEncoder(["abc"])

        interactions = encoder.encode(a=2, b=3, c=4)

        self.assertEqual([24], interactions)

if __name__ == '__main__':
    unittest.main()
