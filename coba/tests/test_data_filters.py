import unittest

from coba.data.filters import CsvReader, Flatten, Transpose, Encode
from coba.data.encoders import NumericEncoder, OneHotEncoder
from coba.tools import NoneLogger, CobaConfig

CobaConfig.Logger = NoneLogger()

class CsvReader_Tests(unittest.TestCase):
    def test_simple_sans_empty(self):
        self.assertEqual([['a','b','c'],['1','2','3']], list(CsvReader().filter(['a,b,c', '1,2,3'])))
    
    def test_simple_with_empty(self):
        self.assertEqual([['a','b','c'],['1','2','3']], list(CsvReader().filter(['a,b,c', '', '1,2,3', ''])))

class Transpose_Tests(unittest.TestCase):

    def test_dense_transpose(self):
        given    = [(1,2,3), (4,5,6)]
        expected = [(1,4),(2,5),(3,6)]

        self.assertEqual(expected, list(Transpose().filter(given)))
        #self.assertEqual(given   , list(Transpose().filter(expected)))

    def test_dense_transpose_not_strict(self):
        given    = [(1,2,3), (4,5,6,7,8)]
        expected = [(1,4),(2,5),(3,6)]
        
        self.assertEqual(expected, list(Transpose().filter(given)))

    def test_sparse_all_column_transpose(self):
        given    = [((0,1),(0,0)), ((2,),(0,))]
        expected = [((0,),(0,)),((0,),(0,)),((1,),(0,))]

        self.assertEqual(expected, list(Transpose().filter(given)))
        self.assertEqual(given   , list(Transpose().filter(expected)))

    def test_sparse_disordered_column_transpose(self):
        disordered_given = [((1,0),(1,0)), ((2,),(0,))]
        col_expected     = [((0,),(0,))  , ((0,),(1,)), ((1,),(0,))]
        row_expected     = [((0,1),(0,1)), ((2,),(0,))]

        self.assertEqual(col_expected, list(Transpose().filter(disordered_given)))
        self.assertEqual(row_expected, list(Transpose().filter(col_expected)))

    def test_sparse_missing_column_transpose(self):
        given    = [((0,1),(0,0)),((3,),(0,))]
        expected = [((0,),(0,)),((0,),(0,)),((),()),((1,),(0,))]

        self.assertEqual(expected, list(Transpose().filter(given)))
        self.assertEqual(given   , list(Transpose().filter(expected)))

    def test_sparse_transpose_tuples(self):
        given    = [((0,1),((0,1),0)),((2,),((1,1),))]
        expected = [((0,),((0,1),))  ,((0,),(0,)),((1,),((1,1),))]
        
        self.assertEqual(expected, list(Transpose().filter(given)))
        self.assertEqual(given   , list(Transpose().filter(expected)))

class Flatten_Tests(unittest.TestCase):

    def test_flatten_flat_list(self):
        self.assertEqual( [(1,2,3),(4,5,6)], list(Flatten().filter([[1,2,3],[4,5,6]])) )

    def test_flatten_deep_list(self):
        self.assertEqual( [(1,2,3),(4,5,6)], list(Flatten().filter([[1,[2,[3]]],[[4,5,6]]])) )

class Encode_Tests(unittest.TestCase):

    def test_dense_encode_numeric(self):
        encode = Encode([NumericEncoder(), NumericEncoder()])
        self.assertEqual( [[1,2,3],[4,5,6]], list(encode.filter([["1","2","3"],["4","5","6"]])))

    def test_dense_encode_onehot(self):
        encode = Encode([OneHotEncoder([1,2,3]), OneHotEncoder()])
        self.assertEqual( [[(1,0,0),(0,1,0),(0,1,0)],[(1,0,0),(0,1,0),(0,0,1)]], list(encode.filter([[1,2,2],[4,5,6]])))

    def test_dense_encode_mixed(self):
        encode = Encode([NumericEncoder(), OneHotEncoder()])
        self.assertEqual( [[1,2,3],[(1,0),(0,1),(0,1)]], list(encode.filter([[1,2,3],[4,5,5]])))

    def test_sparse_encode_numeric(self):
        encode = Encode([NumericEncoder(), NumericEncoder()])
        given    = [([0,1,2],["1","2","3"]),([0,1,2],["4","5","6"])]
        expected = [([0,1,2],[1,2,3]),([0,1,2],[4,5,6])]

        self.assertEqual(expected, list(encode.filter(given)))

    def test_sparse_encode_onehot(self):
        encode = Encode([OneHotEncoder([1,2,3]), OneHotEncoder()])
        given    = [([0,1,2],[1,2,2]),([0,1,2],[4,5,6])]
        expected = [([0,1,2],[(1,0,0),(0,1,0),(0,1,0)]),([0,1,2],[(1,0,0),(0,1,0),(0,0,1)])]

        self.assertEqual(expected, list(encode.filter(given)))

    def test_sparse_encode_mixed(self):
        encode = Encode([NumericEncoder(), OneHotEncoder()])
        given    = [([0,1,2],[1,2,3]),([0,1,2],[4,5,5])]
        expected = [([0,1,2],[1,2,3]),([0,1,2],[(1,0),(0,1),(0,1)])]

        self.assertEqual(expected, list(encode.filter(given)))

if __name__ == '__main__':
    unittest.main()