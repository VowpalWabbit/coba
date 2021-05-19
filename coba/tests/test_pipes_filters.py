import unittest

from coba.pipes import LibSvmReader, ArffReader, CsvReader, Flatten, Transpose, Encode
from coba.encodings import NumericEncoder, OneHotEncoder
from coba.config import NoneLogger, CobaConfig

CobaConfig.Logger = NoneLogger()

class CsvReader_Tests(unittest.TestCase):
    def test_dense_sans_empty(self):
        self.assertEqual([['a','b','c'],['1','2','3']], list(CsvReader().filter(['a,b,c', '1,2,3'])))
    
    def test_dense_with_empty(self):
        self.assertEqual([['a','b','c'],['1','2','3']], list(CsvReader().filter(['a,b,c', '', '1,2,3', ''])))

    def test_sparse(self):
        self.assertEqual([((0,1,2),('a','b','c')), ((0,2),('1','2')), ((1,),('3',))], list(CsvReader().filter(['a,b,c', '{0 1,2 2}', '{1 3}'])))

class ArffReader_Tests(unittest.TestCase):

    def test_dense_sans_empty(self):
        lines = [
            "@relation news20",
            "@attribute a numeric",
            "@attribute b numeric",
            "@attribute c {0, class_B, class_C, class_D}",
            "@data",
            "1,2,class_B",
            "2,3,0",
        ]

        expected = [
            ['a','b','c'],
            ['1','2','class_B'],
            ['2','3','0']
        ]
        
        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_dense_with_empty(self):
        lines = [
            "@relation news20",
            "@attribute a numeric",
            "@attribute b numeric",
            "@attribute c {0, class_B, class_C, class_D}",
            "@data",
            "",
            "",
            "1,2,class_B",
            "2,3,0",
            ""
        ]

        expected = [
            ['a','b','c'],
            ['1','2','class_B'],
            ['2','3','0']
        ]
        
        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_dense_with_comments(self):
        lines = [
            "%This is a comment",
            "@relation news20",
            "@attribute a numeric",
            "@attribute b numeric",
            "@attribute c {0, class_B, class_C, class_D}",
            "@data",
            "1,2,class_B",
            "2,3,0"
        ]

        expected = [
            ['a','b','c'],
            ['1','2','class_B'],
            ['2','3','0']
        ]
        
        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_simple_with_empty(self):
        self.assertEqual([['a','b','c'],['1','2','3']], list(CsvReader().filter(['a,b,c', '', '1,2,3', ''])))

    def test_sparse(self):
        lines = [
            "@relation news20",
            "@attribute a numeric",
            "@attribute b numeric",
            "@attribute c {0, class_B, class_C, class_D}",
            "@data",
            "{0 2,1 3}",
            "{0 1,1 1,2 class_B}",
            "{1 1}",
            "{0 1,2 class_D}",
        ]

        expected = [
            ((0,1,2),('a','b','c')), 
            ((0,1),('2','3')), 
            ((0,1,2),('1','1','class_B')),
            ((1,),('1',)),
            ((0,2),('1','class_D'))
        ]
        
        self.assertEqual(expected, list(ArffReader().filter(lines)))

class LibsvmReader_Tests(unittest.TestCase):
    def test_sparse(self):
        lines = [
            "0 1:2 2:3",
            "1 1:1 2:1",
            "2 2:1",
            "1 1:1",
        ]

        expected = [
            ((0,1,2),(0, 2, 3)),
            ((0,1,2),(1, 1, 1)),
            ((0,2)  ,(2, 1   )),
            ((0,1)  ,(1, 1   )),
        ]
        
        self.assertEqual(expected, list(LibSvmReader().filter(lines)))

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

    def test_sparse_with_all_columns_transpose(self):
        row0 = ((0,1),(0,1))
        row1 = ((2, ),(0, ))

        col0 = ((0,),(0,))
        col1 = ((0,),(1,))
        col2 = ((1,),(0,))

        self.assertEqual([col0,col1,col2], list(Transpose().filter([row0,row1])))
        self.assertEqual([row0,row1], list(Transpose().filter([col0,col1,col2])))

    def test_sparse_with_disordered_column_transpose(self):
        row0 = ((1,0),(1,0))
        row1 = ((2, ),(0, ))

        col0 = ((0,),(0,))
        col1 = ((0,),(1,))
        col2 = ((1,),(0,))

        self.assertEqual([col0,col1,col2], list(Transpose().filter([row0,row1])))

    def test_sparse_with_missing_column_transpose(self):
        row0 = ((0,1),(0,0))
        row1 = ((3, ),(0, ))

        col0 = ((0,),(0,))
        col1 = ((0,),(0,))
        col2 = ((),())
        col3 = ((1,),(0,))

        self.assertEqual([col0,col1,col2,col3], list(Transpose().filter([row0,row1])))
        self.assertEqual([row0,row1], list(Transpose().filter([col0,col1,col2,col3])))

    def test_sparse_with_tuples_transpose(self):

        row0 = ((0,1),((0,1),0))
        row1 = ((2,),((1,1),))

        col0 = ((0,),((0,1),))
        col1 = ((0,),(0,))
        col2 = ((1,),((1,1),))

        self.assertEqual([col0,col1,col2], list(Transpose().filter([row0,row1])))
        self.assertEqual([row0,row1]     , list(Transpose().filter([col0,col1,col2])))

class Flatten_Tests(unittest.TestCase):

    def test_dense_numeric_col_flatten(self):

        given_col0 = [1,2,3]
        given_col1 = [4,5,6]

        expected_col0 = (1,2,3)
        expected_col1 = (4,5,6)

        given    = [given_col0, given_col1]
        expected = [expected_col0, expected_col1]

        self.assertEqual( expected, list(Flatten().filter(given)) )

    def test_dense_onehot_col_flatten(self):

        given_col0 = [(0,1), (1,0), (1,0)]
        given_col1 = [1    , 2    , 3    ]

        expected_col0 = (0, 1, 1)
        expected_col1 = (1, 0, 0)
        expected_col2 = (1, 2, 3)

        given    = [given_col0, given_col1]
        expected = [expected_col0, expected_col1, expected_col2]

        self.assertEqual(expected, list(Flatten().filter(given)) )

    def test_sparse_numeric_col_flatten(self):

        given_col0 = [ [0,1,2], [2,3,4] ]
        given_col1 = [ [0,1,2], [1,2,3] ]

        expected_col0 = ( (0,1,2), (2, 3, 4) )
        expected_col1 = ( (0,1,2), (1, 2, 3) )

        given    = [given_col0, given_col1]
        expected = [expected_col0, expected_col1]

        self.assertEqual(expected, list(Flatten().filter(given)) )


    def test_sparse_onehot_col_flatten(self):

        given_col0 = [ [0,1,2], [(0,1), (1,0), (1,0)] ]
        given_col1 = [ [0,1,2], [1    , 2    , 3    ] ]

        expected_col0 = ( (0,1,2), (0, 1, 1) )
        expected_col1 = ( (0,1,2), (1, 0, 0) )
        expected_col2 = ( (0,1,2), (1, 2, 3) )

        given    = [given_col0, given_col1]
        expected = [expected_col0, expected_col1, expected_col2]

        self.assertEqual(expected, list(Flatten().filter(given)) )

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
        encode   = Encode([OneHotEncoder([1,2,3]), OneHotEncoder()])
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