import timeit
import unittest

from coba.data.filters import CsvReader, ColEncoder, ColRemover, Flatten, CsvTranspose, LabeledCsvCleaner, Transpose, Encode
from coba.data.encoders import NumericEncoder, OneHotEncoder, StringEncoder
from coba.tools import NoneLogger, CobaConfig

CobaConfig.Logger = NoneLogger()

class CsvReader_Tests(unittest.TestCase):
    def test_simple_sans_empty(self):
        self.assertEqual([['a','b','c'],['1','2','3']], list(CsvReader().filter(['a,b,c', '1,2,3'])))
    
    def test_simple_with_empty(self):
        self.assertEqual([['a','b','c'],['1','2','3']], list(CsvReader().filter(['a,b,c', '', '1,2,3', ''])))

class CsvTransposer_Tests(unittest.TestCase):

    def test_simple_sans_empty(self):
        self.assertEqual([('a','1'),('b','2'),('c','3')], list(CsvTranspose().filter([['a','b','c'],['1','2','3']])))

    def test_simple_with_empty(self):
        self.assertEqual([('a','1'),('b','2'),('c','3')], list(CsvTranspose().filter([['a','b','c'],['1','2','3'],[]])))

class ColEncoder_Tests(unittest.TestCase):

    def test_with_headers_1(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        decoder = ColEncoder(['a','b','c'],[NumericEncoder(),NumericEncoder(),NumericEncoder()])
        self.assertEqual([['a',1,4],['b',2,5],['c',3,6]], list(decoder.filter(csv)))

    def test_with_headers_2(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        decoder = ColEncoder(['a','b','c'],[NumericEncoder(),StringEncoder(),NumericEncoder()])
        self.assertEqual([['a',1,4],['b','2','5'],['c',3,6]], list(decoder.filter(csv)))

    def test_with_headers_3(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        decoder = ColEncoder(['a','c','b'],[NumericEncoder(),NumericEncoder(),StringEncoder()])
        self.assertEqual([['a',1,4],['b','2','5'],['c',3,6]], list(decoder.filter(csv)))

    def test_with_indexes_1(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        decoder = ColEncoder([],[NumericEncoder(),NumericEncoder(),NumericEncoder()])
        self.assertEqual([['a',1,4],['b',2,5],['c',3,6]], list(decoder.filter(csv)))

    def test_with_indexes_2(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        decoder = ColEncoder([],[NumericEncoder(),StringEncoder(),NumericEncoder()])
        self.assertEqual([['a',1,4],['b','2','5'],['c',3,6]], list(decoder.filter(csv)))
    
    def test_with_indexes_3(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        decoder = ColEncoder([],[OneHotEncoder(['1','2','3','4']),StringEncoder(),NumericEncoder()])
        out     = list(decoder.filter(csv))

        print(list(CsvTranspose().filter(out)))

        #self.assertEqual([['a',[1,0,0,0],[0,0,0,1]],['b','2','5'],['c',3,6]], )

class ColRemover_Tests(unittest.TestCase):

    def test_with_headers_1(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        remover = ColRemover()
        self.assertEqual([['a','1','4'],['b','2','5'],['c','3','6']], list(remover.filter(csv)))

    def test_with_headers_2(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        remover = ColRemover(['b'])
        self.assertEqual([['a','1','4'],['c','3','6']], list(remover.filter(csv)))

    def test_with_headers_3(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        remover = ColRemover(['b','a'])
        self.assertEqual([['c','3','6']], list(remover.filter(csv)))

    def test_with_indexes_1(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        remover = ColRemover()
        self.assertEqual([['a','1','4'],['b','2','5'],['c','3','6']], list(remover.filter(csv)))

    def test_with_indexes_2(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        remover = ColRemover([1])
        self.assertEqual([['a','1','4'],['c','3','6']], list(remover.filter(csv)))

    def test_with_indexes_3(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        remover = ColRemover([1,0])
        self.assertEqual([['c','3','6']], list(remover.filter(csv)))

class LabeledCsvCleaner_Tests(unittest.TestCase):

    def test_large_from_memory(self):
        headers      = list(map(str,range(15)))
        values       = [["1","0"]*15]*100000
        table        = [headers]+values
        label_col    = 0
        encoders     = [NumericEncoder()]*15

        clean_table = lambda: LabeledCsvCleaner(label_col, headers, encoders, [], rmv_header=True).filter(table)

        time = min(timeit.repeat(clean_table, repeat=2, number=1))

        #was approximately 0.5 at best performance
        self.assertLess(time, 3)

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

    def test_sparse_transpose(self):
        given    = [([0,1],[0,0]), ([2],[0])]
        expected = [([0],[0]),([0],[0]),([1],[0])]
        
        self.assertEqual(expected, list(Transpose().filter(given)))
        self.assertEqual(given   , list(Transpose().filter(expected)))
    
    def test_sparse_transpose_tuples(self):
        given    = [([0,1],[(0,1),0]),([2],[(1,1)])]
        expected = [([0],[(0,1)]),([0],[0]),([1],[(1,1)])]
        
        self.assertEqual(expected, list(Transpose().filter(given)))
        self.assertEqual(given   , list(Transpose().filter(expected)))

class Flatten_Tests(unittest.TestCase):

    def test_flatten_flat_list(self):
        self.assertEqual( [[1,2,3],[4,5,6]], list(Flatten().filter([[1,2,3],[4,5,6]])) )

    def test_flatten_deep_list(self):
        self.assertEqual( [[1,2,3],[4,5,6]], list(Flatten().filter([[1,[2,[3]]],[[4,5,6]]])) )

class Encode_Tests(unittest.TestCase):

    def test_dense_encode_numeric(self):
        encode = Encode([NumericEncoder(), NumericEncoder()])
        self.assertEqual( [[1,2,3],[4,5,6]], list(encode.filter([[1,2,3],[4,5,6]])))

    def test_dense_encode_onehot(self):
        encode = Encode([OneHotEncoder([1,2,3]), OneHotEncoder()])
        self.assertEqual( [[(1,0,0),(0,1,0),(0,1,0)],[(1,0,0),(0,1,0),(0,0,1)]], list(encode.filter([[1,2,2],[4,5,6]])))

    def test_dense_encode_mixed(self):
        encode = Encode([NumericEncoder(), OneHotEncoder()])
        self.assertEqual( [[1,2,3],[(1,0),(0,1),(0,1)]], list(encode.filter([[1,2,3],[4,5,5]])))

    def test_sparse_encode_numeric(self):
        encode = Encode([NumericEncoder(), NumericEncoder()])
        given    = [([0,1,2],[1,2,3]),([0,1,2],[4,5,6])]
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