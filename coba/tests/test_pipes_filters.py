import unittest

from coba.pipes import LibSvmReader, ArffReader, CsvReader, Flatten, Encode, JsonEncode, Structure, Drop, Take, Identity, Shuffle
from coba.encodings import NumericEncoder, OneHotEncoder, StringEncoder
from coba.config import NullLogger, CobaConfig

CobaConfig.logger = NullLogger()

class Identity_Tests(unittest.TestCase):
    
    def test_ident(self):
        items = [1,2,3]

        mem_interactions = items
        idn_interactions = list(Identity().filter(items))

        self.assertEqual(idn_interactions, mem_interactions)

class Shuffle_Tests(unittest.TestCase):
    
    def test_shuffle(self):
        interactions = [ 1,2,3 ]
        shuffled_interactions = list(Shuffle(40).filter(interactions))

        self.assertEqual(3, len(interactions))
        self.assertEqual(interactions[0], interactions[0])
        self.assertEqual(interactions[1], interactions[1])
        self.assertEqual(interactions[2], interactions[2])

        self.assertEqual(3, len(shuffled_interactions))
        self.assertEqual(interactions[1], shuffled_interactions[0])
        self.assertEqual(interactions[2], shuffled_interactions[1])
        self.assertEqual(interactions[0], shuffled_interactions[2])

class Take_Tests(unittest.TestCase):
    
    def test_take1_no_seed(self):

        items = [ 1,2,3 ]
        take_items = list(Take(1).filter(items))

        self.assertEqual(1, len(take_items))
        self.assertEqual(items[0], take_items[0])

        self.assertEqual(3, len(items))
        self.assertEqual(items[0], items[0])
        self.assertEqual(items[1], items[1])
        self.assertEqual(items[2], items[2])

    def test_take2_no_seed(self):
        items = [ 1,2,3 ]
        take_items = list(Take(2).filter(items))

        self.assertEqual(2, len(take_items))
        self.assertEqual(items[0], take_items[0])
        self.assertEqual(items[1], take_items[1])

        self.assertEqual(3, len(items))
        self.assertEqual(items[0], items[0])
        self.assertEqual(items[1], items[1])
        self.assertEqual(items[2], items[2])

    def test_take3_no_seed(self):
        items = [ 1,2,3 ]
        take_items = list(Take(3).filter(items))

        self.assertEqual(3, len(take_items))
        self.assertEqual(items[0], take_items[0])
        self.assertEqual(items[1], take_items[1])
        self.assertEqual(items[2], take_items[2])

        self.assertEqual(3, len(items))
        self.assertEqual(items[0], items[0])
        self.assertEqual(items[1], items[1])
        self.assertEqual(items[2], items[2])

    def test_take4_no_seed(self):
        items = [ 1,2,3 ]
        take_items = list(Take(4).filter(items))

        self.assertEqual(3, len(items))
        self.assertEqual(0, len(take_items))

    def test_take2_seed(self):
        take_items = list(Take(2,seed=1).filter(range(10000)))

        self.assertEqual(2, len(take_items))
        self.assertLess(1, take_items[0])
        self.assertLess(1, take_items[1])

class CsvReader_Tests(unittest.TestCase):
    def test_dense_sans_empty(self):
        self.assertEqual([['a','b','c'],['1','2','3']], list(CsvReader(has_header=True).filter(['a,b,c', '1,2,3'])))
    
    def test_dense_with_empty(self):
        self.assertEqual([['a','b','c'],['1','2','3']], list(CsvReader(has_header=True).filter(['a,b,c', '', '1,2,3', ''])))

    def test_sparse(self):
        self.assertCountEqual([{'a':0,'b':1,'c':2},{0:'1',2:'2'},{1:'3'}], list(CsvReader(has_header=True).filter(['a,b,c', '{0 1,2 2}', '{1 3}'])))

class ArffReader_Tests(unittest.TestCase):

    def test_dense_sans_empty(self):
        lines = [
            "@relation news20",
            "@attribute a numeric",
            "@attribute B numeric",
            "@attribute c {0, class_B, class_C, class_D}",
            "@data",
            "1,2,class_B",
            "2,3,0",
        ]

        expected = [
            ['a','b','c'],
            [1,2,(0,1,0,0)],
            [2,3,(1,0,0,0)]
        ]
        
        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_dense_with_empty(self):
        lines = [
            "@relation news20",
            "@attribute A numeric",
            "@attribute B numeric",
            "@attribute C {0, class_B, class_C, class_D}",
            "@data",
            "",
            "",
            "1,2,class_B",
            "2,3,0",
            ""
        ]

        expected = [
            ['a','b','c'],
            [1,2,(0,1,0,0)],
            [2,3,(1,0,0,0)]
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
            [1,2,(0,1,0,0)],
            [2,3,(1,0,0,0)]
        ]
        
        self.assertEqual(expected, list(ArffReader().filter(lines)))

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
            {'a':0, 'b':1, 'c':2},
            {0:2, 1:3},
            {0:1, 1:1, 2:(0,1,0,0)},
            {1:1},
            {0:1,2:(0,0,0,1)}
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
            None,
            {0:['0'], 1:2, 2:3},
            {0:['1'], 1:1, 2:1},
            {0:['2'], 2:1},
            {0:['1'], 1:1}
        ]
        
        self.assertEqual(expected, list(LibSvmReader().filter(lines)))

    def test_trailing_whitespace(self):
        lines = [
            "0 1:2 2:3",
            "1 1:1 2:1   ",
            "2 2:1",
            "1 1:1",
        ]

        expected = [
            None,
            {0:['0'], 1:2, 2:3},
            {0:['1'], 1:1, 2:1},
            {0:['2'], 2:1},
            {0:['1'], 1:1}
        ]
        
        self.assertEqual(expected, list(LibSvmReader().filter(lines)))

class Flatten_Tests(unittest.TestCase):

    def test_dense_numeric_row_flatten(self):

        given_row0 = [1,2,3]
        given_row1 = [4,5,6]

        expected_row0 = [1,2,3]
        expected_row1 = [4,5,6]

        given    = [given_row0, given_row1]
        expected = [expected_row0, expected_row1]

        self.assertEqual( expected, list(Flatten().filter(given)) )

    def test_dense_onehot_row_flatten(self):

        given_row0 = [(0,1), 1]
        given_row1 = [(1,0), 2]

        expected_row0 = [1, 0, 1]
        expected_row1 = [2, 1, 0]

        given    = [given_row0   , given_row1   ]
        expected = [expected_row0, expected_row1]

        self.assertEqual(expected, list(Flatten().filter(given)) )

    def test_sparse_numeric_row_flatten(self):

        given_row0 = { 0:2, 1:3, 2:4 }
        given_row1 = { 0:1, 1:2, 2:3 }

        expected_row0 = { 0:2, 1:3, 2:4 }
        expected_row1 = { 0:1, 1:2, 2:3 }

        given    = [given_row0, given_row1]
        expected = [expected_row0, expected_row1]

        self.assertEqual(expected, list(Flatten().filter(given)) )

    def test_sparse_onehot_row_flatten(self):

        given_row0 = {0: (0,1), 1:1 }
        given_row1 = {0: (1,0), 1:1 }

        expected_row0 = { (0,0): 0, (0,1): 1, 1:1}
        expected_row1 = { (0,0): 1, (0,1): 0, 1:1}

        given    = [given_row0, given_row1]
        expected = [expected_row0, expected_row1]

        self.assertEqual(expected, list(Flatten().filter(given)) )

    def test_string_row_flatten(self):
        given_row0 = [ "abc", "def", "ghi" ]
        expected_row0 = [ "abc", "def", "ghi" ]

        given    = [ given_row0    ]
        expected = [ expected_row0 ]

        self.assertEqual(expected, list(Flatten().filter(given)) )

class Encode_Tests(unittest.TestCase):

    def test_dense_encode_numeric_sans_header(self):
        encode = Encode({0:NumericEncoder(), 1:NumericEncoder()}, has_header=False)
        self.assertEqual( [[1,2],[4,5]], list(encode.filter([["1","2"],["4","5"]])))

    def test_dense_encode_onehot_sans_header(self):
        encode = Encode({0:OneHotEncoder([1,2,3]), 1:OneHotEncoder()}, has_header=False)
        self.assertEqual([[(1,0,0),(1,0,0)],[(0,1,0),(0,1,0)], [(0,1,0),(0,0,1)]], list(encode.filter([[1,4], [2,5], [2,6]])))

    def test_dense_encode_mixed_sans_header(self):
        encode = Encode({0:NumericEncoder(), 1:OneHotEncoder()}, has_header=False)
        self.assertEqual( [[1,(1,0)],[2,(0,1)],[3,(0,1)]], list(encode.filter([[1,4],[2,5],[3,5]])))

    def test_sparse_encode_numeric_sans_header(self):
        encode = Encode({0:NumericEncoder(), 1:NumericEncoder()}, has_header=False)
        given    = [ {0:"1",1:"4"}, {0:"2",1:"5"}, {0:"3",1:"6"}]
        expected = [ {0:1,1:4}, {0:2,1:5}, {0:3,1:6}]

        self.assertEqual(expected, list(encode.filter(given)))

    def test_sparse_encode_onehot_sans_header(self):
        encode   = Encode({0:OneHotEncoder([1,2,3]), 1:OneHotEncoder()},has_header=False)
        given    = [{0:1,1:4}, {0:2,1:5}, {0:2,1:6}]
        expected = [ {0:(1,0,0), 1:(1,0,0)}, {0:(0,1,0), 1:(0,1,0)}, {0:(0,1,0), 1:(0,0,1)}]

        self.assertEqual(expected, list(encode.filter(given)))

    def test_sparse_encode_mixed_sans_header(self):
        encode = Encode({0:NumericEncoder(), 1:OneHotEncoder()}, has_header=False)
        given    = [{0:"1",1:4},{0:"2",1:5},{0:"3",1:5}]
        expected = [{0:1,1:(1,0)},{0:2,1:(0,1)},{0:3,1:(0,1)}]

        self.assertEqual(expected, list(encode.filter(given)))
    
    def test_dense_encode_onehot_with_header(self):
        encode = Encode({'a':OneHotEncoder([1,2,3]), 'b':OneHotEncoder()}, has_header=True)
        self.assertEqual([['a','b'], [(1,0,0),(1,0,0)],[(0,1,0),(0,1,0)], [(0,1,0),(0,0,1)]], list(encode.filter([['a','b'], [1,4], [2,5], [2,6]])))

    def test_sparse_encode_onehot_sans_header(self):
        encode   = Encode({'a':OneHotEncoder([1,2,3]), 'b':OneHotEncoder()},has_header=True)
        given    = [{'a':0, 'b':1}, {0:1,1:4}, {0:2,1:5}, {0:2,1:6}]
        expected = [{'a':0, 'b':1}, {0:(1,0,0), 1:(1,0,0)}, {0:(0,1,0), 1:(0,1,0)}, {0:(0,1,0), 1:(0,0,1)}]

        self.assertEqual(expected, list(encode.filter(given)))

    def test_dense_encode_onehot_with_header_and_extra_encoder(self):
        encode = Encode({'a':OneHotEncoder([1,2,3]), 'b':OneHotEncoder(), 'c':StringEncoder()}, has_header=True)
        self.assertEqual([['a','b'], [(1,0,0),(1,0,0)],[(0,1,0),(0,1,0)], [(0,1,0),(0,0,1)]], list(encode.filter([['a','b'], [1,4], [2,5], [2,6]])))

class JsonEncode_Tests(unittest.TestCase):
    def test_list_minified(self):
        self.assertEqual('[1,2]',JsonEncode().filter([1,2.]))

    def test_list_list_minified(self):
        self.assertEqual('[1,[2,[1,2]]]',JsonEncode().filter([1,[2.,[1,2.]]]))
    
    def test_tuple_minified(self):
        self.assertEqual('[1,2]',JsonEncode().filter((1,2.)))

    def test_dict_minified(self):
        self.assertEqual('{"a":[1.23,2],"b":{"c":1}}',JsonEncode().filter({'a':[1.23,2],'b':{'c':1.}}))

class Structures_Tests(unittest.TestCase):

    def test_dense_numeric_row_structure(self):

        given_row0 = [1,2,3]
        given_row1 = [4,5,6]

        expected_row0 = [ [1,2] ,3]
        expected_row1 = [ [4,5] ,6]

        given    = [None, given_row0, given_row1]
        expected = [expected_row0, expected_row1]

        self.assertEqual( expected, list(Structure([None, 2]).filter(given)) )

    def test_sparse_numeric_row_structure(self):

        given_row0 = { 0:2, 1:3, 2:4 }
        given_row1 = { 0:1, 1:2, 2:3 }

        expected_row0 = [ {0:2,1:3}, 4 ]
        expected_row1 = [ {0:1,1:2}, 3 ]
        
        given    = [None, given_row0, given_row1]
        expected = [expected_row0, expected_row1]

        self.assertEqual( expected, list(Structure([None, 2]).filter(given)) )

class Drops_Tests(unittest.TestCase):

    def test_dense_sans_header_drop_single_col(self):

        given_row0 = [1,2,3]
        given_row1 = [4,5,6]

        expected_row0 = [ 1, 3 ]
        expected_row1 = [ 4, 6 ]

        given    = [None, given_row0, given_row1]
        expected = [None, expected_row0, expected_row1]

        self.assertEqual( expected, list(Drop(drop_cols=[1]).filter(given)) )

    def test_dense_sans_header_drop_double_col(self):

        given_row0 = [1,2,3]
        given_row1 = [4,5,6]

        expected_row0 = [ 1 ]
        expected_row1 = [ 4 ]

        given    = [None, given_row0, given_row1]
        expected = [None, expected_row0, expected_row1]

        self.assertEqual( expected, list(Drop(drop_cols=[1,2]).filter(given)) )

    def test_dense_with_header_drop_single_col(self):

        given_row0 = [1,2,3]
        given_row1 = [4,5,6]

        expected_row0 = [ 1, 3 ]
        expected_row1 = [ 4, 6 ]

        given    = [['a','b','c'], given_row0, given_row1]
        expected = [['a','c'], expected_row0, expected_row1]

        self.assertEqual( expected, list(Drop(drop_cols=['b']).filter(given)) )

    def test_dense_with_header_drop_double_col(self):

        given_row0 = [1,2,3]
        given_row1 = [4,5,6]

        expected_row0 = [ 1 ]
        expected_row1 = [ 4 ]

        given    = [['a','b','c'], given_row0, given_row1]
        expected = [['a'], expected_row0, expected_row1]

        self.assertEqual( expected, list(Drop(drop_cols=['b','c']).filter(given)) )

    def test_sparse_sans_header_drop_single_col(self):

        given_row0 = {0:1,1:2,2:3}
        given_row1 = {0:4,1:5,2:6}

        expected_row0 = { 0:1, 2:3 }
        expected_row1 = { 0:4, 2:6 }

        given    = [None, given_row0, given_row1]
        expected = [None, expected_row0, expected_row1]

        self.assertEqual( expected, list(Drop(drop_cols=[1]).filter(given)) )

    def test_sparse_sans_header_drop_double_col(self):

        given_row0 = {0:1,1:2,2:3}
        given_row1 = {0:4,1:5,2:6}

        expected_row0 = { 0:1 }
        expected_row1 = { 0:4 }

        given    = [None, given_row0, given_row1]
        expected = [None, expected_row0, expected_row1]

        self.assertEqual( expected, list(Drop(drop_cols=[1,2]).filter(given)) )

    def test_sparse_with_header_drop_single_col(self):

        given_row0 = {0:1,1:2,2:3}
        given_row1 = {0:4,1:5,2:6}

        expected_row0 = { 0:1, 2:3 }
        expected_row1 = { 0:4, 2:6 }

        given    = [{'a':0,'b':1,'c':2}, given_row0, given_row1]
        expected = [{'a':0      ,'c':2}, expected_row0, expected_row1]

        self.assertEqual( expected, list(Drop(drop_cols=['b']).filter(given)) )

    def test_sparse_with_header_drop_double_col(self):

        given_row0 = {0:1,1:2,2:3}
        given_row1 = {0:4,1:5,2:6}

        expected_row0 = { 0:1 }
        expected_row1 = { 0:4 }

        given    = [{'a':0,'b':1,'c':2}, given_row0, given_row1]
        expected = [{'a':0}, expected_row0, expected_row1]

        self.assertEqual( expected, list(Drop(drop_cols=['b','c']).filter(given)) )

if __name__ == '__main__':
    unittest.main()