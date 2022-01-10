import unittest
from coba.exceptions import CobaException

from coba.pipes import Flatten, Encode, JsonEncode, Structure, Drop, Take, Identity, Shuffle
from coba.encodings import NumericEncoder, OneHotEncoder, StringEncoder
from coba.contexts import NullLogger, CobaContext
from coba.pipes.filters import Default, Reservoir

CobaContext.logger = NullLogger()

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

    def test_bad_seed(self):
        with self.assertRaises(ValueError):
            Shuffle(-1)

        with self.assertRaises(ValueError):
            Shuffle('A')

class Take_Tests(unittest.TestCase):
    
    def test_bad_count(self):
        with self.assertRaises(ValueError):
            Take(-1)

        with self.assertRaises(ValueError):
            Take('A')

    def test_take1(self):

        items = [ 1,2,3 ]
        take_items = list(Take(1).filter(items))

        self.assertEqual(1, len(take_items))
        self.assertEqual(items[0], take_items[0])

        self.assertEqual(3, len(items))
        self.assertEqual(items[0], items[0])
        self.assertEqual(items[1], items[1])
        self.assertEqual(items[2], items[2])

    def test_take2(self):
        items = [ 1,2,3 ]
        take_items = list(Take(2).filter(items))

        self.assertEqual(2, len(take_items))
        self.assertEqual(items[0], take_items[0])
        self.assertEqual(items[1], take_items[1])

        self.assertEqual(3, len(items))
        self.assertEqual(items[0], items[0])
        self.assertEqual(items[1], items[1])
        self.assertEqual(items[2], items[2])

    def test_take3(self):
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

    def test_take4(self):
        items = [ 1,2,3 ]
        take_items = list(Take(4).filter(items))

        self.assertEqual(3, len(items))
        self.assertEqual(0, len(take_items))

class Resevoir_Tests(unittest.TestCase):

    def test_bad_count(self):
        with self.assertRaises(ValueError):
            Reservoir(-1)

        with self.assertRaises(ValueError):
            Reservoir('A')

    def test_take_seed(self):
        take_items = list(Reservoir(2,seed=1).filter(range(10000)))
        self.assertEqual(2, len(take_items))
        self.assertLess(0, take_items[0])
        self.assertLess(0, take_items[1])

    def test_take_none_seed(self):
        self.assertEqual(list(range(10)), list(Reservoir(None,seed=1).filter(range(10))))

    def test_take_0_seed(self):
        self.assertEqual(0, len(list(Reservoir(0,seed=1).filter(range(10)))))

    def test_take_10_has_5(self):
        self.assertEqual(0, len(list(Reservoir(10,seed=1).filter(range(5)))))

class Flatten_Tests(unittest.TestCase):

    def test_dense_numeric_row_flatten_1(self):

        given_row0 = [1,2,3]
        given_row1 = [4,5,6]

        expected_row0 = [1,2,3]
        expected_row1 = [4,5,6]

        given    = [given_row0, given_row1]
        expected = [expected_row0, expected_row1]

        self.assertEqual( expected, list(Flatten().filter(given)) )

    def test_dense_numeric_row_flatten_2(self):

        given_row0 = [[1],2,3]
        given_row1 = [[4],5,6]

        expected_row0 = [1,2,3]
        expected_row1 = [4,5,6]

        given    = [given_row0, given_row1]
        expected = [expected_row0, expected_row1]

        self.assertEqual( expected, list(Flatten().filter(given)) )


    def test_dense_onehot_row_flatten(self):

        given_row0 = [(0,1), 1]
        given_row1 = [(1,0), 2]

        expected_row0 = [0, 1, 1]
        expected_row1 = [1, 0, 2]

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

    def test_bad_data(self):

        with self.assertRaises(CobaException):
            list(Flatten().filter(['abc']))

class Encode_Tests(unittest.TestCase):

    def test_encode_empty(self):
        encode = Encode({0:NumericEncoder()})
        self.assertEqual([],list(encode.filter([])))

    def test_dense_encode_numeric(self):
        encode = Encode({0:NumericEncoder(), 1:NumericEncoder()})
        self.assertEqual([[1,2],[4,5]], list(encode.filter([["1","2"],["4","5"]])))

    def test_dense_encode_onehot(self):
        encode = Encode({0:OneHotEncoder([1,2,3]), 1:OneHotEncoder()})
        self.assertEqual([[(1,0,0),(1,0,0)],[(0,1,0),(0,1,0)], [(0,1,0),(0,0,1)]], list(encode.filter([[1,4], [2,5], [2,6]])))

    def test_dense_encode_mixed(self):
        encode = Encode({0:NumericEncoder(), 1:OneHotEncoder()})
        self.assertEqual([[1,(1,0)],[2,(0,1)],[3,(0,1)]], list(encode.filter([[1,4],[2,5],[3,5]])))

    def test_sparse_encode_numeric(self):
        encode   = Encode({0:NumericEncoder(), 1:NumericEncoder()})
        given    = [ {0:"1",1:"4"}, {0:"2",1:"5"}, {0:"3",1:"6"}]
        expected = [ {0:1,1:4}, {0:2,1:5}, {0:3,1:6}]

        self.assertEqual(expected, list(encode.filter(given)))

    def test_sparse_encode_onehot(self):
        encode   = Encode({0:OneHotEncoder([1,2,3]), 1:OneHotEncoder()})
        given    = [{0:1}, {0:2,1:5}, {0:2,1:6}]
        expected = [{0:(1,0,0)}, {0:(0,1,0), 1:(0,1,0)}, {0:(0,1,0), 1:(0,0,1)}]

        self.assertEqual(expected, list(encode.filter(given)))

    def test_sparse_encode_mixed(self):
        encode   = Encode({0:NumericEncoder(), 1:OneHotEncoder()})
        given    = [{0:"1",1:4},{0:"2",1:5},{0:"3",1:5}]
        expected = [{0:1,1:(1,0)},{0:2,1:(0,1)},{0:3,1:(0,1)}]

        self.assertEqual(expected, list(encode.filter(given)))

    def test_dense_encode_onehot_with_header_and_extra_encoder(self):
        encode = Encode({0:OneHotEncoder([1,2,3]), 1:OneHotEncoder(), 2:StringEncoder()})
        self.assertEqual([[(1,0,0),(1,0,0)],[(0,1,0),(0,1,0)],[(0,1,0),(0,0,1)]], list(encode.filter([[1,4], [2,5], [2,6]])))

    def test_ignore_missing_value(self):
        encode = Encode({0:OneHotEncoder([1,2,3]), 1:OneHotEncoder()}, missing_val="?")
        self.assertEqual([[(1,0,0),'?'],[(0,1,0),(1,0)],[(0,1,0),(0,1)]], list(encode.filter([[1,'?'], [2,5], [2,6]])))

class JsonEncode_Tests(unittest.TestCase):
    def test_bool_minified(self):
        self.assertEqual('true',JsonEncode().filter(True))

    def test_list_minified(self):
        self.assertEqual('[1,2]',JsonEncode().filter([1,2.]))

    def test_list_list_minified(self):
        self.assertEqual('[1,[2,[1,2]]]',JsonEncode().filter([1,[2.,[1,2.]]]))

    def test_tuple_minified(self):
        self.assertEqual('[1,2]',JsonEncode().filter((1,2.)))

    def test_dict_minified(self):
        self.assertEqual('{"a":[1.23,2],"b":{"c":1}}',JsonEncode().filter({'a':[1.23,2],'b':{'c':1.}}))

    def test_inf(self):
        self.assertEqual('Infinity',JsonEncode().filter(float('inf')))
        self.assertEqual('-Infinity',JsonEncode().filter(-float('inf')))

    def test_nan(self):
        self.assertEqual('NaN',JsonEncode().filter(float('nan')))

    def test_not_serializable(self):
        with self.assertRaises(TypeError) as e:
            JsonEncode().filter({1,2,3})
        self.assertIn("set", str(e.exception))
        self.assertIn("not JSON serializable", str(e.exception))

    def test_not_minified_list(self):
        self.assertEqual('[1.0, 2.0]',JsonEncode(minify=False).filter([1.,2.]))

class Structures_Tests(unittest.TestCase):

    def test_dense_numeric_row_list_structure(self):

        given_row0 = [1,2,3]
        given_row1 = [4,5,6]

        expected_row0 = [ [1,2] ,3]
        expected_row1 = [ [4,5] ,6]

        given    = [given_row0, given_row1]
        expected = [expected_row0, expected_row1]

        self.assertEqual(expected, list(Structure([None, 2]).filter(given)) )

    def test_dense_numeric_row_tuple_structure(self):

        given_row0 = [1,2,3]
        given_row1 = [4,5,6]

        expected_row0 = ([1,2], 3)
        expected_row1 = ([4,5], 6)

        given    = [given_row0, given_row1]
        expected = [expected_row0, expected_row1]

        self.assertEqual(expected, list(Structure((None, 2)).filter(given)) )


    def test_dense_numeric_row_value_structure(self):

        given_row0 = [1,2,3]
        given_row1 = [4,5,6]

        expected_row0 = 3
        expected_row1 = 6

        given    = [given_row0, given_row1]
        expected = [expected_row0, expected_row1]

        self.assertEqual(expected, list(Structure(2).filter(given)) )


    def test_sparse_numeric_row_structure(self):

        given_row0 = { 0:2, 1:3, 2:4 }
        given_row1 = { 0:1, 1:2, 2:3 }

        expected_row0 = [ {0:2,1:3}, 4 ]
        expected_row1 = [ {0:1,1:2}, 3 ]

        given    = [given_row0, given_row1]
        expected = [expected_row0, expected_row1]

        self.assertEqual( expected, list(Structure([None, 2]).filter(given)) )

class Drops_Tests(unittest.TestCase):

    def test_dense_sans_header_drop_single_col(self):

        given_row0 = [1,2,3]
        given_row1 = [4,5,6]

        expected_row0 = [ 1, 3 ]
        expected_row1 = [ 4, 6 ]

        given    = [given_row0, given_row1]
        expected = [expected_row0, expected_row1]

        self.assertEqual( expected, list(Drop(drop_cols=[1]).filter(given)) )

    def test_dense_sans_header_drop_double_col(self):

        given_row0 = [1,2,3]
        given_row1 = [4,5,6]

        expected_row0 = [ 1 ]
        expected_row1 = [ 4 ]

        given    = [given_row0, given_row1]
        expected = [expected_row0, expected_row1]

        self.assertEqual( expected, list(Drop(drop_cols=[1,2]).filter(given)) )

    def test_dense_with_header_drop_single_col(self):

        given_row0 = [1,2,3]
        given_row1 = [4,5,6]

        expected_row0 = [ 1, 3 ]
        expected_row1 = [ 4, 6 ]

        given    = [['a','b','c'], given_row0, given_row1]
        expected = [['a','c'], expected_row0, expected_row1]

        self.assertEqual( expected, list(Drop(drop_cols=[1]).filter(given)) )

    def test_dense_with_header_drop_double_col(self):

        given_row0 = [1,2,3]
        given_row1 = [4,5,6]

        expected_row0 = [ 1 ]
        expected_row1 = [ 4 ]

        given    = [['a','b','c'], given_row0, given_row1]
        expected = [['a'], expected_row0, expected_row1]

        self.assertEqual( expected, list(Drop(drop_cols=[1,2]).filter(given)) )

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

        given    = [['a','b','c'], given_row0, given_row1]
        expected = [['a',    'c'], expected_row0, expected_row1]

        self.assertEqual( expected, list(Drop(drop_cols=[1]).filter(given)) )

    def test_sparse_with_header_drop_double_col(self):

        given_row0 = {0:1,1:2,2:3}
        given_row1 = {0:4,1:5,2:6}

        expected_row0 = { 0:1 }
        expected_row1 = { 0:4 }

        given    = [['a','b','c'], given_row0, given_row1]
        expected = [['a'        ], expected_row0, expected_row1]

        self.assertEqual( expected, list(Drop(drop_cols=[1,2]).filter(given)) )

class Default_Tests(unittest.TestCase):

    def test_default_sparse(self):

        self.assertEqual([{"A":1},{"A":2}], list(Default({"A":1}).filter([{},{"A":2}])))

    def test_default_dense(self):

        self.assertEqual(['abc','def'], list(Default({"A":1}).filter(['abc', 'def']))) 

if __name__ == '__main__':
    unittest.main()