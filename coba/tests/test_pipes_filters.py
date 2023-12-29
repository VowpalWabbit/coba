import unittest
import unittest.mock

from coba.pipes import LazyDense, LazySparse
from coba.encodings import NumericEncoder, OneHotEncoder, StringEncoder
from coba.context import NullLogger, CobaContext
from coba.utilities import peek_first

from coba.pipes.filters import Flatten, Encode, Structure, Take, Identity
from coba.pipes.filters import Shuffle, Default, Reservoir, Cache, FiltersFilter

CobaContext.logger = NullLogger()

class ReprFilter:
    def __init__(self,id=""):
        self._id = id

    def __str__(self):
        return f"ReprFilter{self._id}"

    def filter(self, item):
        return item

class ParamsFilter:
    @property
    def params(self):
        return {'filter':"ParamsFilter"}

    def filter(self,item):
        return item

class NoParamsFilter:
    def filter(self,item):
        return item

class FiltersFilter_Tests(unittest.TestCase):

    def test_init_filters(self):

        filter = FiltersFilter(ReprFilter("1"), ReprFilter("2"))

        self.assertEqual(2, len(filter._filters))
        self.assertIsInstance(filter._filters[0], ReprFilter)
        self.assertIsInstance(filter._filters[1], ReprFilter)
        self.assertEqual("ReprFilter1,ReprFilter2", str(filter))

    def test_init_filtersfilter(self):

        filter = FiltersFilter(FiltersFilter(ReprFilter(), ReprFilter()), ReprFilter())

        self.assertEqual(3, len(filter._filters))
        self.assertIsInstance(filter._filters[0], ReprFilter)
        self.assertIsInstance(filter._filters[1], ReprFilter)
        self.assertIsInstance(filter._filters[2], ReprFilter)
        self.assertEqual("ReprFilter,ReprFilter,ReprFilter", str(filter))

    def test_read1(self):

        self.assertEqual([0,1,2], list(FiltersFilter(ReprFilter(), ReprFilter()).filter(range(3))))

    def test_read2(self):

        self.assertEqual([0,1,2], list(FiltersFilter(FiltersFilter(ReprFilter(), ReprFilter()), ReprFilter()).filter(range(3))))

    def test_params(self):
        source = FiltersFilter(NoParamsFilter(), NoParamsFilter())
        self.assertEqual({}, source.params)

        source = FiltersFilter(NoParamsFilter(), ParamsFilter())
        self.assertEqual({'filter':'ParamsFilter'}, source.params)

        source = FiltersFilter(ParamsFilter(), NoParamsFilter())
        self.assertEqual({'filter':'ParamsFilter'}, source.params)

        source = FiltersFilter(ParamsFilter(), ParamsFilter())
        self.assertEqual({'filter1':'ParamsFilter','filter2':'ParamsFilter'}, source.params)

    def test_len(self):
        self.assertEqual(2,len(FiltersFilter(ReprFilter("1"), ReprFilter("2"))))

    def test_getitem(self):
        filter1 = ReprFilter("1")
        filter2 = ReprFilter("2")

        pipe = FiltersFilter(filter1, filter2)

        self.assertIs(pipe[0],filter1)
        self.assertIs(pipe[1],filter2)

        self.assertIs(pipe[-1],filter2)
        self.assertIs(pipe[-2],filter1)

    def test_iter(self):
        filter1 = ReprFilter("1")
        filter2 = ReprFilter("2")

        pipes = list(FiltersFilter(filter1, filter2))
        self.assertEqual(pipes, [filter1,filter2])

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

        with self.assertRaises(ValueError):
            Take((-1,5))

    def test_take_exact_1(self):

        items = [ 1,2,3 ]
        take_items = list(Take(1).filter(items))

        self.assertEqual([1    ], take_items)
        self.assertEqual([1,2,3], items     )

    def test_take_exact_2(self):
        items = [ 1,2,3 ]
        take_items = list(Take(2).filter(items))

        self.assertEqual([1,2  ], take_items)
        self.assertEqual([1,2,3], items     )

    def test_take_exact_3(self):
        items = [ 1,2,3 ]
        take_items = list(Take(3).filter(items))

        self.assertEqual([1,2,3], take_items)
        self.assertEqual([1,2,3], items     )

    def test_take_exact_4(self):
        items = [ 1,2,3 ]
        take_items = list(Take(4).filter(items))

        self.assertEqual([1,2,3], take_items)
        self.assertEqual([1,2,3], items     )

class Resevoir_Tests(unittest.TestCase):

    def test_bad_count(self):
        with self.assertRaises(ValueError):
            Reservoir(-1)

        with self.assertRaises(ValueError):
            Reservoir('A')

        with self.assertRaises(ValueError):
            Reservoir((-1,5))

    def test_take_exacts(self):
        items = [1,2,3,4,5]

        take_items = list(Reservoir(2,seed=1).filter(items))
        self.assertEqual([1,2,3,4,5], items)
        self.assertEqual([4, 2], take_items)

        take_items = list(Reservoir(None,seed=1).filter(items))
        self.assertEqual([1,2,3,4,5], items)
        self.assertEqual([1,5,4,3,2], take_items)

        take_items = list(Reservoir(5,seed=1).filter(items))
        self.assertEqual([1,2,3,4,5], items)
        self.assertEqual([1,5,4,3,2], take_items)

        take_items = list(Reservoir(6,seed=1).filter(items))
        self.assertEqual([1,2,3,4,5], items)
        self.assertEqual([1,5,4,3,2], take_items)

        take_items = list(Reservoir(0,seed=1).filter(items))
        self.assertEqual([1,2,3,4,5], items)
        self.assertEqual([]         , take_items)

    def test_take_exacts_iterable(self):
        items = [1,2,3,4,5]

        take_items = list(Reservoir(2,seed=1).filter(iter(items)))
        self.assertEqual([1,2,3,4,5], items)
        self.assertEqual([4, 2], take_items)

        take_items = list(Reservoir(None,seed=1).filter(iter(items)))
        self.assertEqual([1,2,3,4,5], items)
        self.assertEqual([1,5,4,3,2], take_items)

        take_items = list(Reservoir(5,seed=1).filter(iter(items)))
        self.assertEqual([1,2,3,4,5], items)
        self.assertEqual([1,5,4,3,2], take_items)

        take_items = list(Reservoir(6,seed=1).filter(iter(items)))
        self.assertEqual([1,2,3,4,5], items)
        self.assertEqual([1,5,4,3,2], take_items)

        take_items = list(Reservoir(0,seed=1).filter(iter(items)))
        self.assertEqual([1,2,3,4,5], items)
        self.assertEqual([]         , take_items)

class Flatten_Tests(unittest.TestCase):

    def test_number_flatten(self):

        given_row0 = 1
        given_row1 = 'abc'
        given_row2 = None

        expected_row0 = 1
        expected_row1 = 'abc'
        expected_row2 = None

        given    = [given_row0, given_row1, given_row2]
        expected = [expected_row0, expected_row1, expected_row2]

        self.assertEqual( expected, list(Flatten().filter(given)) )

    def test_dense_list_numeric_flatten(self):

        given_row0 = (1,2)
        given_row1 = (4,3)

        expected_row0 = (1,2)
        expected_row1 = (4,3)

        given    = [given_row0, given_row1]
        expected = [expected_row0, expected_row1]

        self.assertEqual(expected, list(Flatten().filter(given)) )

    def test_dense_tuple_onehot_row_flatten(self):

        given_row0 = ((0,1), 1)
        given_row1 = ((1,0), 2)

        expected_row0 = (0, 1, 1)
        expected_row1 = (1, 0, 2)

        given    = [given_row0   , given_row1   ]
        expected = [expected_row0, expected_row1]

        self.assertEqual(expected, list(Flatten().filter(given)) )

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

        expected_row0 = { '0_1': 1, 1:1}
        expected_row1 = { '0_0': 1, 1:1}

        given    = [given_row0, given_row1]
        expected = [expected_row0, expected_row1]

        self.assertEqual(expected, list(Flatten().filter(given)) )

    def test_string_row_flatten(self):
        given_row0 = [ "abc", "def", "ghi" ]
        expected_row0 = [ "abc", "def", "ghi" ]

        given    = [ given_row0    ]
        expected = [ expected_row0 ]

        self.assertEqual(expected, list(Flatten().filter(given)) )

    def test_iterable_row_flatten(self):
        given_row0 = [ range(2), range(2) ]
        expected_row0 = [ 0,1,0,1 ]

        given    = [ given_row0    ]
        expected = [ expected_row0 ]

        self.assertEqual(expected, list(Flatten().filter(given)) )

    def test_lazy_dense_onehot_flatten(self):

        given_row0 = LazyDense([(0,1), 1])
        given_row1 = LazyDense([(1,0), 2])

        expected_row0 = (0, 1, 1)
        expected_row1 = (1, 0, 2)

        given    = [given_row0   , given_row1   ]
        expected = [expected_row0, expected_row1]

        self.assertEqual(expected, list(Flatten().filter(given)) )

    def test_lazy_sparse_onehot_flatten(self):

        given_row0 = LazySparse({0: (0,1), 1:1 })
        given_row1 = LazySparse({0: (1,0), 1:1 })

        expected_row0 = { '0_1': 1, 1:1}
        expected_row1 = { '0_0': 1, 1:1}

        given    = [given_row0, given_row1]
        expected = [expected_row0, expected_row1]

        self.assertEqual(expected, list(Flatten().filter(given)) )

    def test_empty_flatten(self):
        self.assertEqual([], list(Flatten().filter([])) )

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

class Structure_Tests(unittest.TestCase):

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

class Default_Tests(unittest.TestCase):

    def test_default_sparse(self):
        self.assertEqual([{"A":1},{"A":2}], list(Default({"A":1}).filter([{},{"A":2}])))

    def test_default_dense(self):
        self.assertEqual(['abc','def'], list(Default({"A":1}).filter(['abc', 'def'])))

class Cache_Tests(unittest.TestCase):

    def test_cache(self):
        cache = Cache()
        self.assertEqual(None, cache._cache)
        self.assertEqual([1,2,3],list(cache.filter([1,2,3])))
        self.assertEqual([1,2,3],cache._cache)
        self.assertEqual([1,2,3],list(cache.filter([4,5,6])))

    def test_cache_with_peek_first(self):
        cache = Cache(2)

        iterable = iter(cache.filter([1,2,3,4]))
        iterable = peek_first(iterable)[1]

        self.assertEqual([1,2,3,4],list(cache.filter([1,2,3,4])))
        self.assertEqual([1,2,3,4],cache._cache)
        self.assertEqual([1,2,3,4],list(cache.filter([4,5,6])))

if __name__ == '__main__':
    unittest.main()