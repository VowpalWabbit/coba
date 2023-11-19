import unittest
import pickle

from collections import OrderedDict
from coba.primitives.rows import Sparse, Dense, HashableSparse, HashableDense, Sparse_, Dense_

class DummySparse(Sparse):

    def __init__(self, row) -> None:
        self._row = row

    def __getitem__(self, key):
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self):
        ...

    def keys(self):
        ...

    def items(self):
        return self._row.items()

class DummyDense(Dense):

    def __init__(self, row) -> None:
        self._row = row

    def __getitem__(self, key):
        ...

    def __len__(self) -> int:
        return len(self._row)

    def __iter__(self):
        return iter(self._row)

class DummySparse_(Sparse_):

    def __init__(self, row) -> None:
        self._row = row

    def __getitem__(self, key):
        ...

    def __len__(self) -> int:
        ...

    def __iter__(self):
        ...

    def keys(self):
        ...

    def items(self):
        return self._row.items()

class DummyDense_(Dense_):

    def __init__(self, row) -> None:
        self._row = row

    def __getitem__(self, key):
        ...

    def __len__(self) -> int:
        return len(self._row)

    def __iter__(self):
        return iter(self._row)

class Dense_Tests(unittest.TestCase):

    def test_getattr(self):

        class DummyClass:
            def __init__(self) -> None:
                self.missing = True

        self.assertEqual(True, DummyDense(DummyClass()).missing)

        with self.assertRaises(AttributeError):
            self.assertEqual(True, DummyDense({'a':1}).missing)

    def test_eq(self):

        self.assertEqual(DummyDense([1,2,3]),[1,2,3])

    def test_bad_eq(self):

        self.assertNotEqual(DummyDense([1,2,3]),1)

    def test_copy(self):
        dense = DummyDense((1,2,3))
        dense_copy = dense.copy()
        self.assertEqual(dense,dense_copy)
        self.assertIsNot(dense,dense_copy)

    def test_isinstance(self):
        self.assertTrue(isinstance([],Dense))
        self.assertTrue(isinstance((),Dense))

    def test_not_isinstance(self):
        self.assertFalse(isinstance({},Dense))
        self.assertFalse(isinstance("",Dense))
        self.assertFalse(isinstance(1 ,Dense))

class Sparse_Tests(unittest.TestCase):

    def test_getattr(self):

        class DummyClass:
            def __init__(self) -> None:
                self.missing = True

        self.assertEqual(True, DummySparse(DummyClass()).missing)

        with self.assertRaises(AttributeError):
            self.assertEqual(True, DummySparse({'a':1}).missing)

    def test_eq(self):

        self.assertEqual(DummySparse({'a':1}),{'a':1})

    def test_bad_eq(self):

        self.assertNotEqual(DummySparse({'a':1}),1)

    def test_copy(self):
        sparse = DummySparse({'a':1})
        sparse_copy = sparse.copy()
        self.assertEqual(sparse,sparse_copy)
        self.assertIsNot(sparse,sparse_copy)

    def test_isinstance(self):
        self.assertTrue(isinstance({},Sparse))

    def test_not_isinstance(self):
        self.assertFalse(isinstance([],Sparse))
        self.assertFalse(isinstance("",Sparse))
        self.assertFalse(isinstance(1 ,Sparse))

class Dense__Tests(unittest.TestCase):

    def test_getattr(self):

        class DummyClass:
            def __init__(self) -> None:
                self.missing = True

        self.assertEqual(True, DummyDense_(DummyClass()).missing)

        with self.assertRaises(AttributeError):
            self.assertEqual(True, DummyDense_({'a':1}).missing)

    def test_eq(self):

        self.assertEqual(DummyDense_([1,2,3]),[1,2,3])

    def test_bad_eq(self):

        self.assertNotEqual(DummyDense_([1,2,3]),1)

    def test_copy(self):
        dense = DummyDense_((1,2,3))
        dense_copy = dense.copy()
        self.assertEqual(dense,dense_copy)
        self.assertIsNot(dense,dense_copy)

    def test_isinstance(self):
        self.assertIsInstance(DummyDense_((1,2,3)),Dense)

class Sparse__Tests(unittest.TestCase):

    def test_getattr(self):

        class DummyClass:
            def __init__(self) -> None:
                self.missing = True

        self.assertEqual(True, DummySparse_(DummyClass()).missing)

        with self.assertRaises(AttributeError):
            self.assertEqual(True, DummySparse_({'a':1}).missing)

    def test_eq(self):

        self.assertEqual(DummySparse_({'a':1}),{'a':1})

    def test_bad_eq(self):

        self.assertNotEqual(DummySparse_({'a':1}),1)

    def test_copy(self):
        sparse = DummySparse_({'a':1})
        sparse_copy = sparse.copy()
        self.assertEqual(sparse,sparse_copy)
        self.assertIsNot(sparse,sparse_copy)

    def test_isinstance(self):
        self.assertIsInstance(DummySparse_({'a':1}),Sparse)

class HashableSparse_Tests(unittest.TestCase):

    def test_get(self):
        hash_dict = HashableSparse({'a':1,'b':2})
        self.assertEqual(1,hash_dict['a'])

    def test_iter(self):
        hash_dict = HashableSparse({'a':1,'b':2})
        self.assertEqual(['a','b'],list(hash_dict))

    def test_len(self):
        hash_dict = HashableSparse({'a':1,'b':2})
        self.assertEqual(2,len(hash_dict))

    def test_hash(self):
        hash_dict = HashableSparse({'a':1,'b':2})
        self.assertEqual(hash(hash_dict), hash(hash_dict))
        self.assertEqual(hash_dict,hash_dict)

    def test_hash_is_order_agnostic(self):
        hash_dict1 = HashableSparse(OrderedDict({'a':1,'b':2}))
        hash_dict2 = HashableSparse(OrderedDict({'b':2,'a':1}))

        self.assertEqual(hash(hash_dict1), hash(hash_dict2))
        self.assertEqual(hash_dict1,hash_dict2)

    def test_eq_good(self):
        hash_dict = HashableSparse({'a':1,'b':2})
        self.assertEqual({'a':1,'b':2},hash_dict)

    def test_eq_bad(self):
        hash_dict = HashableSparse({'a':1,'b':2})
        self.assertNotEqual(1,hash_dict)

    def test_repr(self):
        hash_dict = HashableSparse({'a':1,'b':2})
        self.assertEqual("{'a': 1, 'b': 2}",repr(hash_dict))

    def test_str(self):
        hash_dict = HashableSparse({'a':1,'b':2})
        self.assertEqual("{'a': 1, 'b': 2}",str(hash_dict))

    def test_copy(self):
        hash_dict = HashableSparse({'a':1,'b':2})
        hash_dict_copy = hash_dict.copy()
        self.assertEqual(hash_dict,hash_dict_copy)
        self.assertIsNot(hash_dict,hash_dict_copy)

    def test_pickle(self):
        dump_dict = HashableSparse(OrderedDict({'a':1,'b':2}))
        load_dict = pickle.loads(pickle.dumps(dump_dict))

class HashableDense_Tests(unittest.TestCase):

    def test_get(self):
        hash_seq = HashableDense([1,2,3])
        self.assertEqual(2,hash_seq[1])

    def test_len(self):
        hash_seq = HashableDense([1,2,3])
        self.assertEqual(3,len(hash_seq))

    def test_hash(self):
        hash_seq = HashableDense([1,2,3])
        self.assertEqual(hash(hash_seq), hash(hash_seq))
        self.assertEqual(hash_seq,hash_seq)

    def test_eq(self):
        hash_seq = HashableDense([1,2,3])
        self.assertEqual([1,2,3],hash_seq)
        self.assertEqual((1,2,3),hash_seq)

    def test_ne(self):
        hash_seq = HashableDense([1,2,3])
        self.assertNotEqual([1,2,4],hash_seq)
        self.assertNotEqual([1,2,3,4],hash_seq)
        self.assertNotEqual(1,hash_seq)

    def test_repr(self):
        hash_seq = HashableDense([1,2,3])
        self.assertEqual("[1, 2, 3]",repr(hash_seq))

    def test_str(self):
        hash_seq = HashableDense([1,2,3])
        self.assertEqual("[1, 2, 3]",str(hash_seq))

    def test_pickle(self):
        dump = HashableDense([1,2,3])
        load = pickle.loads(pickle.dumps(dump))

if __name__ == '__main__':
    unittest.main()
