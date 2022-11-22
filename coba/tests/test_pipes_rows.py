
import unittest

from coba.pipes.rows import EncodeRows, HeadRows, LabelRows, DropRows
from coba.pipes.rows import LazyDense, EncodeDense, HeadDense, LabelDense, KeepDense
from coba.pipes.rows import LazySparse, EncodeSparse, HeadSparse, LabelSparse, DropSparse

class EncodeRows_Tests(unittest.TestCase):

    def test_encode_dense_index_map(self):
        row = next(EncodeRows({0:int,2:float}).filter([['0','1','2.2']]))
        self.assertEqual([0,'1',2.2],row)

    def test_encode_dense_header_map(self):
        row = next(EncodeRows({'a':int,'c':float}).filter([HeadDense(['0','1','2.2'],['a','b','c'])]))
        self.assertEqual([0,'1',2.2],row)

    def test_encode_dense_sequence(self):
        row = next(EncodeRows([int,int,float]).filter([['0','1','2.2']]))
        self.assertEqual([0,1,2.2],row)

    def test_encode_sparse_map(self):
        row = next(EncodeRows({'a':int,'b':int,'c':float}).filter([{'a':'0','b':'1','c':'2.2'}]))
        self.assertEqual({'a':0,'b':1,'c':2.2}, row)

    def test_encode_sparse_seq(self):
        row = next(EncodeRows([int,int,float]).filter([{0:'0',1:'1',2:'2.2'}]))
        self.assertEqual({0:0,1:1,2:2.2}, row)

class HeadRows_Tests(unittest.TestCase):
    
    def test_list(self):
        row = next(HeadRows({'a':0,'b':1}).filter([[1,2]]))
        self.assertEqual(row.headers, ['a','b'])
        self.assertEqual(1, row['a'])
        self.assertEqual(2, row['b'])
        self.assertEqual(1, row[0])
        self.assertEqual(2, row[1])
    
    def test_dict(self):
        row = next(HeadRows({'a':0,'b':1}).filter([{0:1,1:2}]))
        self.assertCountEqual(['a','b'], list(row))
        self.assertEqual((('a',1),('b',2)), row.items())
        self.assertEqual(1, row['a'])
        self.assertEqual(2, row['b'])

class LabelRows_Tests(unittest.TestCase):
    
    def test_lazy_dense(self):
        row = next(LabelRows(1,'c').filter([LazyDense([1,2,3])]))
        self.assertEqual([1,2,3],row)
        self.assertEqual(([1,3],2,'c'),row.labeled)

    def test_head_dense_str(self):
        row = next(LabelRows('b','c').filter([HeadDense([1,2,3],['a','b','c'])]))
        self.assertEqual([1,2,3],row)
        self.assertEqual(([1,3],2,'c'),row.labeled)

    def test_head_dense_int(self):
        row = next(LabelRows(1,'c').filter([HeadDense([1,2,3],['a','b','c'])]))
        self.assertEqual([1,2,3],row)
        self.assertEqual(([1,3],2,'c'),row.labeled)

    def test_list(self):
        row = next(LabelRows(1,'c').filter([[1,2,3]]))
        self.assertEqual([1,2,3],row)
        self.assertEqual(([1,3],2,'c'),row.labeled)

    def test_lazy_sparse(self):
        row = next(LabelRows('b','c').filter([LazySparse({'a':1,'b':2,})]))
        self.assertEqual({'a':1,'b':2},row)
        self.assertEqual(({'a':1},2,'c'),row.labeled)

    def test_dict_sans_encode(self):
        row = next(LabelRows('b','c').filter([{'a':1,'b':2,}]))
        self.assertEqual({'a':1,'b':2},row)
        self.assertEqual(({'a':1},2,'c'),row.labeled)
    
    def test_dict_with_encode(self):
        row = next(LabelRows('b','c').filter([{'a':1,'b':2}]))
        self.assertEqual({'a':1,'b':2},row)
        self.assertEqual(({'a':1},2,'c'),row.labeled)

class DropRows_Tests(unittest.TestCase):

    def test_dense_sans_header_drop(self):

        given_row0 = [1,2,3]
        given_row1 = [4,5,6]

        expected_row0 = [ 2 ]
        expected_row1 = [ 5 ]

        given    = [given_row0, given_row1]
        expected = [expected_row0, expected_row1]

        drop_rows = list(DropRows(drop_cols=[0,2]).filter(given))

        self.assertEqual(expected, drop_rows)
        self.assertEqual(2, drop_rows[0][0])
        self.assertEqual(5, drop_rows[1][0])

    def test_head_dense_with_header(self):

        given     = [HeadDense([4,5,6],['a','b','c'])]
        expected  = [[4,6]]
        drop_rows = list(DropRows(drop_cols=[1]).filter(given))

        self.assertEqual(expected, drop_rows)
        self.assertEqual(4, drop_rows[0]['a'])
        self.assertEqual(6, drop_rows[0]['c'])
        self.assertEqual(4, drop_rows[0][0])
        self.assertEqual(6, drop_rows[0][1])

    def test_dense_with_header_drop_double_col(self):

        given_row0 = [1,2,3]
        given_row1 = [4,5,6]

        expected_row0 = [ 1 ]
        expected_row1 = [ 4 ]

        given    = [['a','b','c'], given_row0, given_row1]
        expected = [['a'], expected_row0, expected_row1]

        self.assertEqual( expected, list(DropRows(drop_cols=[1,2]).filter(given)) )

    def test_sparse_sans_header_drop_single_col(self):

        given_row0 = {0:1,1:2,2:3}
        given_row1 = {0:4,1:5,2:6}

        expected_row0 = { 0:1, 2:3 }
        expected_row1 = { 0:4, 2:6 }

        given    = [given_row0, given_row1]
        expected = [expected_row0, expected_row1]

        self.assertEqual( expected, list(map(dict,DropRows(drop_cols=[1]).filter(given))) )

    def test_sparse_sans_header_drop_double_col(self):

        given_row0 = {0:1,1:2,2:3}
        given_row1 = {0:4,1:5,2:6}

        expected_row0 = { 0:1 }
        expected_row1 = { 0:4 }

        given    = [given_row0, given_row1]
        expected = [expected_row0, expected_row1]

        self.assertEqual( expected, list(map(dict,DropRows(drop_cols=[1,2]).filter(given))) )

    def test_sparse_with_header_drop_single_col(self):

        given_row0 = {0:1,1:2,2:3}
        given_row1 = {0:4,1:5,2:6}

        expected_row0 = { 0:1, 2:3 }
        expected_row1 = { 0:4, 2:6 }

        given    = [given_row0, given_row1]
        expected = [expected_row0, expected_row1]

        self.assertEqual( expected, list(map(dict,DropRows(drop_cols=[1]).filter(given))) )

    def test_sparse_with_header_drop_double_col(self):

        given_row0 = {0:1,1:2,2:3}
        given_row1 = {0:4,1:5,2:6}

        expected_row0 = { 0:1 }
        expected_row1 = { 0:4 }

        given    = [given_row0, given_row1]
        expected = [expected_row0, expected_row1]

        self.assertEqual( expected, list(map(dict,DropRows(drop_cols=[1,2]).filter(given))) )

    def test_empty_drop(self):

        given_row0 = [1,2,3]
        given_row1 = [4,5,6]

        expected_row0 = given_row0
        expected_row1 = given_row1

        given    = [given_row0, given_row1]
        expected = [expected_row0, expected_row1]

        self.assertEqual( expected, list(DropRows(drop_cols=[]).filter(given)) )

class LazyDense_Tests(unittest.TestCase):

    def test_loaded_get(self):
        l = LazyDense([1,2,3])
        self.assertEqual(1, l[0])
        self.assertEqual(2, l[1])
        self.assertEqual(3, l[2])

    def test_loader_get(self):
        l = LazyDense(lambda:[1,2,3])
        self.assertEqual(1, l[0])
        self.assertEqual(2, l[1])
        self.assertEqual(3, l[2])

    def test_loader_len(self):
        l = LazyDense(lambda:[1,2,3])
        self.assertEqual(3,len(l))

    def test_iter(self):
        self.assertEqual([1,2,3],list(LazyDense(lambda:[1,2,3])))

    def test_eq(self):
        self.assertEqual([1,2,3],LazyDense(lambda:[1,2,3]))
        self.assertEqual((1,2,3),LazyDense(lambda:[1,2,3]))
    
    def test_neq(self):
        self.assertNotEqual([1,2,3,4],LazyDense(lambda:[1,2,3]))
        self.assertNotEqual([1,2],LazyDense(lambda:[1,2,3]))
        self.assertNotEqual(1,LazyDense(lambda:[1,2,3]))

class HeadDense_Tests(unittest.TestCase):

    def test_get(self):
        r = HeadDense([1,2,3], ['a','b','c'])
        self.assertEqual(1,r["a"])
        self.assertEqual(2,r["b"])
        self.assertEqual(3,r["c"])
        self.assertEqual(1,r[0])
        self.assertEqual(2,r[1])
        self.assertEqual(3,r[2])

    def test_iter(self):
        r = HeadDense([1,2,3], ['a','b','c'])
        self.assertEqual([1,2,3],list(r))

    def test_len(self):
        r = HeadDense([1,2,3], ['a','b','c'])
        self.assertEqual(3,len(r))

    def test_eq(self):
        r = HeadDense([1,2,3], ['a','b','c'])
        self.assertEqual([1,2,3],r)

class EncodeDense_Tests(unittest.TestCase):

    def test_get(self):
        r = EncodeDense(['1',2,'3'], [int,str,int])
        self.assertEqual(1  ,r[0])
        self.assertEqual('2',r[1])
        self.assertEqual(3  ,r[2])

    def test_len(self):
        r = EncodeDense(['1',2,'3'], [int,str,int])
        self.assertEqual(3,len(r))

    def test_iter(self):
        r = EncodeDense(['1',2,'3'], [int,str,int])
        self.assertEqual([1,'2',3],list(r))

    def test_eq(self):
        r = EncodeDense(['1',2,'3'], [int,str,int])
        self.assertEqual([1,'2',3],r)

    def test_none(self):
        r = EncodeDense(['1',2,'3'], [int,lambda x:x,int])
        self.assertEqual(1,r[0])
        self.assertEqual(2,r[1])
        self.assertEqual(3,r[2])
        self.assertEqual([1,2,3],r)
        self.assertEqual([1,2,3],list(r))

class KeepDense_Tests(unittest.TestCase):

    def test_get_by_index(self):
        r = KeepDense([1,2,3], {0:0,1:2}, [True,False,True], 2)
        self.assertEqual(1,r[0])
        self.assertEqual(3,r[1])

        with self.assertRaises(IndexError):
            r[2]

    def test_get_by_header(self):
        r = KeepDense(HeadDense([1,2],{'a':0, 'b':1}), {'b':1, 0:1}, [False, True], 1)
        self.assertEqual(2,r[0])
        self.assertEqual(2,r['b'])

        with self.assertRaises(IndexError):
            r[1]
        with self.assertRaises(IndexError):
            r['a']

    def test_len(self):
        r = KeepDense([1,2,3], [0,2], [True,False,True], 2)
        self.assertEqual(2,len(r))

    def test_iter(self):
        r = KeepDense([1,2,3], [0,2], [True,False,True], 2)
        self.assertEqual([1,3],list(r))

    def test_eq(self):
        r = KeepDense([1,2,3], [0,2], [True,False,True], 2)
        self.assertEqual([1,3],r)

class LabelDense_Tests(unittest.TestCase):

    def test_header(self):
        r = LabelDense(HeadDense([1,2,3],['a','b','c']), 'c', None, [1,3]) 
        self.assertEqual(1, r[0])
        self.assertEqual(2, r[1])
        self.assertEqual(3, r[2])
        self.assertEqual(1, r['a'])
        self.assertEqual(2, r['b'])
        self.assertEqual(3, r['c'])

    def test_get_sans_encode(self):
        r = LabelDense([1,2,3], 2, None, [1,3])
        self.assertEqual(1, r[0])
        self.assertEqual(2, r[1])
        self.assertEqual(3, r[2])

    def test_get_with_encode(self):
        r = LabelDense([1,2,3], 2, 'c', [1,3])
        self.assertEqual(1, r[0])
        self.assertEqual(2, r[1])
        self.assertEqual(3, r[2])

    def test_feats_and_label_sans_encode(self):
        r = LabelDense([1,2,3], 2, 'c', [1,2])
        self.assertEqual(([1,2],3,'c'),r.labeled)

    def test_feats_and_label_with_encode(self):
        r = LabelDense([1,2,3], 2, 'c', [1,2])
        self.assertEqual(([1,2],3,'c'),r.labeled)

    def test_iter_sans_encode(self):
        r = LabelDense([1,2,3],  2, None, [1,2])
        self.assertEqual([1,2,3], list(r))

    def test_iter_with_encode(self):
        r = LabelDense([1,2,3],  2, 'c', [1,2])
        self.assertEqual([1,2,3], list(r))

    def test_len(self):
        r = LabelDense([1,2,3], 2, None, [1,2])
        self.assertEqual(3,len(r))

    def test_eq(self):
        r = LabelDense([1,2,3], 2, None, [1,2])
        self.assertEqual([1,2,3], r)

class LazySparse_Tests(unittest.TestCase):

    def test_loaded_get(self):
        l = LazySparse({'a':1,'b':2})
        self.assertEqual(l['a'],1)
        self.assertEqual(l['b'],2)

    def test_loader_get(self):
        l = LazySparse(lambda:{'a':1,'b':2})
        self.assertEqual(l['a'],1)
        self.assertEqual(l['b'],2)

    def test_loader_len(self):
        l = LazySparse(lambda:{'a':1,'b':2})
        self.assertEqual(2,len(l))

    def test_iter(self):
        l = LazySparse(lambda:{'a':1,'b':2})
        self.assertEqual(set(l), {'a', 'b'})

    def test_keys(self):
        l = LazySparse(lambda:{'a':1,'b':2})
        self.assertEqual(l.keys(), {'a', 'b'})
    
    def test_items(self):
        l = LazySparse(lambda:{'a':1,'b':2})
        self.assertEqual((('a',1),('b',2)),l.items())

    def test_eq(self):
        l = LazySparse(lambda:{'a':1,'b':2})
        self.assertEqual({'a':1,'b':2},l)

    def test_neq(self):
        l = LazySparse(lambda:{'a':1,'b':2})
        self.assertNotEqual({'b':2},l)
        self.assertNotEqual(1,l)

class HeadSparse_Tests(unittest.TestCase):

    def test_get(self):
        r = HeadSparse({0:1,1:2}, {'a':0,'b':1},{0:'a',1:'b'})
        self.assertEqual(1,r["a"])
        self.assertEqual(2,r["b"])

    def test_iter(self):
        r = HeadSparse({0:1,1:2}, {'a':0,'b':1},{0:'a',1:'b'})
        self.assertEqual({'a','b'},set(r))

    def test_keys(self):
        r = HeadSparse({0:1,1:2}, {'a':0,'b':1},{0:'a',1:'b'})
        self.assertEqual({'a','b'},r.keys())

    def test_items(self):
        r = HeadSparse({0:1,1:2}, {'a':0,'b':1},{0:'a',1:'b'})
        self.assertEqual({'a':1,'b':2},dict(r.items()))

    def test_len(self):
        r = HeadSparse({0:1,1:2}, {'a':0,'b':1},{0:'a',1:'b'})
        self.assertEqual(2,len(r))

    def test_eq(self):
        r = HeadSparse({0:1,1:2}, {'a':0,'b':1},{0:'a',1:'b'})
        self.assertEqual({'a':1,'b':2},r)

class EncodeSparse_Tests(unittest.TestCase):

    def test_get(self):
        r = EncodeSparse({0:'1',1:2}, {0:int,1:str})
        self.assertEqual(1  ,r[0])
        self.assertEqual('2',r[1]) 

    def test_len(self):
        r = EncodeSparse({0:'1',1:2}, {0:int,1:str})
        self.assertEqual(2,len(r))

    def test_iter(self):
        r = EncodeSparse({0:'1',1:2}, {0:int,1:str})
        self.assertCountEqual([0,1],list(r))

    def test_keys(self):
        r = EncodeSparse({0:'1',1:2}, {0:int,1:str})
        self.assertEqual({0,1},r.keys())

    def test_items(self):
        r = EncodeSparse({0:'1',1:2}, {0:int,1:str})
        self.assertCountEqual(((0,1),(1,'2')),r.items())

    def test_eq(self):
        r = EncodeSparse({0:'1',1:2}, {0:int,1:str})
        self.assertEqual({0:1,1:'2'},r)

class DropSparse_Tests(unittest.TestCase):

    def test_get(self):
        r = DropSparse({0:'1',1:2}, {0,5})
        self.assertEqual(2,r[1]) 
        
        with self.assertRaises(KeyError):
            r[0]

    def test_len(self):
        r = DropSparse({0:'1',1:2,2:3}, {0,2,5})
        self.assertEqual(1,len(r))

    def test_iter(self):
        r = DropSparse({0:'1',1:2,2:3}, {0,2,5})
        self.assertCountEqual([1],list(r))

    def test_keys(self):
        r = DropSparse({0:'1',1:2,2:3}, {0,2,5})
        self.assertEqual({1},r.keys())

    def test_items(self):
        r = DropSparse({0:'1',1:2,2:3}, {1,5})
        self.assertCountEqual(((0,'1'),(2,3)),r.items())

    def test_eq(self):
        r = DropSparse({0:'1',1:2,2:3}, {1,5})
        self.assertEqual({0:'1',2:3},r)

class LabelSparse_Tests(unittest.TestCase):

    def test_get(self):
        r = LabelSparse({'a':1,'b':2}, 'b', 'c', {'a':1})
        self.assertEqual(1, r['a'])
        self.assertEqual(2, r['b'])

    def test_items(self):
        r = LabelSparse({'a':1,'b':2}, 'b', 'c', {'a':1})
        self.assertCountEqual((('a',1),('b',2)), r.items())

    def test_labeled(self):
        r = LabelSparse({'a':1,'b':2}, 'b', 'c', {'a':1})
        self.assertEqual(({'a':1},2,'c'),r.labeled)

    def test_keys(self):
        r = LabelSparse({'a':1,'b':2}, 'b', None, {'a':1})
        self.assertEqual({'a','b'},r.keys())

    def test_iter(self):
        r = LabelSparse({'a':1,'b':2}, 'b', str, {'a':1})
        self.assertCountEqual(['a','b'], list(r))

    def test_len(self):
        r = LabelSparse({'a':1,'b':2}, 'b', str, {'a':1})
        self.assertEqual(2,len(r))

    def test_eq(self):
        r = LabelSparse({'a':1,'b':2}, 'b', 'c', {'a':1})
        self.assertEqual({'a':1,'b':2}, r)

    def test_bad_get(self):
        r = LabelSparse({'a':1,'b':2}, 'b', 'c', {'a':1})
        with self.assertRaises(KeyError):
            r['not_real_key']

    def test_keys_sparse_label(self):
        r = LabelSparse({'a':1,'b':2}, 'c', 'c', {'a':1})
        self.assertEqual({'a','b','c'},r.keys())

    def test_sparse_label_column(self):
        r = LabelSparse({'a':1,'b':2}, 'c', 'c', {'a':1,'b':2})
        self.assertEqual(1, r['a'])
        self.assertEqual(2, r['b'])
        self.assertEqual(0, r['c'])
        self.assertEqual(({'a':1,'b':2},0,'c'),r.labeled)
        self.assertEqual({'a':1,'b':2,'c':0},r)
        self.assertCountEqual(['a','b','c'],list(r))
        self.assertCountEqual((('a',1),('b',2),('c',0)),r.items())

if __name__ == '__main__':
    unittest.main()
