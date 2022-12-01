
import unittest

from coba.pipes.rows import EncodeRows, HeadRows, LabelRows, DropRows, EncodeCatRows, Categorical
from coba.pipes.rows import LazyDense, EncodeDense, HeadDense, LabelDense, KeepDense
from coba.pipes.rows import LazySparse, EncodeSparse, HeadSparse, LabelSparse, DropSparse, DropOne

class Categorical_Tests(unittest.TestCase):
    def test_value(self):
        self.assertEqual("A", Categorical("A",["A","B"]))
    
    def test_levels(self):
        self.assertEqual(["A","B"], Categorical("A",["A","B"]).levels)

    def test_eq(self):
        self.assertEqual(Categorical("A",["A","B"]), Categorical("A",["A","B"]))

    def test_ne(self):
        self.assertNotEqual(1, Categorical("A",["A","B"]))

    def test_str(self):
        self.assertEqual("A", str(Categorical("A",["A","B"])))

    def test_repr(self):
        self.assertEqual("Categorical('A',['A', 'B'])", repr(Categorical("A",["A","B"])))

class EncodeCatRows_Tests(unittest.TestCase):

    def test_none_dense_with_categorical(self):

        given = [[1,2,Categorical('1',['1','2'])], [4,5,Categorical('2',['1','2'])]]
        expected = [[1,2,Categorical('1',['1','2'])], [4,5,Categorical('2',['1','2'])]]
        actual = list(EncodeCatRows(None).filter(given))

        self.assertEqual(actual,expected)

    def test_onehot_dense_with_categorical(self):

        given = [(1,2,Categorical('1',['1','2'])), [4,5,Categorical('2',['1','2'])]]
        expected = [[1,2,1,0],[4,5,0,1]]
        actual = list(EncodeCatRows("onehot").filter(given))

        self.assertEqual(actual,expected)

    def test_string_dense_with_categorical(self):

        given = [[1,2,Categorical('1',['1','2'])], [4,5,Categorical('2',['1','2'])]]
        expected = [[1,2,"1"],[4,5,"2"]]
        actual = list(EncodeCatRows("string").filter(given))

        self.assertEqual(actual,expected)

    def test_onehot_dense_sans_categorical(self):

        given = [[1,2], [4,5]]
        expected = [[1,2],[4,5]]
        actual = list(EncodeCatRows("onehot").filter(given))

        self.assertEqual(actual,expected)
    
    def test_onehot_tuple_dense_with_categorical(self):

        given = [[1,2,Categorical('1',['1','2'])], [4,5,Categorical('2',['1','2'])]]
        expected = [[1,2,(1,0)],[4,5,(0,1)]]
        actual = list(EncodeCatRows("onehot_tuple").filter(given))

        self.assertEqual(actual,expected)

    def test_onehot_tuple_dense_sans_categorical(self):

        given = [[1,2], [4,5]]
        expected = [[1,2],[4,5]]
        actual = list(EncodeCatRows("onehot_tuple").filter(given))

        self.assertEqual(actual,expected)

    def test_onehot_sparse_with_categorical(self):

        given = [{1:2, 2:Categorical('1',['1','2'])}, {4:5, 2:Categorical('2',['1','2'])}]
        expected = [{1:2, "2_0":1}, {4:5, "2_1":1}]
        actual = list(EncodeCatRows("onehot").filter(given))

        self.assertEqual(actual,expected)

    def test_string_sparse_with_categorical(self):

        given = [{1:2, 2:Categorical(1,[1,2])}, {4:5, 2:Categorical(2,[1,2])}]
        expected = [{1:2, 2:"1"}, {4:5, 2:"2"}]
        actual = list(EncodeCatRows("string").filter(given))

        self.assertEqual(actual,expected)

    def test_onehot_sparse_sans_categorical(self):

        given = [{1:2}, {4:5}]
        expected = [{1:2}, {4:5}]
        actual = list(EncodeCatRows("onehot").filter(given))

        self.assertEqual(actual,expected)

    def test_onehot_tuple_sparse_with_categorical(self):

        given = [{1:2, 2:Categorical('1',['1','2'])}, {4:5, 2:Categorical('2',['1','2'])}]
        expected = [{1:2, 2:(1,0)}, {4:5, 2:(0,1)}]
        actual = list(EncodeCatRows("onehot_tuple").filter(given))

        self.assertEqual(actual,expected)

    def test_onehot_tuple_sparse_sans_categorical(self):

        given = [{1:2}, {4:5}]
        expected = [{1:2}, {4:5}]
        actual = list(EncodeCatRows("onehot_tuple").filter(given))

        self.assertEqual(actual,expected)

    def test_empty(self):

        given = []
        expected = []
        actual = list(EncodeCatRows(False).filter(given))

        self.assertEqual(actual,expected)

    def test_value_not_categorical(self):

        given = [(1,0),2,3]
        expected = [(1,0),2,3]
        actual = list(EncodeCatRows(value_rows=True).filter(given))

        self.assertEqual(actual,expected)

    def test_value_categorical_to_onehot(self):

        given = [Categorical('1',['1','2']),Categorical('2',['1','2'])]
        expected = [(1,0),(0,1)]
        actual = list(EncodeCatRows("onehot", value_rows=True).filter(given))

        self.assertEqual(actual,expected)

    def test_value_categorical_to_str(self):

        given = [Categorical('1',['1','2']),Categorical('2',['1','2'])]
        expected = ['1','2']
        actual = list(EncodeCatRows("string", value_rows=True).filter(given))

        self.assertEqual(actual,expected)

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

    def test_encode_empty_sequence(self):
        self.assertEqual([], EncodeRows([int,str]).filter([]))

class HeadRows_Tests(unittest.TestCase):
    
    def test_list(self):
        row = next(HeadRows(['a','b']).filter([[1,2]]))
        self.assertEqual(row.headers, {'a':0,'b':1})
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
        row = next(LabelRows('b','c').filter([HeadDense([1,2,3],{'a':0,'b':1,'c':2})]))
        self.assertEqual([1,2,3],row)
        self.assertEqual(([1,3],2,'c'),row.labeled)

    def test_head_dense_int(self):
        row = next(LabelRows(1,'c').filter([HeadDense([1,2,3],{'a':0,'b':1,'c':2})]))
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
        row = next(LabelRows('b','c').filter([{'a':1,'b':2}]))
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

        given     = [HeadDense([4,5,6],{'a':0,'b':1,'c':2})]
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
        r = HeadDense([1,2,3], {'a':0,'b':1,'c':2})
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

    @unittest.skip("This functionality was removed for performance reasons")
    def test_header_get(self):
        r = LabelDense(HeadDense([1,2,3],['a','b','c']), 1, 'c', 'b') 
        self.assertEqual(1, r[0])
        self.assertEqual(2, r[1])
        self.assertEqual(3, r[2])
        self.assertEqual(1, r['a'])
        self.assertEqual(2, r['b'])
        self.assertEqual(3, r['c'])

    @unittest.skip("This functionality was removed for performance reasons")
    def test_header_labeled(self):
        r = LabelDense(HeadDense([1,2,3],['a','b','c']), 1, 'c', 'b')
        feats,label,tipe = r.labeled
        
        self.assertEqual(([1,3],2,'c'),(feats,label,tipe))
        self.assertEqual(3, feats[1])
        self.assertEqual(1, feats['a'])
        self.assertEqual(3, feats['c'])
        with self.assertRaises(KeyError):
            feats['b']

    def test_get(self):
        r = LabelDense([1,2,3],1,'c')
        self.assertEqual(r[0],1)
        self.assertEqual(r[1],2)
        self.assertEqual(r[2],3)

    def test_labeled(self):
        self.assertEqual(([1,2],3,'c'),LabelDense([1,2,3], 2, 'c').labeled)

    def test_iter(self):
        self.assertEqual([1,2,3], list(LabelDense([1,2,3],  2, 'c')))

    def test_len(self):
        self.assertEqual(3, len(LabelDense([1,2,3], 2, 'c')))

    def test_eq(self):
        self.assertEqual([1,2,3], LabelDense([1,2,3], 2, 'c'))

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

    def test_items_enc(self):
        l = LazySparse(lambda:{'a':1,'b':2},enc={'a':str,'b':str})
        self.assertEqual((('a','1'),('b','2')),l.items())

    def test_items_enc_nsp(self):
        l = LazySparse(lambda:{'a':1,'b':2}, enc={'a':str,'b':str,'c':str}, nsp={'c'})
        self.assertEqual((('a','1'),('b','2'),('c','0')),l.items())

    def test_items_enc_inv(self):
        l = LazySparse(lambda:{'a':1,'b':2}, inv={'a':'1'}, enc={'a':str,'b':str})
        self.assertEqual((('1','1'),('b','2')),l.items())

    def test_items_inv(self):
        l = LazySparse(lambda:{'a':1,'b':2}, inv={'a':'1'})
        self.assertEqual((('1',1),('b',2)),l.items())

    def test_eq(self):
        l = LazySparse(lambda:{'a':1,'b':2})
        self.assertEqual({'a':1,'b':2},l)

    def test_neq(self):
        l = LazySparse(lambda:{'a':1,'b':2})
        self.assertNotEqual({'b':2},l)
        self.assertNotEqual(1,l)

class Sparse_Tests(unittest.TestCase):

    def test_simple(self):
        self.assertEqual(True, LazySparse(LazySparse({'a':1,'b':2}, missing=True)).missing)

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
        r = EncodeSparse({0:'1',1:2}, {0:int,1:str}, {1})
        self.assertEqual(1  ,r[0])
        self.assertEqual('2',r[1])

    def test_get_with_non_zero_sparse_encoding(self):
        r = EncodeSparse({0:'1',1:2}, {0:int,1:str,2:str}, {1,2})
        self.assertEqual(1  ,r[0])
        self.assertEqual('2',r[1]) 
        self.assertEqual('0',r[2])

    def test_len(self):
        r = EncodeSparse({0:'1',1:2}, {0:int,1:str}, {1})
        self.assertEqual(2,len(r))

    def test_len_with_non_zero_sparse_encoding(self):
        r = EncodeSparse({0:'1',1:2}, {0:int,1:str,2:str}, {1,2})
        self.assertEqual(3,len(r))

    def test_iter(self):
        r = EncodeSparse({0:'1',1:2}, {0:int,1:str}, {1})
        self.assertCountEqual([0,1],list(r))

    def test_iter_with_non_zero_sparse_encoding(self):
        r = EncodeSparse({0:'1',1:2}, {0:int,1:str,2:str}, {1,2})
        self.assertCountEqual([0,1,2],list(r))

    def test_keys(self):
        r = EncodeSparse({0:'1',1:2}, {0:int,1:str}, {1})
        self.assertEqual({0,1},r.keys())

    def test_keys_with_non_zero_sparse_encoding(self):
        r = EncodeSparse({0:'1',1:2}, {0:int,1:str,2:str}, {1,2})
        self.assertEqual({0,1,2},r.keys())

    def test_items(self):
        r = EncodeSparse({0:'1',1:2}, {0:int,1:str}, {1})
        self.assertCountEqual(((0,1),(1,'2')),r.items())

    def test_items_with_non_zero_sparse_encoding(self):
        r = EncodeSparse({0:'1',1:2}, {0:int,1:str,2:str}, {1,2})
        self.assertCountEqual(((0,1),(1,'2'),(2,"0")),r.items())

    def test_eq(self):
        r = EncodeSparse({0:'1',1:2}, {0:int,1:str}, {1})
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

class DropOne_Tests(unittest.TestCase):

    def test_get(self):
        r = DropOne([1,2,3,4,5],2)
        self.assertEqual(2,r[1])
        self.assertEqual(4,r[2])

    def test_len(self):
        r = DropOne([1,2,3,4,5],2)
        self.assertEqual(4,len(r))

    def test_iter(self):
        r = DropOne([1,2,3,4,5],2)
        self.assertEqual((1,2,4,5),tuple(r))

    def test_eq(self):
        r = DropOne([1,2,3,4,5],2)
        self.assertEqual([1,2,4,5],r)

class LabelSparse_Tests(unittest.TestCase):

    def test_get(self):
        r = LabelSparse({'a':1,'b':2}, 'b', 'c')
        self.assertEqual(1, r['a'])
        self.assertEqual(2, r['b'])

    def test_items(self):
        r = LabelSparse({'a':1,'b':2}, 'b', 'c')
        self.assertCountEqual((('a',1),('b',2)), r.items())

    def test_labeled(self):
        r = LabelSparse({'a':1,'b':2}, 'b', 'c')
        self.assertEqual(({'a':1},2,'c'),r.labeled)

    def test_feats(self):
        r = LabelSparse({'a':1,'b':2}, 'b', 'c')
        self.assertEqual({'a':1},r.feats)

    def test_tipe(self):
        r = LabelSparse({'a':1,'b':2}, 'b', 'c')
        self.assertEqual('c',r.tipe)

    def test_label(self):
        r = LabelSparse({'a':1,'b':2}, 'b', 'c')
        self.assertEqual(2,r.label)

    def test_keys(self):
        r = LabelSparse({'a':1,'b':2}, 'b', 'c')
        self.assertEqual({'a','b'},r.keys())

    def test_iter(self):
        r = LabelSparse({'a':1,'b':2}, 'b', 'c')
        self.assertCountEqual(['a','b'], list(r))

    def test_len(self):
        r = LabelSparse({'a':1,'b':2}, 'b', 'c')
        self.assertEqual(2,len(r))

    def test_eq(self):
        r = LabelSparse({'a':1,'b':2}, 'b', 'c')
        self.assertEqual({'a':1,'b':2}, r)

    def test_bad_get(self):
        r = LabelSparse({'a':1,'b':2}, 'b', 'c')
        with self.assertRaises(KeyError):
            r['not_real_key']

    def test_keys_sparse_label(self):
        r = LabelSparse({'a':1,'b':2}, 'c', 'c')
        self.assertEqual({'a','b','c'},r.keys())

    def test_sparse_label_column(self):
        r = LabelSparse({'a':1,'b':2}, 'c', 'c')
        self.assertEqual(1, r['a'])
        self.assertEqual(2, r['b'])
        self.assertEqual(0, r['c'])
        self.assertEqual(({'a':1,'b':2},0,'c'),r.labeled)
        self.assertEqual({'a':1,'b':2,'c':0},r)
        self.assertCountEqual(['a','b','c'],list(r))
        self.assertCountEqual((('a',1),('b',2),('c',0)),r.items())

if __name__ == '__main__':
    unittest.main()
