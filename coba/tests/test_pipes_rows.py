
import unittest

from coba.pipes.rows import EncodeRows, HeadRows, LabelRows
from coba.pipes.rows import LazyDense, EncodeDense, HeadDense, LabelDense, KeepDense
from coba.pipes.rows import LazySparse, HeadSparse

class EncodeRow_Tests(unittest.TestCase):

    def test_encode_dense(self):
        row = next(EncodeRows([int,int,float]).filter([['0','1','2.2']]))
        self.assertEqual([0,1,2.2],list(row))
    
    def test_encode_sparse(self):
        row = next(EncodeRows({'a':int,'b':int,'c':float}).filter([{'a':'0','b':'1','c':'2.2'}]))
        self.assertEqual({'a':0,'b':1,'c':2.2}, dict(row) )

class HeadRows_Tests(unittest.TestCase):
    def test_index(self):
        row = next(HeadRows({'a':0,'b':1}).filter([[1,2]]))

        self.assertEqual(1, row['a'])
        self.assertEqual(2, row['b'])
        self.assertEqual(1, row[0])
        self.assertEqual(2, row[1])

class LabelRows_Tests(unittest.TestCase):
    def test_lazy_dense(self):
        row = next(LabelRows(1).filter([LazyDense([1,2,3])]))
        self.assertEqual([1,2,3],list(row))
        self.assertEqual(2,row.labeled[1])
        self.assertEqual([1,3],list(row.labeled[0]))

    def test_list(self):
        row = next(LabelRows(1).filter([[1,2,3]]))
        self.assertEqual([1,2,3],list(row))
        self.assertEqual(2,row.labeled[1])
        self.assertEqual([1,3],list(row.labeled[0]))

    def test_lazy_sparse(self):
        row = LazySparse({'a':1,'b':2,'c':3})
        row = next(LabelRows('b').filter([row]))
        self.assertEqual({'a':1,'b':2,'c':3},dict(row))
        self.assertEqual(2,row.labeled[1])
        self.assertEqual({'a':1,'c':3},dict(row.labeled[0]))

    def test_dict(self):
        row = next(LabelRows('b').filter([{'a':1,'b':2,'c':3}]))
        self.assertEqual({'a':1,'b':2,'c':3},dict(row))
        self.assertEqual(2,row.labeled[1])
        self.assertEqual({'a':1,'c':3},dict(row.labeled[0]))

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
        items = [1,2,3]
        for v1,v2 in zip(items,LazyDense(lambda:items)):
            self.assertEqual(v1,v2)

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

class KeepDense_Tests(unittest.TestCase):

    def test_get_by_index(self):
        r = KeepDense([1,2,3], [0,2])
        self.assertEqual(1,r[0])
        self.assertEqual(3,r[1])

        with self.assertRaises(IndexError):
            r[2]

    def test_get_by_header(self):
        r = KeepDense(HeadDense([1,2],{'a':0,'b':1}), [1], ['b'])
        self.assertEqual(2,r[0])
        self.assertEqual(2,r['b'])
        
        with self.assertRaises(IndexError):
            r[1]
        with self.assertRaises(IndexError):
            r['a']

    def test_len(self):
        r = KeepDense([1,2,3], [0,2])
        self.assertEqual(2,len(r))
    
    def test_iter(self):
        r = KeepDense([1,2,3], [0,2])
        self.assertEqual([1,3],list(r))

class LabelDense_Tests(unittest.TestCase):

    def test_get(self):
        r = LabelDense([1,2,3],2)
        self.assertEqual(1, r[0])
        self.assertEqual(2, r[1])
        self.assertEqual(3, r[2])

    def test_feats_and_label(self):
        r = LabelDense([1,2,3],2)
        self.assertEqual(3,r.labeled[1])
        self.assertEqual([1,2],list(r.labeled[0]))

    def test_iter(self):
        r = LabelDense([1,2,3],2)
        self.assertEqual([1,2,3], list(r))

    def test_len(self):
        r = LabelDense([1,2,3],2)
        self.assertEqual(3,len(r))

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

class HeadSparse_Tests(unittest.TestCase):

    def test_get(self):
        r = HeadSparse({0:1,1:2}, {'a':0,'b':1,'c':2})
        self.assertEqual(1,r["a"])
        self.assertEqual(2,r["b"])

    def test_iter(self):
        r = HeadSparse({0:1,1:2}, {'a':0,'b':1,'c':2})
        self.assertEqual({'a','b'},set(r))

    def test_keys(self):
        r = HeadSparse({0:1,1:2}, {'a':0,'b':1,'c':2})
        self.assertEqual({'a','b'},r.keys())

    def test_items(self):
        r = HeadSparse({0:1,1:2}, {'a':0,'b':1,'c':2})
        self.assertEqual({'a':1,'b':2},dict(r.items()))

    def test_len(self):
        r = HeadSparse({0:1,1:2}, {'a':0,'b':1,'c':2})
        self.assertEqual(2,len(r))

if __name__ == '__main__':
    unittest.main()
