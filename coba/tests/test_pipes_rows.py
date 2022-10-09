
import unittest

from coba.exceptions import CobaException
from coba.pipes.rows import EncodeRow, IndexRow, LabelRow, DenseRow, SparseRow

class EncodeRow_Tests(unittest.TestCase):
    def test_encode_dense(self):
        self.assertEqual([0,1,2.2],next(EncodeRow([int,int,float]).filter([['0','1','2.2']])))

    def test_encode_sparse(self):
        self.assertEqual({'a':0,'b':1,'c':2.2},next(EncodeRow({'a':int,'b':int,'c':float}).filter([{'a':'0','b':'1','c':'2.2'}])))

class IndexRow_Tests(unittest.TestCase):
    def test_index(self):
        row = next(IndexRow({'a':0,'b':1}).filter([[1,2]]))

        self.assertEqual(1, row['a'])
        self.assertEqual(2, row['b'])
        self.assertEqual(1, row[0])
        self.assertEqual(2, row[1])

class LabelRow_Tests(unittest.TestCase):
    def test_dense(self):
        row = DenseRow(loaded=[1,2,3])
        self.assertEqual([1,2,3],row)
        row = next(LabelRow(1,'lbl').filter([[1,2,3]]))
        self.assertEqual([1,2,3],row)
        self.assertEqual(2,row['lbl'])
    
    def test_dense_encoding(self):
        row = DenseRow(loaded=[1,2,3])
        self.assertEqual([1,2,3],row)
        row = next(LabelRow(1,'lbl',str).filter([[1,2,3]]))
        self.assertEqual([1,'2',3],row)
        self.assertEqual('2',row['lbl'])

    def test_list(self):
        row = next(LabelRow(1,'lbl').filter([[1,2,3]]))
        self.assertEqual([1,2,3],row)
        self.assertEqual(2,row['lbl'])

    def test_list_encoding(self):
        row = next(LabelRow(1,'lbl',str).filter([[1,2,3]]))
        self.assertEqual([1,'2',3],row)
        self.assertEqual('2',row['lbl'])

    def test_sparse(self):
        row = SparseRow(loaded={'a':1,'b':2,'c':3})
        row = next(LabelRow('b','lbl').filter([row]))
        self.assertEqual({'a':1,'b':2,'c':3},dict(row))
        self.assertEqual(2,row['lbl'])

    def test_sparse_encoding(self):
        row = SparseRow(loaded={'a':1,'b':2,'c':3})
        row = next(LabelRow('b','lbl',str).filter([row]))
        self.assertEqual({'a':1,'b':'2','c':3},dict(row))
        self.assertEqual('2',row['lbl'])

    def test_dict(self):
        row = next(LabelRow('b','lbl').filter([{'a':1,'b':2,'c':3}]))
        self.assertEqual({'a':1,'b':2,'c':3},dict(row))
        self.assertEqual(2,row['lbl'])

    def test_dict_encoding(self):
        row = next(LabelRow('b','lbl',str).filter([{'a':1,'b':2,'c':3}]))
        self.assertEqual({'a':1,'b':'2','c':3},dict(row))
        self.assertEqual('2',row['lbl'])

class DenseRow_Tests(unittest.TestCase):

    def test_loaded(self):
        l = DenseRow(loaded=[1,2,3])
        self.assertEqual([1,2,3],l)

    def test_loader_get(self):
        l = DenseRow(loader=lambda:[1,2,3])
        self.assertEqual([1,2,3],l)

    def test_loader_set(self):
        l = DenseRow(loader=lambda:[1,2,3])
        l[1] = 5
        self.assertEqual([1,5,3],l)

    def test_loader_del(self):
        l = DenseRow(loader=lambda:[1,2,3])
        del l[1]
        self.assertEqual([1,3],l)

    def test_loader_len(self):
        l = DenseRow(loader=lambda:[1,2,3])
        self.assertEqual(3,len(l))

    def test_loader_del_del(self):
        l = DenseRow(loader=lambda:[1,2,3,4])

        del l[1]
        del l[2]

        self.assertFalse(l._loaded)
        self.assertEqual([1,3],l)
        self.assertTrue(l._loaded)

    def test_headers(self):
        l = DenseRow(loader=lambda:[1,2,3])
        l.headers = {'a':0,'b':1,'c':2}
        self.assertEqual(1,l['a'])
        self.assertEqual(2,l['b'])
        self.assertEqual(3,l['c'])

    def test_headers_del(self):
        l = DenseRow(loader=lambda:[1,2,3])
        l.headers = {'a':0,'b':1,'c':2}
        self.assertEqual(1,l['a'])
        del l['b']
        self.assertEqual(3,l['c'])

    def test_encoders(self):
        l = DenseRow(loader=lambda:['1','2','3'])
        l.encoders = [int,int,int]
        self.assertEqual(1,l[0])
        self.assertEqual(2,l[1])
        self.assertEqual(3,l[2])

    def test_encoders_del(self):
        l = DenseRow(loader=lambda:['1','2','3'])
        l.encoders = [int,str,int]
        
        self.assertEqual(1,l[0])
        self.assertEqual('2',l[1])
        self.assertEqual(3,l[2])

        del l[1]

        self.assertEqual(1,l[0])
        self.assertEqual(3,l[1])

    def test_insert(self):
        l = DenseRow(loader=lambda:['1','2','3'])
        with self.assertRaises(NotImplementedError):
            l.insert(1,2)
    
    def test_set_label(self):
        l = DenseRow(loader=lambda:['1','2','3'])
        l.set_label(1,'lbl',None)
        self.assertEqual(['1','2','3'],l)
        self.assertEqual(['1','3'],l.feats)
        self.assertEqual('2',l.label)

    def test_no_label(self):
        l = DenseRow(loader=lambda:['1','2','3'])
        with self.assertRaises(CobaException):
            l.label

    def test_headers_set_label1(self):
        l = DenseRow(loader=lambda:['1','2','3'])
        l.headers = {'a':0,'b':1,'c':2}
        l.headers_inv = {0:'a',1:'b',2:'c'}
    
        self.assertEqual('1',l['a'])
        self.assertEqual('2',l['b'])
        self.assertEqual('3',l['c'])
    
        l.set_label(1,'lbl',None)
    
        self.assertEqual('1',l['a'])
        self.assertEqual('2',l['b'])
        self.assertEqual('3',l['c'])
        self.assertEqual(['1','2','3'],l)
        self.assertEqual('2',l['lbl'])

        self.assertEqual(['1','3'],l.feats)
        self.assertEqual('2',l.label)
    
    def test_headers_set_label2(self):
        l = DenseRow(loader=lambda:['1','2','3'])
        l.headers = {'a':0,'b':1,'c':2}
        l.headers_inv = {0:'a',1:'b',2:'c'}

        self.assertEqual('1',l['a'])
        self.assertEqual('2',l['b'])
        self.assertEqual('3',l['c'])

        l.set_label('b','lbl',None)
        
        self.assertEqual('1',l['a'])
        self.assertEqual('2',l['b'])
        self.assertEqual('3',l['c'])
        self.assertEqual(['1','2','3'],l)
        self.assertEqual('2',l['lbl'])

        self.assertEqual(['1','3'],l.feats)
        self.assertEqual('2',l.label)

    def test_set_label_pop(self):
        l = DenseRow(loader=lambda:['1','2','3'])
        l.set_label(1,'lbl',None)
        self.assertEqual('2',l.pop('lbl'))
        self.assertEqual(['1','3'],l)

    def test_str_and_repr(self):
        l = DenseRow(loader=lambda:[1,2,3])

        self.assertEqual("DenseRow: Unloaded", str(l))
        self.assertEqual("DenseRow: Unloaded", l.__repr__())

        l[1]

        self.assertEqual("DenseRow: [1, 2, 3]", str(l))
        self.assertEqual("DenseRow: [1, 2, 3]", l.__repr__())

class SparseRow_Tests(unittest.TestCase):

    def test_loaded(self):
        l = SparseRow(loaded={'a':1,'b':2,'c':3})
        self.assertEqual({'a':1,'b':2,'c':3},dict(l))

    def test_loader(self):
        l = SparseRow(loader=lambda:{'a':1,'b':2,'c':3})
        self.assertEqual({'a':1,'b':2,'c':3},dict(l))

    def test_loader_get(self):
        l = SparseRow(loader=lambda:{'a':1,'b':2,'c':3})
        self.assertEqual(2,l['b'])

    def test_loader_set(self):
        l = SparseRow(loader=lambda:{'a':1,'b':2,'c':3})
        
        l['b'] = 5

        self.assertEqual(5,l['b'])

    def test_loader_del(self):
        l = SparseRow(loader=lambda:{'a':1,'b':2,'c':3})
        del l['b']
        self.assertEqual({'a':1,'c':3},dict(l))

    def test_loader_iter(self):
        l = SparseRow(loader=lambda:{'a':1,'b':2,'c':3})
        self.assertEqual(['a','b','c'],list(l))

    def test_loader_len(self):
        l = SparseRow(loader=lambda:{'a':1,'b':2,'c':3})
        self.assertEqual(3,len(l))

    def test_loader_del_del(self):
        l = SparseRow(loader=lambda:{'a':1,'b':2,'c':3})

        del l['b']
        del l['c']

        self.assertFalse(l._loaded)
        self.assertEqual({'a':1},dict(l))
        self.assertTrue(l._loaded)

    def test_headers(self):
        l = SparseRow(loader=lambda:{'a':1,'b':2,'c':3})
        l.headers = {'0':'a','1':'b','2':'c'}
        self.assertEqual(1,l['0'])
        self.assertEqual(2,l['1'])
        self.assertEqual(3,l['2'])

    def test_headers_del(self):
        l = SparseRow(loader=lambda:{'0':1,'1':2,'2':3})
        l.headers = {'a':'0','b':'1','c':'2'}
        l.headers_inv = {v:k for k,v in l.headers.items()}
        del l['b']
        self.assertEqual({'a':1,'c':3},dict(l))

    def test_set_label(self):
        l = SparseRow(loader=lambda:{'a':1,'b':2,'c':3})
        l.set_label('b','lbl',None)
        self.assertEqual(['a','b','c'],list(l))
        self.assertEqual({'a':1,'c':3},l.feats)
        self.assertEqual(2,l.label)

    def test_set_label_encoders(self):
        l = SparseRow(loader=lambda:{'a':1,'b':2,'c':3})
        l.encoders = {'a':int,'b':str,'c':int}
        l.set_label('b','lbl',None)
        self.assertEqual(['a','b','c'],list(l))
        self.assertEqual({'a':1,'c':3},l.feats)
        self.assertEqual('2',l.label)

    def test_set_label_headers(self):
        l = SparseRow(loader=lambda:{'0':1,'1':2,'2':3})
        l.headers = {'a':'0','b':'1','c':'2'}
        l.headers_inv = {v:k for k,v in l.headers.items()}
        l.set_label('b','lbl',None)
        self.assertEqual(['a','b','c'],list(l))
        self.assertEqual({'a':1,'c':3},l.feats)
        self.assertEqual(2,l.label)

    def test_set_label_del(self):
        l = SparseRow(loader=lambda:{'0':1,'1':2,'2':3})
        l.headers = {'a':'0','b':'1','c':'2'}
        l.headers_inv = {v:k for k,v in l.headers.items()}
        l.set_label('b','lbl',None)
        del l['lbl']
        self.assertEqual({'a':1,'c':3},dict(l))

    def test_no_label(self):
        l = SparseRow(loader=lambda:{'0':1,'1':2,'2':3})
        with self.assertRaises(CobaException):
            l.label

    def test_str_and_repr(self):
        l = SparseRow(loader=lambda:{'a':1,'b':2,'c':3})

        self.assertEqual("SparseRow: Unloaded", str(l))
        self.assertEqual("SparseRow: Unloaded", l.__repr__())

        l[1]

        self.assertEqual("SparseRow: {'a': 1, 'b': 2, 'c': 3}", str(l))
        self.assertEqual("SparseRow: {'a': 1, 'b': 2, 'c': 3}", l.__repr__())

if __name__ == '__main__':
    unittest.main()
