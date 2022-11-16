
import unittest

from coba.utilities import HashableDict
from coba.exceptions import CobaException
from coba.pipes.rows import EncodeRow, IndexRow, LabelRow, DenseRow, SparseRow
from coba.pipes.rows import DenseRow2, HeaderRow, EncoderRow, SelectRow, SparseRow2, LabelRow2

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

    def test_to_builtin_no_encoders(self):
        l = DenseRow(loader=lambda:['1','2','3'])
        self.assertEqual(('1','2','3'),l.to_builtin())
        self.assertIsInstance(l.to_builtin(),tuple)

    def test_to_builtin_encoders(self):
        l = DenseRow(loader=lambda:['1','2','3'])
        l.encoders = [int,int,int]
        self.assertEqual((1,2,3),l.to_builtin())

    def test_to_builtin_lbl_encoders(self):
        l = DenseRow(loader=lambda:['1','2','3'])
        l.encoders = [int,int,int]
        l.set_label(1,None,lambda v: v*2)
        self.assertEqual((1,4,3),l.to_builtin())

    def test_str_and_repr(self):
        l = DenseRow(loader=lambda:[1,2,3])

        self.assertEqual("DenseRow: Unloaded", str(l))
        self.assertEqual("DenseRow: Unloaded", repr(l))

        l[1]

        self.assertEqual("DenseRow: [1, 2, 3]", str(l))
        self.assertEqual("DenseRow: [1, 2, 3]", repr(l))

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

    def test_to_builtin_no_encoders(self):
        l = SparseRow(loader=lambda:{'0':'1','1':'2','2':'3'})
        self.assertEqual({'0':'1','1':'2','2':'3'},l.to_builtin())

    def test_to_builtin_encoders(self):
        l = SparseRow(loader=lambda:{'0':'1','1':'2','2':'3'})
        l.encoders = {'0':int,'1':int,'2':int}
        self.assertEqual({'0':1,'1':2,'2':3},l.to_builtin())
        self.assertIsInstance(l.to_builtin(), HashableDict)

    def test_to_builtin_lbl_encoders(self):
        l = SparseRow(loader=lambda:{'0':'1','1':'2','2':'3'})
        l.encoders = {'0':int,'1':int,'2':int}
        l.set_label('1',None,lambda v: v*2)
        self.assertEqual({'0':1,'1':4,'2':3},l.to_builtin())

    def test_str_and_repr(self):
        l = SparseRow(loader=lambda:{'a':1,'b':2,'c':3})

        self.assertEqual("SparseRow: Unloaded", str(l))
        self.assertEqual("SparseRow: Unloaded", repr(l))

        l[1]

        self.assertEqual("SparseRow: {'a': 1, 'b': 2, 'c': 3}", str(l))
        self.assertEqual("SparseRow: {'a': 1, 'b': 2, 'c': 3}", repr(l))

class DenseRow2_Tests(unittest.TestCase):

    def test_loaded(self):
        l = DenseRow2(loaded=[1,2,3])
        self.assertEqual([1,2,3],l)

    def test_loader_get(self):
        l = DenseRow2(loader=lambda:[1,2,3])
        self.assertEqual([1,2,3],l)

    def test_loader_set(self):
        l = DenseRow2(loader=lambda:[1,2,3])
        l[1] = 5
        self.assertEqual([1,5,3],l)

    def test_loader_len(self):
        l = DenseRow2(loader=lambda:[1,2,3])
        self.assertEqual(3,len(l))

    def test_bad_eq(self):
        l = DenseRow2(loader=lambda:[1,2,3])
        self.assertNotEqual(3,l)

    def test_str_and_repr(self):
        l = DenseRow2(loader=lambda:[1,2,3])

        self.assertEqual("DenseRow: Unloaded", str(l))
        self.assertEqual("DenseRow: Unloaded", repr(l))

        l[1]

        self.assertEqual("DenseRow: [1, 2, 3]", str(l))
        self.assertEqual("DenseRow: [1, 2, 3]", repr(l))

class SparseRow2_Tests(unittest.TestCase):

    def test_loaded_equal(self):
        sparse = {'a':1,'b':2,'c':3}
        self.assertEqual(SparseRow2(loaded=sparse),sparse) 

    def test_loader_get(self):
        r = SparseRow2(loader=lambda:{'a':1,'b':2,'c':3})
        self.assertEqual(1,r['a'])
        self.assertEqual(2,r['b'])
        self.assertEqual(3,r['c'])
        self.assertEqual({'a':1,'b':2,'c':3},r)

    def test_loader_set(self):
        r = SparseRow2(loader=lambda:{'a':1,'b':2,'c':3})
        r['a'] = 5
        self.assertEqual({'a':5,'b':2,'c':3},r)

    def test_loader_len(self):
        r = SparseRow2(loader=lambda:{'a':1,'b':2,'c':3})
        self.assertEqual(3,len(r))

    def test_bad_eq(self):
        l = SparseRow2(loader=lambda:{'a':1})
        self.assertNotEqual(3,l)

    def test_iter(self):
        r = SparseRow2(loader=lambda:{'a':1,'b':2,'c':3})
        self.assertEqual(['a','b','c'],list(r))

    def test_str_and_repr(self):
        r = SparseRow2(loader=lambda:{'a':1,'b':2,'c':3})

        self.assertEqual("SparseRow: Unloaded", str(r))
        self.assertEqual("SparseRow: Unloaded", repr(r))

        r['a']

        self.assertEqual("SparseRow: {'a': 1, 'b': 2, 'c': 3}", str(r))
        self.assertEqual("SparseRow: {'a': 1, 'b': 2, 'c': 3}", repr(r))

class HeaderRow_Tests(unittest.TestCase):

    def test_get(self):

        r = HeaderRow([1,2,3], {"a":0, "b":1, "c": 2})
        self.assertEqual([1,2,3],r)
        self.assertEqual(1,r["a"])
        self.assertEqual(2,r["b"])
        self.assertEqual(3,r["c"])
        self.assertEqual(1,r[0])
        self.assertEqual(2,r[1])
        self.assertEqual(3,r[2])

    def test_set(self):
        r = HeaderRow([1,2,3], {"a":0, "b":1, "c": 2})

        r[1] = 5
        self.assertEqual([1,5,3],r)

        r["a"] = 8
        self.assertEqual([8,5,3],r)

    def test_iter(self):
        r = HeaderRow([1,2,3], {"a":0, "b":1, "c": 2})
        self.assertEqual([1,2,3],list(r))

    def test_len(self):
        r = HeaderRow([1,2,3], {"a":0, "b":1, "c": 2})
        self.assertEqual(3,len(r))

    def test_str_repr(self):
        r = HeaderRow([1,2,3], {"a":0, "b":1, "c": 2})
        self.assertEqual('[1, 2, 3]',str(r))
        self.assertEqual('[1, 2, 3]',repr(r))

    def test_cascading_getattr(self):
        r = HeaderRow([1,2,3], {"a":0, "b":1, "c": 2})
        self.assertEqual(1,r.index(2))

class EncoderRow_Tests(unittest.TestCase):

    def test_get(self):

        r = EncoderRow(['1',2,'3'], [int,str,int])
        self.assertEqual([1,'2',3],r)
        self.assertEqual(1  ,r[0])
        self.assertEqual('2',r[1])
        self.assertEqual(3  ,r[2])

    def test_set(self):
        r = EncoderRow(['1',2,'3'], [int,str,int])

        r[1] = 5
        self.assertEqual([1,5,3],r)

        r[0] = '8'
        self.assertEqual(['8',5,3],list(r))

    def test_len(self):
        r = EncoderRow(['1',2,'3'], [int,str,int])
        self.assertEqual(3,len(r))

    def test_str_repr(self):
        r = EncoderRow([1,2,3], [str,str,str])
        self.assertEqual('[1, 2, 3]',str(r))
        self.assertEqual('[1, 2, 3]',repr(r))

    def test_cascading_getattr(self):
        r = EncoderRow([1,2,3], [str,str,str])
        self.assertEqual(1,r.index(2))

class SelectRow_Tests(unittest.TestCase):

    def test_get_int(self):
        r = SelectRow([1,2,3], [0,2])
        self.assertEqual([1,3],r)
        self.assertEqual(1,r[0])
        self.assertEqual(3,r[1])

    def test_get_str(self):
        r = SelectRow(HeaderRow([1,2],{'a':0,'b':1}), ['a'])
        self.assertEqual([1],r)
        self.assertEqual(1,r[0])
        self.assertEqual(1,r['a'])

    def test_get_str_bad(self):
        r = SelectRow(HeaderRow([1,2],{'a':0,'b':1}), ['a'])

        with self.assertRaises(KeyError):
            self.assertEqual(1,r['b'])

    def test_set(self):
        r = SelectRow([1,2,3], [0,2])
        r[1] = 5
        self.assertEqual([1,5],r)
        r[0] = '8'
        self.assertEqual(['8',5],r)

    def test_len(self):
        r = SelectRow([1,2,3], [0,2])
        self.assertEqual(2,len(r))

    def test_str_repr(self):
        r = SelectRow([1,2,3], [0,2])
        self.assertEqual('[1, 3]',str(r))
        self.assertEqual('[1, 3]',repr(r))

    def test_cascading_getattr(self):
        r = SelectRow([1,2,3], [0,2])
        self.assertEqual(1,r.index(2))

class LabelRow2_Tests(unittest.TestCase):

    def test_get(self):
        r = LabelRow2([1,2,3],2,'lbl')
        self.assertEqual(1, r[0])
        self.assertEqual(2, r[1])
        self.assertEqual(3, r[2])
        self.assertEqual(3, r['lbl'])
        self.assertEqual([1,2,3],list(r))

    def test_set(self):
        r = LabelRow2([1,2,3],2,'lbl')
        r[1] = 5
        r['lbl'] = 10
        self.assertEqual([1,5,10],r)

    def test_len(self):
        r = LabelRow2([1,2,3],2,'lbl')
        self.assertEqual(3,len(r))

    def test_eq(self):
        r = LabelRow2([1,2,3],2,'lbl')
        self.assertEqual(r,[1,2,3])

    def test_cascading_getattr(self):
        r = LabelRow2([1,2,3],2,'lbl')
        self.assertEqual(1,r.index(2))

    def test_str_and_repr(self):
        r = LabelRow2([1,2,3],2,'lbl')
        self.assertEqual("[1, 2, 3]", str(r))
        self.assertEqual("[1, 2, 3]", repr(r))

if __name__ == '__main__':
    unittest.main()
