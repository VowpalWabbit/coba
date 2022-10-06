
import unittest

from coba.exceptions import CobaException
from coba.pipes.rows import ParseRow, DropRow, EncodeRow, DenseRow, IndexRow, SparseRow, LabelRow

class DenseRow_Tests(unittest.TestCase):

    def test_getitem_parser(self):
        row = DenseRow("1,2,3", lambda l: l.split(','), False)

        self.assertEqual('1',row[0])
        self.assertEqual('2',row[1])
        self.assertEqual('3',row[2])

    def test_getitem_direct(self):
        row = DenseRow([0,1,2], False)

        self.assertEqual(0,row[0])
        self.assertEqual(1,row[1])
        self.assertEqual(2,row[2])

    def test_has_missing(self):
        row1 = DenseRow([0,1,2],False)
        row2 = DenseRow([0,1,2],True)

        self.assertFalse(row1.any_missing)
        self.assertTrue( row2.any_missing)

    def test_equal(self):
        row = DenseRow([0,1,2])

        self.assertEqual([0,1,2], row)
        self.assertNotEqual({'a':0,'b':1,'c':2}, row)

    def test_str(self):
        row = DenseRow([0,1,2])
        self.assertEqual(str([0,1,2]), str(row))

    def test_repr(self):
        row = DenseRow([0,1,2])
        self.assertEqual(repr([0,1,2]), repr(row))

class SparseRow_Tests(unittest.TestCase):
    
    def test_getitem_parse(self):
        row = SparseRow('a:1,b:2', lambda l: { kv[0]:kv[2] for kv in l.split(',') }, False)

        self.assertEqual('1',row['a'])
        self.assertEqual('2',row['b'])

    def test_getitem_direct(self):
        row = SparseRow({'a':0,'b':1,'c':2})

        self.assertEqual(0,row['a'])
        self.assertEqual(1,row['b'])
        self.assertEqual(2,row['c'])
        #self.assertEqual(0,row['d'])

    def test_any_missing(self):
        row1 = SparseRow({'a':0,'b':1,'c':2},False)
        row2 = SparseRow({'a':0,'b':1,'c':2},True)

        self.assertFalse(row1.any_missing)
        self.assertTrue(row2.any_missing)

    def test_equal(self):
        row = SparseRow({'a':0,'b':1,'c':2},False)
        self.assertEqual({'a':0,'b':1,'c':2}, row)
        self.assertNotEqual([0,1,2], row)

    def test_str(self):
        row = SparseRow({'a':0,'b':1,'c':2},['a','b','c'])
        self.assertEqual(str({'a':0,'b':1,'c':2}), str(row))

    def test_repr(self):
        row = SparseRow({'a':0,'b':1,'c':2})
        self.assertEqual(repr({'a':0,'b':1,'c':2}), repr(row))

class DropRow_Tests(unittest.TestCase):

    def test_no_drop_list(self):
        row = DropRow([]).filter([1,2,3])
        self.assertEqual([1,2,3],row)
        self.assertEqual([1,2,3],row)

    def test_one_drop_list(self):
        self.assertEqual([1,3],DropRow([1]).filter([1,2,3]))

    def test_two_drop_list(self):
        self.assertEqual([3,5],DropRow([0,2]).filter(DropRow([1]).filter([1,2,3,4,5])))

    def test_one_drop_dict(self):
        self.assertEqual({'a':1,'c':3},DropRow(['b']).filter({'a':1,'b':2,'c':3}))

    def test_no_drop_dense(self):
        row = DropRow([]).filter(DenseRow([1,2,3]))
        self.assertEqual([1,2,3],row)

    def test_one_drop_dense(self):
        self.assertEqual([1,3],DropRow([1]).filter(DenseRow([1,2,3])))

    def test_two_drop_dense(self):
        self.assertEqual([3,5],DropRow([0,2]).filter(DropRow([1]).filter(DenseRow([1,2,3,4,5]))))

    def test_one_drop_sparse(self):
        self.assertEqual({'a':1,'c':3},DropRow(['b']).filter(SparseRow({'a':1,'b':2,'c':3})))

    def test_bad_row(self):
        with self.assertRaises(CobaException) as e:
            DropRow(['b']).filter(1)
        
        self.assertEqual("Unrecognized row type passed to DropRow.", str(e.exception))

class EncoderRow_Tests(unittest.TestCase):
    def test_encode_dense(self):
        self.assertEqual([0,1,2.2],EncodeRow([int,int,float]).filter(['0','1','2.2']))

    def test_encode_sparse(self):
        self.assertEqual({'a':0,'b':1,'c':2.2},EncodeRow({'a':int,'b':int,'c':float}).filter({'a':'0','b':'1','c':'2.2'}))

class IndexRow_Tests(unittest.TestCase):
    def test_index(self):
        row = IndexRow({'a':0,'b':1}).filter([1,2])

        self.assertEqual(1, row['a'])
        self.assertEqual(2, row['b'])
        self.assertEqual(1, row[0])
        self.assertEqual(2, row[1])

class LabelRow_Tests(unittest.TestCase):
    def test_dense(self):
        row = LabelRow(1).filter([1,2,3])
        
        self.assertEqual([2,1,3],row)
        self.assertEqual([1,0,2],row.keys())

    def test_sparse(self):
        row = SparseRow({'a':1,'b':2,'c':3})
        self.assertNotEqual(list(row.keys())[0],'b')
        row = LabelRow('b').filter(row)
        self.assertEqual(list(row.keys())[0],'b')

class ParserRow_Tests(unittest.TestCase):

    def test_getitem(self):
        row = ParseRow('1,2,3', lambda l: l.split(','))

        self.assertEqual('1',row[0])
        self.assertEqual('2',row[1])
        self.assertEqual('3',row[2])

    def test_parse_once(self):
        
        parse_count=[0]

        def parse(line:str):
            parse_count[0] += 1
            return line.split(',')

        row = ParseRow('1,2,3', parse)

        self.assertEqual(0, parse_count[0])

        self.assertEqual('1',row[0])
        self.assertEqual(1, parse_count[0])

        self.assertEqual('2',row[1])
        self.assertEqual(1, parse_count[0])

if __name__ == '__main__':
    unittest.main()
