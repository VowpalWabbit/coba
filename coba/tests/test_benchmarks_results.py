import math
import unittest

from coba.benchmarks.transactions import Transaction
from coba.benchmarks.results import Result, Table, PackedTable

class Table_Tests(unittest.TestCase):

    def test_insert_item(self):
        table = Table("test", ['a'])

        table['A'] = dict(b='B')
        table['a'] = dict(b='B')

        self.assertTrue('A' in table)
        self.assertTrue('a' in table)

        self.assertEqual(table['a'], {'a':'a', 'b':'B'})
        self.assertEqual(table['A'], {'a':'A', 'b':'B'})

        self.assertEqual(2, len(table))

        self.assertEqual([('A', 'B'), ('a', 'B')], table.to_tuples())

    def test_update_item(self):
        table = Table("test", ['a'])

        table['a'] = dict(b='B')
        table['a'] = dict(b='C')

        self.assertTrue('a' in table)

        self.assertEqual(table['a'], {'a':'a', 'b':'C'})
        self.assertEqual(1, len(table))
        self.assertEqual([('a','C')], table.to_tuples())

    def test_missing_columns(self):
        table = Table("test", ['a'])

        table['A'] = dict(b='B')
        table['a'] = dict(c='C')

        self.assertTrue('A' in table)
        self.assertTrue('a' in table)

        self.assertEqual(table['A'], {'a':'A', 'b':'B'})
        self.assertEqual(table['a'], {'a':'a', 'c':'C'})

        self.assertEqual(2, len(table))

        expected = [('A', 'B', float('nan')), ('a', float('nan'), 'C')]
        actual   = table.to_tuples()

        for tuple1, tuple2 in zip(expected,actual):
            for val1, val2 in zip(tuple1,tuple2):
                if isinstance(val1,float) and math.isnan(val1): 
                    self.assertTrue(math.isnan(val2))
                else:
                    self.assertEqual(val1,val2)

    def test_pandas(self):
        
        import pandas as pd
        import pandas.testing

        table = Table("test", ['a'])

        table['A'] = dict(b='B',c=1,d='d')
        table['B'] = dict(e='E')

        expected_df = pd.DataFrame([
            dict(a='A',b='B',c=1,d='d'),
            dict(a='B',e='E')
        ])

        actual_df = table.to_pandas()

        pandas.testing.assert_frame_equal(expected_df,actual_df)

class PackedTable_Tests(unittest.TestCase):
    def test_insert_no_pack_item(self):
        table = PackedTable("test", ['a'])

        table['A'] = dict(b='B')

        self.assertTrue('A' in table)

        self.assertEqual(table['A'], [{'a':'A', 'index':1, 'b':'B'}])

        self.assertEqual(1, len(table))

        self.assertEqual([('A', 1, 'B')], table.to_tuples())

    def test_insert_two_pack_item(self):
        table = PackedTable("test", ['a'])

        table['A'] = dict(b=['B','b'],c=1,d=['D','d'])

        self.assertTrue('A' in table)

        self.assertEqual(table['A'], [{'a':'A', 'index':1, 'b':'B', 'c':1, 'd':'D'}, {'a':'A', 'index':2, 'b':'b', 'c':1, 'd':'d'}])

        self.assertEqual(2, len(table))

        self.assertEqual([('A', 1, 'B', 1, 'D'), ('A', 2, 'b', 1, 'd')], table.to_tuples())

    def test_pandas_two_pack_item(self):
        
        import pandas as pd
        import pandas.testing

        table = PackedTable("test", ['a'])

        table['A'] = dict(b=['B','b'],c=1,d=['D','d'])
        table['B'] = dict(e='E')

        expected_df = pd.DataFrame([
            dict(a='A',index=1,b='B',c=1,d='D'),
            dict(a='A',index=2,b='b',c=1,d='d'),
            dict(a='B',index=1,e='E')
        ])

        actual_df = table.to_pandas()

        pandas.testing.assert_frame_equal(expected_df,actual_df)

    def test_unequal_pack_exception(self):
        with self.assertRaises(Exception):
            table = PackedTable("test", ['a'])
            table['A'] = dict(b=['B','b'],c=1,d=['D','d','e'])

class Result_Tests(unittest.TestCase):

    def test_has_batches_key(self):
        result = Result.from_transactions([
            Transaction.batch(0, 1, a='A', reward=[1,1]),
            Transaction.batch(0, 2, b='B', reward=[1,1])
        ])

        self.assertEqual("{'Learners': 0, 'Simulations': 0, 'Interactions': 4}", str(result))

        self.assertTrue( (0,1) in result.interactions)
        self.assertTrue( (0,2) in result.interactions)

        self.assertEqual(len(result.interactions), 4)

    def test_has_version(self):
        result = Result.from_transactions([Transaction.version(1)])
        self.assertEqual(result.version, 1)

if __name__ == '__main__':
    unittest.main()