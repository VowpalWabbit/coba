import unittest

from coba.benchmarks.transactions import Transaction
from coba.benchmarks.results import Result, Table

class Table_Tests(unittest.TestCase):

    def test_add_row(self):
        table = Table("test", ['a'])

        table.add_row(a='A', b='B')
        table.add_row(a='a', b='B')

        self.assertTrue('A' in table)
        self.assertTrue('a' in table)
        self.assertTrue({'a':'A'} in table)
        self.assertTrue({'a':'a'} in table)

        self.assertEqual(table.get_row('a'), {'a':'a', 'b':'B'})
        self.assertEqual(table.get_row('A'), {'a':'A', 'b':'B'})

    def test_update_row(self):
        table = Table("test", ['a'])

        table.add_row(a='a', b='B')
        table.add_row('a','C')

        self.assertTrue('a' in table)
        self.assertTrue({'a':'a'} in table)
        self.assertFalse({'a':'C'} in table)

        self.assertEqual(table.get_row('a'), {'a':'a', 'b':'C'})

    def test_to_indexed_tuples(self):
        table = Table("test", ['a'])

        table.add_row(a='A', b='B')
        table.add_row(a='a', b='b')

        t = table.to_indexed_tuples()

        self.assertTrue('a' in t)
        self.assertTrue('A' in t)

        self.assertEqual(t['a'].a, 'a')
        self.assertEqual(t['a'].b, 'b')

        self.assertEqual(t['A'].a, 'A')
        self.assertEqual(t['A'].b, 'B')

class Result_Tests(unittest.TestCase):

    def test_has_batches_key(self):
        result = Result.from_transactions([
            Transaction.batch(0, 1, a='A', reward=[1,1]),
            Transaction.batch(0, 2, b='B', reward=[1,1])
        ])

        self.assertEqual("{'Learners': 0, 'Simulations': 0, 'Interactions': 4}", str(result))

        self.assertTrue( (0,1) in result.batches)
        self.assertTrue( (0,2) in result.batches)

        self.assertEqual(len(result.batches), 2)

    def test_has_version(self):
        result = Result.from_transactions([Transaction.version(1)])
        self.assertEqual(result.version, 1)

if __name__ == '__main__':
    unittest.main()