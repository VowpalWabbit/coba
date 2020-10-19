import unittest

from coba.data import Table

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

if __name__ == '__main__':
    unittest.main()