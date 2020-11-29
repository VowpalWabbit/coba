
import unittest

from coba.data.filters import CsvReader, ColEncoder, ColRemover, CsvTransposer
from coba.data.encoders import NumericEncoder, StringEncoder

class CsvReader_Tests(unittest.TestCase):
    def test_simple_sans_empty(self):
        self.assertEqual([['a','b','c'],['1','2','3']], list(CsvReader().filter(['a,b,c', '1,2,3'])))
    
    def test_simple_with_empty(self):
        self.assertEqual([['a','b','c'],['1','2','3']], list(CsvReader().filter(['a,b,c', '', '1,2,3', ''])))

class CsvTransposer_Tests(unittest.TestCase):

    def test_simple_sans_empty(self):
        self.assertEqual([('a','1'),('b','2'),('c','3')], list(CsvTransposer().filter([['a','b','c'],['1','2','3']])))

    def test_simple_with_empty(self):
        self.assertEqual([('a','1'),('b','2'),('c','3')], list(CsvTransposer().filter([['a','b','c'],['1','2','3'],[]])))

class ColEncoder_Tests(unittest.TestCase):

    def test_with_headers_1(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        decoder = ColEncoder([NumericEncoder(),NumericEncoder(),NumericEncoder()], ['a','b','c'])
        self.assertEqual([['a',1,4],['b',2,5],['c',3,6]], list(decoder.filter(csv)))

    def test_with_headers_2(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        decoder = ColEncoder([NumericEncoder(),StringEncoder(),NumericEncoder()], ['a','b','c'])
        self.assertEqual([['a',1,4],['b','2','5'],['c',3,6]], list(decoder.filter(csv)))

    def test_with_headers_3(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        decoder = ColEncoder([NumericEncoder(),NumericEncoder(),StringEncoder()], ['a','c','b'])
        self.assertEqual([['a',1,4],['b','2','5'],['c',3,6]], list(decoder.filter(csv)))

    def test_with_indexes_1(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        decoder = ColEncoder([NumericEncoder(),NumericEncoder(),NumericEncoder()])
        self.assertEqual([['a',1,4],['b',2,5],['c',3,6]], list(decoder.filter(csv)))

    def test_with_indexes_2(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        decoder = ColEncoder([NumericEncoder(),StringEncoder(),NumericEncoder()])
        self.assertEqual([['a',1,4],['b','2','5'],['c',3,6]], list(decoder.filter(csv)))

class ColRemover_Tests(unittest.TestCase):

    def test_with_headers_1(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        remover = ColRemover()
        self.assertEqual([['a','1','4'],['b','2','5'],['c','3','6']], list(remover.filter(csv)))

    def test_with_headers_2(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        remover = ColRemover(['b'])
        self.assertEqual([['a','1','4'],['c','3','6']], list(remover.filter(csv)))

    def test_with_headers_3(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        remover = ColRemover(['b','a'])
        self.assertEqual([['c','3','6']], list(remover.filter(csv)))

    def test_with_indexes_1(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        remover = ColRemover()
        self.assertEqual([['a','1','4'],['b','2','5'],['c','3','6']], list(remover.filter(csv)))

    def test_with_indexes_2(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        remover = ColRemover([1])
        self.assertEqual([['a','1','4'],['c','3','6']], list(remover.filter(csv)))

    def test_with_indexes_3(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        remover = ColRemover([1,0])
        self.assertEqual([['c','3','6']], list(remover.filter(csv)))

if __name__ == '__main__':
    unittest.main()