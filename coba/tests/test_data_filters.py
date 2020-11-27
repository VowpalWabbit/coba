
import unittest

from coba.data.filters import ColumnDecoder, ColumnRemover
from coba.data.encoders import NumericEncoder, StringEncoder

class ColumnDecoder_Tests(unittest.TestCase):

    def test_with_headers_1(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        decoder = ColumnDecoder([NumericEncoder(),NumericEncoder(),NumericEncoder()], ['a','b','c'])
        self.assertEqual([['a',1,4],['b',2,5],['c',3,6]], list(decoder.filter(csv)))

    def test_with_headers_2(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        decoder = ColumnDecoder([NumericEncoder(),StringEncoder(),NumericEncoder()], ['a','b','c'])
        self.assertEqual([['a',1,4],['b','2','5'],['c',3,6]], list(decoder.filter(csv)))

    def test_with_headers_3(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        decoder = ColumnDecoder([NumericEncoder(),NumericEncoder(),StringEncoder()], ['a','c','b'])
        self.assertEqual([['a',1,4],['b','2','5'],['c',3,6]], list(decoder.filter(csv)))

    def test_with_indexes_1(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        decoder = ColumnDecoder([NumericEncoder(),NumericEncoder(),NumericEncoder()])
        self.assertEqual([['a',1,4],['b',2,5],['c',3,6]], list(decoder.filter(csv)))

    def test_with_indexes_2(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        decoder = ColumnDecoder([NumericEncoder(),StringEncoder(),NumericEncoder()])
        self.assertEqual([['a',1,4],['b','2','5'],['c',3,6]], list(decoder.filter(csv)))

class ColumnRemover_Tests(unittest.TestCase):

    def test_with_headers_1(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        remover = ColumnRemover()
        self.assertEqual([['a','1','4'],['b','2','5'],['c','3','6']], list(remover.filter(csv)))

    def test_with_headers_2(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        remover = ColumnRemover(['b'])
        self.assertEqual([['a','1','4'],['c','3','6']], list(remover.filter(csv)))

    def test_with_headers_3(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        remover = ColumnRemover(['b','a'])
        self.assertEqual([['c','3','6']], list(remover.filter(csv)))

    def test_with_indexes_1(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        remover = ColumnRemover()
        self.assertEqual([['a','1','4'],['b','2','5'],['c','3','6']], list(remover.filter(csv)))

    def test_with_indexes_2(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        remover = ColumnRemover([1])
        self.assertEqual([['a','1','4'],['c','3','6']], list(remover.filter(csv)))

    def test_with_indexes_3(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        remover = ColumnRemover([1,0])
        self.assertEqual([['c','3','6']], list(remover.filter(csv)))

if __name__ == '__main__':
    unittest.main()