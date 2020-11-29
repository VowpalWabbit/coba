import timeit
import unittest

from coba.data.filters import CsvReader, ColEncoder, ColRemover, CsvTransposer, LabeledCsvCleaner
from coba.data.encoders import NumericEncoder, StringEncoder
from coba.execution import ExecutionContext, NoneLogger

ExecutionContext.Logger = NoneLogger()

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

        decoder = ColEncoder(['a','b','c'],[NumericEncoder(),NumericEncoder(),NumericEncoder()])
        self.assertEqual([['a',1,4],['b',2,5],['c',3,6]], list(decoder.filter(csv)))

    def test_with_headers_2(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        decoder = ColEncoder(['a','b','c'],[NumericEncoder(),StringEncoder(),NumericEncoder()])
        self.assertEqual([['a',1,4],['b','2','5'],['c',3,6]], list(decoder.filter(csv)))

    def test_with_headers_3(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        decoder = ColEncoder(['a','c','b'],[NumericEncoder(),NumericEncoder(),StringEncoder()])
        self.assertEqual([['a',1,4],['b','2','5'],['c',3,6]], list(decoder.filter(csv)))

    def test_with_indexes_1(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        decoder = ColEncoder([],[NumericEncoder(),NumericEncoder(),NumericEncoder()])
        self.assertEqual([['a',1,4],['b',2,5],['c',3,6]], list(decoder.filter(csv)))

    def test_with_indexes_2(self):
        csv = [
            ['a','1','4'],
            ['b','2','5'],
            ['c','3','6']
        ]

        decoder = ColEncoder([],[NumericEncoder(),StringEncoder(),NumericEncoder()])
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

class LabeledCsvCleaner_Tests(unittest.TestCase):

    def test_large_from_memory(self):
        headers      = list(map(str,range(15)))
        values       = [["1","0"]*15]*100000
        table        = [headers]+values
        label_col    = 0
        encoders     = [NumericEncoder()]*15

        clean_table = lambda: LabeledCsvCleaner(label_col, headers, encoders, [], rmv_header=True).filter(table)

        time = min(timeit.repeat(clean_table, repeat=2, number=1))

        #was approximately 0.5 at best performance
        self.assertLess(time, 3)

if __name__ == '__main__':
    unittest.main()