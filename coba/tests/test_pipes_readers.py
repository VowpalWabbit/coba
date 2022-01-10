import unittest
from coba.exceptions import CobaException

from coba.pipes import LibSvmReader, ArffReader, CsvReader, ManikReader
from coba.contexts import NullLogger, CobaContext

CobaContext.logger = NullLogger()

class CsvReader_Tests(unittest.TestCase):
    def test_dense_with_header(self):

        parsed = list(CsvReader(has_header=True).filter(['a,b,c', '1,2,3']))

        self.assertEqual(1, len(parsed))
        self.assertEqual('1', parsed[0][0])
        self.assertEqual('1', parsed[0]['a'])
        self.assertEqual('2', parsed[0][1])
        self.assertEqual('2', parsed[0]['b'])
        self.assertEqual('3', parsed[0][2])
        self.assertEqual('3', parsed[0]['c'])

    def test_dense_sans_empty(self):
        self.assertEqual([['a','b','c'],['1','2','3']], list(CsvReader().filter(['a,b,c', '1,2,3'])))
    
    def test_dense_with_empty(self):
        self.assertEqual([['a','b','c'],['1','2','3']], list(CsvReader().filter(['a,b,c', '', '1,2,3', ''])))

class ArffReader_Tests(unittest.TestCase):

    def test_dense_sans_data(self):
        lines = [
            "@relation news20",
            "@attribute a numeric",
            "@attribute B numeric",
            "@attribute c {0, class_B, class_C, class_D}",
            "@data",
        ]

        expected = []
        
        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_dense_with_spaces_after_commas(self):
        lines = [
            "@relation news20",
            "@attribute a numeric",
            "@attribute B numeric",
            "@attribute c {class_B, class_C, class_D}",
            "@data",
            "1,  2,  class_B",
            "2,  3,  class_C",
        ]

        expected = [
            [1,2,(1,0,0)],
            [2,3,(0,1,0)]
        ]
        
        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_dense_sans_empty_lines(self):
        lines = [
            "@relation news20",
            "@attribute a numeric",
            "@attribute B numeric",
            "@attribute c {0, class_B, class_C, class_D}",
            "@data",
            "1,2,class_B",
            "2,3,0",
        ]

        expected = [
            [1,2,(0,1,0,0)],
            [2,3,(1,0,0,0)]
        ]
        
        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_dense_with_missing_value(self):
        lines = [
            "@relation news20",
            "@attribute a numeric",
            "@attribute B numeric",
            "@attribute c {class_B, class_C, class_D}",
            "@data",
            "1,2,class_B",
            "2,3,?",
        ]

        expected = [
            [1,2,(1,0,0)],
            [2,3,'?']
        ]
        
        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_dense_with_empty_lines(self):
        lines = [
            "@relation news20",
            "@attribute A numeric",
            "@attribute B numeric",
            "@attribute C {0, class_B, class_C, class_D}",
            "@data",
            "",
            "",
            "1,2,class_B",
            "2,3,0",
            ""
        ]

        expected = [
            [1,2,(0,1,0,0)],
            [2,3,(1,0,0,0)]
        ]
        
        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_dense_with_comments(self):
        lines = [
            "%This is a comment",
            "@relation news20",
            "@attribute a numeric",
            "@attribute b numeric",
            "@attribute c {0, class_B, class_C, class_D}",
            "@data",
            "1,2,class_B",
            "2,3,0"
        ]

        expected = [
            [1,2,(0,1,0,0)],
            [2,3,(1,0,0,0)]
        ]
        
        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_dense_with_strings(self):
        lines = [
            "@relation news20",
            "@attribute a string",
            "@attribute b string",
            "@attribute c {0, class_B, class_C, class_D}",
            "@data",
            "1,2,class_B",
            "2,3,0"
        ]

        expected = [
            ['1','2',(0,1,0,0)],
            ['2','3',(1,0,0,0)]
        ]
        
        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_sparse(self):
        lines = [
            "@relation news20",
            "@attribute a numeric",
            "@attribute b numeric",
            "@attribute c {0, class_B, class_C, class_D}",
            "@data",
            "{0 2,1 3}",
            "{0 1,1 1,2 class_B}",
            "{1 1}",
            "{0 1,2 class_D}",
        ]

        expected = [
            {0:2, 1:3},
            {0:1, 1:1, 2:(0,1,0,0)},
            {1:1},
            {0:1,2:(0,0,0,1)}
        ]
        
        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_sparse_with_empty_lines(self):
        lines = [
            "@relation news20",
            "@attribute a numeric",
            "@attribute b numeric",
            "@attribute c {0, class_B, class_C, class_D}",
            "@data",
            "",
            "{0 2,1 3}",
            "{0 1,1 1,2 class_B}",
            "",
            "",
            "{1 1}",
            "{0 1,2 class_D}",
        ]

        expected = [
            {0:2, 1:3},
            {0:1, 1:1, 2:(0,1,0,0)},
            {1:1},
            {0:1,2:(0,0,0,1)}
        ]
        
        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_sparse_with_spaces_after_comma(self):
        lines = [
            "@relation news20",
            "@attribute a numeric",
            "@attribute b numeric",
            "@attribute c {0, class_B, class_C, class_D}",
            "@data",
            "{0 2, 1 3}",
            "{0 1, 1 1,2 class_B}",
            "{1 1}",
            "{0 1, 2 class_D}",
        ]

        expected = [
            {0:2, 1:3},
            {0:1, 1:1, 2:(0,1,0,0)},
            {1:1},
            {0:1,2:(0,0,0,1)}
        ]
        
        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_leading_and_trailing_comments(self):
        lines = [
            "%",
            "%",
            "@relation news20",
            "@attribute a string",
            "@attribute b string",
            "@attribute c {0, class_B, class_C, class_D}",
            "@data",
            "1,2,class_B",
            "2,3,0",
            "%"
        ]

        expected = [
            ['1','2',(0,1,0,0)],
            ['2','3',(1,0,0,0)]
        ]
        
        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_headers_with_quotes_and_pct(self):
        lines = [
            "@relation news20",
            "@attribute 'a%3' string",
            "@attribute 'b%4' string",
            "@attribute 'c%5' {class_B, class_C, class_D}",
            "@data",
            "1,2,class_B",
            "2,3,class_C",
        ]

        expected = [
            ['1','2',(1,0,0)],
            ['2','3',(0,1,0)]
        ]

        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_bad_class_labels_throws_exception(self):
        lines = [
            "@relation news20",
            "@attribute 'a' string",
            "@attribute 'b' string",
            "@attribute 'c' {class_B, class_C, class_D}",
            "@data",
            "1,2,class_A",
        ]

        with self.assertRaises(CobaException) as e:
            list(ArffReader().filter(lines))

        self.assertIn("We were unable to find 'class_A' in ['class_B', 'class_C', 'class_D']", str(e.exception))

class LibsvmReader_Tests(unittest.TestCase):
    def test_sparse(self):
        lines = [
            "0 1:2 2:3",
            "1 1:1 2:1",
            "2 2:1",
            "1 1:1",
        ]

        expected = [
            {0:['0'], 1:2, 2:3},
            {0:['1'], 1:1, 2:1},
            {0:['2'], 2:1},
            {0:['1'], 1:1}
        ]
        
        self.assertEqual(expected, list(LibSvmReader().filter(lines)))

    def test_trailing_whitespace(self):
        lines = [
            "0 1:2 2:3",
            "1 1:1 2:1   ",
            "2 2:1",
            "1 1:1",
        ]

        expected = [
            {0:['0'], 1:2, 2:3},
            {0:['1'], 1:1, 2:1},
            {0:['2'], 2:1},
            {0:['1'], 1:1}
        ]
        
        self.assertEqual(expected, list(LibSvmReader().filter(lines)))

class ManikReader_Tests(unittest.TestCase):
    def test_sparse(self):
        lines = [
            "abcde",
            "0 1:2 2:3",
            "1 1:1 2:1",
            "2 2:1",
            "1 1:1",
        ]

        expected = [
            {0:['0'], 1:2, 2:3},
            {0:['1'], 1:1, 2:1},
            {0:['2'], 2:1},
            {0:['1'], 1:1}
        ]
        
        self.assertEqual(expected, list(ManikReader().filter(lines)))

    def test_trailing_whitespace(self):
        lines = [
            "abcde",
            "0 1:2 2:3",
            "1 1:1 2:1   ",
            "2 2:1",
            "1 1:1",
        ]

        expected = [
            {0:['0'], 1:2, 2:3},
            {0:['1'], 1:1, 2:1},
            {0:['2'], 2:1},
            {0:['1'], 1:1}
        ]
        
        self.assertEqual(expected, list(ManikReader().filter(lines)))

if __name__ == '__main__':
    unittest.main()