import unittest

from coba.encodings import NumericEncoder, StringEncoder
from coba.exceptions import CobaException

from coba.pipes import LibSvmReader, ArffReader, CsvReader, ManikReader
from coba.contexts import NullLogger, CobaContext

from coba.pipes.readers import LazyDense, LazySparse

CobaContext.logger = NullLogger()

class LazyDense_Tests(unittest.TestCase):

    def test_no_headers_no_encoders(self):
        a = LazyDense([1,2])

        self.assertEqual([1,2], a)
        self.assertEqual(2, len(a))

        a[1] = 3
        self.assertEqual(3, a[1])

        del a[1]
        self.assertEqual(1, len(a))
        self.assertEqual([1], a)

    def test_headers_no_encoders(self):
        a = LazyDense([1,2,3],['a','b','c'])

        self.assertEqual([1,2,3], a)
        self.assertEqual(3, len(a))

        self.assertEqual(1,a[0])
        self.assertEqual(2,a[1])
        self.assertEqual(3,a[2])
        self.assertEqual(1,a['a'])
        self.assertEqual(2,a['b'])
        self.assertEqual(3,a['c'])

        a['b'] = 3
        self.assertEqual(3, a['b'])

        del a['b']
        self.assertEqual(2, len(a))
        self.assertEqual([1,3], a)
        self.assertEqual(3, a['c'])

    def test_headers_and_encoders(self):
        a = LazyDense(['1','2'],['a','b'], encoders=[NumericEncoder(), StringEncoder()])

        self.assertEqual([1,'2'], a)
        self.assertEqual(2, len(a))

        self.assertEqual(1,a['a'])
        self.assertEqual('2',a['b'])

        a['b'] = 3
        self.assertEqual(3, a['b'])

        del a['b']
        self.assertEqual(1, len(a))
        self.assertEqual([1], a)

    def test_insert_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            LazyDense([1,2]).insert(0,1)

    def test_str(self):
        self.assertEqual('[1, 2, 3]', str(LazyDense([1,2,3])))

    def test_repr(self):
        self.assertEqual('[1, 2, 3]', LazyDense([1,2,3]).__repr__())

class LazySparse_Tests(unittest.TestCase):

    def test_no_headers_no_encoders(self):
        a = LazySparse({'a':1,'b':2})

        self.assertEqual({'a':1,'b':2}, a)
        self.assertEqual(2, len(a))

        a['b'] = 3
        self.assertEqual(3, a['b'])

        del a['b']
        self.assertEqual(1, len(a))
        self.assertEqual({'a':1}, a)

    def test_headers_no_encoders(self):
        a = LazySparse({'a':1,'b':2}, {'aa':'a', 'bb':'b'})

        self.assertEqual({'a':1,'b':2}, a)
        self.assertEqual(2, len(a))

        self.assertEqual(1,a['aa'])
        self.assertEqual(2,a['bb'])

        a['bb'] = 3
        self.assertEqual(3, a['b'])
        self.assertEqual(3, a['bb'])

        del a['bb']
        self.assertEqual(1, len(a))
        self.assertEqual({'a':1}, a)

    def test_headers_and_encoders(self):
        a = LazySparse({'a':'1','b':'2'}, {'aa':'a', 'bb':'b'}, {'a':NumericEncoder(), 'b': StringEncoder()})

        self.assertEqual({'a':1,'b':'2'}, a)
        self.assertEqual(2, len(a))

        self.assertEqual(1,a['aa'])
        self.assertEqual('2',a['bb'])

        a['bb'] = 3
        self.assertEqual(3, a['b'])
        self.assertEqual(3, a['bb'])

        del a['bb']
        self.assertEqual(1, len(a))
        self.assertEqual({'a':1}, a)

    def test_encode_sparse_value(self):
        values   = {'a':'1','b':'2'}
        headers  = {'A':'a', 'B':'b', 'C':'c'}
        encoders = {'a':NumericEncoder(), 'b': StringEncoder(), 'c': StringEncoder()}
        
        a = LazySparse(values, headers, encoders)

        self.assertEqual('0', a['c'])

    def test_encode_sparse_value_with_modifier(self):
        values   = {'a':'1','b':'2'}
        headers  = {'A':'a', 'B':'b', 'C':'c'}
        encoders = {'a':NumericEncoder(), 'b': StringEncoder(), 'c': StringEncoder()}
        
        a = LazySparse(values, headers, encoders, ['b','c'])

        self.assertEqual('0', a['c'])

    def test_str(self):
        self.assertEqual("{'a': 2}", str(LazySparse({'a':2})))

    def test_repr(self):
        self.assertEqual("{'a': 2}", LazySparse({'a':2}).__repr__())

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

    def test_sans_data(self):
        lines = [
            "@relation news20",
            "@attribute a numeric",
            "@attribute B numeric",
            "@attribute c {0, class_B, class_C, class_D}",
            "@data",
        ]

        expected = []
        
        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_skip_encoding(self):
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
            ['1', '2', 'class_B'],
            ['2', '3', 'class_C']
        ]

        self.assertEqual(expected, list(ArffReader(skip_encoding=True).filter(lines)))

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
            [2,3,None]
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
            "@attribute c {class_B, class_C, class_D}",
            "@data",
            "{0 2,1 3}",
            "{0 1,1 1,2 class_B}",
            "{1 1}",
            "{0 1,2 class_D}",
        ]

        expected = [
            {0:2, 1:3, 2:(1,0,0,0)},
            {0:1, 1:1, 2:(0,1,0,0)},
            {1:1, 2:(1,0,0,0)},
            {0:1,2:(0,0,0,1)}
        ]
        
        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_sparse_categorical_0_value(self):
        
        #this is a bug in ARFF, it is not uncommon for the first class value in an ARFF class list
        #to be dropped from the actual data because it is encoded as 0. Therefore our ARFF reader
        #automatically adds a 0 value to all categorical one-hot encoders to protect against this.
        #Below is what a dataset with this bug would look like, there is no class_B, instead all
        #class_B's are encoded as 0.
        lines = [
            "@relation news20",
            "@attribute a numeric",
            "@attribute b numeric",
            "@attribute c {class_B, class_C, class_D}",
            "@data",
            "{0 2,1 3}",
            "{0 1,1 1,2 class_C}",
            "{1 1}",
            "{0 1,2 class_D}",
        ]

        expected = [
            {0:2, 1:3, 2:(1,0,0,0)},
            {0:1, 1:1, 2:(0,0,1,0)},
            {1:1, 2:(1,0,0,0)},
            {0:1,2:(0,0,0,1)}
        ]
        
        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_sparse_with_empty_lines(self):
        lines = [
            "@relation news20",
            "@attribute a numeric",
            "@attribute b numeric",
            "@attribute c {class_B, class_C, class_D}",
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
            {0:2, 1:3, 2:(1,0,0,0)},
            {0:1, 1:1, 2:(0,1,0,0)},
            {1:1, 2:(1,0,0,0)},
            {0:1,2:(0,0,0,1)}
        ]
        
        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_sparse_with_spaces_after_comma(self):
        lines = [
            "@relation news20",
            "@attribute a numeric",
            "@attribute b numeric",
            "@attribute c {class_B, class_C, class_D}",
            "@data",
            "{0 2, 1 3}",
            "{0 1, 1 1,2 class_B}",
            "{1 1}",
            "{0 1, 2 class_D}",
        ]

        expected = [
            {0:2, 1:3, 2:(1,0,0,0)},
            {0:1, 1:1, 2:(0,1,0,0)},
            {1:1, 2:(1,0,0,0)},
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
            list(list(ArffReader().filter(lines))[0])

        self.assertIn("We were unable to find 'class_A' in ['class_B', 'class_C', 'class_D']", str(e.exception))

    def test_spaces_in_attribute_name(self):
        lines = [
            "@relation news20",
            "@attribute 'a a' string",
            "@attribute 'b b' string",
            "@attribute 'c c' {class_B, class_C, class_D}",
            "@data",
            "1,2,class_B",
        ]

        expected = [
            ['1','2',(1,0,0)],
        ]
        
        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_escaped_quote_in_attribute_name(self):
        lines = [
            "@relation news20",
            "@attribute 'a\\'a' numeric",
            "@attribute 'b b' string",
            "@attribute 'c c' {class_B, class_C, class_D}",
            "@data",
            "1,2,class_B",
        ]

        expected = [
            [1,'2',(1,0,0)],
        ]
        
        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_capitalized_attribute(self):
        lines = [
            "@relation news20",
            "@ATTRIBUTE 'a\\'a' numeric",
            "@attribute 'b b' string",
            "@attribute 'c c' {class_B, class_C, class_D}",
            "@data",
            "1,2,class_B",
        ]

        expected = [
            [1,'2',(1,0,0)],
        ]

        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_bad_tipe_raises_exception(self):
        lines = [
            "@relation news20",
            "@ATTRIBUTE 'a\\'a' numeric",
            "@attribute 'b b' abcd",
            "@attribute 'c c' {class_B, class_C, class_D}",
            "@data",
            "1,2,class_B",
        ]
        
        with self.assertRaises(CobaException) as ex:
            list(ArffReader().filter(lines))

        self.assertEqual('An unrecognized type was found in the arff attributes: abcd.', str(ex.exception))

    def test_all_good_tipes_do_not_raise_exception(self):
        lines = [
            "@relation news20",
            "@ATTRIBUTE a numeric",
            "@ATTRIBUTE b integer",
            "@ATTRIBUTE c real",
            "@attribute d    date",
            "@attribute e   {class_B, class_C, class_D}",
            "@attribute f relational",
            "@data",
        ]
        
        list(ArffReader().filter(lines))

    def test_str_as_cat(self):
        lines = [
            "@relation news20",
            "@attribute A numeric",
            "@attribute C {0, class_B, class_C, class_D}",
            "@data",
            "1,class_B",
            "2,0",
        ]

        expected = [
            [1,"class_B"],
            [2,"0"]
        ]

        self.assertEqual(expected, list(ArffReader(cat_as_str=True).filter(lines)))

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
