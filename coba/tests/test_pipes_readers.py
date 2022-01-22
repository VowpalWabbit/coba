import unittest
from itertools import count

from coba.encodings import NumericEncoder, StringEncoder
from coba.exceptions import CobaException

from coba.pipes import LibSvmReader, ArffReader, CsvReader, ManikReader
from coba.contexts import NullLogger, CobaContext

from coba.pipes.readers import LazyDualDense, LazyDualSparse

CobaContext.logger = NullLogger()

class LazyDualDense_Tests(unittest.TestCase):

    def test_no_headers_no_encoders(self):
        a = LazyDualDense([1,2])

        self.assertEqual([1,2], a)
        self.assertEqual(2, len(a))

        a[1] = 3
        self.assertEqual(3, a[1])

        del a[1]
        self.assertEqual(1, len(a))
        self.assertEqual([1], a)

    def test_headers_no_encoders(self):
        a = LazyDualDense([1,2,3],dict(zip(['a','b','c'],count())))

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
        a = LazyDualDense(['1','2'],dict(zip(['a','b','c'],count())), encoders=[float, str])

        self.assertEqual([1,'2'], a)
        self.assertEqual(2, len(a))

        self.assertEqual(1,a['a'])
        self.assertEqual('2',a['b'])

        a['b'] = 3
        self.assertEqual(3, a['b'])

        del a['a']
        self.assertEqual(1, len(a))
        self.assertEqual([3], a)
        self.assertEqual(3, a['b'])
        self.assertEqual(3, a[0])

    def test_insert_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            LazyDualDense([1,2]).insert(0,1)

    def test_str(self):
        self.assertEqual('[1, 2, 3]', str(LazyDualDense([1,2,3])))

    def test_repr(self):
        self.assertEqual('[1, 2, 3]', LazyDualDense([1,2,3]).__repr__())

class LazyDualSparse_Tests(unittest.TestCase):

    def test_no_headers_no_encoders(self):
        a = LazyDualSparse({'a':1,'b':2})

        self.assertEqual({'a':1,'b':2}, a)
        self.assertEqual(2, len(a))

        a['b'] = 3
        self.assertEqual(3, a['b'])

        del a['b']
        self.assertEqual(1, len(a))
        self.assertEqual({'a':1}, a)

    def test_headers_no_encoders(self):
        a = LazyDualSparse({'a':1,'b':2}, {'aa':'a', 'bb':'b'})

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
        a = LazyDualSparse({'a':'1','b':'2'}, {'aa':'a', 'bb':'b'}, {'a':float, 'b': str})

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

    def test_str(self):
        self.assertEqual("{'a': 2}", str(LazyDualSparse({'a':2})))

    def test_repr(self):
        self.assertEqual("{'a': 2}", LazyDualSparse({'a':2}).__repr__())

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
            {     1:1, 2:(1,0,0,0)},
            {0:1,      2:(0,0,0,1)}
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

        self.assertIn("We were unable to find one of the categorical values in the arff data.", str(e.exception))

    def test_spaces_in_attribute_name(self):
        lines = [
            "@relation news20",
            "@attribute 'a a' string",
            '@attribute "b b" string',
            "@attribute 'c c' {class_B, class_C, class_D}",
            "@data",
            "1,2,class_B",
        ]

        expected = [
            ['1','2',(1,0,0)],
        ]
        
        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_tab_delimieted_attributes(self):
        lines = [
            "@relation news20",
            "@attribute a	string",
            '@attribute b	string',
            "@attribute c	{class_B, class_C, class_D}",
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

        self.assertEqual('An unrecognized encoding was found in the arff attributes: abcd.', str(ex.exception))

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

    def test_too_many_dense_elements(self):
        lines = [
            "@relation news20",
            "@attribute A numeric",
            "@attribute C {0, class_B, class_C, class_D}",
            "@data",
            "1,0,class_B",
        ]

        with self.assertRaises(CobaException) as e:
            list(ArffReader().filter(lines))

        self.assertEqual(str(e.exception), "We were unable to parse line 0 in a way that matched the expected attributes.")

    def test_too_few_dense_elements(self):
        lines = [
            "@relation news20",
            "@attribute A numeric",
            "@attribute C {0, class_B, class_C, class_D}",
            "@data",
            "1",
        ]

        with self.assertRaises(CobaException) as e:
            list(ArffReader().filter(lines))

        self.assertEqual(str(e.exception), "We were unable to parse line 0 in a way that matched the expected attributes.")

    def test_min_unknown_sparse_elements(self):
        lines = [
            "@relation news20",
            "@attribute A numeric",
            "@attribute C {0, class_B, class_C, class_D}",
            "@data",
            "{-1 2,0 2,1 3}",
        ]

        with self.assertRaises(CobaException) as e:
            list(ArffReader().filter(lines))

        self.assertEqual(str(e.exception), "We were unable to parse line 0 in a way that matched the expected attributes.")

    def test_max_unknown_sparse_elements(self):
        lines = [
            "@relation news20",
            "@attribute A numeric",
            "@attribute C {0, class_B, class_C, class_D}",
            "@data",
            "{0 2,1 3,2 4}",
        ]

        with self.assertRaises(CobaException) as e:
            list(ArffReader().filter(lines))

        self.assertEqual(str(e.exception), "We were unable to parse line 0 in a way that matched the expected attributes.")

    def test_quotes_from_hell_dense(self):
        lines = [
            "@relation news20",
            "@attribute 'A  a' numeric",
            "@attribute '\"' {0, \"class'B\", '\"class_C\"', 'class\",D'}",
            "@attribute '\'' {0, \"class'B\", '\"class_C\"', 'class\",D'}",
            "@attribute ',' {0, \"class'B\", '\"class_C\"', 'class\",D'}",
            "@data",
            "1,    'class\'B', '\"class_C\"', 'class\",D'",
        ]

        expected = [
            [1, "class'B", '"class_C"', 'class",D']
        ]

        items = list(ArffReader(cat_as_str=True).filter(lines))

        self.assertEqual(expected, items)
        self.assertEqual(1, items[0]['A  a'])
        self.assertEqual("class'B", items[0]['"'])
        self.assertEqual('"class_C"', items[0]["'"])
        self.assertEqual('class",D', items[0][","])

    def test_no_lazy_encoding_no_header_indexes_dense(self):
        lines = [
            "@relation news20",
            "@attribute a numeric",
            "@attribute b numeric",
            "@attribute c {class_B, class_C, class_D}",
            "@data",
            "1,2,class_B",
            "2,3,class_C",
        ]

        expected = [
            [1,2,(1,0,0)],
            [2,3,(0,1,0)]
        ]

        actual = list(ArffReader(lazy_encoding=False,header_indexing=False).filter(lines))

        self.assertEqual(expected, actual)
        self.assertIsInstance(actual[0], list)
        self.assertIsInstance(actual[1], list)

    def test_no_lazy_encoding_no_header_indexes_sparse(self):
        lines = [
            "@relation news20",
            "@attribute a numeric",
            "@attribute b numeric",
            "@attribute c {class_B, class_C, class_D}",
            "@data",
            "{0 1,1 2,2 class_B}",
            "{0 2,1 3,2 class_C}",
        ]

        expected = [
            {0:1, 1:2, 2:(0,1,0,0)},
            {0:2, 1:3, 2:(0,0,1,0)}
        ]

        actual = list(ArffReader(lazy_encoding=False,header_indexing=False).filter(lines))

        self.assertEqual(expected, actual)
        self.assertIsInstance(actual[0], dict)
        self.assertIsInstance(actual[1], dict)

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
