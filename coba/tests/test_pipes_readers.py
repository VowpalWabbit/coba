import unittest

from coba.exceptions import CobaException

from coba.pipes import LibsvmReader, ArffReader, CsvReader, ManikReader
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

    def test_sans_data(self):
        lines = [
            "@relation test",
            "@attribute a numeric",
            "@attribute B numeric",
            "@attribute c {0, class_B, class_C, class_D}",
            "@data",
        ]

        expected = []

        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_dense_with_spaces_after_commas(self):
        lines = [
            "@relation test",
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

    def test_dense_with_spaces_after_tabs(self):
        lines = [
            "@relation test",
            "@attribute a numeric",
            "@attribute B numeric",
            "@attribute c {class_B, class_C, class_D}",
            "@data",
            "1\t  2\t  class_B",
            "2\t  3\t  class_C",
        ]

        expected = [
            [1,2,(1,0,0)],
            [2,3,(0,1,0)]
        ]

        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_dense_with_spaces_in_quotes(self):
        lines = [
            "@relation test",
            "@attribute a numeric",
            "@attribute B numeric",
            "@attribute c {' B', ' C', ' D'}",
            "@data",
            "1, 2, ' B'",
            "2, 3, ' C'",
        ]

        expected = [
            [1,2,(1,0,0)],
            [2,3,(0,1,0)]
        ]

        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_dense_duplicate_headers(self):
        lines = [
            "@relation test",
            "@attribute a numeric",
            "@attribute a numeric",
            "@attribute c {0, class_B, class_C, class_D}",
            "@data",
            "1,2,class_B",
            "2,3,0",
        ]

        expected = [
            [1,2,(0,1,0,0)],
            [2,3,(1,0,0,0)]
        ]

        with self.assertRaises(CobaException) as e:
            list(ArffReader().filter(lines))
        
        self.assertEqual("Two columns in the ARFF file had identical header values.", str(e.exception))

    def test_dense_with_missing_value(self):
        lines = [
            "@relation test",
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

        self.assertEqual(expected, list(ArffReader(missing_value=None).filter(lines)))

    def test_dense_with_empty_lines(self):
        lines = [
            "@relation test",
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
            "@relation test",
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
            "@relation test",
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
            "@relation test",
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
            {'a':2, 'b':3, 'c':(1,0,0,0)},
            {'a':1, 'b':1, 'c':(0,1,0,0)},
            {       'b':1, 'c':(1,0,0,0)},
            {       'a':1, 'c':(0,0,0,1)}
        ]

        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_sparse_full_sparse(self):
        lines = [
            "@relation test",
            "@attribute a numeric",
            "@attribute b numeric",
            "@attribute c {class_B, class_C, class_D}",
            "@data",
            "{0 2,1 3}",
            "{ }",
            "{}",
        ]

        expected = [
            {'a':2, 'b':3, 'c':(1,0,0,0)},
            {'c':(1,0,0,0)},
            {'c':(1,0,0,0)}
        ]

        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_sparse_categorical_0_value(self):

        #this is a bug in ARFF, it is not uncommon for the first class value in an ARFF class list
        #to be dropped from the actual data because it is encoded as 0. Therefore our ARFF reader
        #automatically adds a 0 value to all categorical one-hot encoders to protect against this.
        #Below is what a dataset with this bug would look like, there is no class_B, instead all
        #class_B's are encoded as 0.
        lines = [
            "@relation test",
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
            {'a':2, 'b':3, 'c':(1,0,0,0)},
            {'a':1, 'b':1, 'c':(0,0,1,0)},
            {'b':1,        'c':(1,0,0,0)},
            {'a':1,        'c':(0,0,0,1)}
        ]

        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_sparse_with_empty_lines(self):
        lines = [
            "@relation test",
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
            {'a':2, 'b':3, 'c':(1,0,0,0)},
            {'a':1, 'b':1, 'c':(0,1,0,0)},
            {'b':1,        'c':(1,0,0,0)},
            {'a':1,        'c':(0,0,0,1)}
        ]

        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_sparse_with_spaces_after_comma(self):
        lines = [
            "@relation test",
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
            {'a':2, 'b':3, 'c':(1,0,0,0)},
            {'a':1, 'b':1, 'c':(0,1,0,0)},
            {     'b':1, 'c':(1,0,0,0)},
            {'a':1,      'c':(0,0,0,1)}
        ]

        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_leading_and_trailing_comments(self):
        lines = [
            "%",
            "%",
            "@relation test",
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

    def test_bad_class_labels_throws_exception(self):
        lines = [
            "@relation test",
            "@attribute 'a' string",
            "@attribute 'b' string",
            "@attribute 'c' {class_B, class_C, class_D}",
            "@data",
            "1,2,class_A",
        ]

        with self.assertRaises(CobaException) as e:
            list(list(ArffReader().filter(lines))[0])

        self.assertIn("We were unable to find one of the categorical values in the arff data.", str(e.exception))

    def test_percent_in_attribute_name(self):
        lines = [
            "@relation test",
            "@attribute 'a%3' numeric",
            "@data",
            "1",
        ]

        expected = [ [1] ]

        items = list(ArffReader().filter(lines))
        self.assertEqual(expected, items)
        self.assertEqual(1, items[0]["a%3"])

    def test_spaces_in_attribute_name(self):
        lines = [
            "@relation test",
            "@attribute 'a a' numeric",
            "@data",
            "1",
        ]

        expected = [ [1] ]

        items = list(ArffReader().filter(lines))
        self.assertEqual(expected, items)
        self.assertEqual(1, items[0]["a a"])

    def test_escaped_single_quote_in_attribute_name(self):
        lines = [
            "@relation test",
            "@attribute 'a\\'a' numeric",
            "@data",
            "1",
        ]

        expected = [ [1] ]
        items = list(ArffReader().filter(lines))

        self.assertEqual(expected, items)
        self.assertEqual(1, items[0]["a'a"])

    def test_escaped_double_quote_in_attribute_name(self):
        lines = [
            "@relation test",
            '@attribute "a\\"a" numeric',
            "@data",
            "1",
        ]

        expected = [ [1] ]
        items = list(ArffReader().filter(lines))

        self.assertEqual(expected, items)
        self.assertEqual(1, items[0]['a"a'])

    def test_tab_delimieted_attributes(self):
        lines = [
            "@relation test",
            "@attribute a\tstring",
            '@attribute b\tstring',
            "@attribute c\t{class_B, class_C, class_D}",
            "@data",
            "1,2,class_B",
        ]

        expected = [
            ['1','2',(1,0,0)],
        ]

        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_capitalized_attribute_tag(self):
        lines = [
            "@relation test",
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

    def test_capitalized_attribute_type(self):
        lines = [
            "@relation test",
            "@attribute a REAL",
            "@attribute b STRING",
            "@data",
            "1,a",
        ]

        expected = [
            [1,'a'],
        ]

        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_bad_attribute_type_raises_exception(self):
        lines = [
            "@relation test",
            "@attribute a abcd",
            "@data",
            "1",
        ]

        with self.assertRaises(CobaException) as ex:
            list(ArffReader().filter(lines))

        self.assertEqual('An unrecognized encoding was found in the arff attributes: abcd.', str(ex.exception))

    def test_good_attribute_types_do_not_raise_exception(self):
        lines = [
            "@relation test",
            "@ATTRIBUTE a numeric",
            "@ATTRIBUTE b integer",
            "@ATTRIBUTE c real",
            "@attribute d date",
            "@attribute e {class_B, class_C, class_D}",
            "@attribute f relational",
            "@data",
        ]

        list(ArffReader().filter(lines))

    def test_str_as_cat(self):

        lines = [
            "@relation test",
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
            "@relation test",
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
            "@relation test",
            "@attribute A numeric",
            "@attribute C {0, class_B, class_C, class_D}",
            "@data",
            "1",
        ]

        with self.assertRaises(CobaException) as e:
            list(ArffReader().filter(lines))

        self.assertEqual(str(e.exception), "We were unable to parse line 0 in a way that matched the expected attributes.")

    def test_too_min_sparse_element(self):
        lines = [
            "@relation test",
            "@attribute A numeric",
            "@attribute C {0, class_B, class_C, class_D}",
            "@data",
            "{-1 2,0 2,1 3}",
        ]

        with self.assertRaises(CobaException) as e:
            list(ArffReader().filter(lines))

        self.assertEqual(str(e.exception), "We were unable to parse line 0 in a way that matched the expected attributes.")

    def test_too_max_sparse_element(self):
        lines = [
            "@relation test",
            "@attribute A numeric",
            "@attribute C {0, class_B, class_C, class_D}",
            "@data",
            "{0 2,1 3,2 4}",
        ]

        with self.assertRaises(CobaException) as e:
            list(ArffReader().filter(lines))

        self.assertEqual(str(e.exception), "We were unable to parse line 0 in a way that matched the expected attributes.")

    def test_escaped_quotes_in_categorical_values(self):
        lines = [
            "@relation test",
            "@attribute A numeric",
            "@attribute B {'\\'classA\\'', '\\'classB\\''}",
            "@data",
            "1, '\\'classB\\''",
        ]

        expected = [
            [1, "'classB'"]
        ]

        items = list(ArffReader(cat_as_str=True).filter(lines))

        self.assertEqual(expected, items)
        self.assertEqual(1         , items[0]['A'])
        self.assertEqual("'classB'", items[0]['B'])

    def test_quotes_from_hell_dense_cat_as_str_true_good_categories(self):
        lines = [
            "@relation test",
            "@attribute 'A  a' numeric",
            "@attribute '\"' {0, \"class'B\", '\"class_C\"', 'class\",D'}",
            "@attribute '\'' {0, \"class'B\", '\"class_C\"', 'class\",D'}",
            "@attribute ',' {0, \"class'B\", '\"class_C\"', 'class\",D'}",
            "@data",
            "1,    'class\\'B', '\"class_C\"', \"class\\\",D\"",
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

    def test_quotes_from_hell_dense_cat_as_str_true_bad_categories(self):
        lines = [
            "@relation test",
            "@attribute 'A  a' numeric",
            "@attribute '\"' {0, \"class'B\", '\"class_C\"', 'class\",D'}",
            "@attribute '\'' {0, \"class'B\", '\"class_C\"', 'class\",D'}",
            "@attribute ',' {0, \"class'B\", '\"class_C\"', 'class\",D'}",
            "@data",
            "1,    'class\\'B', '\"class_C\"', 'class\",G'",
        ]

        with self.assertRaises(CobaException):
            [ list(l) for l in ArffReader(cat_as_str=True).filter(lines) ]

    def test_quotes_from_hell_dense_cat_as_str_false(self):
        lines = [
            "@relation test",
            "@attribute 'A  a' numeric",
            "@attribute '\"' {0, \"class'B\", '\"class_C\"', 'class\",D', 'class\\',E', 'class\\'   ,F'}",
            "@attribute '\'' {0, \"class'B\", '\"class_C\"', 'class\",D', 'class\\',E', 'class\\'   ,F'}",
            "@attribute ','  {0, \"class'B\", '\"class_C\"', 'class\",D', 'class\\',E', 'class\\'   ,F'}",
            "@data",
            "1,    'class\\'B', '\"class_C\"', 'class\",D'",
        ]

        expected = [
            [1, (0,1,0,0,0,0),(0,0,1,0,0,0),(0,0,0,1,0,0)]
        ]

        items = list(ArffReader(cat_as_str=False).filter(lines))

        self.assertEqual(expected, items)
        self.assertEqual(1, items[0]['A  a'])
        self.assertEqual((0,1,0,0,0,0), items[0]['"'])
        self.assertEqual((0,0,1,0,0,0), items[0]["'"])
        self.assertEqual((0,0,0,1,0,0), items[0][","])

    def test_quotes_with_csv(self):
        lines = [
            "@relation test",
            "@attribute 'value' numeric",
            "@attribute 'class' {'0','1'}",
            "@data",
            "1,'0'",
        ]

        expected = [
            [1, (1,0)]
        ]

        items = list(ArffReader(cat_as_str=False).filter(lines))

        self.assertEqual(expected, items)
        self.assertEqual(1    , items[0]['value'])
        self.assertEqual((1,0), items[0]['class'])

class LibsvmReader_Tests(unittest.TestCase):
    def test_sparse(self):
        lines = [
            "0 1:2 2:3",
            "1 1:1 2:1",
            "2 2:1",
            "1 1:1",
        ]

        expected = [
            ({1:2, 2:3} ,['0']),
            ({1:1, 2:1}, ['1']),
            ({     2:1}, ['2']),
            ({1:1     }, ['1'])
        ]

        self.assertEqual(expected, list(LibsvmReader().filter(lines)))

    def test_no_label_lines(self):
        lines = [
            "0 1:2 2:3",
            "1:1 2:1",
            "2 2:1",
            "1:1",
        ]

        expected = [
            ({1:2, 2:3} ,['0']),
            ({     2:1}, ['2']),
        ]

        self.assertEqual(expected, list(LibsvmReader().filter(lines)))


    def test_trailing_whitespace(self):
        lines = [
            "0 0:2 2:3",
            "1 0:1 2:1   ",
            "2 2:1",
            "1 0:1",
        ]

        expected = [
            ({0:2, 2:3}, ['0']),
            ({0:1, 2:1}, ['1']),
            ({     2:1}, ['2']),
            ({0:1     }, ['1'])
        ]

        self.assertEqual(expected, list(LibsvmReader().filter(lines)))

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
            ({1:2, 2:3} ,['0']),
            ({1:1, 2:1}, ['1']),
            ({     2:1}, ['2']),
            ({1:1     }, ['1'])
        ]

        self.assertEqual(expected, list(ManikReader().filter(lines)))

    def test_no_label_lines(self):
        lines = [
            "abcde",
            "0 1:2 2:3",
            "1:1 2:1",
            "2 2:1",
            "1:1",
        ]

        expected = [
            ({1:2, 2:3} ,['0']),
            ({     2:1}, ['2']),
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
            ({1:2, 2:3}, ['0']),
            ({1:1, 2:1}, ['1']),
            ({     2:1}, ['2']),
            ({1:1     }, ['1'])
        ]

        self.assertEqual(expected, list(ManikReader().filter(lines)))

if __name__ == '__main__':
    unittest.main()
