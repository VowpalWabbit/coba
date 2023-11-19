import unittest

from coba.exceptions import CobaException

from coba.pipes import LibsvmReader, ArffReader, CsvReader, ManikReader
from coba.pipes.readers import ArffAttrReader, ArffDataReader, ArffLineReader
from coba.context import NullLogger, CobaContext
from coba.primitives import Categorical

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
            "@attribute c {0, B, C, D}",
            "@data",
        ]

        expected = []

        self.assertEqual(expected, list(ArffReader().filter(lines)))

    def test_capital_data(self):
        lines = [
            "@relation test",
            "@attribute A numeric",
            "@attribute B numeric",
            "@attribute C {0, B, C, D}",
            "@attribute D numeric",
            "@DATA",
            "1,2,B,?",
            "2,3,0,?",
        ]

        expected = [
            [1,2,Categorical("B", ["0","B","C","D"]),None],
            [2,3,Categorical("0", ["0","B","C","D"]),None]
        ]

        rows = list(ArffReader().filter(lines))
        self.assertEqual(rows,expected)
        self.assertEqual(rows[0]["D"],None)

    def test_dense_data(self):
        lines = [
            "@relation test",
            "@attribute A numeric",
            "@attribute B numeric",
            "@attribute C {0, B, C, D}",
            "@attribute D numeric",
            "@data",
            "1,2,B,?",
            "2,3,0,?",
        ]

        expected = [
            [1,2,Categorical("B", ["0","B","C","D"]),None],
            [2,3,Categorical("0", ["0","B","C","D"]),None]
        ]

        rows = list(ArffReader().filter(lines))
        self.assertEqual(rows,expected)
        self.assertEqual(rows[0]["D"],None)

    def test_sparse_data(self):
        lines = [
            "@relation test",
            "@attribute A numeric",
            "@attribute B numeric",
            "@attribute C {B, C, D}",
            "@attribute D numeric",
            "@data",
            "{0 1,1 2,2 B,3 ?}",
            "{ }",
        ]

        expected = [
            {"A":1,"B":2,"C":Categorical("B", ["0","B","C","D"]), "D":None},
            {            "C":Categorical("0", ["0","B","C","D"])          }
        ]

        rows = list(ArffReader().filter(lines))

        self.assertEqual(rows,expected)
        self.assertEqual(rows[0]["D"],None)

    def test_empty_lines(self):
        lines = [
            "@relation test",
            "",
            "@attribute A numeric",
            "",
            "@attribute B numeric",
            "@attribute C {0, B, C, D}",
            "@data",
            "",
            "",
            "1,2,B",
            "2,3,0",
            ""
        ]

        expected = [
            [1,2,Categorical("B", ["0","B","C","D"])],
            [2,3,Categorical("0", ["0","B","C","D"])]
        ]

        self.assertEqual(expected, list(map(list,ArffReader().filter(lines))))

    def test_comment_lines(self):
        lines = [
            "%",
            "@relation test",
            "@attribute a numeric",
            "%",
            "@attribute b numeric",
            "@attribute c {0, B, C, D}",
            "@data",
            "%",
            "1,2,B",
            "%",
            "2,3,0",
            "%"
        ]

        expected = [
            [1,2,Categorical("B", ["0","B","C","D"])],
            [2,3,Categorical("0", ["0","B","C","D"])]
        ]

        self.assertEqual(expected, list(map(list,ArffReader().filter(lines))))

    def test_dense_missing_data(self):
        lines = [
            "@relation test",
            "@attribute A numeric",
            "@attribute B numeric",
            "@attribute C {0, B, C, D}",
            "@data",
            "1,2,B",
            "2,3,0",
        ]

        expected = [
            [1,2,Categorical("B", ["0","B","C","D"])],
            [2,3,Categorical("0", ["0","B","C","D"])]
        ]

        self.assertEqual(expected, list(ArffReader().filter(lines)))

    #In practice many of these tests are now redundant thanks to splitting the
    #reader into the three smaller readers (AttrReader,DataReader,and LineReader).
    #I believe the remaining tests in this class could be removed without compromising safety.

    def test_dense_with_spaces_after_commas(self):
        lines = [
            "@relation test",
            "@attribute a numeric",
            "@attribute B numeric",
            "@attribute c {B, C, D}",
            "@data",
            "1,  2,  B",
            "2,  3,  C",
        ]

        expected = [
            [1,2,Categorical("B", ["B","C","D"])],
            [2,3,Categorical("C", ["B","C","D"])]
        ]

        self.assertEqual(expected, list(map(list,ArffReader().filter(lines))))

    def test_dense_with_spaces_after_tabs(self):
        lines = [
            "@relation test",
            "@attribute a numeric",
            "@attribute B numeric",
            "@attribute c {B, C, D}",
            "@data",
            "1\t  2\t  B",
            "2\t  3\t  C",
        ]

        expected = [
            [1,2,Categorical("B", ["B","C","D"])],
            [2,3,Categorical("C", ["B","C","D"])]
        ]

        self.assertEqual(expected, list(map(list,ArffReader().filter(lines))))

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
            [1,2,Categorical(" B", [" B"," C"," D"])],
            [2,3,Categorical(" C", [" B"," C"," D"])]
        ]

        self.assertEqual(expected, list(map(list,ArffReader().filter(lines))))

    def test_dense_duplicate_headers(self):
        lines = [
            "@relation test",
            "@attribute a numeric",
            "@attribute a numeric",
            "@attribute c {0, B, C, D}",
            "@data",
            "1,2,B",
            "2,3,0",
        ]

        with self.assertRaises(CobaException) as e:
            list(ArffReader().filter(lines))

        self.assertEqual("Two columns in the ARFF file had identical header values.", str(e.exception))

    def test_dense_with_missing_value1(self):
        lines = [
            "@relation test",
            "@attribute a numeric",
            "@attribute B numeric",
            "@attribute c {B, C, D}",
            "@data",
            "1,2,B",
            "4,3,?",
        ]

        expected = [
            [1,2,Categorical("B", ["B","C","D"])],
            [4,3,None]
        ]

        self.assertEqual(expected, list(map(list,ArffReader().filter(lines))))

    def test_dense_with_missing_value2(self):
        lines = [
            "@relation test",
            "@attribute a numeric",
            "@attribute B numeric",
            "@attribute c {B, C, D}",
            "@data",
            "1,2,B",
            "?,3,?",
        ]

        expected = [
            [1,2,Categorical("B", ["B","C","D"])],
            [None,3,None]
        ]

        self.assertEqual(expected, list(map(list,ArffReader().filter(lines))))

    def test_dense_with_strings(self):
        lines = [
            "@relation test",
            "@attribute a string",
            "@attribute b string",
            "@attribute c {0, B, C, D}",
            "@data",
            "1,2,B",
            "2,3,0"
        ]

        expected = [
            ['1','2',Categorical("B", ["0","B","C","D"])],
            ['2','3',Categorical("0", ["0","B","C","D"])]
        ]

        self.assertEqual(expected, list(map(list,ArffReader().filter(lines))))

    def test_dense_categorical_with_spaces(self):
        lines = [
            "@relation test",
            "@attribute a string",
            "@attribute b string",
            "@attribute c {0, B, C, D }",
            "@data",
            "1,2,B",
            "2,3,D"
        ]

        expected = [
            ['1','2',Categorical("B", ["0","B","C","D"])],
            ['2','3',Categorical("D", ["0","B","C","D"])]
        ]

        self.assertEqual(expected, list(map(list,ArffReader().filter(lines))))

    def test_sparse(self):
        lines = [
            "@relation test",
            "@attribute a numeric",
            "@attribute b numeric",
            "@attribute c {B, C, D}",
            "@data",
            "{0 2,1 3}",
            "{0 1,1 1,2 B}",
            "{1 1}",
            "{0 1,2 D}",
        ]

        expected = [
            {'a':2, 'b':3, 'c':Categorical("0", ["0","B","C","D"])},
            {'a':1, 'b':1, 'c':Categorical("B", ["0","B","C","D"])},
            {       'b':1, 'c':Categorical("0", ["0","B","C","D"])},
            {       'a':1, 'c':Categorical("D", ["0","B","C","D"])}
        ]

        self.assertEqual(expected, list(map(dict,ArffReader().filter(lines))))

    def test_sparse_full_sparse(self):
        lines = [
            "@relation test",
            "@attribute a numeric",
            "@attribute b numeric",
            "@attribute c {B, C, D}",
            "@data",
            "{0 2,1 3}",
            "{ }",
            "{}",
        ]

        expected = [
            {'a':2, 'b':3, 'c':Categorical("0", ["0","B","C","D"])},
            {              'c':Categorical("0", ["0","B","C","D"])},
            {              'c':Categorical("0", ["0","B","C","D"])}
        ]

        self.assertEqual(expected, list(map(dict,ArffReader().filter(lines))))

    def test_sparse_categorical_0_value(self):

        #this is a bug in ARFF, it is not uncommon for the first class value in an ARFF class list
        #to be dropped from the actual data because it is encoded as 0. Therefore our ARFF reader
        #automatically adds a 0 value to all categorical one-hot encoders to protect against this.
        #Below is what a dataset with this bug would look like, there is no B, instead all
        #B's are encoded as 0.
        lines = [
            "@relation test",
            "@attribute a numeric",
            "@attribute b numeric",
            "@attribute c {B, C, D}",
            "@data",
            "{0 2,1 3}",
            "{0 1,1 1,2 C}",
            "{1 1}",
            "{0 1,2 D}",
        ]

        expected = [
            {'a':2, 'b':3, 'c':Categorical("0", ["0","B","C","D"])},
            {'a':1, 'b':1, 'c':Categorical("C", ["0","B","C","D"])},
            {'b':1,        'c':Categorical("0", ["0","B","C","D"])},
            {'a':1,        'c':Categorical("D", ["0","B","C","D"])}
        ]

        self.assertEqual(expected, list(map(dict,ArffReader().filter(lines))))

    def test_sparse_with_empty_lines(self):
        lines = [
            "@relation test",
            "@attribute a numeric",
            "@attribute b numeric",
            "@attribute c {B, C, D}",
            "@data",
            "",
            "{0 2,1 3}",
            "{0 1,1 1,2 B}",
            "",
            "",
            "{1 1}",
            "{0 1,2 D}",
        ]

        expected = [
            {'a':2, 'b':3, 'c':Categorical("0", ["0","B","C","D"])},
            {'a':1, 'b':1, 'c':Categorical("B", ["0","B","C","D"])},
            {'b':1,        'c':Categorical("0", ["0","B","C","D"])},
            {'a':1,        'c':Categorical("D", ["0","B","C","D"])}
        ]

        self.assertEqual(expected, list(map(dict,ArffReader().filter(lines))))

    def test_sparse_with_spaces_after_comma(self):
        lines = [
            "@relation test",
            "@attribute a numeric",
            "@attribute b numeric",
            "@attribute c {B, C, D}",
            "@data",
            "{0 2, 1 3}",
            "{0 1, 1 1,2 B}",
            "{1 1}",
            "{0 1, 2 D}",
        ]

        expected = [
            {'a':2, 'b':3, 'c':Categorical("0", ["0","B","C","D"])},
            {'a':1, 'b':1, 'c':Categorical("B", ["0","B","C","D"])},
            {       'b':1, 'c':Categorical("0", ["0","B","C","D"])},
            {'a':1,        'c':Categorical("D", ["0","B","C","D"])}
        ]

        self.assertEqual(expected, list(map(dict,ArffReader().filter(lines))))

    def test_bad_labels_throws_exception(self):
        lines = [
            "@relation test",
            "@attribute 'a' string",
            "@attribute 'b' string",
            "@attribute 'c' {B, C, D}",
            "@data",
            "1,2,A",
        ]

        with self.assertRaises(CobaException) as e:
            list(list(ArffReader().filter(lines))[0])

        self.assertIn("We were unable to find 'A' in ['B', 'C', 'D'].", str(e.exception))

    def test_percent_in_attribute_name(self):
        lines = [
            "@relation test",
            "@attribute 'a%3' numeric",
            "@data",
            "1",
        ]

        expected = [ [1] ]

        items = list(ArffReader().filter(lines))
        self.assertEqual(expected, list(map(list,items)))
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
        self.assertEqual(expected, list(map(list,items)))
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

        self.assertEqual(expected, list(map(list,items)))
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

        self.assertEqual(expected, list(map(list,items)))
        self.assertEqual(1, items[0]['a"a'])

    def test_tab_delimieted_attributes(self):
        lines = [
            "@relation test",
            "@attribute a\tstring",
            '@attribute b\tstring',
            "@attribute c\t{B, C, D}",
            "@data",
            "1,2,B",
        ]

        expected = [
            ['1','2',Categorical("B", ["B","C","D"])],
        ]

        self.assertEqual(expected, list(map(list,ArffReader().filter(lines))))

    def test_capitalized_attribute_tag(self):
        lines = [
            "@relation test",
            "@ATTRIBUTE 'a\\'a' numeric",
            "@attribute 'b b' string",
            "@attribute 'c c' {B, C, D}",
            "@data",
            "1,2,B",
        ]

        expected = [
            [1,'2',Categorical("B", ["B","C","D"])],
        ]

        self.assertEqual(expected, list(map(list,ArffReader().filter(lines))))

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

        self.assertEqual(expected, list(map(list,ArffReader().filter(lines))))

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
            "@attribute e {B, C, D}",
            "@attribute f relational",
            "@data",
        ]

        list(ArffReader().filter(lines))

    def test_too_many_dense_elements(self):
        lines = [
            "@relation test",
            "@attribute A numeric",
            "@attribute C {0, B, C, D}",
            "@data",
            "1,0,B",
        ]

        with self.assertRaises(CobaException) as e:
            [ list(l) for l in ArffReader().filter(lines) ]

        self.assertEqual(str(e.exception), "We were unable to parse a line in a way that matched the expected attributes.")

    def test_too_few_dense_elements(self):
        lines = [
            "@relation test",
            "@attribute A numeric",
            "@attribute C {0, B, C, D}",
            "@data",
            "1",
        ]

        with self.assertRaises(CobaException) as e:
            [ list(l) for l in ArffReader().filter(lines) ]

        self.assertEqual(str(e.exception), "We were unable to parse a line in a way that matched the expected attributes.")

    def test_too_min_sparse_element(self):
        lines = [
            "@relation test",
            "@attribute A numeric",
            "@attribute C {0, B, C, D}",
            "@data",
            "{-1 2,0 2,1 3}",
        ]

        with self.assertRaises(CobaException) as e:
            [ dict(l) for l in ArffReader().filter(lines) ]

        self.assertEqual(str(e.exception), "We were unable to parse a line in a way that matched the expected attributes.")

    def test_too_max_sparse_element(self):
        lines = [
            "@relation test",
            "@attribute A numeric",
            "@attribute C {0, B, C, D}",
            "@data",
            "{0 2,1 3,2 4}",
        ]

        with self.assertRaises(CobaException) as e:
            [ dict(l) for l in ArffReader().filter(lines) ]

        self.assertEqual(str(e.exception), "We were unable to parse a line in a way that matched the expected attributes.")

    def test_escaped_quotes_in_categorical_values(self):
        lines = [
            "@relation test",
            "@attribute A numeric",
            "@attribute B {'\\'A\\'', '\\'B\\''}",
            "@data",
            "1, '\\'B\\''",
        ]

        expected = [
            [1, Categorical("'B'",["'A'","'B'"])]
        ]

        items = list(ArffReader().filter(lines))

        self.assertEqual(expected                        , items        )
        self.assertEqual(1                               , items[0]['A'])
        self.assertEqual(Categorical("'B'",["'A'","'B'"]), items[0]['B'])

    def test_quotes_from_hell_dense_good_categories1(self):
        lines = [
            "@relation test",
            "@attribute 'A  a' numeric",
            "@attribute '\"' {0, \"class'B\", '\"C\"', 'class\",D'}",
            "@attribute '\'' {0, \"class'B\", '\"C\"', 'class\",D'}",
            "@attribute ','  {0, \"class'B\", '\"C\"', 'class\",D'}",
            "@data",
            "1,    'class\\'B', '\"C\"', \"class\\\",D\"",
        ]

        cats = ['0',"class'B",'"C"','class",D']

        expected = [
            [1, Categorical(cats[1],cats), Categorical(cats[2],cats), Categorical(cats[3],cats)]
        ]

        items = list(ArffReader().filter(lines))

        self.assertEqual(expected, items)
        self.assertEqual(1                        , items[0]['A  a'])
        self.assertEqual(Categorical(cats[1],cats), items[0]['"'])
        self.assertEqual(Categorical(cats[2],cats), items[0]["'"])
        self.assertEqual(Categorical(cats[3],cats), items[0][","])

    def test_quotes_from_hell_dense_good_categories2(self):
        lines = [
            "@relation test",
            "@attribute 'A  a' numeric",
            "@attribute '\"' {0, \"class B\", 'C', 'class D'}",
            "@attribute '\'' {0, \"class B\", 'C', 'class D'}",
            "@attribute ','  {0, \"class B\", 'C', 'class D'}",
            "@data",
            "1,    'class B', 'C', 'class D'",
            '1,    "class B", "C", "class D"',
        ]

        cats = ['0', "class B",'C','class D']

        expected = [
            [1, Categorical(cats[1],cats), Categorical(cats[2],cats), Categorical(cats[3],cats)],
            [1, Categorical(cats[1],cats), Categorical(cats[2],cats), Categorical(cats[3],cats)]
        ]

        items = list(ArffReader().filter(lines))

        self.assertEqual(expected, items)
        self.assertEqual(1                        , items[0]['A  a'])
        self.assertEqual(Categorical(cats[1],cats), items[0]['"'])
        self.assertEqual(Categorical(cats[2],cats), items[0]["'"])
        self.assertEqual(Categorical(cats[3],cats), items[0][","])

    def test_quotes_from_hell_dense(self):
        lines = [
            "@relation test",
            "@attribute 'A  a' numeric",
            "@attribute '\"' {0, \"class'B\", '\"C\"', 'class\",D', 'class\\',E', 'class\\'   ,F'}",
            "@attribute '\'' {0, \"class'B\", '\"C\"', 'class\",D', 'class\\',E', 'class\\'   ,F'}",
            "@attribute ','  {0, \"class'B\", '\"C\"', 'class\",D', 'class\\',E', 'class\\'   ,F'}",
            "@data",
            "1,    'class\\'B', '\"C\"', 'class\",D'",
        ]

        cats = ['0',"class'B",'"C"','class",D',"class',E","class'   ,F"]

        expected = [
            [1, Categorical(cats[1],cats),Categorical(cats[2],cats),Categorical(cats[3],cats)]
        ]

        items = list(ArffReader().filter(lines))

        self.assertEqual(expected, items)
        self.assertEqual(1                        , items[0]['A  a'])
        self.assertEqual(Categorical(cats[1],cats), items[0]['"'])
        self.assertEqual(Categorical(cats[2],cats), items[0]["'"])
        self.assertEqual(Categorical(cats[3],cats), items[0][","])

    def test_quotes_with_csv(self):
        lines = [
            "@relation test",
            "@attribute 'value' numeric",
            "@attribute 'class' {'0','1'}",
            "@data",
            "1,'0'",
        ]

        expected = [
            [1, Categorical('0',['0','1'])]
        ]

        items = list(ArffReader().filter(lines))

        self.assertEqual(expected                  , items)
        self.assertEqual(1                         , items[0]['value'])
        self.assertEqual(Categorical('0',['0','1']), items[0]['class'])

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

class ArffAttrReader_Tests(unittest.TestCase):

    def test_numeric_attribute(self):
        items = list(ArffAttrReader(True).filter(["@attribute a numeric"]))

        self.assertEqual(len(items),1)

        self.assertEqual(items[0][0], 'a'   )
        self.assertEqual(items[0][1]("1"), 1)

    def test_integer_attribute(self):
        items = list(ArffAttrReader(True).filter(["@attribute a integer"]))

        self.assertEqual(len(items),1)

        self.assertEqual(items[0][0], 'a'   )
        self.assertEqual(items[0][1]("1"), 1)

    def test_real_attribute(self):
        items = list(ArffAttrReader(True).filter(["@attribute a real"]))

        self.assertEqual(len(items),1)

        self.assertEqual(items[0][0], 'a'   )
        self.assertEqual(items[0][1]("1"), 1)

    def test_string_attribute(self):
        items = list(ArffAttrReader(True).filter(["@attribute a string"]))

        self.assertEqual(len(items),1)

        self.assertEqual(items[0][0], 'a'   )
        self.assertEqual(items[0][1]('1'), '1')

    def test_date_attribute(self):
        items = list(ArffAttrReader(True).filter(["@attribute a date"]))

        self.assertEqual(len(items),1)

        self.assertEqual(items[0][0], 'a'   )
        self.assertEqual(items[0][1]('1'), '1')

    def test_relational_attribute(self):
        items = list(ArffAttrReader(True).filter(["@attribute a relational"]))

        self.assertEqual(len(items),1)

        self.assertEqual(items[0][0], 'a'      )
        self.assertEqual(items[0][1]('1'), '1' )

    def test_dense_categorical_attribute(self):
        items = list(ArffAttrReader(True).filter(["@attribute a {A, B}"]))

        self.assertEqual(len(items),1)

        self.assertEqual(items[0][0], 'a'      )
        self.assertEqual(items[0][1]('A'), Categorical("A",['A','B']))

        with self.assertRaises(CobaException) as e:
            items[0][1]('C')

        self.assertIn("We were unable to find 'C' in ['A', 'B'].", str(e.exception))

    def test_sparse_categorical_attribute(self):
        items = list(ArffAttrReader(False).filter(["@attribute a {A, B}"]))

        self.assertEqual(len(items),1)

        self.assertEqual(items[0][0], 'a'      )
        self.assertEqual(items[0][1]('A'), Categorical("A",['0','A','B']))

        with self.assertRaises(CobaException) as e:
            items[0][1]('C')

        self.assertIn("We were unable to find 'C' in ['0', 'A', 'B'].", str(e.exception))

    def test_capitalized_attributes(self):
        items = list(ArffAttrReader(True).filter([
            "@ATTRIBUTE a NUMERIC",
            "@ATTRIBUTE A STRING",
        ]))

        self.assertEqual(len(items),2)

        self.assertEqual(items[0][0],'a')
        self.assertEqual(items[0][1]("1"), 1)

        self.assertEqual(items[1][0],'A')
        self.assertEqual(items[1][1]("1"), '1')

    def test_spaced_attribute(self):
        items = list(ArffAttrReader(True).filter(["@attribute  a  {  A,  B }"]))

        self.assertEqual(len(items),1)

        self.assertEqual(items[0][0], 'a'      )
        self.assertEqual(items[0][1]('A'), Categorical("A",['A','B']))

    def test_tabbed_attribute(self):
        items = list(ArffAttrReader(True).filter(["@attribute\ta\t{A,\tB}"]))

        self.assertEqual(len(items),1)

        self.assertEqual(items[0][0], 'a'      )
        self.assertEqual(items[0][1]('A'), Categorical("A",['A','B']))

    def test_quoted_attribute_names(self):
        items = list(ArffAttrReader(True).filter([
            "@attribute 'A a' {A, B}",
            "@attribute '\"' {A, B}",
            "@attribute '\'' {A, B}",
            "@attribute ','  {A, B}",

        ]))

        self.assertEqual(len(items),4)

        self.assertEqual(items[0][0], 'A a'                          )
        self.assertEqual(items[0][1]('A'), Categorical("A",['A','B']))

        self.assertEqual(items[1][0], '"'                            )
        self.assertEqual(items[1][1]('A'), Categorical("A",['A','B']))

        self.assertEqual(items[2][0], "'"                            )
        self.assertEqual(items[2][1]('A'), Categorical("A",['A','B']))

        self.assertEqual(items[3][0], ","                            )
        self.assertEqual(items[3][1]('A'), Categorical("A",['A','B']))

    def test_percent_in_attribute_name(self):
        items = list(ArffAttrReader(True).filter(["@attribute 'a%3' {A,B}"]))

        self.assertEqual(len(items),1)

        self.assertEqual(items[0][0], 'a%3'      )
        self.assertEqual(items[0][1]('A'), Categorical("A",['A','B']))

    def test_escaped_single_quote_in_attribute_name(self):
        items = list(ArffAttrReader(True).filter(["@attribute 'a\\'a' {A,B}"]))

        self.assertEqual(len(items),1)

        self.assertEqual(items[0][0]     , "a'a"                     )
        self.assertEqual(items[0][1]('A'), Categorical("A",['A','B']))

    def test_escaped_double_quote_in_attribute_name(self):
        items = list(ArffAttrReader(True).filter(['@attribute "a\\"a" {A,B}']))

        self.assertEqual(len(items),1)

        self.assertEqual(items[0][0]     , 'a"a'                     )
        self.assertEqual(items[0][1]('A'), Categorical("A",['A','B']))

    def test_single_quoted_categories_with_spaces(self):
        items = list(ArffAttrReader(True).filter(["@attribute 'a' { ' A  ', 'B ' }"]))

        self.assertEqual(len(items),1)

        self.assertEqual(items[0][0]        , 'a'                              )
        self.assertEqual(items[0][1](' A  '), Categorical(" A  ",[' A  ','B ']))

    def test_single_quoted_categories_with_comma(self):
        items = list(ArffAttrReader(True).filter(["@attribute 'a' {'A,a','B'}"]))

        self.assertEqual(len(items),1)

        self.assertEqual(items[0][0]       , 'a'                           )
        self.assertEqual(items[0][1]('A,a'), Categorical("A,a",['A,a','B']))

    def test_single_quoted_categories_with_comma(self):
        items = list(ArffAttrReader(True).filter(["@attribute 'a' {'A\ta','B'}"]))

        self.assertEqual(len(items),1)

        self.assertEqual(items[0][0]       , 'a'                              )
        self.assertEqual(items[0][1]('A\ta'), Categorical("A\ta",['A\ta','B']))

    def test_single_quoted_category_with_double_quote(self):
        items = list(ArffAttrReader(True).filter(["@attribute 'a' {\"A's\",B}"]))

        self.assertEqual(len(items),1)

        self.assertEqual(items[0][0]       , 'a'                            )
        self.assertEqual(items[0][1]("A's"), Categorical("A's",['A\'s','B']))

    def test_single_quoted_category_with_escaped_single_quote(self):
        items = list(ArffAttrReader(True).filter(["@attribute 'a' {'A\\'s',B}"]))

        self.assertEqual(len(items),1)

        self.assertEqual(items[0][0]       , 'a'                           )
        self.assertEqual(items[0][1]("A's"), Categorical("A's",["A's",'B']))

    def test_double_quoted_categories_with_spaces(self):
        items = list(ArffAttrReader(True).filter(["@attribute a { \" A  \", \"B \" }"]))

        self.assertEqual(len(items),1)

        self.assertEqual(items[0][0]        , 'a'                              )
        self.assertEqual(items[0][1](' A  '), Categorical(" A  ",[' A  ','B ']))

    def test_double_quoted_categories_with_comma(self):
        items = list(ArffAttrReader(True).filter(['@attribute a {"A,a", B}']))

        self.assertEqual(len(items),1)

        self.assertEqual(items[0][0]       , 'a'                           )
        self.assertEqual(items[0][1]('A,a'), Categorical("A,a",['A,a','B']))

    def test_double_quoted_category_with_single_quote(self):
        items = list(ArffAttrReader(True).filter(["@attribute a {\"A's\",B}"]))

        self.assertEqual(len(items),1)

        self.assertEqual(items[0][0]       , 'a'                            )
        self.assertEqual(items[0][1]("A's"), Categorical("A's",['A\'s','B']))

    def test_double_quoted_category_with_escaped_double_quote(self):
        items = list(ArffAttrReader(True).filter(["@attribute 'a' {\"A\\\"s\",'B'}"]))

        self.assertEqual(len(items),1)

        self.assertEqual(items[0][0]       , 'a'                           )
        self.assertEqual(items[0][1]('A"s'), Categorical('A"s',['A"s','B']))

    def test_single_and_double_quoted_category_with_escaped_quotes(self):
        items = list(ArffAttrReader(True).filter(["@attribute 'a' {\"A\\\"s\",'B\\'s'}"]))

        self.assertEqual(len(items),1)

        self.assertEqual(items[0][0]       , 'a'                             )
        self.assertEqual(items[0][1]('A"s'), Categorical('A"s',['A"s',"B's"]))

    def test_dense_duplicate_headers(self):
        with self.assertRaises(CobaException) as e:
            list(ArffAttrReader(True).filter([
                "@attribute 'a' string",
                "@attribute 'a' string",
            ]))

        self.assertEqual("Two columns in the ARFF file had identical header values.", str(e.exception))

    def test_bad_attribute_type_raises_exception(self):
        with self.assertRaises(CobaException) as ex:
            list(ArffAttrReader(True).filter(["@attribute 'a' abcd"]))

        self.assertEqual('An unrecognized encoding was found in the arff attributes: abcd.', str(ex.exception))

    def test_good_attribute_types_do_not_raise_exception(self):
        list(ArffReader().filter([
            "@attribute a numeric",
            "@attribute b integer",
            "@attribute c real",
            "@attribute d date",
            "@attribute e string",
            "@attribute f {B, C, D}",
            "@attribute g relational",
        ]))

class ArffDataReader_Tests(unittest.TestCase):
    def test_one_dense_line_no_missing_no_comments(self):
        out = list(ArffDataReader(True).filter(['1,2,3']))
        self.assertEqual(out, [('1,2,3',False)])

    def test_one_dense_line_missing_tight_start(self):
        out = list(ArffDataReader(True).filter(['?,2,3']))
        self.assertEqual(out, [('?,2,3',True)])

    def test_one_dense_line_missing_tight_middle(self):
        out = list(ArffDataReader(True).filter(['1,?,3']))
        self.assertEqual(out, [('1,?,3',True)])

    def test_one_dense_line_missing_tight_end(self):
        out = list(ArffDataReader(True).filter(['1,2,?']))
        self.assertEqual(out, [('1,2,?',True)])

    def test_one_dense_line_missing_middle_loose(self):
        out = list(ArffDataReader(True).filter(['1,  ?  ,3']))
        self.assertEqual(out, [('1,  ?  ,3',True)])

    def test_two_dense_line_no_missing_no_comments(self):
        out = list(ArffDataReader(True).filter(['1','2']))
        self.assertEqual(out, [('1',False),('2',False)])

    def test_two_dense_line_no_missing_yes_comments(self):
        out = list(ArffDataReader(True).filter(['1','%','2']))
        self.assertEqual(out, [('1',False),('2',False)])

    def test_one_sparse_line_no_missing_no_comments(self):
        out = list(ArffDataReader(True).filter(['{1 2,3 4}']))
        self.assertEqual(out, [('{1 2,3 4}',False)])

    def test_one_sparse_line_missing_tight_start(self):
        out = list(ArffDataReader(False).filter(['{1 ?}']))
        self.assertEqual(out, [('{1 ?}',True)])

    def test_one_sparse_line_missing_tight_middle(self):
        out = list(ArffDataReader(False).filter(['{1 ?,3 4}']))
        self.assertEqual(out, [('{1 ?,3 4}',True)])

    def test_one_sparse_line_missing_tight_end(self):
        out = list(ArffDataReader(False).filter(['{1 2,3 ?}']))
        self.assertEqual(out, [('{1 2,3 ?}',True)])

    def test_two_sparse_line_no_missing_no_comment(self):
        out = list(ArffDataReader(False).filter(['{1 2}','{3 4}']))
        self.assertEqual(out, [('{1 2}',False),('{3 4}', False)])

    def test_two_sparse_line_no_missing_yes_comments(self):
        out = list(ArffDataReader(False).filter(['{1 2}','%','{3 4}']))
        self.assertEqual(out, [('{1 2}',False),('{3 4}', False)])

class ArffLineReader_Tests(unittest.TestCase):
    def test_dense_1_comma_no_quotes_tight(self):
        self.assertEqual(ArffLineReader(True,1).filter('1'), ['1'])

    def test_dense_2_comma_no_quotes_tight(self):
        self.assertEqual(ArffLineReader(True,2).filter('1,2'), ['1','2'])

    def test_dense_2_tab_no_quotes_tight(self):
        self.assertEqual(ArffLineReader(True,2).filter('1\t2'), ['1','2'])

    def test_dense_2_comma_no_quotes_initialspaces(self):
        self.assertEqual(ArffLineReader(True,2).filter('1, 2'), ['1','2'])

    def test_dense_2_tab_no_quotes_initialspaces(self):
        self.assertEqual(ArffLineReader(True,2).filter('1\t 2'), ['1','2'])

    @unittest.skip("Known shortcoming of csv.reader ")
    def test_dense_2_comma_no_quotes_precedingspaces(self):
        self.assertEqual(ArffLineReader(True,2).filter('1 ,2'), ['1','2'])

    def test_dense_2_comma_double_quotes_with_space(self):
        self.assertEqual(ArffLineReader(True,2).filter('"1  a", " 2 b "'), ['1  a',' 2 b '])

    def test_dense_2_comma_double_quotes_with_comma(self):
        self.assertEqual(ArffLineReader(True,2).filter('"1,a","2,b"'), ['1,a','2,b'])

    def test_dense_2_comma_double_quotes_with_single_quote(self):
        self.assertEqual(ArffLineReader(True,2).filter('"1\'a",2'), ['1\'a','2'])

    def test_dense_2_comma_double_quotes_with_escaped_double_quotes(self):
        self.assertEqual(ArffLineReader(True,2).filter('"1\\"a",2'), ['1"a','2'])

    def test_dense_2_comma_double_quotes_initialspaces(self):
        self.assertEqual(ArffLineReader(True,2).filter('1,  "2\'a"'), ['1','2\'a'])

    def test_dense_2_comma_single_quotes_with_space(self):
        self.assertEqual(ArffLineReader(True,2).filter("'1  a', ' 2 b '"), ["1  a"," 2 b "])

    def test_dense_2_comma_single_quotes_with_comma(self):
        self.assertEqual(ArffLineReader(True,2).filter("'1,a','2,b'"), ["1,a","2,b"])

    def test_dense_2_comma_single_quotes_with_double_quote(self):
        self.assertEqual(ArffLineReader(True,2).filter("'1\"a',2"), ["1\"a","2"])

    def test_dense_2_comma_single_quotes_with_escaped_single_quotes(self):
        self.assertEqual(ArffLineReader(True,2).filter("'1\\'a',2"), ["1'a","2"])

    def test_dense_2_comma_single_quotes_initialspaces(self):
        self.assertEqual(ArffLineReader(True,2).filter("1,  '2\"a'"), ["1","2\"a"])

    def test_dense_2_comma_double_and_single_quotes_tight(self):
        self.assertEqual(ArffLineReader(True,2).filter("'1\"s',\"2's\""), ['1"s',"2's"])

    def test_dense_2_quote_from_hell(self):
        self.assertEqual(ArffLineReader(True,2).filter("'1\" , \\'s',\"2's\""), ['1" , \'s',"2's"])

    def test_sparse_2_no_quotes_tight(self):
        self.assertEqual(ArffLineReader(False,2).filter("{0 2,1 4}"),{0:'2',1:'4'})

    def test_sparse_2_no_quotes_initialspaces(self):
        self.assertEqual(ArffLineReader(False,2).filter("{0 2,  1 4}"),{0:'2',1:'4'})

    def test_sparse_2_no_quotes_empty(self):
        self.assertEqual(ArffLineReader(False,2).filter("{}"),{})

    def test_sparse_2_no_quotes_empty_spaces(self):
        self.assertEqual(ArffLineReader(False,2).filter("{    }"),{})

    def test_good_row_then_too_many_dense_elements(self):
        reader = ArffLineReader(True,2)
        with self.assertRaises(CobaException) as e:
            reader.filter('1,2')
            reader.filter('1,2,3')
        self.assertEqual(str(e.exception), "We were unable to parse a line in a way that matched the expected attributes.")

    def test_too_many_dense_elements(self):
        with self.assertRaises(CobaException) as e:
            ArffLineReader(True,2).filter('1,2,3')
        self.assertEqual(str(e.exception), "We were unable to parse a line in a way that matched the expected attributes.")

    def test_too_few_dense_elements(self):
        with self.assertRaises(CobaException) as e:
            ArffLineReader(True,2).filter('1')
        self.assertEqual(str(e.exception), "We were unable to parse a line in a way that matched the expected attributes.")

    def test_too_min_sparse_element(self):
        with self.assertRaises(CobaException) as e:
            ArffLineReader(False,2).filter("{-1 2,0 2,1 3}")
        self.assertEqual(str(e.exception), "We were unable to parse a line in a way that matched the expected attributes.")

    def test_too_max_sparse_element(self):
        with self.assertRaises(CobaException) as e:
            ArffLineReader(False,2).filter("{0 2,2 3}")
        self.assertEqual(str(e.exception), "We were unable to parse a line in a way that matched the expected attributes.")

    def test_no_quote_then_single_quote(self):
        reader = ArffLineReader(True,2)
        self.assertEqual(reader.filter('1,2'), ['1','2'])
        self.assertEqual(reader.filter("1,'b '"), ['1','b '])

    def test_no_quote_then_double_quote(self):
        reader = ArffLineReader(True,2)
        self.assertEqual(reader.filter('1,2'), ['1','2'])
        self.assertEqual(reader.filter('1,"b "'), ['1','b '])

    def test_single_quote_then_double_quote(self):
        reader = ArffLineReader(True,2)
        self.assertEqual(reader.filter("1,'b '"), ['1','b '])
        self.assertEqual(reader.filter('1,"c "'), ['1','c '])

    def test_double_quote_then_single_quote(self):
        reader = ArffLineReader(True,2)
        self.assertEqual(reader.filter('1,"b "'), ['1','b '])
        self.assertEqual(reader.filter("1,'c '"), ['1','c '])

if __name__ == '__main__':
    unittest.main()
