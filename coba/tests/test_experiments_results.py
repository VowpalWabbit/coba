import unittest
import unittest.mock
import timeit
import importlib.util

from pathlib import Path
from coba.contexts.core import CobaContext
from coba.contexts.loggers import IndentLogger

from coba.exceptions import CobaException
from coba.pipes import DiskIO

from coba.experiments.results import Result, Table, InteractionsTable, TransactionIO, TransactionIO_V3, TransactionIO_V4
from coba.pipes.io import ListIO

class Table_Tests(unittest.TestCase):

    def test_table_name(self):
        self.assertEqual('abc',Table('abc',[],[]).name)

    def test_table_str(self):
        self.assertEqual("{'Table': 'abc', 'Columns': ['id', 'col'], 'Rows': 2}",str(Table('abc',['id'],[{'id':1,'col':2},{'id':2,'col':3}])))

    def test_ipython_display(self):
        with unittest.mock.patch("builtins.print") as mock:
            table = Table('abc',['id'],[{'id':1,'col':2},{'id':2,'col':3}])
            table._ipython_display_()
            mock.assert_called_once_with(str(table))

    def test_insert_item(self):
        table = Table("test", ['a'], [{'a':'a', 'b':'B'}, {'a':'A', 'b':'B'}])

        self.assertTrue('A' in table)
        self.assertTrue('a' in table)

        self.assertEqual(table['a'], {'a':'a', 'b':'B'})
        self.assertEqual(table['A'], {'a':'A', 'b':'B'})

        self.assertEqual(2, len(table))

        self.assertEqual([('A', 'B'),('a', 'B')], list(table.to_tuples()))

    def test_missing_columns(self):
        table = Table("test", ['a'], [{'a':'A', 'b':'B'}, {'a':'a', 'c':'C'}])

        self.assertTrue('A' in table)
        self.assertTrue('a' in table)

        self.assertEqual(table['A'], {'a':'A', 'b':'B'})
        self.assertEqual(table['a'], {'a':'a', 'c':'C'})

        self.assertEqual(2, len(table))

        expected = [('A', 'B', None), ('a', None, 'C')]
        actual   = table.to_tuples()

        self.assertEqual(expected, actual)

    def test_tuples_with_array_column(self):

        table = Table("test", ['a'], [dict(a='A', b='B', c=[1,2], d='d'), dict(a='B', e='E') ])

        expected_tuples = [ ('A', 'B', [1,2], 'd', None), ('B', None, None, None, 'E') ]
        actual_tuples = table.to_tuples()

        self.assertEqual(expected_tuples, actual_tuples)

    def test_tuples_with_dict_column(self):

        table = Table("test", ['a'], [dict(a='A',b='B',c={'z':5},d='d'), dict(a='B',e='E')])

        expected_tuples = [ ('A', 'B', {'z':5}, 'd', None), ('B', None, None, None, 'E') ]
        actual_tuples = table.to_tuples()

        self.assertEqual(expected_tuples, actual_tuples)

    def test_two_packed_items(self):
        table = Table("test", ['a'], [dict(a='A', c=1, _packed=dict(b=['B','b'],d=['D','d']))])

        self.assertTrue('A' in table)

        self.assertEqual(table['A'], {'a':'A', 'index':[1,2], 'b':['B','b'], 'c':1, 'd':['D','d']})

        self.assertEqual(2, len(table))

        self.assertEqual([('A', 1, 'B', 1, 'D'), ('A', 2, 'b', 1, 'd')], list(table.to_tuples()))

    def test_unequal_pack_exception(self):
        with self.assertRaises(Exception):
            table = Table("test", ['a'])
            table['A'] = dict(c=1,_packed=dict(b=['B','b'],d=['D','d','e']))

    def test_filter_kwarg_str(self):
        table = Table("test", ['a'], [{'a':'a', 'b':'b'}, {'a':'A', 'b':'B'}])

        filtered_table = table.filter(b="B")

        self.assertEqual(2, len(table))
        self.assertEqual([('A', 'B'),('a', 'b')], list(table.to_tuples()))

        self.assertEqual(1, len(filtered_table))
        self.assertEqual([('A', 'B')], list(filtered_table.to_tuples()))

    def test_filter_kwarg_int_1(self):
        table = Table("test", ['a'], [{'a':'1', 'b':'b'}, {'a':'12', 'b':'B'}])

        filtered_table = table.filter(a=1)

        self.assertEqual(2, len(table))
        self.assertEqual([('1', 'b'),('12', 'B')], list(table.to_tuples()))

        self.assertEqual(1, len(filtered_table))
        self.assertEqual([('1', 'b')], list(filtered_table.to_tuples()))
    
    def test_filter_kwarg_int_2(self):
        table = Table("test", ['a'], [{'a':1, 'b':'b'}, {'a':12, 'b':'B'}])

        filtered_table = table.filter(a=1)

        self.assertEqual(2, len(table))
        self.assertEqual([(1, 'b'),(12, 'B')], list(table.to_tuples()))

        self.assertEqual(1, len(filtered_table))
        self.assertEqual([(1, 'b')], list(filtered_table.to_tuples()))

    def test_filter_kwarg_pred(self):
        table = Table("test", ['a'], [{'a':'1', 'b':'b'}, {'a':'12', 'b':'B'}])

        filtered_table = table.filter(a= lambda a: a =='1')

        self.assertEqual(2, len(table))
        self.assertEqual([('1', 'b'),('12', 'B')], list(table.to_tuples()))

        self.assertEqual(1, len(filtered_table))
        self.assertEqual([('1', 'b')], list(filtered_table.to_tuples()))

    def test_filter_kwarg_multi(self):
        table = Table("test", ['a'], [
            {'a':'1', 'b':'b', 'c':'c'}, 
            {'a':'2', 'b':'b', 'c':'C'},
            {'a':'3', 'b':'B', 'c':'c'},
            {'a':'4', 'b':'B', 'c':'C'}
        ])

        filtered_table = table.filter(b="b", c="C")

        self.assertEqual(4, len(table))
        self.assertEqual(1, len(filtered_table))
        self.assertEqual([('2', 'b', "C")], list(filtered_table.to_tuples()))

    def test_filter_pred(self):
        table = Table("test", ['a'], [{'a':'a', 'b':'b'}, {'a':'A', 'b':'B'}])

        filtered_table = table.filter(lambda row: row["b"]=="B")

        self.assertEqual(2, len(table))
        self.assertEqual([('A', 'B'),('a', 'b')], list(table.to_tuples()))

        self.assertEqual(1, len(filtered_table))
        self.assertEqual([('A', 'B')], list(filtered_table.to_tuples()))

    def test_filter_sequence_1(self):
        table = Table("test", ['a'], [{'a':'a', 'b':'b'}, {'a':'A', 'b':'B'}, {'a':'1', 'b':'C'}])

        filtered_table = table.filter(a=['a','1'])

        self.assertEqual(3, len(table))
        self.assertCountEqual([('A', 'B'),('a', 'b'),('1','C')], list(table.to_tuples()))

        self.assertEqual(2, len(filtered_table))
        self.assertCountEqual([('a', 'b'),('1','C')], list(filtered_table.to_tuples()))

    def test_filter_sequence_2(self):
        table = Table("test", ['a'], [{'a':'1', 'b':'b'}, {'a':'2', 'b':'B'}, {'a':'3', 'b':'C'}])

        filtered_table = table.filter(a=[1,2])

        self.assertEqual(3, len(table))
        self.assertCountEqual([('1', 'b'),('2', 'B'),('3','C')], list(table.to_tuples()))

        self.assertEqual(2, len(filtered_table))
        self.assertCountEqual([('1', 'b'),('2','B')], list(filtered_table.to_tuples()))

    def test_filter_table_contains(self):
        table = Table("test", ['a'], [{'a':'a', 'b':'b'}, {'a':'A', 'b':'B'}])

        filtered_table = table.filter(lambda row: row["b"]=="B")

        self.assertEqual(2, len(table))
        self.assertEqual([('A', 'B'),('a', 'b')], list(table.to_tuples()))

        self.assertNotIn("a", filtered_table)

    def test_filter_missing_columns(self):
        table = Table("test", ['a'], [{'a':'a', 'b':'B'}, {'a':'A'}])

        filtered_table = table.filter(b="B")

        self.assertEqual(2, len(table))
        self.assertEqual([('A',None),('a', 'B')], list(table.to_tuples()))

        self.assertNotIn("A", filtered_table)
        self.assertIn("a", filtered_table)

    def test_filter_nan_value(self):
        table = Table("test", ['a'], [{'a':'a', 'b':'B'}, {'a':'A'}])

        filtered_table = table.filter(b=None)

        self.assertEqual(2, len(table))
        self.assertEqual([('A',None),('a', 'B')], list(table.to_tuples()))

        self.assertNotIn("a", filtered_table)
        self.assertIn("A", filtered_table)

@unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas is not installed so we must skip pandas tests")
class Table_Pandas_Tests(unittest.TestCase):

    def test_pandas(self):
        
        import pandas as pd #type: ignore
        import pandas.testing #type: ignore

        table = Table("test", ['a'], [dict(a='A',b='B',c=1,d='d'),dict(a='B',e='E')])

        expected_df = pd.DataFrame([
            dict(a='A',b='B',c=1,d='d'),
            dict(a='B',e='E')
        ])

        actual_df = table.to_pandas()

        pandas.testing.assert_frame_equal(expected_df,actual_df)

    def test_pandas_with_array_column(self):
        import pandas as pd   #type: ignore
        import pandas.testing #type: ignore

        table = Table("test", ['a'], [dict(a='A',b='B',c=[1,2],d='d'),dict(a='B',e='E')])

        expected_df = pd.DataFrame([
            dict(a='A',b='B',c=[1,2],d='d'),
            dict(a='B',e='E')
        ])

        actual_df = table.to_pandas()

        pandas.testing.assert_frame_equal(expected_df,actual_df)

    def test_pandas_with_packed_array_column(self):
        import pandas as pd   #type: ignore
        import pandas.testing #type: ignore

        table = Table("test", ['a'], [dict(a='A',b=1.,c=[1,2],d='d',_packed={'z':[[1,2],[3,4]]}),dict(a='B',b=2.,e='E')])

        expected_df = pd.DataFrame([
            dict(a='A',index=1,b=1.,c=[1,2],d='d',e=None,z=[1,2]),
            dict(a='A',index=2,b=1.,c=[1,2],d='d',e=None,z=[3,4]),
            dict(a='B',index=1,b=2.,e='E')
        ])

        actual_df = table.to_pandas()

        pandas.testing.assert_frame_equal(expected_df,actual_df, check_dtype=False)

    def test_pandas_with_dict_column(self):
        import pandas as pd   #type: ignore
        import pandas.testing #type: ignore

        table = Table("test", ['a'],[dict(a='A',b='B',c={'z':10},d='d'),dict(a='B',e='E')])        

        expected_df = pd.DataFrame([
            dict(a='A',b='B',c={'z':10},d='d'),
            dict(a='B',e='E')
        ])

        actual_df = table.to_pandas()

        pandas.testing.assert_frame_equal(expected_df,actual_df)

    def test_pandas_two_pack_item(self):

        import pandas as pd
        import pandas.testing

        table = Table("test", ['a'], [dict(a='A', c=1, _packed=dict(b=['B','b'],d=['D','d'])), dict(a='B', e='E')])

        expected_df = pd.DataFrame([
            dict(a='A',index=1,b='B',c=1,d='D'),
            dict(a='A',index=2,b='b',c=1,d='d'),
            dict(a='B',index=1,e='E')
        ])

        actual_df = table.to_pandas()

        pandas.testing.assert_frame_equal(expected_df,actual_df, check_dtype=False)

    def test_pandas_huge_pack_item(self):

        rows  = [dict(environment_id=i,learner_id=2,C=5,A=5,N=1,_packed=dict(reward=[2]*9000)) for i in range(2) ]
        table = Table("test", ['environment_id', 'learner_id'], rows)
        time = min(timeit.repeat(lambda:table.to_pandas(), repeat=6, number=1))
        
        #best time on my laptop was 0.15
        self.assertLess(time,1)

class InteractionTable_Tests(unittest.TestCase):

    def test_simple_each_span_none(self):
        table = InteractionsTable("ABC", ["environment_id", "learner_id"], rows=[
            {"environment_id":0, "learner_id":0, "_packed": {"reward":[1,2,3]}},
            {"environment_id":1, "learner_id":0, "_packed": {"reward":[3,6,9]}},
            {"environment_id":0, "learner_id":1, "_packed": {"reward":[2,4,6]}}
        ])

        expected = [[0,0,1,1.5,2], [0,1,3,4.5,6], [1,0,2,3,4]]
        actual   = table.to_progressive_lists(each=True)

        self.assertCountEqual(expected,actual)

    def test_simple_not_each_span_none(self):
        table = InteractionsTable("ABC", ["environment_id", "learner_id"], rows=[
            {"environment_id":0, "learner_id":0, "_packed": {"reward":[1,2,3]}},
            {"environment_id":1, "learner_id":0, "_packed": {"reward":[3,6,9]}},
            {"environment_id":0, "learner_id":1, "_packed": {"reward":[2,4,6]}}
        ])

        expected = [[0,2,3,4], [1,2,3,4]]
        actual   = table.to_progressive_lists(each=False)

        self.assertCountEqual(expected,actual)

    def test_simple_each_span_one(self):
        table = InteractionsTable("ABC", ["environment_id", "learner_id"], rows=[
            {"environment_id":0, "learner_id":0, "_packed": {"reward":[1,2,3]}},
            {"environment_id":1, "learner_id":0, "_packed": {"reward":[3,6,9]}},
            {"environment_id":0, "learner_id":1, "_packed": {"reward":[2,4,6]}}
        ])

        expected = [[0,0,1,2,3], [0,1,3,6,9], [1,0,2,4,6]]
        actual   = table.to_progressive_lists(each=True,span=1)

        self.assertCountEqual(expected,actual)

    def test_simple_not_each_span_one(self):
        table = InteractionsTable("ABC", ["environment_id", "learner_id"], rows=[
            {"environment_id":0, "learner_id":0, "_packed": {"reward":[1,2,3]}},
            {"environment_id":1, "learner_id":0, "_packed": {"reward":[3,6,9]}},
            {"environment_id":0, "learner_id":1, "_packed": {"reward":[2,4,6]}}
        ])

        expected = [[0,2,4,6], [1,2,4,6]]
        actual   = table.to_progressive_lists(each=False,span=1)

        self.assertCountEqual(expected,actual)

    def test_simple_each_span_two(self):
        table = InteractionsTable("ABC", ["environment_id", "learner_id"], rows=[
            {"environment_id":0, "learner_id":0, "_packed": {"reward":[1,2,3]}},
            {"environment_id":1, "learner_id":0, "_packed": {"reward":[2,4,6]}},
        ])
        expected = [[0,0,1,3/2,5/2],[0,1,2,6/2,10/2]]
        actual   = table.to_progressive_lists(each=True,span=2)

        self.assertEqual(len(expected), len(actual))

        for E,A in zip(expected, actual):
            for e,a in zip(E,A):
                self.assertAlmostEqual(e,a,places=3)

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas is not installed so we must skip pandas tests")
    def test_simple_each_span_none_pandas(self):
        table = InteractionsTable("ABC", ["environment_id", "learner_id"], rows=[
            {"learner_id":0, "environment_id":0, "_packed": {"reward":[1,2,3]}},
            {"learner_id":0, "environment_id":1, "_packed": {"reward":[3,6,9]}},
            {"learner_id":1, "environment_id":0, "_packed": {"reward":[2,4,6]}}
        ])

        expected = [[0,0,1,1.5,2], [1,0,2,3,4], [0,1,3,4.5,6]]
        actual   = table.to_progressive_pandas(each=True)

        self.assertEqual(actual["learner_id"].tolist(), [0,1,0])
        self.assertEqual(actual["environment_id"].tolist(), [0,0,1])
        self.assertEqual(actual[1].tolist(), [1,2,3])
        self.assertEqual(actual[2].tolist(), [1.5,3,4.5])
        self.assertEqual(actual[3].tolist(), [2,4,6])

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas is not installed so we must skip pandas tests")
    def test_simple_not_each_span_none_pandas(self):
        table = InteractionsTable("ABC", ["environment_id", "learner_id"], rows=[
            {"learner_id":0, "environment_id":0, "_packed": {"reward":[1,2,3]}},
            {"learner_id":0, "environment_id":1, "_packed": {"reward":[3,6,9]}},
            {"learner_id":1, "environment_id":0, "_packed": {"reward":[2,4,6]}}
        ])

        expected = [[0,2,3,4], [1,2,3,4]]
        actual   = table.to_progressive_pandas(each=False)

        self.assertEqual(actual["learner_id"].tolist(), [0,1])
        self.assertEqual(actual[1].tolist(), [2,2])
        self.assertEqual(actual[2].tolist(), [3,3])
        self.assertEqual(actual[3].tolist(), [4,4])

class TransactionIO_V3_Tests(unittest.TestCase):

    def setUp(self) -> None:
        if Path("coba/tests/.temp/transaction_v3.log").exists():
            Path("coba/tests/.temp/transaction_v3.log").unlink()

    def tearDown(self) -> None:
        if Path("coba/tests/.temp/transaction_v3.log").exists():
            Path("coba/tests/.temp/transaction_v3.log").unlink()

    def test_simple_to_and_from_file(self):

        io = TransactionIO_V3("coba/tests/.temp/transaction_v3.log")

        io.write(["T0",1,2])
        io.write(["T1",0,{"name":"lrn1"}])
        io.write(["T2",1,{"source":"test"}])
        io.write(["T3",[0,1], [{"reward":3},{"reward":4}]])

        result = io.read()

        self.assertEqual(result.experiment, {"n_learners":1, "n_environments":2})
        self.assertEqual([(0,"lrn1")], result.learners.to_tuples())
        self.assertEqual([(1,"test")], result.environments.to_tuples())
        self.assertEqual([(0,1,1,3),(0,1,2,4)], result.interactions.to_tuples())

    def test_simple_to_and_from_memory(self):
        io = TransactionIO_V3()

        io.write(["T0",1,2])
        io.write(["T1",0,{"name":"lrn1"}])
        io.write(["T2",1,{"source":"test"}])
        io.write(["T3",[0,1], [{"reward":3},{"reward":4}]])

        result = io.read()

        self.assertEqual(result.experiment, {"n_learners":1, "n_environments":2})
        self.assertEqual([(0,"lrn1")], result.learners.to_tuples())
        self.assertEqual([(1,"test")], result.environments.to_tuples())
        self.assertEqual([(0,1,1,3),(0,1,2,4)], result.interactions.to_tuples())

    def test_simple_to_and_from_memory_unknown_transaction(self):
        io = TransactionIO_V3()

        io.write(["T0",1,2])
        io.write(["T1",0,{"name":"lrn1"}])
        io.write(["T2",1,{"source":"test"}])
        io.write(["T4",'UNKNOWN'])
        io.write(["T3",[0,1], [{"reward":3},{"reward":4}]])

        result = io.read()

        self.assertEqual(result.experiment, {"n_learners":1, "n_environments":2})
        self.assertEqual([(0,"lrn1")], result.learners.to_tuples())
        self.assertEqual([(1,"test")], result.environments.to_tuples())
        self.assertEqual([(0,1,1,3),(0,1,2,4)], result.interactions.to_tuples())

class TransactionIO_V4_Tests(unittest.TestCase):

    def setUp(self) -> None:
        if Path("coba/tests/.temp/transaction_v4.log").exists():
            Path("coba/tests/.temp/transaction_v4.log").unlink()

    def tearDown(self) -> None:
        if Path("coba/tests/.temp/transaction_v4.log").exists():
            Path("coba/tests/.temp/transaction_v4.log").unlink()

    def test_simple_to_and_from_file(self):
        io = TransactionIO_V4("coba/tests/.temp/transaction_v4.log")

        io.write(["T0",1,2])
        io.write(["T1",0,{"name":"lrn1"}])
        io.write(["T2",1,{"source":"test"}])
        io.write(["T3",[0,1], [{"reward":3},{"reward":4}]])

        result = io.read()

        self.assertEqual(result.experiment, {"n_learners":1, "n_environments":2})
        self.assertEqual([(0,"lrn1")], result.learners.to_tuples())
        self.assertEqual([(1,"test")], result.environments.to_tuples())
        self.assertEqual([(0,1,1,3),(0,1,2,4)], result.interactions.to_tuples())

    def test_simple_to_and_from_memory(self):
        io = TransactionIO_V4()

        io.write(["T0",1,2])
        io.write(["T1",0,{"name":"lrn1"}])
        io.write(["T2",1,{"source":"test"}])
        io.write(["T3",[0,1], [{"reward":3},{"reward":4}]])

        result = io.read()

        self.assertEqual(result.experiment, {"n_learners":1, "n_environments":2})
        self.assertEqual([(0,"lrn1")], result.learners.to_tuples())
        self.assertEqual([(1,"test")], result.environments.to_tuples())
        self.assertEqual([(0,1,1,3),(0,1,2,4)], result.interactions.to_tuples())

    def test_simple_to_and_from_memory_unknown_transaction(self):
        io = TransactionIO_V4()

        io.write(["T0",1,2])
        io.write(["T1",0,{"name":"lrn1"}])
        io.write(["T2",1,{"source":"test"}])
        io.write(["T4",'UNKNOWN'])
        io.write(["T3",[0,1], [{"reward":3},{"reward":4}]])

        result = io.read()

        self.assertEqual(result.experiment, {"n_learners":1, "n_environments":2})
        self.assertEqual([(0,"lrn1")], result.learners.to_tuples())
        self.assertEqual([(1,"test")], result.environments.to_tuples())
        self.assertEqual([(0,1,1,3),(0,1,2,4)], result.interactions.to_tuples())

class TransactionIO_Tests(unittest.TestCase):

    def setUp(self) -> None:
        if Path("coba/tests/.temp/transaction.log").exists():
            Path("coba/tests/.temp/transaction.log").unlink()
    
    def tearDown(self) -> None:
        if Path("coba/tests/.temp/transaction.log").exists():
            Path("coba/tests/.temp/transaction.log").unlink()

    def test_simple_to_and_from_file_v2(self):

        DiskIO("coba/tests/.temp/transaction.log").write('["version",2]')

        with self.assertRaises(CobaException):
            io = TransactionIO("coba/tests/.temp/transaction.log")

    def test_simple_to_and_from_file_v3(self):

        io = TransactionIO_V3("coba/tests/.temp/transaction.log")

        io.write(["T0",1,2])
        io.write(["T1",0,{"name":"lrn1"}])
        io.write(["T2",1,{"source":"test"}])
        io.write(["T3",[0,1], [{"reward":3},{"reward":4}]])

        result = TransactionIO("coba/tests/.temp/transaction.log").read()

        self.assertEqual(result.experiment, {"n_learners":1, "n_environments":2})
        self.assertEqual([(0,"lrn1")], result.learners.to_tuples())
        self.assertEqual([(1,"test")], result.environments.to_tuples())
        self.assertEqual([(0,1,1,3),(0,1,2,4)], result.interactions.to_tuples())

    def test_simple_to_and_from_file_v4(self):
        io = TransactionIO("coba/tests/.temp/transaction.log")

        io.write(["T0",1,2])
        io.write(["T1",0,{"name":"lrn1"}])
        io.write(["T2",1,{"source":"test"}])
        io.write(["T3",[0,1], [{"reward":3},{"reward":4}]])

        result = io.read()

        self.assertEqual(result.experiment, {"n_learners":1, "n_environments":2})
        self.assertEqual([(0,"lrn1")], result.learners.to_tuples())
        self.assertEqual([(1,"test")], result.environments.to_tuples())
        self.assertEqual([(0,1,1,3),(0,1,2,4)], result.interactions.to_tuples())

    def test_simple_to_and_from_memory(self):
        io = TransactionIO()

        io.write(["T0",1,2])
        io.write(["T1",0,{"name":"lrn1"}])
        io.write(["T2",1,{"source":"test"}])
        io.write(["T3",[0,1], [{"reward":3},{"reward":4}]])

        result = io.read()

        self.assertEqual(result.experiment, {"n_learners":1, "n_environments":2})
        self.assertEqual([(0,"lrn1")], result.learners.to_tuples())
        self.assertEqual([(1,"test")], result.environments.to_tuples())
        self.assertEqual([(0,1,1,3),(0,1,2,4)], result.interactions.to_tuples())

    def test_simple_resume(self):
        io = TransactionIO("coba/tests/.temp/transaction.log")

        io.write(["T0",1,2])
        io.write(["T1",0,{"name":"lrn1"}])
        io.write(["T2",1,{"source":"test"}])
        io.write(["T3",[0,1], [{"reward":3},{"reward":4}]])

        result = TransactionIO("coba/tests/.temp/transaction.log").read()

        self.assertEqual(result.experiment, {"n_learners":1, "n_environments":2})
        self.assertEqual([(0,"lrn1")], result.learners.to_tuples())
        self.assertEqual([(1,"test")], result.environments.to_tuples())
        self.assertEqual([(0,1,1,3),(0,1,2,4)], result.interactions.to_tuples())

class Result_Tests(unittest.TestCase):

    def test_has_interactions_key(self):

        result = Result(1,2, {}, {}, {(0,1): {"_packed":{"reward":[1,1]}},(0,2):{"_packed":{"reward":[1,1]}}})

        self.assertEqual("{'Learners': 0, 'Environments': 0, 'Interactions': 4}", str(result))

        self.assertTrue( (0,1) in result._interactions)
        self.assertTrue( (0,2) in result._interactions)

        self.assertEqual(len(result._interactions), 4)

    def test_has_preamble(self):

        self.assertDictEqual(Result(1,2,{},{},{}).experiment, {"n_learners":1, "n_environments":2})

    def test_exception_when_no_file(self):
        with self.assertRaises(Exception):
            Result.from_file("abcd")

    def test_filter_fin(self):

        sims = {1:{}, 2:{}}
        lrns = {1:{}, 2:{}}
        ints = {(1,1):{}, (1,2):{}, (2,1):{}}

        original_result = Result(1, 1, sims, lrns, ints)
        filtered_result = original_result.filter_fin()

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(1, len(filtered_result.environments))
        self.assertEqual(2, len(filtered_result.learners))
        self.assertEqual(2, len(filtered_result.interactions))

    def test_filter_fin_no_finished(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListIO()

        sims = {1:{}, 2:{}}
        lrns = {1:{}, 2:{}}
        ints = {(1,1):{}, (2,1):{}}

        original_result = Result(1, 1, sims, lrns, ints)
        filtered_result = original_result.filter_fin()

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(2, len(original_result.interactions))

        self.assertEqual(0, len(filtered_result.environments))
        self.assertEqual(2, len(filtered_result.learners))
        self.assertEqual(0, len(filtered_result.interactions))
        self.assertEqual(["There was no environment which was finished for every learner."], CobaContext.logger.sink.items)

    def test_filter_env(self):

        sims = {1:{}, 2:{}}
        lrns = {1:{}, 2:{}}
        ints = {(1,1):{}, (1,2):{}, (2,1):{}}

        original_result = Result(1, 1, sims, lrns, ints)
        filtered_result = original_result.filter_env(environment_id=2)

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(1, len(filtered_result.environments))
        self.assertEqual(2, len(filtered_result.learners))
        self.assertEqual(1, len(filtered_result.interactions))

    def test_filter_env_no_match(self):

        CobaContext.logger = IndentLogger() 
        CobaContext.logger.sink = ListIO()

        sims = {1:{}, 2:{}}
        lrns = {1:{}, 2:{}}
        ints = {(1,1):{}, (1,2):{}, (2,1):{}}

        original_result = Result(1, 1, sims, lrns, ints)
        filtered_result = original_result.filter_env(environment_id=3)

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(0, len(filtered_result.environments))
        self.assertEqual(2, len(filtered_result.learners))
        self.assertEqual(0, len(filtered_result.interactions))
        self.assertEqual(["No environments matched the given filter."], CobaContext.logger.sink.items)

    def test_filter_lrn_1(self):

        sims = {1:{}, 2:{}}
        lrns = {1:{}, 2:{}}
        ints = {(1,1):{}, (1,2):{}, (2,1):{}}

        original_result = Result(1, 1, sims, lrns, ints)
        filtered_result = original_result.filter_lrn(learner_id=2)

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(2, len(filtered_result.environments))
        self.assertEqual(1, len(filtered_result.learners))
        self.assertEqual(1, len(filtered_result.interactions))

    def test_filter_lrn_2(self):

        sims = {1:{}, 2:{}}
        lrns = {1:{}, 2:{}, 3:{}}
        ints = {(1,1):{}, (1,2):{}, (2,3):{}}

        original_result = Result(1, 1, sims, lrns, ints)
        filtered_result = original_result.filter_lrn(learner_id=[2,1])

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(3, len(original_result.learners))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(2, len(filtered_result.environments))
        self.assertEqual(2, len(filtered_result.learners))
        self.assertEqual(2, len(filtered_result.interactions))

    def test_filter_lrn_no_match(self):
        
        CobaContext.logger=IndentLogger() 
        CobaContext.logger.sink = ListIO()

        sims = {1:{}, 2:{}}
        lrns = {1:{}, 2:{}, 3:{}}
        ints = {(1,1):{}, (1,2):{}, (2,3):{}}

        original_result = Result(1, 1, sims, lrns, ints)
        filtered_result = original_result.filter_lrn(learner_id=5)

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(3, len(original_result.learners))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(2, len(filtered_result.environments))
        self.assertEqual(0, len(filtered_result.learners))
        self.assertEqual(0, len(filtered_result.interactions))
        self.assertEqual(["No learners matched the given filter."], CobaContext.logger.sink.items)

    def test_copy(self):

        sims = {1:{}, 2:{}}
        lrns = {1:{}, 2:{}, 3:{}}
        ints = {(1,1):{}, (1,2):{}, (2,3):{}}

        result = Result(1, 1, sims, lrns, ints)
        result_copy = result.copy()

        self.assertIsNot(result, result_copy)
        self.assertIsNot(result.learners, result_copy.learners)
        self.assertIsNot(result.environments, result_copy.environments)
        self.assertIsNot(result.interactions, result_copy.interactions)

        self.assertEqual(result.learners.keys, result_copy.learners.keys)
        self.assertEqual(result.environments.keys, result_copy.environments.keys)
        self.assertEqual(result.interactions.keys, result_copy.interactions.keys)

    def test_str(self):

        sims = {1:{}, 2:{}}
        lrns = {1:{}, 2:{}, 3:{}}
        ints = {(1,1):{}, (1,2):{}, (2,3):{}}

        self.assertEqual("{'Learners': 3, 'Environments': 2, 'Interactions': 3}", str(Result(1, 1, sims, lrns, ints)))

    def test_ipython_display_(self):

        with unittest.mock.patch("builtins.print") as mock:

            sims = {1:{}, 2:{}}
            lrns = {1:{}, 2:{}, 3:{}}
            ints = {(1,1):{}, (1,2):{}, (2,3):{}}

            result = Result(1, 1, sims, lrns, ints)
            result._ipython_display_()
            mock.assert_called_once_with(str(result))

    def test_plot_learners_data_empty_result_xlim_none(self):
        result = Result(None, None, {}, {}, {})
        self.assertEqual([], list(result._plot_learners_data(xlim=None)))

    def test_plot_learners_data_one_environment_all_default(self):
 
        lrns = {1:{'full_name':'learner_1'}, 2:{ 'full_name':'learner_2'} }
        ints = {(0,1): {"_packed":{"reward":[1,2]}},(0,2):{"_packed":{"reward":[1,2]}}}

        result = Result(None, None, {}, lrns, ints)

        plot_learners_data = list(result._plot_learners_data())

        self.assertEqual(2, len(plot_learners_data))
        self.assertEqual( ('learner_1', [1,2], [1.,3/2], 0, [(1.,),(3/2,)]), plot_learners_data[0])
        self.assertEqual( ('learner_2', [1,2], [1.,3/2], 0, [(1.,),(3/2,)]), plot_learners_data[1])

    def test_plot_learners_data_two_environments_all_default(self):

        lrns = {1:{'full_name':'learner_1'}, 2:{ 'full_name':'learner_2'} }
        ints = {
            (0,1): {"_packed":{"reward":[1,2]}},
            (0,2): {"_packed":{"reward":[1,2]}},
            (1,1): {"_packed":{"reward":[2,3]}},
            (1,2): {"_packed":{"reward":[2,3]}}
        }

        result = Result(None, None, {}, lrns, ints)

        plot_learners_data = list(result._plot_learners_data())

        self.assertEqual(2, len(plot_learners_data))
        self.assertEqual( ('learner_1', [1,2], [3/2,4/2], 0, [(1.,2.),(3/2,5/2)]), plot_learners_data[0])
        self.assertEqual( ('learner_2', [1,2], [3/2,4/2], 0, [(1.,2.),(3/2,5/2)]), plot_learners_data[1])

    def test_plot_learners_data_two_environments_xlim(self):

        lrns = {1:{'full_name':'learner_1'}, 2:{ 'full_name':'learner_2'} }
        ints = {
            (0,1): {"_packed":{"reward":[1,2]}},
            (0,2): {"_packed":{"reward":[1,2]}},
            (1,1): {"_packed":{"reward":[2,3]}},
            (1,2): {"_packed":{"reward":[2,3]}}
        }

        result = Result(None, None, {}, lrns, ints)
        plot_learners_data = list(result._plot_learners_data(xlim=(1,2)))

        self.assertEqual(2, len(plot_learners_data))
        self.assertEqual( ('learner_1', [2], [4/2], 0, [(3/2,5/2)]), plot_learners_data[0])
        self.assertEqual( ('learner_2', [2], [4/2], 0, [(3/2,5/2)]), plot_learners_data[1])

    def test_plot_learners_data_two_environments_err_sd(self):

        lrns = {1:{'full_name':'learner_1'}, 2:{ 'full_name':'learner_2'} }
        ints = {
            (0,1): {"_packed":{"reward":[1,2]}},
            (0,2): {"_packed":{"reward":[1,2]}},
            (1,1): {"_packed":{"reward":[2,3]}},
            (1,2): {"_packed":{"reward":[2,3]}}
        }

        result = Result(None, None, {}, lrns, ints)

        plot_learners_data = list(result._plot_learners_data(err='sd'))

        self.assertEqual(2, len(plot_learners_data))
        self.assertEqual( ('learner_1', [1,2], [3/2,4/2], [1/2,1/2], [(1.,2.),(3/2,5/2)]), plot_learners_data[0])
        self.assertEqual( ('learner_2', [1,2], [3/2,4/2], [1/2,1/2], [(1.,2.),(3/2,5/2)]), plot_learners_data[1])

    def test_plot_learners_data_bad_xlim(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListIO()

        lrns = {1:{'full_name':'learner_1'}, 2:{ 'full_name':'learner_2'} }
        ints = {
            (0,1): {"_packed":{"reward":[1,2]}},
            (0,2): {"_packed":{"reward":[1,2]}},
            (1,1): {"_packed":{"reward":[2,3]}},
            (1,2): {"_packed":{"reward":[2,3]}}
        }

        result = Result(None, None, {}, lrns, ints)

        plot_learners_data = list(result._plot_learners_data(xlim=(1,0)))

        expected_length = 0
        expected_log = "The given x-limit end is less than the x-limit start. Plotting is impossible."

        self.assertEqual(expected_length, len(plot_learners_data))
        self.assertEqual(expected_log, CobaContext.logger.sink.items[0])

    def test_plot_learners_data_sort(self):
        
        lrns = {1:{ 'full_name':'learner_2'}, 2:{'full_name':'learner_1'}}
        ints = {
            (0,1): {"_packed":{"reward":[1,2]}},
            (1,1): {"_packed":{"reward":[2,3]}},
            (0,2): {"_packed":{"reward":[1,3]}},
            (1,2): {"_packed":{"reward":[2,4]}}
        }

        result = Result(None, None, {}, lrns, ints)

        plot_learners_data = list(result._plot_learners_data(sort="name"))
        self.assertEqual(2, len(plot_learners_data))
        self.assertEqual( ('learner_1', [1,2], [3/2,5/2], 0, [(1.,2.),(4/2,6/2)]), plot_learners_data[0])
        self.assertEqual( ('learner_2', [1,2], [3/2,4/2], 0, [(1.,2.),(3/2,5/2)]), plot_learners_data[1])

        plot_learners_data = list(result._plot_learners_data(sort="id"))
        self.assertEqual(2, len(plot_learners_data))
        self.assertEqual( ('learner_2', [1,2], [3/2,4/2], 0, [(1.,2.),(3/2,5/2)]), plot_learners_data[0])
        self.assertEqual( ('learner_1', [1,2], [3/2,5/2], 0, [(1.,2.),(4/2,6/2)]), plot_learners_data[1])

        plot_learners_data = list(result._plot_learners_data(sort="reward"))
        self.assertEqual(2, len(plot_learners_data))
        self.assertEqual( ('learner_1', [1,2], [3/2,5/2], 0, [(1.,2.),(4/2,6/2)]), plot_learners_data[0])
        self.assertEqual( ('learner_2', [1,2], [3/2,4/2], 0, [(1.,2.),(3/2,5/2)]), plot_learners_data[1])

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed so we must skip plotting tests")
    def test_plot_learners_not_each(self):
        with unittest.mock.patch('matplotlib.pyplot.show') as show:
            with unittest.mock.patch('matplotlib.pyplot.savefig') as savefig:
                with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:
                    lrns = {1:{'full_name':'learner_1'}, 2:{ 'full_name':'learner_2'} }
                    ints = {
                        (0,1): {"_packed":{"reward":[1,2]}},
                        (0,2): {"_packed":{"reward":[1,2]}},
                        (1,1): {"_packed":{"reward":[2,3]}},
                        (1,2): {"_packed":{"reward":[2,3]}}
                    }

                    mock_ax = plt_figure().add_subplot()
                    mock_ax.get_legend.return_value=None

                    mock_ax.get_xticks.return_value = [1,2]
                    mock_ax.get_xlim.return_value   = [2,2]

                    Result(None, None, {}, lrns, ints).plot_learners()

                    plt_figure().add_subplot.assert_called_with(111)

                    self.assertEqual(([1,2],[1.5,2.]), mock_ax.errorbar.call_args_list[0][0])
                    self.assertEqual('learner_1'     , mock_ax.errorbar.call_args_list[0][1]["label"])
                    self.assertEqual(0               , mock_ax.errorbar.call_args_list[0][1]["yerr"])

                    self.assertEqual(([1,2],[1.5,2.]), mock_ax.errorbar.call_args_list[1][0])
                    self.assertEqual('learner_2'     , mock_ax.errorbar.call_args_list[1][1]["label"])
                    self.assertEqual(0               , mock_ax.errorbar.call_args_list[1][1]["yerr"])

                    self.assertEqual(0, mock_ax.plot.call_count)
                    self.assertIsNone(mock_ax.get_legend())
                    self.assertEqual(1, mock_ax.legend.call_count)

                    mock_ax.set_xticks.called_once_with([2,2])
                    self.assertEqual(1, show.call_count)
                    self.assertEqual(0, savefig.call_count)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed so we must skip plotting tests")
    def test_plot_learners_each_xlim_ylim(self):
        with unittest.mock.patch('matplotlib.pyplot.show') as show:
            with unittest.mock.patch('matplotlib.pyplot.savefig') as savefig:
                with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:
                    lrns = {1:{'full_name':'learner_1'}, 2:{ 'full_name':'learner_2'} }
                    ints = {
                        (0,1): {"_packed":{"reward":[1,2]}},
                        (0,2): {"_packed":{"reward":[1,2]}},
                        (1,1): {"_packed":{"reward":[2,3]}},
                        (1,2): {"_packed":{"reward":[2,3]}}
                    }

                    mock_ax = plt_figure().add_subplot()

                    mock_ax.get_xticks.return_value = [1,2]
                    mock_ax.get_xlim.return_value   = [2,2]

                    Result(None, None, {}, lrns, ints).plot_learners(each=True,xlim=(0,1),ylim=(0,1))

                    plt_figure().add_subplot.assert_called_with(111)

                    self.assertEqual(([1],[1.5]) , mock_ax.errorbar.call_args_list[0][0])
                    self.assertEqual('learner_1' , mock_ax.errorbar.call_args_list[0][1]["label"])
                    self.assertEqual(0           , mock_ax.errorbar.call_args_list[0][1]["yerr"])

                    self.assertEqual(([1],[1.5]) , mock_ax.errorbar.call_args_list[1][0])
                    self.assertEqual('learner_2' , mock_ax.errorbar.call_args_list[1][1]["label"])
                    self.assertEqual(0           , mock_ax.errorbar.call_args_list[1][1]["yerr"])

                    self.assertEqual(4, mock_ax.plot.call_count)

                    mock_ax.set_xticks.called_once_with([2,2])
                    self.assertEqual(1, show.call_count)
                    self.assertEqual(0, savefig.call_count)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed so we must skip plotting tests")
    def test_plot_learners_each_figname(self):
        with unittest.mock.patch('matplotlib.pyplot.show') as show:
            with unittest.mock.patch('matplotlib.pyplot.savefig') as savefig:
                with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:
                    lrns = {1:{'full_name':'learner_1'}, 2:{ 'full_name':'learner_2'} }
                    ints = {
                        (0,1): {"_packed":{"reward":[1,2]}},
                        (0,2): {"_packed":{"reward":[1,2]}},
                        (1,1): {"_packed":{"reward":[2,3]}},
                        (1,2): {"_packed":{"reward":[2,3]}}
                    }

                    mock_ax = plt_figure().add_subplot()

                    mock_ax.get_xticks.return_value = [1,2]
                    mock_ax.get_xlim.return_value   = [2,2]

                    Result(None, None, {}, lrns, ints).plot_learners(filename='abc')

                    plt_figure().add_subplot.assert_called_with(111)

                    self.assertEqual(([1,2],[1.5,2.]), mock_ax.errorbar.call_args_list[0][0])
                    self.assertEqual('learner_1'     , mock_ax.errorbar.call_args_list[0][1]["label"])
                    self.assertEqual(0               , mock_ax.errorbar.call_args_list[0][1]["yerr"])

                    self.assertEqual(([1,2],[1.5,2.]), mock_ax.errorbar.call_args_list[1][0])
                    self.assertEqual('learner_2'     , mock_ax.errorbar.call_args_list[1][1]["label"])
                    self.assertEqual(0               , mock_ax.errorbar.call_args_list[1][1]["yerr"])

                    self.assertEqual(0, mock_ax.plot.call_count)

                    mock_ax.set_xticks.called_once_with([2,2])
                    self.assertEqual(1, show.call_count)
                    self.assertEqual(1, savefig.call_count)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed so we must skip plotting tests")
    def test_plot_learners_ax_provided(self):
        with unittest.mock.patch('matplotlib.pyplot.show') as show:
            with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:
                
                mock_ax = unittest.mock.MagicMock()

                lrns = {1:{'full_name':'learner_1'}, 2:{ 'full_name':'learner_2'} }
                ints = {
                    (0,1): {"_packed":{"reward":[1,2]}},
                    (0,2): {"_packed":{"reward":[1,2]}},
                    (1,1): {"_packed":{"reward":[2,3]}},
                    (1,2): {"_packed":{"reward":[2,3]}}
                }

                mock_ax.get_xticks.return_value = [1,2]
                mock_ax.get_xlim.return_value   = [2,2]

                Result(None, None, {}, lrns, ints).plot_learners(ax=mock_ax)

                self.assertEqual(0, plt_figure().add_subplot.call_count)

                self.assertEqual(([1,2],[1.5,2.]), mock_ax.errorbar.call_args_list[0][0])
                self.assertEqual('learner_1'     , mock_ax.errorbar.call_args_list[0][1]["label"])
                self.assertEqual(0               , mock_ax.errorbar.call_args_list[0][1]["yerr"])

                self.assertEqual(([1,2],[1.5,2.]), mock_ax.errorbar.call_args_list[1][0])
                self.assertEqual('learner_2'     , mock_ax.errorbar.call_args_list[1][1]["label"])
                self.assertEqual(0               , mock_ax.errorbar.call_args_list[1][1]["yerr"])

                self.assertEqual(1, mock_ax.get_legend().remove.call_count)
                self.assertEqual(1, mock_ax.legend.call_count)
                self.assertEqual(0, mock_ax.plot.call_count)

                mock_ax.set_xticks.called_once_with([2,2])
                self.assertEqual(0, show.call_count)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed so we must skip plotting tests")
    def test_plot_learners_empty_results(self):
        with unittest.mock.patch('matplotlib.pyplot.show') as show:
            with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

                CobaContext.logger = IndentLogger()
                CobaContext.logger.sink = ListIO()

                result = Result(None, None, {}, {}, {})
                result.plot_learners()

                self.assertEqual(0, plt_figure().add_subplot.call_count)
                self.assertEqual([f"No data was found for plotting in the given results: {result}."], CobaContext.logger.sink.items)

if __name__ == '__main__':
    unittest.main()