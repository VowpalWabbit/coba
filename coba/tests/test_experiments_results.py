import unittest
import unittest.mock
import importlib.util

from statistics import mean

from coba.pipes import ListSink
from coba.context import CobaContext, IndentLogger
from coba.exceptions import CobaException, CobaExit
from coba.statistics import BootstrapCI

from coba.experiments.results import TransactionEncode,TransactionDecode,TransactionResult
from coba.experiments.results import Result, Table, View
from coba.experiments.results import MatplotPlotter, Points
from coba.experiments.results import moving_average

class TestPlotter:
    def __init__(self):
        self.plot_calls = []

    def plot(self, *args) -> None:
        self.plot_calls.append(args)

class TransactionResult_Tests(unittest.TestCase):

    def test_empty(self):
        transactions = [
            ["version",4]
        ]

        res = TransactionResult().filter(transactions)
        self.assertEqual(res.environments,Table(columns=['environment_id']))
        self.assertEqual(res.learners,Table(columns=['learner_id']))
        self.assertEqual(res.evaluators  ,Table(columns=['evaluator_id']))
        self.assertEqual(res.interactions,Table(columns=['environment_id', 'learner_id', 'evaluator_id', 'index']))

    def test_multirow(self):
        transactions = [
            ["version",4],
            ["E",0,{'a':1}],
            ["E",0,{'b':2}]
        ]
        res = TransactionResult().filter(transactions)
        self.assertEqual(res.environments,Table(columns=['environment_id','a','b']).insert([[0,1,2]]))
        self.assertEqual(res.learners    ,Table(columns=['learner_id']))
        self.assertEqual(res.evaluators  ,Table(columns=['evaluator_id']))
        self.assertEqual(res.interactions,Table(columns=['environment_id', 'learner_id', 'evaluator_id', 'index']))

    def test_environments(self):
        transactions = [
            ["version",4],
            ["E",0,{'a':1,'b':2}]
        ]
        res = TransactionResult().filter(transactions)
        self.assertEqual(res.environments,Table(columns=['environment_id','a','b']).insert([[0,1,2]]))
        self.assertEqual(res.learners    ,Table(columns=['learner_id']))
        self.assertEqual(res.evaluators  ,Table(columns=['evaluator_id']))
        self.assertEqual(res.interactions,Table(columns=['environment_id', 'learner_id', 'evaluator_id', 'index']))

    def test_learners(self):
        transactions = [
            ["version",4],
            ["L",0,{'a':1,'b':2}]
        ]
        res = TransactionResult().filter(transactions)
        self.assertEqual(res.environments,Table(columns=['environment_id']))
        self.assertEqual(res.learners    ,Table(columns=['learner_id','a','b']).insert([[0,1,2]]))
        self.assertEqual(res.evaluators  ,Table(columns=['evaluator_id']))
        self.assertEqual(res.interactions,Table(columns=['environment_id', 'learner_id', 'evaluator_id', 'index']))

    def test_evaluators(self):
        transactions = [
            ["version",4],
            ["V",0,{'a':1,'b':2}]
        ]
        res = TransactionResult().filter(transactions)
        self.assertEqual(res.environments,Table(columns=['environment_id']))
        self.assertEqual(res.learners    ,Table(columns=['learner_id']))
        self.assertEqual(res.evaluators  ,Table(columns=['evaluator_id','a','b']).insert([[0,1,2]]))
        self.assertEqual(res.interactions,Table(columns=['environment_id', 'learner_id', 'evaluator_id', 'index']))

    def test_same_columns(self):
        transactions = [
            ["version",4],
            ["I",(0,2),{"_packed":{"reward":[1,4]}}],["I",(0,1),{"_packed":{"reward":[1,3]}}]
        ]
        res = TransactionResult().filter(transactions)
        self.assertEqual(res.environments,Table(columns=['environment_id']))
        self.assertEqual(res.learners    ,Table(columns=['learner_id']))
        self.assertEqual(res.evaluators  ,Table(columns=['evaluator_id']))
        self.assertEqual(res.interactions,Table(columns=['environment_id', 'learner_id', 'evaluator_id', 'index', 'reward']).insert([(0,1,0,1,1),(0,1,0,2,3),(0,2,0,1,1),(0,2,0,2,4)]))

    def test_diff_columns(self):
        transactions = [
            ["version",4],
            ["I",(0,1),{"_packed":{"reward":[1,3]}}],["I",(0,2),{"_packed":{"z":[1,4]}}]
        ]
        res = TransactionResult().filter(transactions)
        self.assertEqual(res.environments,Table(columns=['environment_id']))
        self.assertEqual(res.learners,Table(columns=['learner_id']))
        self.assertEqual(res.evaluators  ,Table(columns=['evaluator_id']))
        self.assertEqual(res.interactions,Table(columns=['environment_id', 'learner_id', 'evaluator_id', 'index', 'reward', 'z']).insert([(0,1,0,1,1,None),(0,1,0,2,3,None),(0,2,0,1,None,1),(0,2,0,2,None,4)]))

    def test_old_version(self):
        with self.assertRaises(CobaException):
            TransactionResult().filter([["version",3]])

    def test_unknown_version(self):
        with self.assertRaises(CobaException):
            TransactionResult().filter([["version",5]])

class TransactionEncode_Tests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(list(TransactionEncode().filter([])),['["version",4]'])

    def test_experiment_dict(self):
        self.assertEqual(list(TransactionEncode().filter([['T0',{'a':1.0}]])),['["version",4]','["experiment",{"a":1}]'])

    def test_environment_params(self):
        self.assertEqual(list(TransactionEncode().filter([['T1',0,{'a':1.0}]])),['["version",4]','["E",0,{"a":1}]'])

    def test_learner_params(self):
        self.assertEqual(list(TransactionEncode().filter([['T2',0,{'a':1.0}]])),['["version",4]','["L",0,{"a":1}]'])

    def test_interactions(self):
        self.assertEqual(list(TransactionEncode().filter([['T4',[1,0],[{"R1":3},{"R1":4}]]])),['["version",4]',r'["I",[1,0],{"_packed":{"R1":[3,4]}}]'])

    def test_interaction_uneven_dictionaries(self):
        self.assertEqual(list(TransactionEncode().filter([['T4',[1,0],[{"R1":3},{"R2":4}]]])),['["version",4]',r'["I",[1,0],{"_packed":{"R1":[3,null],"R2":[null,4]}}]'])

class TransactionDecode_Tests(unittest.TestCase):
    def test_get_version(self):
        self.assertEqual(list(TransactionDecode().filter(['["version",4]'])), [["version",4]])

    def test_one_row(self):
        self.assertEqual(list(TransactionDecode().filter(['["version",4]','{"a":1}'])), [["version",4],{"a":1}])

class View_Tests(unittest.TestCase):

    def test_listview(self):
        listview = View.ListView([1,2,3,4],[0,2,3])
        self.assertEqual([1,3], list(listview[:2]))
        self.assertEqual(1, listview[0])
        self.assertEqual(3, len(listview))

    def test_sliceview(self):
        sliceview = View.SliceView([1,2,3,4],slice(3))
        self.assertEqual([1,2], list(sliceview[:2]))
        self.assertEqual(1, sliceview[0])
        self.assertEqual(3, len(sliceview))

    def test_getitem_seq(self):
        view = View({"a":[1,2,3]},[0,2])
        self.assertSequenceEqual(view['a'],(1,3))

    def test_getitem_slice(self):
        view = View({"a":[1,2,3]},slice(0,2))
        self.assertSequenceEqual(view['a'],(1,2))

    def test_values(self):
        view = View({"a":[1,2,3],'b':[4,5,6]},[0,2])
        self.assertCountEqual([tuple(v) for v in view.values()],[(1,3),(4,6)])

    def test_keys(self):
        view = View({"a":[1,2,3],'b':[4,5,6]},[0,2])
        self.assertCountEqual(view.keys(),['a','b'])

    def test_contains(self):
        view = View({"a":[1,2,3],'b':[4,5,6]},[0,2])
        self.assertIn('a',view)

    def test_readonly(self):
        view = View({"a":[1,2,3],'b':[4,5,6]},[0,2])
        with self.assertRaises(CobaException):
            view['a'] = [3,4]

    def test_view_of_view(self):
        view = View(View({"a":[1,2,3],'b':[4,5,6]},[0,2]),[0,1])
        self.assertEqual([1,3],list(view['a']))

    def test_view_of_view_slice_slice(self):
        view = View(View({"a":[1,2,3],'b':[4,5,6]},slice(0,2)),slice(1,2))
        self.assertEqual([2],list(view['a']))

    def test_view_of_view_list_slice(self):
        view = View(View({"a":[1,2,3],'b':[4,5,6]},[0,2]),slice(1,2))
        self.assertEqual([3],list(view['a']))

    def test_view_of_view_slice_list(self):
        view = View(View({"a":[1,2,3],'b':[4,5,6]},slice(0,2)),[1])
        self.assertEqual([2],list(view['a']))

class Table_Tests(unittest.TestCase):

    def test_table_str(self):
        self.assertEqual("{'Columns': ('id', 'col'), 'Rows': 2}",str(Table(columns=['id','col']).insert([[1,2],[2,3]])))

    def test_table_init(self):
        data = {'id':[1,2],'col':[2,3]}
        view = View(data, slice(None,None))

        table = Table(data)
        self.assertEqual(table.columns, ('id','col'))
        self.assertIs(table._data, data)

        table = Table(data,columns=('col',))
        self.assertEqual(table.columns, ('col','id'))
        self.assertIsNot(table._data, data)
        self.assertEqual(table._data,data)

        table = Table(view)
        self.assertEqual(table.columns, ('id','col'))
        self.assertIs(table._data, view)

    def test_ipython_display(self):
        with unittest.mock.patch("builtins.print") as mock:
            table = Table(columns=['id','col']).insert([[1,2],[2,3]])
            table._ipython_display_()
            mock.assert_called_once_with(str(table))

    def test_insert_item(self):
        table = Table(columns=['a','b']).insert({'a':['a','B'],'b':['A','B']})

        self.assertEqual(list(table), [('a','A'),('B','B')])
        self.assertSequenceEqual(table.columns, ['a','b'])
        self.assertEqual(2, len(table))

    def test_bad_index(self):
        table = Table(columns=['a','b']).insert({'a':['a','B'],'b':['A','B']})

        with self.assertRaises(KeyError):
            table['c']

        with self.assertRaises(KeyError):
            table[0]

    def test_where_kwarg_str(self):
        table = Table(columns=['a','b']).insert({'a':['a','A'],'b':['b','B']})

        filtered_table = table.where(b="B")

        self.assertEqual(2, len(table))
        self.assertEqual([('a','b'),('A','B')],list(table))

        self.assertEqual(1, len(filtered_table))
        self.assertEqual([('A','B')], list(filtered_table))

    def test_where_kwarg_int_1(self):
        table = Table(columns=['a','b']).insert([['1','b'],['11','B']])

        filtered_table = table.where(a=1,comparison='match')

        self.assertEqual(2, len(table))
        self.assertEqual([('1','b'),('11','B')], list(table))

        self.assertEqual(1, len(filtered_table))
        self.assertEqual([('1','b')], list(filtered_table))

    def test_where_kwarg_int_2(self):
        table = Table(columns=['a','b']).insert([[1,'b'],[2,'B']])

        filtered_table = table.where(a=1)

        self.assertEqual(2, len(table))
        self.assertEqual([(1,'b'),(2,'B')], list(table))

        self.assertEqual(1, len(filtered_table))
        self.assertEqual([(1,'b')], list(filtered_table))

    def test_where_kwarg_pred(self):
        table = Table(columns=['a','b']).insert([['1','b'],['12','B']])

        filtered_table = table.where(a=lambda v: v=='1')

        self.assertEqual(2, len(table))
        self.assertEqual([('1','b'),('12','B')], list(table))

        self.assertEqual(1, len(filtered_table))
        self.assertEqual([('1','b')], list(filtered_table))

    def test_where_kwarg_multi(self):
        table = Table(columns=['a','b','c']).insert([
            ['1', 'b', 'c'],
            ['2', 'b', 'C'],
            ['3', 'B', 'c'],
            ['4', 'B', 'C']
        ])

        filtered_table = table.where(b="b", c="C")

        self.assertEqual(4, len(table))
        self.assertEqual(3, len(filtered_table))
        self.assertEqual([('1','b','c'),('2','b','C'),('4','B','C')], list(filtered_table))

    def test_where_without_any(self):
        table = Table(columns=['a','b','c']).insert([
            ['1', 'b', 'c'],
            ['2', 'b', 'C'],
        ])

        filtered_table = table.where()

        self.assertEqual(2, len(table))
        self.assertEqual(2, len(filtered_table))
        self.assertEqual([('1','b','c'),('2','b','C')], list(filtered_table))

    def test_where_pred(self):
        table = Table(columns=['a','b']).insert([['A','B'],['a','b']])

        filtered_table = table.where(lambda row: row[1]=="B")

        self.assertEqual(2, len(table))
        self.assertEqual([('A','B'),('a','b')], list(table))

        self.assertEqual(1, len(filtered_table))
        self.assertEqual([('A','B')], list(filtered_table))

    def test_where_in_1(self):
        table = Table(columns=['a','b']).insert([['a','b'], ['A','B'], ['1','C']])

        filtered_table = table.where(a=['a','1'])

        self.assertEqual(3, len(table))
        self.assertEqual([('a','b'),('A','B'),('1','C')], list(table))

        self.assertEqual(2, len(filtered_table))
        self.assertEqual([('a','b'),('1','C')], list(filtered_table))

    def test_where_in_2(self):
        table = Table(columns=['a']).insert([[['1']],[['2']],[['3']]])

        filtered_table = table.where(a=[['1']],comparison='in')

        self.assertEqual(3, len(table))
        self.assertEqual([(['1'],),(['2'],),(['3'],)], list(table))

        self.assertEqual(1, len(filtered_table))
        self.assertEqual([(['1'],)], list(filtered_table))

    def test_where_not_in_foreach(self):
        table = Table(columns=['a','b']).insert([['a','b'], ['A','B'], ['1','C']])

        filtered_table = table.where(a=['a','1'],comparison='!in')

        self.assertEqual(3, len(table))
        self.assertEqual([('a','b'),('A','B'),('1','C')], list(table))

        self.assertEqual(1, len(filtered_table))
        self.assertEqual([('A','B')], list(filtered_table))

    def test_where_not_in_bisect(self):
        table = Table(columns=['a','b']).insert([['a','b'], ['A','B'], ['1','C']]).index('a')

        filtered_table = table.where(a=['a','1'],comparison='!in')

        self.assertEqual(3, len(table))
        self.assertEqual([('1','C'),('A','B'),('a','b')], list(table))

        self.assertEqual(1, len(filtered_table))
        self.assertEqual([('A','B')], list(filtered_table))

    def test_where_le(self):
        table = Table(columns=['a']).insert([[1]]*10)
        table = table.insert([[2]]*10)

        filtered_table = table.where(a=1,comparison='<=')

        self.assertEqual(20, len(table))
        self.assertEqual(10, len(filtered_table))

        filtered_table = table.index('a').where(a=1,comparison='<=')

        self.assertEqual(20, len(table))
        self.assertEqual(10, len(filtered_table))

    def test_where_lt(self):
        table = Table(columns=['a']).insert([[1]]*10)
        table = table.insert([[2]]*10)

        filtered_table = table.where(a=1,comparison='<')

        self.assertEqual(20, len(table))
        self.assertEqual(0, len(filtered_table))

        filtered_table = table.index('a').where(a=1,comparison='<')

        self.assertEqual(20, len(table))
        self.assertEqual(0, len(filtered_table))

    def test_where_gt(self):
        table = Table(columns=['a']).insert([[1]]*10)
        table = table.insert([[2]]*10)

        filtered_table = table.where(a=1,comparison='>')

        self.assertEqual(20, len(table))
        self.assertEqual(10, len(filtered_table))

        filtered_table = table.index('a').where(a=1,comparison='>')

        self.assertEqual(20, len(table))
        self.assertEqual(10, len(filtered_table))

    def test_where_ge(self):
        table = Table(columns=['a']).insert([[1]]*10)
        table = table.insert([[2]]*10)

        filtered_table = table.where(a=1,comparison='>=')

        self.assertEqual(20, len(table))
        self.assertEqual(20, len(filtered_table))

        filtered_table = table.index('a').where(a=1,comparison='>=')

        self.assertEqual(20, len(table))
        self.assertEqual(20, len(filtered_table))

    def test_where_eq(self):
        table = Table(columns=['a']).insert([[1],[1]]).insert([[2],[2]])

        filtered_table = table.where(a=1,comparison='=')

        self.assertEqual(4, len(table))
        self.assertEqual(2, len(filtered_table))

        filtered_table = table.index('a').where(a=1,comparison='=')

        self.assertEqual(4, len(table))
        self.assertEqual(2, len(filtered_table))

    def test_where_ne(self):
        table = Table(columns=['a']).insert([[1],[1]]).insert([[2],[2]])

        filtered_table = table.where(a=1,comparison='!=')

        self.assertEqual(4, len(table))
        self.assertEqual(2, len(filtered_table))

        filtered_table = table.index('a').where(a=1,comparison='!=')

        self.assertEqual(4, len(table))
        self.assertEqual(2, len(filtered_table))

    def test_where_match_number_number(self):
        table = Table(columns=['a']).insert([[1],[1]]).insert([[2],[2]])

        filtered_table = table.where(a=1,comparison='match')

        self.assertEqual(4, len(table))
        self.assertEqual(2, len(filtered_table))

    def test_where_match_number_str(self):
        table = Table(columns=['a']).insert([['1'],['1']]).insert([['2'],['2']])

        filtered_table = table.where(a=1,comparison='match')

        self.assertEqual(4, len(table))
        self.assertEqual(2, len(filtered_table))

    def test_where_match_str_str(self):
        table = Table(columns=['a']).insert([['1'],['1']]).insert([['2'],['2']])
        filtered_table = table.where(a='1',comparison='match')
        self.assertEqual(4, len(table))
        self.assertEqual(2, len(filtered_table))

    def test_where_preserves_order(self):
        table = Table(columns=['a']).insert({'a':list(range(1000))})

        no_index_where = table.where(a=list(reversed([0,10,20,30,40,50,60])))
        self.assertSequenceEqual(no_index_where['a'],[0,10,20,30,40,50,60])

        index_where = table.index('a').where(a=list(reversed([0,10,20,30,40,50,60])))
        self.assertSequenceEqual(index_where['a'],[0,10,20,30,40,50,60])

    def test_where_lt_multilevel_index(self):
        table = Table({'a':[1,1,1,2,2,2],'b':[1,2,3,1,2,3]}).index('a','b')

        filtered_table = table.where(b={'<':3})

        self.assertEqual(6, len(table))
        self.assertEqual(4, len(filtered_table))

    def test_multilevel_index(self):
        table = Table(columns=['a','b','c','d']).insert([(0,1,1,1),(0,1,2,3),(0,2,1,1),(0,2,2,4)])
        table.index('a','b','c')
        self.assertSequenceEqual(['a','b','c'],table.indexes)
        self.assertEqual(list(table), [(0,1,1,1),(0,1,2,3),(0,2,1,1),(0,2,2,4)])

    def test_multilevel_index2(self):
        table = Table(columns=['a','b','c','d']).insert([(0,1,1,1),(0,1,2,3),(1,1,1,1),(1,1,2,4)])
        table.index('a','b','c')
        self.assertEqual(list(table), [(0,1,1,1),(0,1,2,3),(1,1,1,1),(1,1,2,4)])

    def test_groupby_with_index_with_table(self):
        table = Table(columns=['a','b','c','d']).insert([[0,0,1,1],[0,1,1,1],[1,0,1,1],[1,0,2,1],[1,1,1,1],[1,1,2,1]])
        table.index('a','b','c')
        self.assertEqual([len(t) for _,t in table.groupby(2)],[1,1,2,2])
        self.assertEqual([g for g,_ in table.groupby(2)],[(0,0),(0,1),(1,0),(1,1)])

    def test_groupby_with_index_select_none(self):
        table = Table(columns=['a','b','c','d']).insert([[0,0,1,1],[0,1,1,1],[1,0,1,1],[1,0,2,1],[1,1,1,1],[1,1,2,1]])
        table.index('a','b','c')
        self.assertEqual([i for i in table.groupby(2,select=None)],[(0,0),(0,1),(1,0),(1,1)])

    def test_groupby_sans_index_select_table(self):
        table = Table(columns=['a','b','c','d']).insert([[0,0,1,1],[0,1,1,1],[1,0,1,1],[1,0,2,1],[1,1,1,1],[1,1,2,1]])
        table.index('a','b','c')
        self.assertEqual([len(t) for _,t in table.groupby(2,select='table')],[1,1,2,2])
        self.assertEqual([ i     for i,_ in table.groupby(2,select='table')],[(0,0),(0,1),(1,0),(1,1)])

    def test_groupby_sans_index_select_count(self):
        table = Table(columns=['a','b','c','d']).insert([[0,0,1,1],[0,1,1,1],[1,0,1,1],[1,0,2,1],[1,1,1,1],[1,1,2,1]])
        table.index('a','b','c')
        self.assertEqual([ t for _,t in table.groupby(2,select='count')],[1,1,2,2])
        self.assertEqual([ i for i,_ in table.groupby(2,select='count')],[(0,0),(0,1),(1,0),(1,1)])

    def test_groupby_sans_index_select_column(self):
        table = Table(columns=['a','b','c','d']).insert([[0,0,1,1],[0,1,1,1],[1,0,1,1],[1,0,2,1],[1,1,1,1],[1,1,2,1]])
        table.index('a','b','c')
        self.assertEqual([ t for _,t in table.groupby(2,select='b')],[[0],[1],[0,0],[1,1]])
        self.assertEqual([ i for i,_ in table.groupby(2,select='b')],[(0,0),(0,1),(1,0),(1,1)])

    def test_copy(self):
        table = Table(columns=['a','b','c','d']).insert([[0,0,1,1],[0,1,1,1],[1,0,1,1]]).index('a','b','c')
        tcopy = table.copy()

        self.assertIsNot(table,tcopy)
        self.assertIs(table._data, tcopy._data)
        self.assertEqual(table._columns,tcopy._columns)
        self.assertEqual(table._indexes,tcopy._indexes)
        self.assertEqual(table._lohis,tcopy._lohis)

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "this test requires pandas")
    def test_to_pandas(self):

        import pandas as pd
        import pandas.testing

        table = Table(columns=['a','b','c','d','e']).insert([['A','B',1,'d',None],['B',None,None,None,'E']])

        expected_df = pd.DataFrame([
            dict(a='A',b='B',c=1,d='d',e=None),
            dict(a='B',b=None,c=None,d=None,e='E')
        ])

        actual_df = table.to_pandas()

        pandas.testing.assert_frame_equal(expected_df,actual_df)

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "this test requires pandas")
    def test_to_pandas_with_array_column(self):
        import pandas as pd
        import pandas.testing

        table = Table(columns=['a','b','c','d','e']).insert([['A','B',[1,2],'d',None],['B',None,None,None,'E']])

        expected_df = pd.DataFrame([
            dict(a='A',b='B',c=[1,2],d='d',e=None),
            dict(a='B',b=None,c=None,d=None,e='E')
        ])

        actual_df = table.to_pandas()

        pandas.testing.assert_frame_equal(expected_df,actual_df)

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "this test requires pandas")
    def test_to_pandas_with_dict_column(self):
        import pandas as pd
        import pandas.testing

        table = Table(columns=['a','b','c','d','e']).insert([['A','B',{'z':10},'d',None],['B',None,None,None,'E']])

        expected_df = pd.DataFrame([
            dict(a='A',b='B',c={'z':10},d='d',e=None),
            dict(a='B',b=None,c=None,d=None,e='E')
        ])

        actual_df = table.to_pandas()

        pandas.testing.assert_frame_equal(expected_df,actual_df)

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "this test requires pandas")
    def test_to_pandas_with_View(self):
        import pandas as pd
        import pandas.testing

        table = Table(columns=['a','b','c','d','e']).insert([['A','B',1,'d',None],['B',None,None,None,'E']])

        expected_df = pd.DataFrame([
            dict(a='A',b='B',c=1,d='d',e=None),
        ])

        actual_df = table.where(a='A').to_pandas()

        pandas.testing.assert_frame_equal(expected_df,actual_df)

@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "this test requires matplotlib")
class MatplotPlotter_Tests(unittest.TestCase):

    def test_no_matplotlib(self):
        with unittest.mock.patch('importlib.import_module', side_effect=ImportError()):
            with self.assertRaises(CobaExit):
                MatplotPlotter().plot(None,None,None,None,None,None,None,None,None,None,None,None,None)

    def test_plot_lines_title_xlabel_ylabel(self):
        with unittest.mock.patch('matplotlib.pyplot.show') as show:
            with unittest.mock.patch('matplotlib.pyplot.savefig') as savefig:
                with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

                    lines = [
                        Points([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                        Points([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
                    ]

                    mock_ax = plt_figure().add_subplot()
                    MatplotPlotter().plot(None,lines,"title","xlabel","ylabel",None,None,True,True,None,None,None,"screen")

                    plt_figure().add_subplot.assert_called_with(111)

                    self.assertEqual(([1,2],[5,6],'-'), mock_ax.plot.call_args_list[0][0])
                    self.assertEqual('B'              , mock_ax.plot.call_args_list[0][1]["color"])
                    self.assertEqual(1                , mock_ax.plot.call_args_list[0][1]["alpha"])
                    self.assertEqual('L1'             , mock_ax.plot.call_args_list[0][1]["label"])
                    self.assertEqual(1                , mock_ax.plot.call_args_list[0][1]["zorder"])

                    self.assertEqual(([3,4],[7,8],'-'), mock_ax.plot.call_args_list[1][0])
                    self.assertEqual('R'              , mock_ax.plot.call_args_list[1][1]["color"])
                    self.assertEqual(.25              , mock_ax.plot.call_args_list[1][1]["alpha"])
                    self.assertEqual('L2'             , mock_ax.plot.call_args_list[1][1]["label"])
                    self.assertEqual(1                , mock_ax.plot.call_args_list[1][1]["zorder"])

                    mock_ax.set_xticks.called_once_with([2,2])
                    mock_ax.set_title.colled_once_with('title')
                    mock_ax.set_xlabel.called_once_with('xlabel')
                    mock_ax.set_ylabel.called_once_with('ylabel')

                    self.assertEqual(1, show.call_count)
                    self.assertEqual(0, savefig.call_count)

    def test_plot_lines_err_title_xlabel_ylabel_screen(self):
        with unittest.mock.patch('matplotlib.pyplot.show') as show:
            with unittest.mock.patch('matplotlib.pyplot.savefig') as savefig:
                with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

                    lines = [
                        Points([1,2], [5,6], None, [10,11], "B", 1.00, 'L1', '-', 1),
                        Points([3,4], [7,8], None, [12,13], "R", 0.25, 'L2', '-', 1)
                    ]

                    mock_ax = plt_figure().add_subplot()
                    MatplotPlotter().plot(None, lines, "title", "xlabel", "ylabel", None, None, True, True, None, None, None, "screen")

                    plt_figure().add_subplot.assert_called_with(111)

                    self.assertEqual(([1,2],[5,6],[10,11],None,'-'), mock_ax.errorbar.call_args_list[0][0])
                    self.assertEqual('B'                           , mock_ax.errorbar.call_args_list[0][1]["color"])
                    self.assertEqual(1                             , mock_ax.errorbar.call_args_list[0][1]["alpha"])
                    self.assertEqual('L1'                          , mock_ax.errorbar.call_args_list[0][1]["label"])
                    self.assertEqual(1                             , mock_ax.errorbar.call_args_list[0][1]["zorder"])

                    self.assertEqual(([3,4],[7,8],[12,13],None,'-'), mock_ax.errorbar.call_args_list[1][0])
                    self.assertEqual('R'                           , mock_ax.errorbar.call_args_list[1][1]["color"])
                    self.assertEqual(.25                           , mock_ax.errorbar.call_args_list[1][1]["alpha"])
                    self.assertEqual('L2'                          , mock_ax.errorbar.call_args_list[1][1]["label"])
                    self.assertEqual(1                             , mock_ax.errorbar.call_args_list[1][1]["zorder"])

                    mock_ax.set_xticks.called_once_with([2,2])
                    mock_ax.set_title.colled_once_with('title')
                    mock_ax.set_xlabel.called_once_with('xlabel')
                    mock_ax.set_ylabel.called_once_with('ylabel')

                    self.assertEqual(1, show.call_count)
                    self.assertEqual(0, savefig.call_count)

    def test_plot_xlim1(self):
        with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

            lines = [
                Points([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                Points([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
            ]

            mock_ax = plt_figure().add_subplot()
            MatplotPlotter().plot(mock_ax,lines,"title","xlabel","ylabel",(2,3),None,True,True,None,None,None,None)
            self.assertEqual(([2],[6],'-'), mock_ax.plot.call_args_list[0][0])
            self.assertEqual(([3],[7],'-'), mock_ax.plot.call_args_list[1][0])

    def test_plot_xlim2(self):
        with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

            lines = [
                Points([1,2], [5,6], None, [4,3], "B", 1.00, 'L1', '-', 1),
                Points([3,4], [7,8], None, [2,1], "R", 0.25, 'L2', '-', 1)
            ]

            mock_ax = plt_figure().add_subplot()
            MatplotPlotter().plot(mock_ax,lines,"title","xlabel","ylabel",(3,4),None,True,True,None,None,None,None)
            self.assertEqual(([3,4],[7,8],[2,1],None,'-'), mock_ax.errorbar.call_args_list[0][0])

    def test_plot_ylim(self):
        with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

            lines = [
                Points([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                Points([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
            ]

            mock_ax = plt_figure().add_subplot()
            MatplotPlotter().plot(mock_ax,lines,"title","xlabel","ylabel",None,(6,7),True,True,None,None,None,None)
            self.assertEqual(([2],[6],'-'), mock_ax.plot.call_args_list[0][0])
            self.assertEqual(([3],[7],'-'), mock_ax.plot.call_args_list[1][0])

    def test_plot_xrotation(self):
        with unittest.mock.patch('matplotlib.pyplot.xticks') as xticks:
            with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

                lines = [
                    Points([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                    Points([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
                ]

                mock_ax = plt_figure().add_subplot()
                MatplotPlotter().plot(mock_ax,lines,"title","xlabel","ylabel",None,(6,7),True,True,90,None,None,None)

                self.assertEqual(([2],[6],'-'), mock_ax.plot.call_args_list[0][0])
                self.assertEqual(([3],[7],'-'), mock_ax.plot.call_args_list[1][0])
                self.assertEqual({'rotation':90},xticks.call_args[1])

    def test_plot_yrotation(self):
        with unittest.mock.patch('matplotlib.pyplot.yticks') as yticks:
            with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

                lines = [
                    Points([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                    Points([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
                ]

                mock_ax = plt_figure().add_subplot()

                MatplotPlotter().plot(mock_ax,lines,"title","xlabel","ylabel",None,(6,7),True,True,None,90,None,None)

                self.assertEqual(([2],[6],'-'), mock_ax.plot.call_args_list[0][0])
                self.assertEqual(([3],[7],'-'), mock_ax.plot.call_args_list[1][0])
                self.assertEqual({'rotation':90},yticks.call_args[1])

    def test_plot_total_xorder(self):
        with unittest.mock.patch('matplotlib.pyplot.xticks') as xticks:
            with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

                lines = [
                    Points([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                    Points([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
                ]

                mock_ax = plt_figure().add_subplot()
                MatplotPlotter().plot(None,lines,"title","xlabel","ylabel",None,None,True,True,90,None,[1,3,4,2],None)

                self.assertEqual(([0,3],[5,6],'-'), mock_ax.plot.call_args_list[0][0])
                self.assertEqual(([1,2],[7,8],'-'), mock_ax.plot.call_args_list[1][0])
                self.assertEqual(((3,2,1,0),(2,4,3,1)),xticks.call_args[0])

    def test_plot_total_xorder_ascending(self):
        with unittest.mock.patch('matplotlib.pyplot.xticks') as xticks:
            with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

                lines = [
                    Points([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                    Points([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
                ]

                mock_ax = plt_figure().add_subplot()
                MatplotPlotter().plot(None,lines,"title","xlabel","ylabel",None,None,True,True,None,None,[1,2,3,4],None)

                self.assertEqual(([1,2],[5,6],'-'), mock_ax.plot.call_args_list[0][0])
                self.assertEqual(([3,4],[7,8],'-'), mock_ax.plot.call_args_list[1][0])
                self.assertEqual(0,xticks.call_count)

    def test_plot_total_xorder_descending(self):
        with unittest.mock.patch('matplotlib.pyplot.xticks') as xticks:
            with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

                lines = [
                    Points([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                    Points([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
                ]

                mock_ax = plt_figure().add_subplot()
                MatplotPlotter().plot(None,lines,"title","xlabel","ylabel",None,None,True,True,90,None,[4,3,2,1],None)

                self.assertEqual(([-1,-2],[5,6],'-'), mock_ax.plot.call_args_list[0][0])
                self.assertEqual(([-3,-4],[7,8],'-'), mock_ax.plot.call_args_list[1][0])
                self.assertEqual(((-1,-2,-3,-4),(1,2,3,4)),xticks.call_args[0])

    def test_plot_partial_xorder(self):
        with unittest.mock.patch('matplotlib.pyplot.xticks') as xticks:
            with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

                lines = [
                    Points(['1','2'], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                    Points(['3','4'], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
                ]

                mock_ax = plt_figure().add_subplot()
                MatplotPlotter().plot(None,lines,"title","xlabel","ylabel",None,None,True,True,90,None,['1','3'],None)

                self.assertEqual(([0,2],[5,6],'-'), mock_ax.plot.call_args_list[0][0])
                self.assertEqual(([1,3],[7,8],'-'), mock_ax.plot.call_args_list[1][0])
                self.assertEqual(((1,0,2,3),('3','1','2','4')),xticks.call_args[0])

    def test_plot_extra_xorder(self):
        with unittest.mock.patch('matplotlib.pyplot.xticks') as xticks:
            with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

                lines = [
                    Points([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                    Points([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
                ]

                mock_ax = plt_figure().add_subplot()
                MatplotPlotter().plot(None,lines,"title","xlabel","ylabel",None,None,True,True,90,None,[1,3,4,2,5],None)

                self.assertEqual(([0,3],[5,6],'-'), mock_ax.plot.call_args_list[0][0])
                self.assertEqual(([1,2],[7,8],'-'), mock_ax.plot.call_args_list[1][0])
                self.assertEqual(((4,3,2,1,0),(5,2,4,3,1)),xticks.call_args[0])

    def test_plot_no_xticks(self):
        with unittest.mock.patch('matplotlib.pyplot.xticks') as xticks:
            with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

                lines = [
                    Points([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                    Points([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
                ]

                mock_ax = plt_figure().add_subplot()
                MatplotPlotter().plot(None,lines,"title","xlabel","ylabel",None,(6,7),False,True,None,None,None,None)

                self.assertEqual(([2],[6],'-'), mock_ax.plot.call_args_list[0][0])
                self.assertEqual(([3],[7],'-'), mock_ax.plot.call_args_list[1][0])
                self.assertEqual([],xticks.call_args[0][0])

    def test_plot_no_yticks(self):
        with unittest.mock.patch('matplotlib.pyplot.yticks') as yticks:
            with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

                lines = [
                    Points([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                    Points([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
                ]

                mock_ax = plt_figure().add_subplot()
                MatplotPlotter().plot(None,lines,"title","xlabel","ylabel",None,(6,7),True,False,None,None,None,None)

                self.assertEqual(([2],[6],'-'), mock_ax.plot.call_args_list[0][0])
                self.assertEqual(([3],[7],'-'), mock_ax.plot.call_args_list[1][0])
                self.assertEqual([],yticks.call_args[0][0])

    def test_plot_ax(self):
        with unittest.mock.patch('matplotlib.pyplot.show') as show:
            with unittest.mock.patch('matplotlib.pyplot.savefig') as savefig:
                with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

                    lines = [
                        Points([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                        Points([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
                    ]

                    mock_ax = plt_figure().add_subplot()
                    self.assertEqual(1, plt_figure().add_subplot.call_count)
                    MatplotPlotter().plot(mock_ax,lines,"title","xlabel","ylabel",None,None,True,True,None,None,None,None)

                    self.assertEqual(1, plt_figure().add_subplot.call_count)
                    self.assertEqual(([1,2],[5,6],'-'), mock_ax.plot.call_args_list[0][0])
                    self.assertEqual('B'              , mock_ax.plot.call_args_list[0][1]["color"])
                    self.assertEqual(1                , mock_ax.plot.call_args_list[0][1]["alpha"])
                    self.assertEqual('L1'             , mock_ax.plot.call_args_list[0][1]["label"])
                    self.assertEqual(1                , mock_ax.plot.call_args_list[0][1]["zorder"])

                    self.assertEqual(([3,4],[7,8],'-'), mock_ax.plot.call_args_list[1][0])
                    self.assertEqual('R'              , mock_ax.plot.call_args_list[1][1]["color"])
                    self.assertEqual(.25              , mock_ax.plot.call_args_list[1][1]["alpha"])
                    self.assertEqual('L2'             , mock_ax.plot.call_args_list[1][1]["label"])
                    self.assertEqual(1                , mock_ax.plot.call_args_list[1][1]["zorder"])

                    mock_ax.set_xticks.called_once_with([2,2])
                    mock_ax.set_title.colled_once_with('title')
                    mock_ax.set_xlabel.called_once_with('xlabel')
                    mock_ax.set_ylabel.called_once_with('ylabel')

                    self.assertEqual(0, show.call_count)
                    self.assertEqual(0, savefig.call_count)

    def test_plot_bad_xlim(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        plotter = MatplotPlotter()
        plotter.plot(None, [[]], 'abc', 'def', 'efg', (1,0), (0,1), True, True, None, None, None, None)

        expected_log = "The xlim end is less than the xlim start. Plotting is impossible."

        self.assertEqual(expected_log, CobaContext.logger.sink.items[0])

    def test_plot_bad_ylim(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        plotter = MatplotPlotter()
        plotter.plot(None, [[]], 'abc', 'def', 'efg', (0,1), (1,0), True, True, None, None, None, None)

        expected_log = "The ylim end is less than the ylim start. Plotting is impossible."

        self.assertEqual(expected_log, CobaContext.logger.sink.items[0])

    def test_plot_filename(self):
        with unittest.mock.patch('matplotlib.pyplot.show') as show:
            with unittest.mock.patch('matplotlib.pyplot.savefig') as savefig:
                with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

                    mock_ax = plt_figure().add_subplot()

                    mock_ax.get_xticks.return_value = [1,2]
                    mock_ax.get_xlim.return_value   = [2,2]

                    lines = [
                        Points([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                        Points([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
                    ]

                    MatplotPlotter().plot(None,lines,"title","xlabel","ylabel",None,None,True,True,None,None,None,"abc")

                    plt_figure().add_subplot.assert_called_with(111)

                    self.assertEqual(([1,2],[5,6],'-'), mock_ax.plot.call_args_list[0][0])
                    self.assertEqual('B'              , mock_ax.plot.call_args_list[0][1]["color"])
                    self.assertEqual(1                , mock_ax.plot.call_args_list[0][1]["alpha"])
                    self.assertEqual('L1'             , mock_ax.plot.call_args_list[0][1]["label"])
                    self.assertEqual(1                , mock_ax.plot.call_args_list[0][1]["zorder"])

                    self.assertEqual(([3,4],[7,8],'-'), mock_ax.plot.call_args_list[1][0])
                    self.assertEqual('R'              , mock_ax.plot.call_args_list[1][1]["color"])
                    self.assertEqual(.25              , mock_ax.plot.call_args_list[1][1]["alpha"])
                    self.assertEqual('L2'             , mock_ax.plot.call_args_list[1][1]["label"])
                    self.assertEqual(1                , mock_ax.plot.call_args_list[1][1]["zorder"])

                    mock_ax.set_xticks.called_once_with([2,2])
                    mock_ax.set_title.colled_once_with('title')
                    mock_ax.set_xlabel.called_once_with('xlabel')
                    mock_ax.set_ylabel.called_once_with('ylabel')

                    self.assertEqual(1, show.call_count)
                    self.assertEqual(1, savefig.call_count)

    def test_plot_none(self):
        with unittest.mock.patch('matplotlib.pyplot.show') as show:
            with unittest.mock.patch('matplotlib.pyplot.savefig') as savefig:
                with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

                    mock_ax = plt_figure().add_subplot()

                    mock_ax.get_xticks.return_value = [1,2]
                    mock_ax.get_xlim.return_value   = [2,2]

                    lines = [
                        Points([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                        Points([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
                    ]

                    MatplotPlotter().plot(None,lines,"title","xlabel","ylabel",None,None,True,True,None,None,None,None)

                    plt_figure().add_subplot.assert_called_with(111)

                    self.assertEqual(([1,2],[5,6],'-'), mock_ax.plot.call_args_list[0][0])
                    self.assertEqual('B'              , mock_ax.plot.call_args_list[0][1]["color"])
                    self.assertEqual(1                , mock_ax.plot.call_args_list[0][1]["alpha"])
                    self.assertEqual('L1'             , mock_ax.plot.call_args_list[0][1]["label"])
                    self.assertEqual(1                , mock_ax.plot.call_args_list[0][1]["zorder"])

                    self.assertEqual(([3,4],[7,8],'-'), mock_ax.plot.call_args_list[1][0])
                    self.assertEqual('R'              , mock_ax.plot.call_args_list[1][1]["color"])
                    self.assertEqual(.25              , mock_ax.plot.call_args_list[1][1]["alpha"])
                    self.assertEqual('L2'             , mock_ax.plot.call_args_list[1][1]["label"])
                    self.assertEqual(1                , mock_ax.plot.call_args_list[1][1]["zorder"])

                    mock_ax.set_xticks.called_once_with([2,2])
                    mock_ax.set_title.colled_once_with('title')
                    mock_ax.set_xlabel.called_once_with('xlabel')
                    mock_ax.set_ylabel.called_once_with('ylabel')

                    self.assertEqual(0, show.call_count)
                    self.assertEqual(0, savefig.call_count)

    def test_plot_existing_figure(self):
        with unittest.mock.patch('matplotlib.pyplot.get_figlabels') as plt_get_figlabels:
            with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

                plt_get_figlabels.return_value = ['coba']

                lines = [
                    Points([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                    Points([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
                ]

                MatplotPlotter().plot(None,lines,"title","xlabel","ylabel",None,None,True,True,None,None,None,None)

                plt_figure.assert_called_with(num='coba')
                plt_figure().add_subplot.assert_called_with(111)

    def test_no_lines(self):
        with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

            CobaContext.logger = IndentLogger()
            CobaContext.logger.sink = ListSink()

            plotter = MatplotPlotter()
            plotter.plot(None, [], 'abc', 'def', 'efg', None, None, True, True, None, None, None, None)

            self.assertEqual(0, plt_figure().add_subplot.call_count)
            self.assertEqual(["No data was found for plotting."], CobaContext.logger.sink.items)

    def test_plot_with_existing_figure(self):

        import matplotlib.pyplot as plt

        lines = [
            Points([1,2], [5,6], None, None, "blue", 1.00, 'L1', '-', 1),
            Points([3,4], [7,8], None, None, "red", 0.25, 'L2', '-', 2)
        ]

        MatplotPlotter().plot(None,lines[:1],"title","xlabel","ylabel",None,None,True,True,None,None,None,None)
        MatplotPlotter().plot(None,lines[1:],"title","xlabel","ylabel",None,None,True,True,None,None,None,None)

        self.assertEqual('title' ,plt.gca().get_title(loc='left'))
        self.assertEqual('xlabel',plt.gca().get_xlabel())
        self.assertEqual('ylabel',plt.gca().get_ylabel())
        self.assertEqual('L1'    ,plt.gca().get_legend_handles_labels()[1][0])
        self.assertEqual('blue'  ,plt.gca().get_legend_handles_labels()[0][0].get_color())
        self.assertEqual(1.0     ,plt.gca().get_legend_handles_labels()[0][0].get_alpha())
        self.assertEqual('-'     ,plt.gca().get_legend_handles_labels()[0][0].get_linestyle())
        self.assertEqual(1       ,plt.gca().get_legend_handles_labels()[0][0].get_zorder())
        self.assertEqual('L2'    ,plt.gca().get_legend_handles_labels()[1][1])
        self.assertEqual('red'   ,plt.gca().get_legend_handles_labels()[0][1].get_color())
        self.assertEqual(0.25    ,plt.gca().get_legend_handles_labels()[0][1].get_alpha())
        self.assertEqual('-'     ,plt.gca().get_legend_handles_labels()[0][1].get_linestyle())
        self.assertEqual(2       ,plt.gca().get_legend_handles_labels()[0][1].get_zorder())

        plt.clf()

    def test_plot_with_empty_points(self):

        import matplotlib.pyplot as plt

        lines = [
            Points([], [], None, None, "blue", 1.00, 'L1', '-', 1),
            Points([3,4], [7,8], None, [0.1,0.2], "red", 0.25, 'L2', '.', 2),
            Points([], [], None, None, "blue", 1.00, 'L3', '-', 1),
        ]

        MatplotPlotter().plot(None,lines,"title","xlabel","ylabel",None,None,True,True,None,None,None,None)

        l2i = plt.gca().get_legend_handles_labels()[1].index("L2")

        self.assertEqual(True    ,plt.gca().get_legend_handles_labels()[0][l2i].has_yerr)
        self.assertEqual('title' ,plt.gca().get_title(loc='left'))
        self.assertEqual('xlabel',plt.gca().get_xlabel())
        self.assertEqual('ylabel',plt.gca().get_ylabel())
        self.assertEqual(['L1','L2','L3'],[t.get_text() for t in plt.gca().get_legend().get_texts()])

        plt.clf()

class Result_Tests(unittest.TestCase):

    def test_set_plotter(self):
        result = Result()
        self.assertIsNotNone(result._plotter)
        result.set_plotter(None)
        self.assertIsNone(result._plotter)

    def test_interaction_rows(self):
        result = Result(None,None,None,[['environment_id','learner_id','index','reward'],(0,1,1,1),(0,1,2,3),(0,2,1,1),(0,2,2,4)])
        self.assertEqual("{'Learners': 0, 'Environments': 0, 'Interactions': 4}", str(result))
        self.assertEqual( [(0,1,1,1),(0,1,2,3),(0,2,1,1),(0,2,2,4)], list(result.interactions))

    def test_has_preamble(self):
        self.assertDictEqual(Result(None,None,None,None,{"n_learners":1, "n_environments":2}).experiment, {"n_learners":1, "n_environments":2})

    def test_exception_when_no_file(self):
        with self.assertRaises(Exception):
            Result().from_file("abcd")

    def test_from_logged_envs(self):
        class Logged1:
            @property
            def params(self):
                return {"learner":{"family":"lrn1", "a":1}, "source":1, "logged":True, "scale":True }
            def read(self):
                yield {"reward": 1}
                yield {"reward": 2}

        class Logged2:
            @property
            def params(self):
                return {"learner":{"family":"lrn1", "a":1}, "source":[2], "logged":True, "scale":True, "batched": 2}
            def read(self):
                yield {"reward": [1,5]}
                yield {"reward": [2,6]}

        class Logged3:
            @property
            def params(self):
                return {"learner":{"family":"lrn2", "a":2}, "source":1, "logged":True, "scale":True}
            def read(self):
                yield {"reward": 5}
                yield {"reward": 6}

        class Logged4:
            @property
            def params(self):
                return {"learner":{"family":"lrn2", "a":2}, "source":[2], "logged":True, "scale":True, "batched": 2}
            def read(self):
                yield {"reward": 1}
                yield {"reward": 2}

        expected_envs = [
            {"environment_id":0, "source": 1 , "logged":True, "scale":True, "batched": None},
            {"environment_id":1, "source":[2], "logged":True, "scale":True, "batched": 2   },
        ]

        expected_lrns = [
            {"learner_id":0, "family":"lrn1", "a":1},
            {"learner_id":1, "family":"lrn2", "a":2},
        ]

        expected_vals = [
            {'evaluator_id':0, 'eval_type':'unknown'}
        ]

        expected_ints = [
            {'environment_id': 0, 'learner_id': 0, 'evaluator_id': 0, 'index':1, 'reward': 1 },
            {'environment_id': 0, 'learner_id': 0, 'evaluator_id': 0, 'index':2, 'reward': 2 },
            {'environment_id': 0, 'learner_id': 1, 'evaluator_id': 0, 'index':1, 'reward': 5 },
            {'environment_id': 0, 'learner_id': 1, 'evaluator_id': 0, 'index':2, 'reward': 6 },
            {'environment_id': 1, 'learner_id': 0, 'evaluator_id': 0, 'index':1, 'reward': 3 },
            {'environment_id': 1, 'learner_id': 0, 'evaluator_id': 0, 'index':2, 'reward': 4 },
            {'environment_id': 1, 'learner_id': 1, 'evaluator_id': 0, 'index':1, 'reward': 1 },
            {'environment_id': 1, 'learner_id': 1, 'evaluator_id': 0, 'index':2, 'reward': 2 },
        ]

        res = Result().from_logged_envs([Logged1(),Logged2(),Logged3(),Logged4()])
        self.assertCountEqual(res.environments.to_dicts(), expected_envs)
        self.assertCountEqual(res.learners.to_dicts()    , expected_lrns)
        self.assertCountEqual(res.evaluators.to_dicts()  , expected_vals)
        self.assertCountEqual(res.interactions.to_dicts(), expected_ints)

    def test_filter_fin_sans_n_interactions(self):
        envs = [['environment_id'],[1],[2]]
        lrns = [['learner_id'],[1],[2]]
        vals = [['evaluator_id'],[1],[2]]
        ints = [['environment_id','learner_id','evaluator_id','index'],[1,1,1,0],[1,2,1,0],[2,1,1,0]]

        original_result = Result(envs, lrns, vals, ints)
        filtered_result = original_result.filter_fin(l='learner_id',p='environment_id')

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(2, len(original_result.evaluators))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(1, len(filtered_result.environments))
        self.assertEqual(2, len(filtered_result.learners))
        self.assertEqual(1, len(filtered_result.evaluators))
        self.assertEqual(2, len(filtered_result.interactions))

    def test_filter_fin_removes_when_no_interactions(self):
        envs = [['environment_id'],[1],[2]]
        lrns = [['learner_id'],[1],[2]]
        vals = [['evaluator_id'],[1],[2]]
        ints = [['environment_id','learner_id','evaluator_id','index'],[1,1,1,0],[1,2,1,0]]

        original_result = Result(envs, lrns, vals, ints)
        filtered_result = original_result.filter_fin(l='learner_id',p='environment_id')

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(2, len(original_result.evaluators))
        self.assertEqual(2, len(original_result.interactions))

        self.assertEqual(1, len(filtered_result.environments))
        self.assertEqual(2, len(filtered_result.learners))
        self.assertEqual(1, len(filtered_result.evaluators))
        self.assertEqual(2, len(filtered_result.interactions))

    def test_filter_fin_removes_all(self):
        envs = [['environment_id'],[1],[2]]
        lrns = [['learner_id'],[1],[2]]
        vals = [['evaluator_id'],[1],[2]]
        ints = [['environment_id','learner_id','evaluator_id','index'],[1,1,1,0],[2,2,1,0]]

        original_result = Result(envs, lrns, vals, ints)
        filtered_result = original_result.filter_fin(l='learner_id',p='environment_id')

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(2, len(original_result.evaluators))
        self.assertEqual(2, len(original_result.interactions))

        self.assertEqual(0, len(filtered_result.environments))
        self.assertEqual(0, len(filtered_result.learners))
        self.assertEqual(0, len(filtered_result.evaluators))
        self.assertEqual(0, len(filtered_result.interactions))

    def test_filter_fin_with_n_and_default(self):
        envs = [['environment_id'],[1],[2]]
        lrns = [['learner_id'    ],[1],[2]]
        vals = [['evaluator_id'  ],[1]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [1,1,1,1,1],[1,1,1,2,2],
            [1,2,1,1,1],[1,2,1,2,2],[1,2,1,3,3],
            [2,1,1,1,1],
            [2,2,1,1,1]
        ]

        original_result = Result(envs, lrns, vals, ints)
        filtered_result = original_result.filter_fin(2)

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(7, len(original_result.interactions))

        self.assertEqual(1, len(filtered_result.environments))
        self.assertEqual(2, len(filtered_result.learners))
        self.assertEqual(4, len(filtered_result.interactions))

    def test_filter_fin_with_n_1(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[1],[2]]
        lrns = [['learner_id'    ],[1],[2]]
        vals = [['evaluator_id'  ],[1]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [1,1,1,1,1],[1,1,1,2,2],
            [1,2,1,1,1],[1,2,1,2,2],[1,2,1,3,3],
            [2,1,1,1,1],[2,1,1,2,1],
            [2,2,1,1,1]
        ]

        original_result = Result(envs, lrns, vals, ints)
        filtered_result = original_result.filter_fin(2)

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(8, len(original_result.interactions))

        self.assertEqual(2, len(filtered_result.environments))
        self.assertEqual(2, len(filtered_result.learners))
        self.assertEqual(6, len(filtered_result.interactions))

        self.assertEqual("We removed 1 learner evaluation because it was shorter than 2 interactions.", CobaContext.logger.sink.items[0])

    def test_filter_fin_with_n_2(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[1],[2]]
        lrns = [['learner_id'    ],[1],[2],[3]]
        vals = [['evaluator_id'  ],[1],[2]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [1,1,1,1,1],[1,1,1,2,2],
            [1,2,1,1,1],[1,2,1,2,2],[1,2,1,3,3],
            [2,1,1,1,1],
            [2,2,1,1,1],
            [2,3,2,1,1],
        ]

        original_result = Result(envs, lrns, vals, ints)
        filtered_result = original_result.filter_fin(2)

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(3, len(original_result.learners))
        self.assertEqual(2, len(original_result.evaluators))
        self.assertEqual(8, len(original_result.interactions))

        self.assertEqual(1, len(filtered_result.environments))
        self.assertEqual(2, len(filtered_result.learners))
        self.assertEqual(1, len(filtered_result.evaluators))
        self.assertEqual(4, len(filtered_result.interactions))

        self.assertEqual("We removed 3 learner evaluations because they were shorter than 2 interactions.", CobaContext.logger.sink.items[0])

    def test_filter_fin_with_n_3(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[1],[2]]
        lrns = [['learner_id'    ],[1],[2],[3]]
        vals = [['evaluator_id'  ],[1],[2]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [1,1,1,1,1],[1,1,1,2,2],[1,1,1,3,2],
            [1,2,1,1,1],[1,2,1,2,2],[1,2,1,3,3],
            [2,1,1,1,1],
            [2,2,1,1,1],
            [2,3,2,1,1],
        ]

        original_result = Result(envs, lrns, vals, ints)
        filtered_result = original_result.filter_fin(2)

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(3, len(original_result.learners))
        self.assertEqual(2, len(original_result.evaluators))
        self.assertEqual(9, len(original_result.interactions))

        self.assertEqual(1, len(filtered_result.environments))
        self.assertEqual(2, len(filtered_result.learners))
        self.assertEqual(1, len(filtered_result.evaluators))
        self.assertEqual(4, len(filtered_result.interactions))

        self.assertEqual("We removed 3 learner evaluations because they were shorter than 2 interactions.", CobaContext.logger.sink.items[0])

    def test_filter_fin_no_finished(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[1],[2]]
        lrns = [['learner_id'    ],[1],[2]]
        vals = [['evaluator_id'  ],[1]]
        ints = [['environment_id','learner_id','evaluator_id','index'],[1,1,1,0],[2,2,1,0]]

        original_result = Result(envs, lrns, vals, ints)
        filtered_result = original_result.filter_fin(l='learner_id',p='environment_id')

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(2, len(original_result.interactions))

        self.assertEqual(0, len(filtered_result.environments))
        self.assertEqual(0, len(filtered_result.learners))
        self.assertEqual(0, len(filtered_result.interactions))
        self.assertEqual("We removed 2 environment_id because they did not exist for every learner_id.", CobaContext.logger.sink.items[0])
        self.assertEqual("There was no environment_id which was finished for every learner_id.", CobaContext.logger.sink.items[1])

    def test_filter_fin_multi_p(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id','data_id','seed'                   ],[1,2,1],[2,2,2]]
        lrns = [['learner_id'                                        ],[1],[2]]
        vals = [['evaluator_id'                                      ],[1]]
        ints = [['environment_id','learner_id','evaluator_id','index'],[1,1,1,0],[1,2,1,0],[2,1,1,0],[2,2,1,0]]

        original_result = Result(envs, lrns, vals, ints)
        filtered_result = original_result.filter_fin(l='learner_id',p='data_id')

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(1, len(original_result.evaluators))
        self.assertEqual(4, len(original_result.interactions))

        self.assertEqual(0, len(filtered_result.environments))
        self.assertEqual(0, len(filtered_result.learners))
        self.assertEqual(0, len(filtered_result.evaluators))
        self.assertEqual(0, len(filtered_result.interactions))
        self.assertEqual("We removed 1 data_id because more than one existed for each learner_id.", CobaContext.logger.sink.items[0])
        self.assertEqual("There was no data_id which was finished for every learner_id.", CobaContext.logger.sink.items[1])

    def test_filter_env(self):

        envs = [['environment_id'],[1],[2]]
        lrns = [['learner_id'    ],[1],[2]]
        vals = [['evaluator_id'  ],[1],[2]]
        ints = [['environment_id','learner_id','evaluator_id'],[1,1,1],[2,1,1],[1,2,1]]

        original_result = Result(envs, lrns, vals, ints)
        filtered_result = original_result.filter_env(environment_id=2)

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(2, len(original_result.evaluators))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(1, len(filtered_result.environments))
        self.assertEqual(1, len(filtered_result.learners))
        self.assertEqual(1, len(filtered_result.evaluators))
        self.assertEqual(1, len(filtered_result.interactions))

    def test_filter_env_no_change(self):

        envs = [['environment_id'],[1],[2]]
        lrns = [['learner_id'    ],[1],[2]]
        vals = [['evaluator_id'  ],[1]    ]
        ints = [['environment_id','learner_id','evaluator_id'],[1,1,1],[2,1,1],[1,2,1]]

        original_result = Result(envs, lrns, vals, ints)
        self.assertIs(original_result, original_result.filter_env(environment_id=[1,2]))

    def test_filter_env_no_match(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[1],[2]]
        lrns = [['learner_id'    ],[1],[2]]
        vals = [['evaluator_id'  ],[1]    ]
        ints = [['environment_id','learner_id','evaluator_id'],[1,1,1],[2,1,1],[1,2,1]]

        original_result = Result(envs, lrns, vals, ints)
        filtered_result = original_result.filter_env(environment_id=3)

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(0, len(filtered_result.environments))
        self.assertEqual(0, len(filtered_result.learners))
        self.assertEqual(0, len(filtered_result.interactions))
        self.assertEqual(["No environments matched the given filter."], CobaContext.logger.sink.items)

    def test_filter_lrn_1(self):

        envs = [['environment_id'],[1],[2]]
        lrns = [['learner_id'    ],[1],[2]]
        vals = [['evaluator_id'  ],[1],[2]]
        ints = [['environment_id','learner_id','evaluator_id'],[1,1,1],[2,1,1],[1,2,1]]

        original_result = Result(envs, lrns, vals, ints)
        filtered_result = original_result.filter_lrn(learner_id=2)

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(2, len(original_result.evaluators))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(1, len(filtered_result.environments))
        self.assertEqual(1, len(filtered_result.learners))
        self.assertEqual(1, len(filtered_result.evaluators))
        self.assertEqual(1, len(filtered_result.interactions))

    def test_filter_lrn_2(self):

        envs = [['environment_id'],[1],[2]    ]
        lrns = [['learner_id'    ],[1],[2],[3]]
        vals = [['evaluator_id'  ],[1],[2],[3]]
        ints = [['environment_id','learner_id','evaluator_id'],[1,1,1],[1,2,1],[2,3,2]]

        original_result = Result(envs, lrns, vals, ints)
        filtered_result = original_result.filter_lrn(learner_id=[2,1])

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(3, len(original_result.learners))
        self.assertEqual(3, len(original_result.evaluators))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(1, len(filtered_result.environments))
        self.assertEqual(2, len(filtered_result.learners))
        self.assertEqual(1, len(filtered_result.evaluators))
        self.assertEqual(2, len(filtered_result.interactions))

    def test_filter_lrn_no_match(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[1],[2]    ]
        lrns = [['learner_id'    ],[1],[2],[3]]
        vals = [['evaluator_id'  ],[1]        ]
        ints = [['environment_id','learner_id','evaluator_id'],[1,1,1],[1,2,1],[2,3,1]]

        original_result = Result(envs, lrns, vals, ints)
        filtered_result = original_result.filter_lrn(learner_id=5)

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(3, len(original_result.learners))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(0, len(filtered_result.environments))
        self.assertEqual(0, len(filtered_result.learners))
        self.assertEqual(0, len(filtered_result.interactions))
        self.assertEqual(["No learners matched the given filter."], CobaContext.logger.sink.items)

    def test_filter_lrn_no_change(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[1],[2]    ]
        lrns = [['learner_id'    ],[1],[2],[3]]
        vals = [['evaluator_id'  ],[1]        ]
        ints = [['environment_id','learner_id','evaluator_id'],[1,1,1],[1,2,1],[2,3,1]]

        original_result = Result(envs, lrns, vals, ints)
        self.assertIs(original_result, original_result.filter_lrn(learner_id=[1,2,3]))

    def test_filter_val(self):

        envs = [['environment_id'],[1],[2]]
        lrns = [['learner_id'    ],[1],[2]]
        vals = [['evaluator_id'  ],[1],[2]]
        ints = [['environment_id','learner_id','evaluator_id'],[1,1,1],[2,1,1],[1,2,2]]

        original_result = Result(envs, lrns, vals, ints)
        filtered_result = original_result.filter_val(evaluator_id=1)

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(2, len(original_result.evaluators))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(2, len(filtered_result.environments))
        self.assertEqual(1, len(filtered_result.learners))
        self.assertEqual(1, len(filtered_result.evaluators))
        self.assertEqual(2, len(filtered_result.interactions))

    def test_filter_val_no_change(self):

        envs = [['environment_id'],[1],[2]]
        lrns = [['learner_id'    ],[1],[2]]
        vals = [['evaluator_id'  ],[1]    ]
        ints = [['environment_id','learner_id','evaluator_id'],[1,1,1],[2,1,1],[1,2,1]]

        original_result = Result(envs, lrns, vals, ints)
        self.assertIs(original_result, original_result.filter_val(evaluator_id=1))

    def test_filter_val_no_match(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[1],[2]]
        lrns = [['learner_id'    ],[1],[2]]
        vals = [['evaluator_id'  ],[1]    ]
        ints = [['environment_id','learner_id','evaluator_id'],[1,1,1],[2,1,1],[1,2,1]]

        original_result = Result(envs, lrns, vals, ints)
        filtered_result = original_result.filter_val(evaluator_id=3)

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(0, len(filtered_result.environments))
        self.assertEqual(0, len(filtered_result.learners))
        self.assertEqual(0, len(filtered_result.interactions))
        self.assertEqual(["No evaluators matched the given filter."], CobaContext.logger.sink.items)

    def test_filter_int_no_change(self):

        envs = [['environment_id'],[1],[2]]
        lrns = [['learner_id'    ],[1],[2]]
        vals = [['evaluator_id'  ],[1]    ]
        ints = [['environment_id','learner_id','evaluator_id','index'],[1,1,1,0],[2,1,1,0],[1,2,1,0]]

        original_result = Result(envs, lrns, vals, ints)
        self.assertIs(original_result, original_result.filter_int(index=0))

    def test_filter_int_no_match(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[1],[2]]
        lrns = [['learner_id'    ],[1],[2]]
        vals = [['evaluator_id'  ],[1]    ]
        ints = [['environment_id','learner_id','evaluator_id','index'],[1,1,1,0],[2,1,1,0],[1,2,1,0]]

        original_result = Result(envs, lrns, vals, ints)
        filtered_result = original_result.filter_int(index=1)

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(0, len(filtered_result.environments))
        self.assertEqual(0, len(filtered_result.learners))
        self.assertEqual(0, len(filtered_result.interactions))
        self.assertEqual(["No interactions matched the given filter."], CobaContext.logger.sink.items)

    def test_where(self):
        envs = [['environment_id'],[1],[2],[3],[4]]
        lrns = [['learner_id'    ],[1],[2],[3],[4]]
        vals = [['evaluator_id'  ],[1],[2],[3],[4]]
        ints = [['environment_id','learner_id','evaluator_id','index'],[1,1,1,0],[2,2,2,0],[3,3,3,1],[4,4,4,1]]

        result = Result(envs, lrns, vals, ints)

        self.assertEqual(4, len(result.environments))
        self.assertEqual(4, len(result.learners))
        self.assertEqual(4, len(result.evaluators))
        self.assertEqual(4, len(result.interactions))

        result = result.where(environment_id={'!=':1})

        self.assertEqual(3, len(result.environments))
        self.assertEqual(3, len(result.learners))
        self.assertEqual(3, len(result.evaluators))
        self.assertEqual(3, len(result.interactions))

        result = result.where(learner_id={'!=':2})

        self.assertEqual(2, len(result.environments))
        self.assertEqual(2, len(result.learners))
        self.assertEqual(2, len(result.evaluators))
        self.assertEqual(2, len(result.interactions))

        result = result.where(evaluator_id={'!=':3})

        self.assertEqual(1, len(result.environments))
        self.assertEqual(1, len(result.learners))
        self.assertEqual(1, len(result.evaluators))
        self.assertEqual(1, len(result.interactions))

        result = Result(envs, lrns, vals, ints).where(index={'>':0})

        self.assertEqual(2, len(result.environments))
        self.assertEqual(2, len(result.learners))
        self.assertEqual(2, len(result.evaluators))
        self.assertEqual(2, len(result.interactions))

        with self.assertRaises(CobaException):
            Result(envs, lrns, vals, ints).where(abc=0)

    def test_copy(self):

        envs = [['environment_id'],[1],[2]    ]
        lrns = [['learner_id'    ],[1],[2],[3]]
        vals = [['evaluator_id'  ],[1]        ]
        ints = [['environment_id','learner_id','evaluator_id'],[1,1,1],[1,2,1],[2,3,1]]

        result = Result(envs, lrns, vals, ints)
        result_copy = result.copy()

        self.assertIsNot(result, result_copy)

        self.assertIsNot(result.environments, result_copy.environments)
        self.assertIsNot(result.learners, result_copy.learners)
        self.assertIsNot(result.evaluators, result_copy.evaluators)
        self.assertIsNot(result.interactions, result_copy.interactions)

        self.assertEqual(list(result.environments), list(result_copy.environments))
        self.assertEqual(list(result.learners), list(result_copy.learners))
        self.assertEqual(list(result.evaluators), list(result_copy.evaluators))
        self.assertEqual(list(result.interactions), list(result_copy.interactions))

    def test_str(self):

        envs = [['environment_id'],[1],[2]    ]
        lrns = [['learner_id'    ],[1],[2],[3]]
        vals = [['evaluator_id'  ],[1]        ]
        ints = [['environment_id','learner_id','evaluator_id'],[1,1,1],[1,2,1],[2,3,1]]

        self.assertEqual("{'Learners': 3, 'Environments': 2, 'Interactions': 3}", str(Result(envs, lrns, vals, ints)))

    def test_ipython_display_(self):

        with unittest.mock.patch("builtins.print") as mock:

            envs = [['environment_id'],[1],[2]]
            lrns = [['learner_id'    ],[1],[2],[3]]
            vals = [['evaluator_id'],[0]]
            ints = [['environment_id','learner_id','evaluator_id'],[1,1,0],[1,2,0],[2,3,0]]

            result = Result(envs, lrns, vals, ints)
            result._ipython_display_()
            mock.assert_called_once_with(str(result))

    def test_raw_learners_all_default(self):
        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
                [0,1,0,1,1],[0,1,0,2,2],
                [0,2,0,1,1],[0,2,0,2,2],
                [1,1,0,1,1],[1,1,0,2,2],
                [1,2,0,1,1],[1,2,0,2,2],
        ]

        table = Result(envs, lrns, vals, ints).raw_learners()
        self.assertEqual(('p','x','1. learner_1','2. learner_2'), table.columns)
        self.assertEqual([(0,1,1,1),(1,1,1,1),(0,2,1.5,1.5),(1,2,1.5,1.5)], list(table))

    def test_raw_contrast_all_default(self):
        envs = [['environment_id'],[0]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
                [0,1,0,1,1],[0,1,0,2,2],
                [0,2,0,1,1],[0,2,0,2,2]
        ]

        table = Result(envs, lrns, vals, ints).raw_contrast(1,2)
        self.assertEqual(('p','x','1. learner_1','2. learner_2'), table.columns)
        self.assertEqual([(0,0,1.5,1.5)], list(table))

    def test_raw_contrast_index(self):
        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
                [0,1,0,1,1],[0,1,0,2,2],
                [0,2,0,1,1],[0,2,0,2,2],
                [1,1,0,1,1],[1,1,0,2,2],
                [1,2,0,1,1],[1,2,0,2,2]
        ]

        table = Result(envs, lrns, vals, ints).raw_contrast(1,2,x='index')
        self.assertEqual(('p','x','l1','l2'), table.columns)
        self.assertEqual([(0,1,1,1),(1,1,1,1),(0,2,1.5,1.5),(1,2,1.5,1.5)], list(table))

    def test_raw_contrast_bad_l(self):
        envs = [['environment_id'],[0]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
                [0,1,0,1,1],[0,1,0,2,2],
                [0,2,0,1,1],[0,2,0,2,2]
        ]

        with self.assertRaises(CobaException):
            table = Result(envs, lrns, vals, ints).raw_contrast(1,1,x='index')

    def test_raw_contrast_no_pair(self):
        envs = [['environment_id'],[0]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
                [0,1,0,1,1],[0,1,0,2,2],
        ]

        with self.assertRaises(CobaException):
            table = Result(envs, lrns, vals, ints).raw_contrast(1,2,x='index')

    def test_plot_learners_bad_x_index(self):
        CobaContext.logger.sink = ListSink()
        Result().plot_learners(x=['index','a'])
        self.assertIn("The x-axis cannot contain", CobaContext.logger.sink.items[0])

    def test_plot_learners_all_default(self):
        envs = [['environment_id'],[0]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],[0,1,0,1,1],[0,1,0,2,2],[0,2,0,1,1],[0,2,0,2,2]]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_learners()

        expected_lines = [
            Points([1,2],[1,1.5],[],[0,0],0,1,'1. learner_1','-', 1),
            Points([1,2],[1,1.5],[],[0,0],1,1,'2. learner_2','-', 1)
        ]

        self.assertEqual("Progressive Reward (1 Environments)", plotter.plot_calls[0][2])
        self.assertEqual("Interaction", plotter.plot_calls[0][3])
        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_xlabel_ylabel(self):
        envs = [['environment_id'],[0]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],[0,1,0,1,1],[0,1,0,2,2],[0,2,0,1,1],[0,2,0,2,2]]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_learners(xlabel='x',ylabel='y')

        expected_lines = [
            Points([1,2],[1,1.5],[],[0,0],0,1,'1. learner_1','-', 1),
            Points([1,2],[1,1.5],[],[0,0],1,1,'2. learner_2','-', 1)
        ]

        self.assertEqual("Progressive y (1 Environments)", plotter.plot_calls[0][2])
        self.assertEqual("x", plotter.plot_calls[0][3])
        self.assertEqual("y", plotter.plot_calls[0][4])
        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_title(self):
        envs = [['environment_id'],[0]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],[0,1,0,1,1],[0,1,0,2,2],[0,2,0,1,1],[0,2,0,2,2]]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_learners(title='abc')

        expected_lines = [
            Points([1,2],[1,1.5],[],[0,0],0,1,'1. learner_1','-', 1),
            Points([1,2],[1,1.5],[],[0,0],1,1,'2. learner_2','-', 1)
        ]

        self.assertEqual("abc", plotter.plot_calls[0][2])
        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_all_str_to_none(self):
        envs = [['environment_id'],[0]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,None]]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],[0,1,0,1,1],[0,1,0,2,2],[0,2,0,1,1],[0,2,0,2,2]]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_learners(l='family')

        expected_lines = [
            Points([1,2],[1,1.5],[],[0,0],0,1,'learner_1','-', 1),
            Points([1,2],[1,1.5],[],[0,0],1,1,'None','-', 1)
        ]

        self.assertEqual("Progressive Reward (1 Environments)", plotter.plot_calls[0][2])
        self.assertEqual("Interaction", plotter.plot_calls[0][3])
        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_all_str_to_int(self):
        envs = [['environment_id'],[0]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,1337]]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],[0,1,0,1,1],[0,1,0,2,2],[0,2,0,1,1],[0,2,0,2,2]]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_learners(l='family')

        expected_lines = [
            Points([1,2],[1,1.5],[],[0,0],0,1,1337       ,'-', 1),
            Points([1,2],[1,1.5],[],[0,0],1,1,'learner_1','-', 1),
        ]

        self.assertEqual("Progressive Reward (1 Environments)", plotter.plot_calls[0][2])
        self.assertEqual("Interaction", plotter.plot_calls[0][3])
        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_with_l_sequence(self):
        envs = [['environment_id'],[0]]
        lrns = [['learner_id', 'family','a','b'],[1,'learner_1',1,2],[2,'learner_2',3,4]]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],[0,1,0,1,1],[0,1,0,2,2],[0,2,0,1,1],[0,2,0,2,2]]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_learners(l=['a','b'])

        expected_lines = [
            Points([1,2],[1,1.5],[],[0,0],0,1,(1,2),'-', 1),
            Points([1,2],[1,1.5],[],[0,0],1,1,(3,4),'-', 1)
        ]

        self.assertEqual("Progressive Reward (1 Environments)", plotter.plot_calls[0][2])
        self.assertEqual("Interaction", plotter.plot_calls[0][3])
        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_one_environment_err_se(self):
        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],[0,1,0,1,1],[0,1,0,2,2],[1,1,0,1,3],[1,1,0,2,4]]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_learners(err='sd',errevery=2)

        expected_lines = [
            Points([1,2],[2,2.5],[],[(0,0),(1.41421,1.41421)],0,1,'1. learner_1','-', 1),
        ]

        self.assertEqual("Progressive Reward (2 Environments)", plotter.plot_calls[0][2])
        self.assertEqual("Interaction", plotter.plot_calls[0][3])

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_one_environment_lrn_params(self):
        envs = [['environment_id'],[0]]
        lrns = [['learner_id', 'family','i','j','t'],[1,'learner_1',1,2,None],[2,'learner_2',None,None,2]]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],[0,1,0,1,1],[0,1,0,2,2],[0,2,0,1,1],[0,2,0,2,2]]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_learners()

        expected_lines = [
            Points([1,2],[1.,1.5],[],[0,0],0,1,'1. learner_1(i=1,j=2)','-', 1),
            Points([1,2],[1.,1.5],[],[0,0],1,1,'2. learner_2(t=2)','-', 1)
        ]

        self.assertEqual("Progressive Reward (1 Environments)", plotter.plot_calls[0][2])
        self.assertEqual("Interaction", plotter.plot_calls[0][3])

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_one_environment_val_params(self):
        envs = [['environment_id'],[0]]
        lrns = [['learner_id'],[1]]
        vals = [['evaluator_id','eval_type'],[0,'a'],[1,'b']]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],[0,1,0,1,1],[0,1,0,2,2],[0,1,1,1,1],[0,1,1,2,2]]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_learners(l='eval_type')

        expected_lines = [
            Points([1,2],[1.,1.5],[],[0,0],0,1,'a','-', 1),
            Points([1,2],[1.,1.5],[],[0,0],1,1,'b','-', 1)
        ]

        self.assertEqual("Progressive Reward (1 Environments)", plotter.plot_calls[0][2])
        self.assertEqual("Interaction", plotter.plot_calls[0][3])

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_two_environments_all_default(self):
        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,1],[0,1,0,2,2],
            [0,2,0,1,1],[0,2,0,2,2],
            [1,1,0,1,2],[1,1,0,2,3],
            [1,2,0,1,2],[1,2,0,2,3],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_learners()

        expected_lines = [
            Points([1,2],[3/2,4/2],[],[0,0],0,1,'1. learner_1','-', 1),
            Points([1,2],[3/2,4/2],[],[0,0],1,1,'2. learner_2','-', 1)
        ]

        self.assertEqual("Progressive Reward (2 Environments)", plotter.plot_calls[0][2])
        self.assertEqual("Interaction", plotter.plot_calls[0][3])

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_one_environment_x_not_index(self):
        envs = [['environment_id'],[0]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,1],[0,1,0,2,2],
            [0,2,0,1,1],[0,2,0,2,2],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_learners(x='environment_id')

        expected_lines = [
            Points([0],[1.5],[],[0],0,1,'1. learner_1','.',1),
            Points([0],[1.5],[],[0],1,1,'2. learner_2','.',1)
        ]

        self.assertEqual("Total Progressive Reward (1 Environments)", plotter.plot_calls[0][2])
        self.assertEqual("environment_id", plotter.plot_calls[0][3])

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_three_environments_err_sd(self):
        envs = [['environment_id'],[0],[1],[2]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,1],[0,1,0,2,2],
            [0,2,0,1,2],[0,2,0,2,3],
            [1,1,0,1,2],[1,1,0,2,3],
            [1,2,0,1,3],[1,2,0,2,4],
            [2,1,0,1,3],[2,1,0,2,4],
            [2,2,0,1,4],[2,2,0,2,5],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_learners(err='sd')

        expected_lines = [
            Points([1,2],[3,3.5],[],[(1,1),(1,1)],1,1,'2. learner_2','-', 1),
            Points([1,2],[2,2.5],[],[(1,1),(1,1)],0,1,'1. learner_1','-', 1),
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_mixed_env_count(self):
        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,1],[0,1,0,2,2],[0,1,0,3,3],
            [1,1,0,1,2],[1,1,0,2,3],
            [1,2,0,1,2],[1,2,0,2,3],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_learners()

        expected_logs = ["We removed 1 environment_id because it did not exist for every full_name."]
        expected_lines = [
            Points([1,2],[2,5/2],[],[0,0],0,1,'1. learner_1','-', 1),
            Points([1,2],[2,5/2],[],[0,0],1,1,'2. learner_2','-', 1)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])
        self.assertEqual(expected_logs, CobaContext.logger.sink.items)

    def test_plot_learners_mixed_env_length(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,1],[0,1,0,2,2],[0,1,0,3,3],
            [0,2,0,1,1],[0,2,0,2,2],[0,2,0,3,3],
            [1,1,0,1,2],[1,1,0,2,3],
            [1,2,0,1,2],[1,2,0,2,3],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_learners()

        expected_logs = ['We shortened 2 environments because they were longer than the shortest environment.']
        expected_lines = [
            Points([1,2],[3/2,4/2],[],[0,0],0,1,'1. learner_1','-', 1),
            Points([1,2],[3/2,4/2],[],[0,0],1,1,'2. learner_2','-', 1)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])
        self.assertEqual(expected_logs, CobaContext.logger.sink.items)

    def test_plot_learners_filename(self):
        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,1],[0,1,0,2,2],
            [0,2,0,1,1],[0,2,0,2,2],
            [1,1,0,1,2],[1,1,0,2,3],
            [1,2,0,1,2],[1,2,0,2,3],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_learners(out="abc")

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual("abc", plotter.plot_calls[0][12])

    def test_plot_learners_ax(self):
        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,1],[0,1,0,2,2],
            [0,2,0,1,1],[0,2,0,2,2],
            [1,1,0,1,2],[1,1,0,2,3],
            [1,2,0,1,2],[1,2,0,2,3],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_learners(ax=1)

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(1, plotter.plot_calls[0][0])

    def test_plot_learners_xlim_ylim(self):
        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,1],[0,1,0,2,2],
            [0,2,0,1,1],[0,2,0,2,2],
            [1,1,0,1,2],[1,1,0,2,3],
            [1,2,0,1,2],[1,2,0,2,3],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_learners(xlim=(1,2), ylim=(2,3))

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual((1,2), plotter.plot_calls[0][5])
        self.assertEqual((2,3), plotter.plot_calls[0][6])

    def test_plot_learners_labels_int_colors(self):
        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,1],[0,1,0,2,2],
            [0,2,0,1,1],[0,2,0,2,2],
            [1,1,0,1,2],[1,1,0,2,3],
            [1,2,0,1,2],[1,2,0,2,3],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_learners(labels=['a','b'],colors=3)

        expected_lines = [
            Points([1,2],[3/2,4/2],[],[0,0],3,1,'a','-', 1),
            Points([1,2],[3/2,4/2],[],[0,0],4,1,'b','-', 1)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_labels_no_colors(self):
        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,1],[0,1,0,2,2],
            [0,2,0,1,1],[0,2,0,2,2],
            [1,1,0,1,2],[1,1,0,2,3],
            [1,2,0,1,2],[1,2,0,2,3],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_learners(labels=['a','b'])

        expected_lines = [
            Points([1,2],[3/2,4/2],[],[0,0],0,1,'a','-', 1),
            Points([1,2],[3/2,4/2],[],[0,0],1,1,'b','-', 1)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_labels_one_color(self):
        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,1],[0,1,0,2,2],
            [0,2,0,1,1],[0,2,0,2,2],
            [1,1,0,1,2],[1,1,0,2,3],
            [1,2,0,1,2],[1,2,0,2,3],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_learners(labels=['a','b'],colors=1)

        expected_lines = [
            Points([1,2],[3/2,4/2],[],[0,0],1,1,'a','-', 1),
            Points([1,2],[3/2,4/2],[],[0,0],2,1,'b','-', 1)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_bad_labels(self):
        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,1],[0,1,0,2,2],
            [0,2,0,1,1],[0,2,0,2,2],
            [1,1,0,1,2],[1,1,0,2,3],
            [1,2,0,1,2],[1,2,0,2,3],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_learners(labels=['a'],colors=1)

        expected_lines = [
            Points([1,2],[3/2,4/2],[],[0,0],1,1,'a'           ,'-', 1),
            Points([1,2],[3/2,4/2],[],[0,0],2,1,'2. learner_2','-', 1)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_one_int_color(self):
        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,1],[0,1,0,2,2],
            [0,2,0,1,1],[0,2,0,2,2],
            [1,1,0,1,2],[1,1,0,2,3],
            [1,2,0,1,2],[1,2,0,2,3],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_learners(colors=2)

        expected_lines = [
            Points([1,2],[3/2,4/2],[],[0,0],2,1,'1. learner_1','-', 1),
            Points([1,2],[3/2,4/2],[],[0,0],3,1,'2. learner_2','-', 1)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_one_list_color(self):
        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,1],[0,1,0,2,2],
            [0,2,0,1,1],[0,2,0,2,2],
            [1,1,0,1,2],[1,1,0,2,3],
            [1,2,0,1,2],[1,2,0,2,3],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_learners(colors=[2])

        expected_lines = [
            Points([1,2],[3/2,4/2],[],[0,0],2,1,'1. learner_1','-', 1),
            Points([1,2],[3/2,4/2],[],[0,0],3,1,'2. learner_2','-', 1)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_top_n_positive(self):
        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2'],[3,'learner_3']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,1],[0,1,0,2,2],
            [0,2,0,1,3],[0,2,0,2,4],
            [0,3,0,1,5],[0,3,0,2,6],
            [1,1,0,1,1],[1,1,0,2,2],
            [1,2,0,1,3],[1,2,0,2,4],
            [1,3,0,1,5],[1,3,0,2,6],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_learners(top_n=2)

        expected_lines = [
            Points([1,2],[5,5.5],[],[0,0],2,1,'3. learner_3','-', 1),
            Points([1,2],[3,3.5],[],[0,0],1,1,'2. learner_2','-', 1)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_top_n_negative(self):
        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2'],[3,'learner_3']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,1],[0,1,0,2,2],
            [0,2,0,1,3],[0,2,0,2,4],
            [0,3,0,1,5],[0,3,0,2,6],
            [1,1,0,1,1],[1,1,0,2,2],
            [1,2,0,1,3],[1,2,0,2,4],
            [1,3,0,1,5],[1,3,0,2,6],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_learners(top_n=-2)

        expected_lines = [
            Points([1,2],[3,3.5],[],[0,0],1,1,'2. learner_2','-', 1),
            Points([1,2],[1,1.5],[],[0,0],0,1,'1. learner_1','-', 1)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertSequenceEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_empty_results(self):
        plotter = TestPlotter()
        result = Result()

        result.set_plotter(plotter)

        CobaContext.logger.sink = ListSink()
        result.plot_learners()
        self.assertEqual(["This result does not contain any data to plot."],CobaContext.logger.sink.items)

    def test_plot_contrast_no_data(self):
        CobaContext.logger.sink = ListSink()
        Result().plot_contrast(0, 1, x=['a'])
        self.assertEqual(["This result does not contain any data to plot."],CobaContext.logger.sink.items)

    def test_shared_c(self):
        CobaContext.logger.sink = ListSink()
        Result().plot_contrast(1, 1, x=['a'])
        self.assertEqual(["A value cannot be in both `l1` and `l2`. Please make a change and run it again."],CobaContext.logger.sink.items)

    def test_plot_contrast_no_pairings(self):
        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,0],[0,1,0,2,3],[0,1,0,3,9],
            [0,2,0,1,1],[0,2,0,2,2],[0,2,0,3,6],
            [1,1,0,1,1],[1,1,0,2,2],[1,1,0,3,6],
            [1,2,0,1,0],[1,2,0,2,3],[1,2,0,3,9],
        ]

        CobaContext.logger.sink = ListSink()
        Result(envs, lrns, vals, ints).plot_contrast(0, 1, x='index')

        self.assertEqual(f"We were unable to create any pairings to contrast. Make sure l1=0 and l2=1 is correct.",CobaContext.logger.sink.items[0])

    def test_plot_contrast_index(self):
        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,0],[0,1,0,2,3],[0,1,0,3,9],
            [0,2,0,1,1],[0,2,0,2,2],[0,2,0,3,6],
            [1,1,0,1,1],[1,1,0,2,2],[1,1,0,3,9],
            [1,2,0,1,0],[1,2,0,2,3],[1,2,0,3,6],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_contrast(1, 2, x='index')

        expected_lines = [
            Points((1,2,3), (0, 0, -1), None, (0,0,0), 0     , 1, 'l2-l1', '-', 1.),
            Points((1,3)  , (0, 0    ), None,  None  , "#888", 1, None   , '-', .5),
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_contrast_four_environment_all_default(self):
        envs = [['environment_id'],[0],[1],[2],[3]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,0],[0,1,0,2,3],[0,1,0,3,9],
            [0,2,0,1,1],[0,2,0,2,2],[0,2,0,3,6],
            [1,1,0,1,1],[1,1,0,2,2],[1,1,0,3,6],
            [1,2,0,1,0],[1,2,0,2,3],[1,2,0,3,9],
            [2,1,0,1,0],[2,1,0,2,0],[2,1,0,3,6],
            [2,2,0,1,0],[2,2,0,2,3],[2,2,0,3,9],
            [3,1,0,1,0],[3,1,0,2,3],[3,1,0,3,9],
            [3,2,0,1,0],[3,2,0,2,0],[3,2,0,3,6],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_contrast(2,1)

        expected_lines = [
            Points(('2','1'), (-2,-1), None, (0,0), 0     , 1, '2. learner_2 (2)' , '.', 1.),
            Points(()       , ()     , None, None , 1     , 1, 'Tie (0)', '.', 1.),
            Points(('0','3'), ( 1, 2), None, (0,0), 2     , 1, '1. learner_1 (2)' , '.', 1.),
            Points(('2','3'), ( 0, 0), None,  None, "#888", 1, None     , '-', .5),
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_contrast_xlabel_ylabel(self):
        envs = [['environment_id'],[0],[1],[2],[3]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,0],[0,1,0,2,3],[0,1,0,3,9],
            [0,2,0,1,1],[0,2,0,2,2],[0,2,0,3,6],
            [1,1,0,1,1],[1,1,0,2,2],[1,1,0,3,6],
            [1,2,0,1,0],[1,2,0,2,3],[1,2,0,3,9],
            [2,1,0,1,0],[2,1,0,2,0],[2,1,0,3,6],
            [2,2,0,1,0],[2,2,0,2,3],[2,2,0,3,9],
            [3,1,0,1,0],[3,1,0,2,3],[3,1,0,3,9],
            [3,2,0,1,0],[3,2,0,2,0],[3,2,0,3,6],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_contrast(2,1,xlabel='x',ylabel='y')

        expected_lines = [
            Points(('2','1'), (-2,-1), None, (0,0), 0     , 1, '2. learner_2 (2)' , '.', 1.),
            Points(()       , ()     , None, None , 1     , 1, 'Tie (0)', '.', 1.),
            Points(('0','3'), ( 1, 2), None, (0,0), 2     , 1, '1. learner_1 (2)' , '.', 1.),
            Points(('2','3'), ( 0, 0), None,  None, "#888", 1, None     , '-', .5),
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])
        self.assertEqual('y (1 Environments)', plotter.plot_calls[0][2])
        self.assertEqual('x', plotter.plot_calls[0][3])
        self.assertEqual('y', plotter.plot_calls[0][4])

    def test_plot_contrast_title(self):
        envs = [['environment_id'],[0],[1],[2],[3]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,0],[0,1,0,2,3],[0,1,0,3,9],
            [0,2,0,1,1],[0,2,0,2,2],[0,2,0,3,6],
            [1,1,0,1,1],[1,1,0,2,2],[1,1,0,3,6],
            [1,2,0,1,0],[1,2,0,2,3],[1,2,0,3,9],
            [2,1,0,1,0],[2,1,0,2,0],[2,1,0,3,6],
            [2,2,0,1,0],[2,2,0,2,3],[2,2,0,3,9],
            [3,1,0,1,0],[3,1,0,2,3],[3,1,0,3,9],
            [3,2,0,1,0],[3,2,0,2,0],[3,2,0,3,6],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_contrast(2,1,title='abc')

        expected_lines = [
            Points(('2','1'), (-2,-1), None, (0,0), 0     , 1, '2. learner_2 (2)' , '.', 1.),
            Points(()       , ()     , None, None , 1     , 1, 'Tie (0)', '.', 1.),
            Points(('0','3'), ( 1, 2), None, (0,0), 2     , 1, '1. learner_1 (2)' , '.', 1.),
            Points(('2','3'), ( 0, 0), None,  None, "#888", 1, None     , '-', .5),
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])
        self.assertEqual('abc', plotter.plot_calls[0][2])

    def test_plot_contrast_one_environment_env_not_index(self):
        envs = [['environment_id','a'],[0,1],[1,2],[2,3]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,0],[0,1,0,2,3],[0,1,0,3,12],
            [0,2,0,1,1],[0,2,0,2,2],[0,2,0,3,6],
            [1,1,0,1,0],[1,1,0,2,3],[1,1,0,3,6],
            [1,2,0,1,1],[1,2,0,2,2],[1,2,0,3,6],
            [2,1,0,1,0],[2,1,0,2,3],[2,1,0,3,9],
            [2,2,0,1,1],[2,2,0,2,2],[2,2,0,3,6],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_contrast(2,1,'a')

        expected_lines = [
            Points(()       , ()   , None, None , 0     , 1, '2. learner_2 (0)' , '.', 1.),
            Points(('2',)   , (0, ), None, (0, ), 1     , 1, 'Tie (1)', '.', 1.),
            Points(('3','1'), (1,2), None, (0,0), 2     , 1, '1. learner_1 (2)' , '.', 1.),
            Points(('2','1'), (0,0), None, None , "#888", 1, None     , '-', .5)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_contrast_one_environment_env_not_index_mode_prob_mixed_x(self):
        envs = [['environment_id','a'],[0,1],[1,2],[2,None]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,0],[0,1,0,2,3],[0,1,0,3,12],
            [0,2,0,1,1],[0,2,0,2,2],[0,2,0,3,6],
            [1,1,0,1,0],[1,1,0,2,3],[1,1,0,3,6],
            [1,2,0,1,1],[1,2,0,2,2],[1,2,0,3,6],
            [2,1,0,1,0],[2,1,0,2,3],[2,1,0,3,9],
            [2,2,0,1,1],[2,2,0,2,2],[2,2,0,3,6],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_contrast(2,1,x='a',mode='prob')

        expected_lines = [
            Points(('2',)      , (0,   ), None, (0, ), 0     , 1, '2. learner_2 (1)' , '.', 1.),
            Points(()           , ()   , None , None , 1     , 1, 'Tie (0)', '.', 1.),
            Points(('1','None'), (1,1  ), None, (0,0), 2     , 1, '1. learner_1 (2)' , '.', 1.),
            Points(('2','None'), (.5,.5), None, None , "#888", 1, None     , '-', .5)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_contrast_one_environment_env_not_index_mode_prob(self):
        envs = [['environment_id','a'],[0,1],[1,2],[2,3]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,0],[0,1,0,2,3],[0,1,0,3,12],
            [0,2,0,1,1],[0,2,0,2,2],[0,2,0,3,6],
            [1,1,0,1,0],[1,1,0,2,3],[1,1,0,3,6],
            [1,2,0,1,1],[1,2,0,2,2],[1,2,0,3,6],
            [2,1,0,1,0],[2,1,0,2,3],[2,1,0,3,9],
            [2,2,0,1,1],[2,2,0,2,2],[2,2,0,3,6],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_contrast(2,1,x='a',mode='prob')

        expected_lines = [
            Points(('2',)   , (0,   ), None, (0, ), 0     , 1, '2. learner_2 (1)' , '.', 1.),
            Points(()       , ()     , None, None , 1     , 1, 'Tie (0)', '.', 1.),
            Points(('1','3'), (1,1  ), None, (0,0), 2     , 1, '1. learner_1 (2)' , '.', 1.),
            Points(('2','3'), (.5,.5), None, None , "#888", 1, None     , '-', .5)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_contrast_one_environment_env_not_index_with_p(self):
        envs = [['environment_id','a'],[0,1],[1,2],[2,3]]
        lrns = [['learner_id', 'family'],[1,'learner_1']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,0],[0,1,0,2,3],[0,1,0,3,12],
            [1,1,0,1,0],[1,1,0,2,3],[1,1,0,3,6],
            [2,1,0,1,0],[2,1,0,2,3],[2,1,0,3,9],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_contrast(2,1,l='a',p='learner_id')

        expected_lines = [
            Points(()           , ()   , None, None , 0     , 1, 'l1 (0)' , '.', 1.),
            Points(()           , ()   , None, None , 1     , 1, 'Tie (0)', '.', 1.),
            Points(('0-1',     ), (2, ), None, (0, ), 2     , 1, 'l2 (1)' , '.', 1.),
            Points(('0-1','0-1'), (0,0), None, None , "#888", 1, None     , '-', .5)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_contrast_x_eq_l_multi_l1(self):
        envs = [['environment_id','a'],[0,1],[1,2],[2,3]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,0],[0,1,0,2,3],[0,1,0,3,12],
            [0,2,0,1,1],[0,2,0,2,2],[0,2,0,3,6],
            [1,1,0,1,0],[1,1,0,2,3],[1,1,0,3,6],
            [1,2,0,1,1],[1,2,0,2,2],[1,2,0,3,6],
            [2,1,0,1,0],[2,1,0,2,3],[2,1,0,3,9],
            [2,2,0,1,1],[2,2,0,2,2],[2,2,0,3,6],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_contrast([2,3],1,x='a',l='a',p='learner_id',boundary=False)

        expected_lines = [
            Points(('1-2','1-3'), (1,.5), None, (0, 0), 0, 1, None, '.', 1.),
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_contrast_x_eq_l_multi_l2(self):
        envs = [['environment_id','a'],[0,1],[1,2],[2,3]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,0],[0,1,0,2,3],[0,1,0,3,12],
            [0,2,0,1,1],[0,2,0,2,2],[0,2,0,3,6],
            [1,1,0,1,0],[1,1,0,2,3],[1,1,0,3,6],
            [1,2,0,1,1],[1,2,0,2,2],[1,2,0,3,6],
            [2,1,0,1,0],[2,1,0,2,3],[2,1,0,3,9],
            [2,2,0,1,1],[2,2,0,2,2],[2,2,0,3,6],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_contrast(1,[3,2],x='a',l='a',p='learner_id',boundary=False)

        expected_lines = [
            Points(('3-1','2-1'), (-.5,-1), None, (0, 0), 0, 1, None, '.', 1.),
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_contrast_x_eq_c_one_one(self):
        envs = [['environment_id','a'],[0,1],[1,2],[2,3]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
            [0,1,0,1,0],[0,1,0,2,3],[0,1,0,3,12],
            [0,2,0,1,1],[0,2,0,2,2],[0,2,0,3,6],
            [1,1,0,1,0],[1,1,0,2,3],[1,1,0,3,6],
            [1,2,0,1,1],[1,2,0,2,2],[1,2,0,3,6],
            [2,1,0,1,0],[2,1,0,2,3],[2,1,0,3,9],
            [2,2,0,1,1],[2,2,0,2,2],[2,2,0,3,6],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, vals, ints)

        result.set_plotter(plotter)
        result.plot_contrast(1,2,x='a',l='a',p='learner_id',boundary=False)

        expected_lines = [
            Points(('2-1',), (-1,), None, (0,), 0, 1, None, '.', 1.),
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plottable_all_env_finished_with_equal_lengths_and_x_index(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[0]]
        lrns = [['learner_id', 'family'],[0,'learner_1'],[1,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],[0,0,0,1,1],[0,1,0,1,1]]
        result = Result(envs, lrns, vals, ints)

        self.assertEqual(result, result._plottable('index','reward')._finished('index','reward','learner_id','environment_id'))
        self.assertEqual([], CobaContext.logger.sink.items)

    def test_plottable_all_env_finished_with_unequal_lengths_and_x_index(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[0,'learner_1'],[1,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
                [0,0,0,1,1],
                [0,1,0,1,1],
                [1,0,0,1,1],[1,0,0,2,1],
                [1,1,0,1,1],[1,1,0,2,1]
        ]
        result = Result(envs, lrns, vals, ints)

        expected_logs = ['We shortened 2 environments because they were longer than the shortest environment.']
        expected_rows = [(0,0,0,1,1),(0,1,0,1,1),(1,0,0,1,1),(1,1,0,1,1)]

        plottable = result._plottable('index','reward')._finished('index','reward','learner_id','environment_id')

        self.assertEqual(expected_rows,list(plottable.interactions))
        self.assertEqual(expected_logs, CobaContext.logger.sink.items)

    def test_plottable_not_all_env_finished_with_unequal_lengths_and_x_index(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[0,'learner_1'],[1,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],
                [0,0,0,1,1],
                [1,0,0,1,1],[1,0,0,2,1],
                [1,1,0,1,1],[1,1,0,2,1],
        ]
        result = Result(envs, lrns, vals, ints)

        expected_logs = ["We removed 1 environment_id because it did not exist for every learner_id."]
        expected_rows = [(1,0,0,1,1),(1,0,0,2,1),(1,1,0,1,1),(1,1,0,2,1)]
        plottable = result._plottable('index','reward')._finished('index','reward','learner_id','environment_id')

        self.assertEqual(expected_rows,list(plottable.interactions))
        self.assertEqual(expected_logs, CobaContext.logger.sink.items)

    def test_plottable_no_env_finished_for_all_learners(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[0,'learner_1'],[1,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],[0,0,0,1,1],[1,1,0,1,1],[1,1,0,2,1]]
        result = Result(envs, lrns, vals, ints)

        with self.assertRaises(CobaException) as e:
            result._plottable('index','reward')._finished('index','reward','learner_id','environment_id')

        self.assertEqual(str(e.exception),"This result does not contain a environment_id that has been finished for every learner_id.")

    def test_plottable_all_env_finished_with_unequal_lengths_and_not_x_index(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[0,'learner_1'],[1,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],[0,0,0,1,1],[0,1,0,1,1],[1,0,0,1,1],[1,0,0,2,1],[1,1,0,1,1],[1,1,0,2,1]]
        result = Result(envs, lrns, vals, ints)

        expected_rows = [
           (0,0,0,1,1),
           (0,1,0,1,1),
           (1,0,0,1,1),(1,0,0,2,1),
           (1,1,0,1,1),(1,1,0,2,1)
        ]

        plottable = result._plottable('openml_task','reward')._finished('openml_task','reward','learner_id','environment_id')
        self.assertEqual(expected_rows,list(plottable.interactions))

    def test_plottable_no_data(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        with self.assertRaises(CobaException) as e:
            Result()._plottable('index','reward')._finished('index','reward','learner_id','environment_id')

        self.assertEqual(str(e.exception),"This result does not contain any data to plot.")

    def test_plottable_no_y(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[0]]
        lrns = [['learner_id', 'family'],[0,'learner_1'],[1,'learner_2']]
        vals = [['evaluator_id'],[0]]
        ints = [['environment_id','learner_id','evaluator_id','index','reward'],[0,0,0,1,1],[0,1,0,1,1]]
        result = Result(envs, lrns, vals, ints)

        with self.assertRaises(CobaException) as e:
            result._plottable('index','a')._finished('index','a','learner_id','environment_id')

        self.assertEqual(str(e.exception),"This result does not contain column 'a' in interactions.")

    def test_confidence_skip_err(self):
        self.assertEqual((2,(0,0)),Result()._confidence('sd',2)([1,2,3],0))
        self.assertEqual((2,(1,1)),Result()._confidence('sd',2)([1,2,3],1))

    def test_confidence_sd(self):
        self.assertEqual((2,(1,1)),Result()._confidence('sd')([1,2,3]))

    def test_confidence_se(self):
        self.assertEqual((2,(1.96,1.96)),Result()._confidence('se')([1,3]))

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "this test requires scipy")
    def test_confidence_bs(self):
        self.assertEqual((2.5, (1.0,1.0)),Result()._confidence('bs')([1,2,3,4]))

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "this test requires scipy")
    def test_confidence_ci(self):
        self.assertEqual((2.5, (1.5,0.5)),Result()._confidence(BootstrapCI(.95,mean))([1,2,3,4]))

    def test_confidence_none(self):
        self.assertEqual((2,0),Result()._confidence(None)([1,2,3]))

    def test_confidence_bi(self):
        mu,(l,h) = Result()._confidence('bi')([0,0,1,1])

        self.assertEqual(.5,mu)
        self.assertAlmostEqual(l,0.34996429)
        self.assertAlmostEqual(h,0.34996429)

class moving_average_Tests(unittest.TestCase):

    def test_weights(self):
        self.assertEqual([0,1/2,1/2,0/2,1/2],list(moving_average([0,1,0,0,1],span=2,weights=[.5,.5,.5,.5,.5])))
        self.assertEqual([0,1/2,1/3,1/3,1/3],list(moving_average([0,1,0,0,1],span=3,weights=[.5,.5,.5,.5,.5])))
        self.assertEqual([0,1/2,1/3,1/4,2/4],list(moving_average([0,1,0,0,1],span=4,weights=[.5,.5,.5,.5,.5])))

    def test_sliding_windows(self):
        self.assertEqual([0,1/2,1/2,0/2,1/2], list(moving_average([0,1,0,0,1],span=2)))
        self.assertEqual([0,1/2,1/3,1/3,1/3], list(moving_average([0,1,0,0,1],span=3)))
        self.assertEqual([0,1/2,1/3,1/4,2/4], list(moving_average([0,1,0,0,1],span=4)))

    def test_rolling_windows(self):
        self.assertEqual([0,1/2,1/3,1/4,2/5], list(moving_average([0,1,0,0,1],span=None)))
        self.assertEqual([0,1/2,1/3,1/4,2/5], list(moving_average([0,1,0,0,1],span=5   )))
        self.assertEqual([0,1/2,1/3,1/4,2/5], list(moving_average([0,1,0,0,1],span=6   )))

    def test_no_window(self):
        self.assertEqual([0,1,0,0,1], list(moving_average([0,1,0,0,1],span=1)))

    def test_exponential_span_2(self):
        self.assertEqual([1,1.75,2.62,3.55], [round(v,2) for v in list(moving_average([1,2,3,4],span=2,weights='exp'))])

if __name__ == '__main__':
    unittest.main()
