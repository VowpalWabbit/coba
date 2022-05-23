import unittest
import unittest.mock
import timeit
import importlib.util

from pathlib import Path

from coba.pipes import ListSink
from coba.contexts import CobaContext, IndentLogger
from coba.exceptions import CobaException, CobaExit

from coba.experiments.results import TransactionIO, TransactionIO_V3, TransactionIO_V4
from coba.experiments.results import Result, Table, InteractionsTable
from coba.experiments.results import MatplotlibPlotter

class TestPlotter:
    def __init__(self):
        self.plot_calls = []
    def plot(self, *args) -> None:
        self.plot_calls.append(args)

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
        self.assertEqual([{'a':'A', 'b':'B'},{'a':'a', 'b':'B'}], list(table.to_dicts()))

    def test_missing_columns(self):
        table = Table("test", ['a'], [{'a':'A', 'b':'B'}, {'a':'a', 'c':'C'}])

        self.assertTrue('A' in table)
        self.assertTrue('a' in table)

        self.assertEqual(table['A'], {'a':'A', 'b':'B'})
        self.assertEqual(table['a'], {'a':'a', 'c':'C'})

        self.assertEqual(2, len(table))
        self.assertEqual([{'a':'A', 'b':'B', 'c':None}, {'a':'a', 'b':None, 'c':'C'}], table.to_dicts())

    def test_tuples_with_array_column(self):

        table = Table("test", ['a'], [dict(a='A', b='B', c=[1,2], d='d'), dict(a='B', e='E') ])

        expected = [{'a':'A','b':'B','c':[1,2],'d':'d','e':None},{'a':'B','b':None,'c':None,'d':None,'e':'E'}]
        actual = table.to_dicts()

        self.assertEqual(expected, actual)

    def test_tuples_with_dict_column(self):

        table = Table("test", ['a'], [dict(a='A',b='B',c={'z':5},d='d'), dict(a='B',e='E')])

        expected = [{'a':'A','b':'B','c':{'z':5},'d':'d','e':None},{'a':'B','b':None,'c':None,'d':None,'e':'E'} ]
        actual = table.to_dicts()

        self.assertEqual(expected, actual)

    def test_two_packed_items(self):
        table = Table("test", ['a'], [dict(a='A', c=1, _packed=dict(b=['B','b'],d=['D','d']))])

        self.assertTrue('A' in table)
        self.assertEqual(table['A'], {'a':'A', 'index':[1,2], 'b':['B','b'], 'c':1, 'd':['D','d']})
        self.assertEqual(2, len(table))

        expected = [{'a':'A','index':1,'b':'B','c':1,'d':'D'},{'a':'A','index':2,'b':'b','c':1,'d':'d'}]
        actual = table.to_dicts()

        self.assertEqual(expected,actual)

    def test_unequal_pack_exception(self):
        with self.assertRaises(Exception):
            table = Table("test", ['a'])
            table['A'] = dict(c=1,_packed=dict(b=['B','b'],d=['D','d','e']))

    def test_filter_kwarg_str(self):
        table = Table("test", ['a'], [{'a':'a', 'b':'b'}, {'a':'A', 'b':'B'}])

        filtered_table = table.filter(b="B")

        self.assertEqual(2, len(table))
        self.assertEqual([{'a':'A', 'b':'B'},{'a':'a', 'b':'b'}], table.to_dicts())

        self.assertEqual(1, len(filtered_table))
        self.assertEqual([{'a':'A', 'b':'B'}], filtered_table.to_dicts())

    def test_filter_kwarg_int_1(self):
        table = Table("test", ['a'], [{'a':'1', 'b':'b'}, {'a':'12', 'b':'B'}])

        filtered_table = table.filter(a=1)

        self.assertEqual(2, len(table))
        self.assertEqual([{'a':'1','b':'b'},{'a':'12','b':'B'}], table.to_dicts())

        self.assertEqual(1, len(filtered_table))
        self.assertEqual([{'a':'1','b':'b'}], filtered_table.to_dicts())

    def test_filter_kwarg_int_2(self):
        table = Table("test", ['a'], [{'a':1, 'b':'b'}, {'a':12, 'b':'B'}])

        filtered_table = table.filter(a=1)

        self.assertEqual(2, len(table))
        self.assertEqual([{'a':1,'b':'b'},{'a':12,'b':'B'}], table.to_dicts())

        self.assertEqual(1, len(filtered_table))
        self.assertEqual([{'a':1, 'b':'b'}], filtered_table.to_dicts())

    def test_filter_kwarg_pred(self):
        table = Table("test", ['a'], [{'a':'1', 'b':'b'}, {'a':'12', 'b':'B'}])

        filtered_table = table.filter(a= lambda a: a =='1')

        self.assertEqual(2, len(table))
        self.assertEqual([{'a':'1','b':'b'},{'a':'12','b':'B'}], table.to_dicts())

        self.assertEqual(1, len(filtered_table))
        self.assertEqual([{'a':'1','b':'b'}], filtered_table.to_dicts())

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
        self.assertEqual([{'a':'2','b':'b','c':"C"}], filtered_table.to_dicts())

    def test_filter_pred(self):
        table = Table("test", ['a'], [{'a':'a', 'b':'b'}, {'a':'A', 'b':'B'}])

        filtered_table = table.filter(lambda row: row["b"]=="B")

        self.assertEqual(2, len(table))
        self.assertEqual([{'a':'A','b':'B'},{'a':'a','b':'b'}], table.to_dicts())

        self.assertEqual(1, len(filtered_table))
        self.assertEqual([{'a':'A','b':'B'}], filtered_table.to_dicts())

    def test_filter_sequence_1(self):
        table = Table("test", ['a'], [{'a':'a', 'b':'b'}, {'a':'A', 'b':'B'}, {'a':'1', 'b':'C'}])

        filtered_table = table.filter(a=['a','1'])

        self.assertEqual(3, len(table))
        self.assertCountEqual([{'a':'A','b':'B'},{'a':'a','b':'b'},{'a':'1','b':'C'}], table.to_dicts())

        self.assertEqual(2, len(filtered_table))
        self.assertCountEqual([{'a':'a','b':'b'},{'a':'1','b':'C'}], filtered_table.to_dicts())

    def test_filter_sequence_2(self):
        table = Table("test", ['a'], [{'a':'1', 'b':'b'}, {'a':'2', 'b':'B'}, {'a':'3', 'b':'C'}])

        filtered_table = table.filter(a=[1,2])

        self.assertEqual(3, len(table))
        self.assertCountEqual([{'a':'1','b':'b'},{'a':'2','b':'B'},{'a':'3','b':'C'}], table.to_dicts())

        self.assertEqual(2, len(filtered_table))
        self.assertCountEqual([{'a':'1','b':'b'},{'a':'2','b':'B'}], filtered_table.to_dicts())

    def test_filter_table_contains(self):
        table = Table("test", ['a'], [{'a':'a', 'b':'b'}, {'a':'A', 'b':'B'}])

        filtered_table = table.filter(lambda row: row["b"]=="B")

        self.assertEqual(2, len(table))
        self.assertEqual([{'a':'A','b':'B'},{'a':'a','b':'b'}], table.to_dicts())

        self.assertNotIn("a", filtered_table)

    def test_filter_missing_columns(self):
        table = Table("test", ['a'], [{'a':'a', 'b':'B'}, {'a':'A'}])

        filtered_table = table.filter(b="B")

        self.assertEqual(2, len(table))
        self.assertEqual([{'a':'A','b':None},{'a':'a','b':'B'}], table.to_dicts())

        self.assertNotIn("A", filtered_table)
        self.assertIn("a", filtered_table)

    def test_filter_nan_value(self):
        table = Table("test", ['a'], [{'a':'a', 'b':'B'}, {'a':'A'}])

        filtered_table = table.filter(b=None)

        self.assertEqual(2, len(table))
        self.assertEqual([{'a':'A','b':None},{'a':'a','b':'B'}], table.to_dicts())

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

@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed so we must skip plotting tests")
class MatplotlibPlotter_Tests(unittest.TestCase):

    def test_no_matplotlib(self):
        with unittest.mock.patch('importlib.import_module', side_effect=ImportError()):
            with self.assertRaises(CobaExit):
                MatplotlibPlotter().plot(None,None,None,None,None,None,None,None)

    def test_plot_lines_title_xlabel_ylabel(self):
        with unittest.mock.patch('matplotlib.pyplot.show') as show:
            with unittest.mock.patch('matplotlib.pyplot.savefig') as savefig:
                with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

                    mock_ax = plt_figure().add_subplot()

                    mock_ax.get_xticks.return_value = [1,2]
                    mock_ax.get_xlim.return_value   = [2,2]

                    lines = [
                        ([1,2], [5,6], None, "B", 1.00, 'L1'),
                        ([3,4], [7,8], None, "R", 0.25, 'L2')
                    ]

                    MatplotlibPlotter().plot(None,lines,"title","xlabel","ylabel",None,None,None)

                    plt_figure().add_subplot.assert_called_with(111)

                    self.assertEqual(([1,2],[5,6]), mock_ax.plot.call_args_list[0][0])
                    self.assertEqual('B'          , mock_ax.plot.call_args_list[0][1]["color"])
                    self.assertEqual(1            , mock_ax.plot.call_args_list[0][1]["alpha"])
                    self.assertEqual('L1'         , mock_ax.plot.call_args_list[0][1]["label"])

                    self.assertEqual(([3,4],[7,8]), mock_ax.plot.call_args_list[1][0])
                    self.assertEqual('R'          , mock_ax.plot.call_args_list[1][1]["color"])
                    self.assertEqual(.25          , mock_ax.plot.call_args_list[1][1]["alpha"])
                    self.assertEqual('L2'         , mock_ax.plot.call_args_list[1][1]["label"])

                    mock_ax.set_xticks.called_once_with([2,2])
                    mock_ax.set_title.colled_once_with('title')
                    mock_ax.set_xlabel.called_once_with('xlabel')
                    mock_ax.set_ylabel.called_once_with('ylabel')

                    self.assertEqual(1, show.call_count)
                    self.assertEqual(0, savefig.call_count)

    def test_plot_lines_err_title_xlabel_ylabel(self):
        with unittest.mock.patch('matplotlib.pyplot.show') as show:
            with unittest.mock.patch('matplotlib.pyplot.savefig') as savefig:
                with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

                    mock_ax = plt_figure().add_subplot()

                    mock_ax.get_xticks.return_value = [1,2]
                    mock_ax.get_xlim.return_value   = [2,2]

                    lines = [
                        ([1,2], [5,6], [10,11], "B", 1.00, 'L1'),
                        ([3,4], [7,8], [12,13], "R", 0.25, 'L2')
                    ]

                    MatplotlibPlotter().plot(None,lines,"title","xlabel","ylabel",None,None,None)

                    plt_figure().add_subplot.assert_called_with(111)

                    self.assertEqual(([1,2],[5,6]), mock_ax.errorbar.call_args_list[0][0])
                    self.assertEqual([10,11]      , mock_ax.errorbar.call_args_list[0][1]["yerr"])
                    self.assertEqual('B'          , mock_ax.errorbar.call_args_list[0][1]["color"])
                    self.assertEqual(1            , mock_ax.errorbar.call_args_list[0][1]["alpha"])
                    self.assertEqual('L1'         , mock_ax.errorbar.call_args_list[0][1]["label"])

                    self.assertEqual(([3,4],[7,8]), mock_ax.errorbar.call_args_list[1][0])
                    self.assertEqual([12,13]      , mock_ax.errorbar.call_args_list[1][1]["yerr"])
                    self.assertEqual('R'          , mock_ax.errorbar.call_args_list[1][1]["color"])
                    self.assertEqual(.25          , mock_ax.errorbar.call_args_list[1][1]["alpha"])
                    self.assertEqual('L2'         , mock_ax.errorbar.call_args_list[1][1]["label"])

                    mock_ax.set_xticks.called_once_with([2,2])
                    mock_ax.set_title.colled_once_with('title')
                    mock_ax.set_xlabel.called_once_with('xlabel')
                    mock_ax.set_ylabel.called_once_with('ylabel')

                    self.assertEqual(1, show.call_count)
                    self.assertEqual(0, savefig.call_count)

    def test_plot_xlim1(self):
        with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

            lines = [
                ([1,2], [5,6], None, "B", 1.00, 'L1'),
                ([3,4], [7,8], None, "R", 0.25, 'L2')
            ]

            mock_ax = plt_figure().add_subplot()
            mock_ax.get_xticks.return_value = [1,2]
            mock_ax.get_xlim.return_value   = [2,2]
            MatplotlibPlotter().plot(None,lines,"title","xlabel","ylabel",(2,3),None,None)
            self.assertEqual(([2],[6]), mock_ax.plot.call_args_list[0][0])
            self.assertEqual(([3],[7]), mock_ax.plot.call_args_list[1][0])

            lines = [
                ([1,2], [5,6], [4,3], "B", 1.00, 'L1'),
                ([3,4], [7,8], [2,1], "R", 0.25, 'L2')
            ]

    def test_plot_xlim2(self):
        with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

            lines = [
                ([1,2], [5,6], [4,3], "B", 1.00, 'L1'),
                ([3,4], [7,8], [2,1], "R", 0.25, 'L2')
            ]

            mock_ax = plt_figure().add_subplot()
            mock_ax.get_xticks.return_value = [1,2]
            mock_ax.get_xlim.return_value   = [2,2]
            MatplotlibPlotter().plot(None,lines,"title","xlabel","ylabel",(3,4),None,None)
            self.assertEqual(([3,4],[7,8]), mock_ax.errorbar.call_args_list[0][0])
            self.assertEqual([2,1]        , mock_ax.errorbar.call_args_list[0][1]["yerr"])

    def test_plot_ylim(self):
        with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

            lines = [
                ([1,2], [5,6], None, "B", 1.00, 'L1'),
                ([3,4], [7,8], None, "R", 0.25, 'L2')
            ]

            mock_ax = plt_figure().add_subplot()
            mock_ax.get_xticks.return_value = [1,2]
            mock_ax.get_xlim.return_value   = [2,2]
            MatplotlibPlotter().plot(None,lines,"title","xlabel","ylabel",None,(6,7),None)
            self.assertEqual(([2],[6]), mock_ax.plot.call_args_list[0][0])
            self.assertEqual(([3],[7]), mock_ax.plot.call_args_list[1][0])

    def test_plot_ax(self):
        with unittest.mock.patch('matplotlib.pyplot.show') as show:
            with unittest.mock.patch('matplotlib.pyplot.savefig') as savefig:
                with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

                    mock_ax = plt_figure().add_subplot()
                    self.assertEqual(1, plt_figure().add_subplot.call_count)

                    mock_ax.get_xticks.return_value = [1,2]
                    mock_ax.get_xlim.return_value   = [2,2]

                    lines = [
                        ([1,2], [5,6], None, "B", 1.00, 'L1'),
                        ([3,4], [7,8], None, "R", 0.25, 'L2')
                    ]

                    MatplotlibPlotter().plot(mock_ax,lines,"title","xlabel","ylabel",None,None,None)

                    self.assertEqual(1, plt_figure().add_subplot.call_count)

                    self.assertEqual(([1,2],[5,6]), mock_ax.plot.call_args_list[0][0])
                    self.assertEqual('B'          , mock_ax.plot.call_args_list[0][1]["color"])
                    self.assertEqual(1            , mock_ax.plot.call_args_list[0][1]["alpha"])
                    self.assertEqual('L1'         , mock_ax.plot.call_args_list[0][1]["label"])

                    self.assertEqual(([3,4],[7,8]), mock_ax.plot.call_args_list[1][0])
                    self.assertEqual('R'          , mock_ax.plot.call_args_list[1][1]["color"])
                    self.assertEqual(.25          , mock_ax.plot.call_args_list[1][1]["alpha"])
                    self.assertEqual('L2'         , mock_ax.plot.call_args_list[1][1]["label"])

                    mock_ax.set_xticks.called_once_with([2,2])
                    mock_ax.set_title.colled_once_with('title')
                    mock_ax.set_xlabel.called_once_with('xlabel')
                    mock_ax.set_ylabel.called_once_with('ylabel')

                    self.assertEqual(0, show.call_count)
                    self.assertEqual(0, savefig.call_count)

    def test_plot_bad_xlim(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        plotter = MatplotlibPlotter()
        plotter.plot(None, [[]], 'abc', 'def', 'efg', (1,0), (0,1), None)

        expected_log = "The xlim end is less than the xlim start. Plotting is impossible."

        self.assertEqual(expected_log, CobaContext.logger.sink.items[0])

    def test_plot_bad_ylim(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        plotter = MatplotlibPlotter()
        plotter.plot(None, [[]], 'abc', 'def', 'efg', (0,1), (1,0), None)

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
                        ([1,2], [5,6], None, "B", 1.00, 'L1'),
                        ([3,4], [7,8], None, "R", 0.25, 'L2')
                    ]

                    MatplotlibPlotter().plot(None,lines,"title","xlabel","ylabel",None,None,"abc")

                    plt_figure().add_subplot.assert_called_with(111)

                    self.assertEqual(([1,2],[5,6]), mock_ax.plot.call_args_list[0][0])
                    self.assertEqual('B'          , mock_ax.plot.call_args_list[0][1]["color"])
                    self.assertEqual(1            , mock_ax.plot.call_args_list[0][1]["alpha"])
                    self.assertEqual('L1'         , mock_ax.plot.call_args_list[0][1]["label"])

                    self.assertEqual(([3,4],[7,8]), mock_ax.plot.call_args_list[1][0])
                    self.assertEqual('R'          , mock_ax.plot.call_args_list[1][1]["color"])
                    self.assertEqual(.25          , mock_ax.plot.call_args_list[1][1]["alpha"])
                    self.assertEqual('L2'         , mock_ax.plot.call_args_list[1][1]["label"])

                    mock_ax.set_xticks.called_once_with([2,2])
                    mock_ax.set_title.colled_once_with('title')
                    mock_ax.set_xlabel.called_once_with('xlabel')
                    mock_ax.set_ylabel.called_once_with('ylabel')

                    self.assertEqual(1, show.call_count)
                    self.assertEqual(1, savefig.call_count)

    def test_no_lines(self):
        with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

            CobaContext.logger = IndentLogger()
            CobaContext.logger.sink = ListSink()

            plotter = MatplotlibPlotter()
            plotter.plot(None, [], 'abc', 'def', 'efg', None, None, None)

            self.assertEqual(0, plt_figure().add_subplot.call_count)
            self.assertEqual(["No data was found for plotting in the given results."], CobaContext.logger.sink.items)

class InteractionTable_Tests(unittest.TestCase):

    def test_simple_each_span_none(self):
        table = InteractionsTable("ABC", ["environment_id", "learner_id"], rows=[
            {"environment_id":0, "learner_id":0, "_packed": {"reward":[1,2,3]}},
            {"environment_id":1, "learner_id":0, "_packed": {"reward":[3,6,9]}},
            {"environment_id":0, "learner_id":1, "_packed": {"reward":[2,4,6]}}
        ])

        expected = [
            {"learner_id":0,"environment_id":0,"values":[1,1.5,2]},
            {"learner_id":0,"environment_id":1,"values":[3,4.5,6]},
            {"learner_id":1,"environment_id":0,"values":[2,3.0,4]}
        ]
        actual = table.to_progressive_dicts(each=True)

        self.assertCountEqual(expected,actual)

    def test_simple_not_each_span_none(self):
        table = InteractionsTable("ABC", ["environment_id", "learner_id"], rows=[
            {"environment_id":0, "learner_id":0, "_packed": {"reward":[1,2,3]}},
            {"environment_id":1, "learner_id":0, "_packed": {"reward":[3,6,9]}},
            {"environment_id":0, "learner_id":1, "_packed": {"reward":[2,4,6]}}
        ])

        expected = [
            {"learner_id":0,"values":[2,3,4]},
            {"learner_id":1,"values":[2,3,4]}
        ]
        actual = table.to_progressive_dicts(each=False)

        self.assertCountEqual(expected,actual)

    def test_simple_each_span_one(self):
        table = InteractionsTable("ABC", ["environment_id", "learner_id"], rows=[
            {"environment_id":0, "learner_id":0, "_packed": {"reward":[1,2,3]}},
            {"environment_id":1, "learner_id":0, "_packed": {"reward":[3,6,9]}},
            {"environment_id":0, "learner_id":1, "_packed": {"reward":[2,4,6]}}
        ])
        expected = [
            {"learner_id":0,"environment_id":0,"values":[1,2,3]},
            {"learner_id":0,"environment_id":1,"values":[3,6,9]},
            {"learner_id":1,"environment_id":0,"values":[2,4,6]}
        ]
        actual = table.to_progressive_dicts(each=True,span=1)

        self.assertCountEqual(expected,actual)

    def test_simple_not_each_span_one(self):
        table = InteractionsTable("ABC", ["environment_id", "learner_id"], rows=[
            {"environment_id":0, "learner_id":0, "_packed": {"reward":[1,2,3]}},
            {"environment_id":1, "learner_id":0, "_packed": {"reward":[3,6,9]}},
            {"environment_id":0, "learner_id":1, "_packed": {"reward":[2,4,6]}}
        ])
        expected = [
            {"learner_id":0,"values":[2,4,6]},
            {"learner_id":1,"values":[2,4,6]}
        ]
        actual = table.to_progressive_dicts(each=False,span=1)

        self.assertCountEqual(expected,actual)

    def test_simple_each_span_two(self):
        table = InteractionsTable("ABC", ["environment_id", "learner_id"], rows=[
            {"environment_id":0, "learner_id":0, "_packed": {"reward":[1,2,3]}},
            {"environment_id":1, "learner_id":0, "_packed": {"reward":[2,4,6]}},
        ])
        expected = [
            {"learner_id":0,"environment_id":0,"values":[1,3/2,5/2]},
            {"learner_id":1,"environment_id":1,"values":[2,6/2,10/2]}
        ]
        actual = table.to_progressive_dicts(each=True,span=2)

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

        actual = table.to_progressive_pandas(each=True)

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

        actual = table.to_progressive_pandas(each=False)

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
        io.write(["T3",[1,0], [{"reward":3},{"reward":4}]])

        result = io.read()

        self.assertEqual({"n_learners":1, "n_environments":2}, result.experiment)
        self.assertEqual([{'learner_id':0,'name':"lrn1"}], result.learners.to_dicts())
        self.assertEqual([{'environment_id':1,'source':"test"}], result.environments.to_dicts())
        self.assertEqual([{'learner_id':0,'environment_id':1,'index':1,'reward':3},{'learner_id':0,'environment_id':1,'index':2,'reward':4}], result.interactions.to_dicts())

    def test_simple_to_and_from_memory(self):
        io = TransactionIO_V3()

        io.write(["T0",1,2])
        io.write(["T1",0,{"name":"lrn1"}])
        io.write(["T2",1,{"source":"test"}])
        io.write(["T3",[1,0], [{"reward":3},{"reward":4}]])

        result = io.read()

        self.assertEqual({"n_learners":1, "n_environments":2}, result.experiment)
        self.assertEqual([{'learner_id':0,'name':"lrn1"}], result.learners.to_dicts())
        self.assertEqual([{'environment_id':1,'source':"test"}], result.environments.to_dicts())
        self.assertEqual([{'learner_id':0,'environment_id':1,'index':1,'reward':3},{'learner_id':0,'environment_id':1,'index':2,'reward':4}], result.interactions.to_dicts())

    def test_simple_to_and_from_memory_unknown_transaction(self):
        io = TransactionIO_V3()

        io.write(["T0",1,2])
        io.write(["T1",0,{"name":"lrn1"}])
        io.write(["T2",1,{"source":"test"}])
        io.write(["T4",'UNKNOWN'])
        io.write(["T3",[1,0], [{"reward":3},{"reward":4}]])

        result = io.read()

        self.assertEqual({"n_learners":1, "n_environments":2}, result.experiment)
        self.assertEqual([{'learner_id':0,'name':"lrn1"}], result.learners.to_dicts())
        self.assertEqual([{'environment_id':1,'source':"test"}], result.environments.to_dicts())
        self.assertEqual([{'learner_id':0,'environment_id':1,'index':1,'reward':3},{'learner_id':0,'environment_id':1,'index':2,'reward':4}], result.interactions.to_dicts())

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
        io.write(["T3",[1,0], [{"reward":3},{"reward":4}]])

        result = io.read()

        self.assertEqual({**result.experiment, "n_learners":1, "n_environments":2}, result.experiment)
        self.assertEqual([{'learner_id':0,'name':"lrn1"}], result.learners.to_dicts())
        self.assertEqual([{'environment_id':1,'source':"test"}], result.environments.to_dicts())
        self.assertEqual([{'learner_id':0,'environment_id':1,'index':1,'reward':3},{'learner_id':0,'environment_id':1,'index':2,'reward':4}], result.interactions.to_dicts())

    def test_simple_to_and_from_memory_1(self):
        io = TransactionIO_V4()

        io.write(["T0",1,2])
        io.write(["T1",0,{"name":"lrn1"}])
        io.write(["T2",1,{"source":"test"}])
        io.write(["T3",[1,0], [{"reward":3},{"reward":4}]])

        result = io.read()

        self.assertEqual({**result.experiment, "n_learners":1, "n_environments":2}, result.experiment)
        self.assertEqual([{'learner_id':0,'name':"lrn1"}], result.learners.to_dicts())
        self.assertEqual([{'environment_id':1,'source':"test"}], result.environments.to_dicts())
        self.assertEqual([{'learner_id':0,'environment_id':1,'index':1,'reward':3},{'learner_id':0,'environment_id':1,'index':2,'reward':4}], result.interactions.to_dicts())

    def test_simple_to_and_from_memory_2(self):
        io = TransactionIO_V4()

        io.write(["T0",1,2])
        io.write(["T1",0,{"name":"lrn1"}])
        io.write(["T2",1,{"source":"test"}])
        io.write(["T3",[1,0], [{"reward":3},{"reward":4}]])

        result = io.read()

        self.assertEqual({**result.experiment, "n_learners":1, "n_environments":2}, result.experiment)
        self.assertEqual([{'learner_id':0,'name':"lrn1"}], result.learners.to_dicts())
        self.assertEqual([{'environment_id':1,'source':"test"}], result.environments.to_dicts())
        self.assertEqual([{'learner_id':0,'environment_id':1,'index':1,'reward':3},{'learner_id':0,'environment_id':1,'index':2,'reward':4}], result.interactions.to_dicts())

    def test_simple_to_and_from_memory_unknown_transaction(self):
        io = TransactionIO_V4()

        io.write(["T0",{"n_learners":1,"n_environments":2}])
        io.write(["T1",0,{"name":"lrn1"}])
        io.write(["T2",1,{"source":"test"}])
        io.write(["T4",'UNKNOWN'])
        io.write(["T3",[1,0], [{"reward":3},{"reward":4}]])

        result = io.read()

        self.assertEqual({**result.experiment, "n_learners":1, "n_environments":2}, result.experiment)
        self.assertEqual([{'learner_id':0,'name':"lrn1"}], result.learners.to_dicts())
        self.assertEqual([{'environment_id':1,'source':"test"}], result.environments.to_dicts())
        self.assertEqual([{'learner_id':0,'environment_id':1,'index':1,'reward':3},{'learner_id':0,'environment_id':1,'index':2,'reward':4}], result.interactions.to_dicts())

class TransactionIO_Tests(unittest.TestCase):

    def setUp(self) -> None:
        if Path("coba/tests/.temp/transaction.log").exists():
            Path("coba/tests/.temp/transaction.log").unlink()

    def tearDown(self) -> None:
        if Path("coba/tests/.temp/transaction.log").exists():
            Path("coba/tests/.temp/transaction.log").unlink()

    def test_simple_to_and_from_file_v2(self):

        Path("coba/tests/.temp/transaction.log").write_text('["version",2]')

        with self.assertRaises(CobaException):
            TransactionIO("coba/tests/.temp/transaction.log")

    def test_simple_to_and_from_file_v3(self):

        io = TransactionIO_V3("coba/tests/.temp/transaction.log")

        io.write(["T0",1,2])
        io.write(["T1",0,{"name":"lrn1"}])
        io.write(["T2",1,{"source":"test"}])
        io.write(["T3",[1,0], [{"reward":3},{"reward":4}]])

        result = TransactionIO("coba/tests/.temp/transaction.log").read()

        self.assertEqual({"n_learners":1, "n_environments":2}, result.experiment)
        self.assertEqual([{'learner_id':0,'name':"lrn1"}], result.learners.to_dicts())
        self.assertEqual([{'environment_id':1,'source':"test"}], result.environments.to_dicts())
        self.assertEqual([{'learner_id':0,'environment_id':1,'index':1,'reward':3},{'learner_id':0,'environment_id':1,'index':2,'reward':4}], result.interactions.to_dicts())

    def test_simple_to_and_from_file_v4(self):
        io = TransactionIO_V4("coba/tests/.temp/transaction.log")

        io.write(["T0",1,2])
        io.write(["T1",0,{"name":"lrn1"}])
        io.write(["T2",1,{"source":"test"}])
        io.write(["T3",[1,0], [{"reward":3},{"reward":4}]])

        result = io.read()

        self.assertEqual({**result.experiment, "n_learners":1, "n_environments":2}, result.experiment)
        self.assertEqual([{'learner_id':0,'name':"lrn1"}], result.learners.to_dicts())
        self.assertEqual([{'environment_id':1,'source':"test"}], result.environments.to_dicts())
        self.assertEqual([{'learner_id':0,'environment_id':1,'index':1,'reward':3},{'learner_id':0,'environment_id':1,'index':2,'reward':4}], result.interactions.to_dicts())

    def test_simple_to_and_from_memory(self):
        io = TransactionIO()

        io.write(["T0",1,2])
        io.write(["T1",0,{"name":"lrn1"}])
        io.write(["T2",1,{"source":"test"}])
        io.write(["T3",[1,0], [{"reward":3},{"reward":4}]])

        result = io.read()

        self.assertEqual({**result.experiment, "n_learners":1, "n_environments":2}, result.experiment)
        self.assertEqual([{'learner_id':0,'name':"lrn1"}], result.learners.to_dicts())
        self.assertEqual([{'environment_id':1,'source':"test"}], result.environments.to_dicts())
        self.assertEqual([{'learner_id':0,'environment_id':1,'index':1,'reward':3},{'learner_id':0,'environment_id':1,'index':2,'reward':4}], result.interactions.to_dicts())

    def test_simple_resume(self):
        io = TransactionIO("coba/tests/.temp/transaction.log")

        io.write(["T0",1,2])
        io.write(["T1",0,{"name":"lrn1"}])
        io.write(["T2",1,{"source":"test"}])
        io.write(["T3",[1,0], [{"reward":3},{"reward":4}]])

        result = TransactionIO("coba/tests/.temp/transaction.log").read()

        self.assertEqual({**result.experiment, "n_learners":1, "n_environments":2}, result.experiment)
        self.assertEqual([{'learner_id':0,'name':"lrn1"}], result.learners.to_dicts())
        self.assertEqual([{'environment_id':1,'source':"test"}], result.environments.to_dicts())
        self.assertEqual([{'learner_id':0,'environment_id':1,'index':1,'reward':3},{'learner_id':0,'environment_id':1,'index':2,'reward':4}], result.interactions.to_dicts())

class Result_Tests(unittest.TestCase):

    def test_set_plotter(self):
        result = Result()
        
        self.assertIsNotNone(result._plotter)
        result.set_plotter(None)
        self.assertIsNone(result._plotter)

    def test_has_interactions_key(self):

        result = Result({}, {}, {(0,1): {"_packed":{"reward":[1,1]}},(0,2):{"_packed":{"reward":[1,1]}}})

        self.assertEqual("{'Learners': 0, 'Environments': 0, 'Interactions': 4}", str(result))

        self.assertTrue( (0,1) in result._interactions)
        self.assertTrue( (0,2) in result._interactions)

        self.assertEqual(len(result._interactions), 4)

    def test_has_preamble(self):
        self.assertDictEqual(Result({},{},{},{"n_learners":1, "n_environments":2}).experiment, {"n_learners":1, "n_environments":2})

    def test_exception_when_no_file(self):
        with self.assertRaises(Exception):
            Result.from_file("abcd")

    def test_filter_fin_sans_n_interactions(self):
        sims = {1:{}, 2:{}}
        lrns = {1:{}, 2:{}}
        ints = {(1,1):{}, (1,2):{}, (2,1):{}}

        original_result = Result(sims, lrns, ints)
        filtered_result = original_result.filter_fin()

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(1, len(filtered_result.environments))
        self.assertEqual(2, len(filtered_result.learners))
        self.assertEqual(2, len(filtered_result.interactions))

    def test_filter_fin_with_n_interactions(self):

        sims = {1:{}, 2:{}}
        lrns = {1:{}, 2:{}}
        ints = {
            (1,1):{"_packed":{"reward":[1,2  ]}},
            (1,2):{"_packed":{"reward":[1,2,3]}},
            (2,1):{"_packed":{"reward":[1    ]}},
            (2,2):{"_packed":{"reward":[1    ]}}
        }

        original_result = Result(sims, lrns, ints)
        filtered_result = original_result.filter_fin(2)

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(7, len(original_result.interactions))

        self.assertEqual(1, len(filtered_result.environments))
        self.assertEqual(2, len(filtered_result.learners))
        self.assertEqual(4, len(filtered_result.interactions))

    def test_filter_fin_no_finished(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        sims = {1:{}, 2:{}}
        lrns = {1:{}, 2:{}}
        ints = {(1,1):{}, (2,1):{}}

        original_result = Result(sims, lrns, ints)
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

        original_result = Result(sims, lrns, ints)
        filtered_result = original_result.filter_env(environment_id=2)

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(1, len(filtered_result.environments))
        self.assertEqual(2, len(filtered_result.learners))
        self.assertEqual(1, len(filtered_result.interactions))

    def test_filter_env_no_match(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        sims = {1:{}, 2:{}}
        lrns = {1:{}, 2:{}}
        ints = {(1,1):{}, (1,2):{}, (2,1):{}}

        original_result = Result(sims, lrns, ints)
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

        original_result = Result(sims, lrns, ints)
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

        original_result = Result(sims, lrns, ints)
        filtered_result = original_result.filter_lrn(learner_id=[2,1])

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(3, len(original_result.learners))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(2, len(filtered_result.environments))
        self.assertEqual(2, len(filtered_result.learners))
        self.assertEqual(2, len(filtered_result.interactions))

    def test_filter_lrn_no_match(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        sims = {1:{}, 2:{}}
        lrns = {1:{}, 2:{}, 3:{}}
        ints = {(1,1):{}, (1,2):{}, (2,3):{}}

        original_result = Result(sims, lrns, ints)
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

        result = Result(sims, lrns, ints)
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

        self.assertEqual("{'Learners': 3, 'Environments': 2, 'Interactions': 3}", str(Result(sims, lrns, ints)))

    def test_ipython_display_(self):

        with unittest.mock.patch("builtins.print") as mock:

            sims = {1:{}, 2:{}}
            lrns = {1:{}, 2:{}, 3:{}}
            ints = {(1,1):{}, (1,2):{}, (2,3):{}}

            result = Result(sims, lrns, ints)
            result._ipython_display_()
            mock.assert_called_once_with(str(result))

    def test_plot_learners_one_environment_all_default(self):

        lrns = {1:{'full_name':'learner_1'}, 2:{ 'full_name':'learner_2'} }
        ints = {(0,1): {"_packed":{"reward":[1,2]}},(0,2):{"_packed":{"reward":[1,2]}}}

        plotter = TestPlotter()
        result = Result({}, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners()

        expected_lines = [
            ([1,2],[1,1.5],None,"#1f77b4",1,'learner_1'),
            ([1,2],[1,1.5],None,"#ff7f0e",1,'learner_2')
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_two_environments_all_default(self):

        lrns = {1:{'full_name':'learner_1'}, 2:{ 'full_name':'learner_2'} }
        ints = {
            (0,1): {"_packed":{"reward":[1,2]}},
            (0,2): {"_packed":{"reward":[1,2]}},
            (1,1): {"_packed":{"reward":[2,3]}},
            (1,2): {"_packed":{"reward":[2,3]}}
        }

        plotter = TestPlotter()
        result = Result({}, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners()

        expected_lines = [
            ([1,2],[3/2,4/2],None,"#1f77b4",1,'learner_1'),
            ([1,2],[3/2,4/2],None,"#ff7f0e",1,'learner_2')
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_two_environments_err_sd(self):

        lrns = {1:{'full_name':'learner_1'}, 2:{ 'full_name':'learner_2'} }
        ints = {
            (0,1): {"_packed":{"reward":[1,2]}},
            (0,2): {"_packed":{"reward":[1,2]}},
            (1,1): {"_packed":{"reward":[2,3]}},
            (1,2): {"_packed":{"reward":[2,3]}}
        }

        plotter = TestPlotter()
        result = Result({}, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners(err='sd')

        expected_lines = [
            ([1,2],[3/2,4/2],[1/2,1/2],"#1f77b4",1,'learner_1'),
            ([1,2],[3/2,4/2],[1/2,1/2],"#ff7f0e",1,'learner_2')
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_mixed_env_count(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        lrns = {1:{'full_name':'learner_1'}, 2:{ 'full_name':'learner_2'} }
        ints = {
            (0,1): {"_packed":{"reward":[1,2,3]}},
            (1,1): {"_packed":{"reward":[2,3]}},
            (1,2): {"_packed":{"reward":[2,3]}}
        }

        plotter = TestPlotter()
        result = Result({}, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners()

        expected_log = "This result contains environments not present for all learners. Environments not present for all learners have been excluded. To supress this warning in the future call <result>.filter_fin() before plotting."
        expected_lines = [
            ([1,2],[2,5/2],None,"#1f77b4",1,'learner_1'),
            ([1,2],[2,5/2],None,"#ff7f0e",1,'learner_2')
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])
        self.assertEqual(expected_log, CobaContext.logger.sink.items[0])

    def test_plot_learners_mixed_env_length(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        lrns = {1:{'full_name':'learner_1'}, 2:{ 'full_name':'learner_2'} }
        ints = {
            (0,1): {"_packed":{"reward":[1,2,3]}},
            (0,2): {"_packed":{"reward":[1,2]}},
            (1,1): {"_packed":{"reward":[2,3]}},
            (1,2): {"_packed":{"reward":[2,3]}}
        }

        plotter = TestPlotter()
        result = Result({}, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners()

        expected_log = "The result contains environments of different lengths. The plot only includes data which is present in all environments. To only plot environments with a minimum number of interactions call <result>.filter_fin(n_interactions)."
        expected_lines = [
            ([1,2],[3/2,4/2],None,"#1f77b4",1,'learner_1'),
            ([1,2],[3/2,4/2],None,"#ff7f0e",1,'learner_2')
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])
        self.assertEqual(expected_log, CobaContext.logger.sink.items[0])

    def test_plot_learners_sort(self):

        lrns = {1:{ 'full_name':'learner_2'}, 2:{'full_name':'learner_1'}}
        ints = {
            (0,1): {"_packed":{"reward":[1,2]}},
            (1,1): {"_packed":{"reward":[2,3]}},
            (0,2): {"_packed":{"reward":[1,3]}},
            (1,2): {"_packed":{"reward":[2,4]}}
        }

        plotter = TestPlotter()
        result  = Result({}, lrns, ints)

        result.set_plotter(plotter)
        
        result.plot_learners(sort='name')
        result.plot_learners(sort='id')
        result.plot_learners(sort='y')

        sort0 = [
            ([1,2],[3/2,5/2],None,"#1f77b4",1,'learner_1'),
            ([1,2],[3/2,4/2],None,"#ff7f0e",1,'learner_2')
        ]

        sort1 = [
            ([1,2],[3/2,4/2],None,"#1f77b4",1,'learner_2'),
            ([1,2],[3/2,5/2],None,"#ff7f0e",1,'learner_1')
        ]

        self.assertEqual(3, len(plotter.plot_calls))
        
        self.assertEqual(sort0, plotter.plot_calls[0][1])
        self.assertEqual(sort1, plotter.plot_calls[1][1])
        self.assertEqual(sort0, plotter.plot_calls[2][1])

    def test_plot_learners_each(self):

        lrns = {1:{'full_name':'learner_1'}, 2:{ 'full_name':'learner_2'} }
        ints = {
            (0,1): {"_packed":{"reward":[1,2]}},
            (0,2): {"_packed":{"reward":[1,2]}},
            (1,1): {"_packed":{"reward":[2,3]}},
            (1,2): {"_packed":{"reward":[2,3]}}
        }

        plotter = TestPlotter()
        result = Result({}, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners(each=True)

        expected_lines = [
            ([1,2],[1.5,2],None,"#1f77b4",1,'learner_1'),
            ([1,2],[1,1.5],None,"#1f77b4",.15,None),
            ([1,2],[2,2.5],None,"#1f77b4",.15,None),
            ([1,2],[1.5,2],None,"#ff7f0e",1,'learner_2'),
            ([1,2],[1,1.5],None,"#ff7f0e",.15,None),
            ([1,2],[2,2.5],None,"#ff7f0e",.15,None)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_filename(self):

        lrns = {1:{'full_name':'learner_1'}, 2:{ 'full_name':'learner_2'} }
        ints = {
            (0,1): {"_packed":{"reward":[1,2]}},
            (0,2): {"_packed":{"reward":[1,2]}},
            (1,1): {"_packed":{"reward":[2,3]}},
            (1,2): {"_packed":{"reward":[2,3]}}
        }

        plotter = TestPlotter()
        result = Result({}, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners(filename="abc")

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual("abc", plotter.plot_calls[0][7])

    def test_plot_learners_ax(self):

        lrns = {1:{'full_name':'learner_1'}, 2:{ 'full_name':'learner_2'} }
        ints = {
            (0,1): {"_packed":{"reward":[1,2]}},
            (0,2): {"_packed":{"reward":[1,2]}},
            (1,1): {"_packed":{"reward":[2,3]}},
            (1,2): {"_packed":{"reward":[2,3]}}
        }

        plotter = TestPlotter()
        result = Result({}, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners(ax=1)

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(1, plotter.plot_calls[0][0])

    def test_plot_learners_xlim_ylim(self):

        lrns = {1:{'full_name':'learner_1'}, 2:{ 'full_name':'learner_2'} }
        ints = {
            (0,1): {"_packed":{"reward":[1,2]}},
            (0,2): {"_packed":{"reward":[1,2]}},
            (1,1): {"_packed":{"reward":[2,3]}},
            (1,2): {"_packed":{"reward":[2,3]}}
        }

        plotter = TestPlotter()
        result = Result({}, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners(xlim=(1,2), ylim=(2,3))

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual((1,2), plotter.plot_calls[0][5])
        self.assertEqual((2,3), plotter.plot_calls[0][6])

    def test_plot_learners_labels(self):
        lrns = {1:{'full_name':'learner_1'}, 2:{ 'full_name':'learner_2'} }
        ints = {
            (0,1): {"_packed":{"reward":[1,2]}},
            (0,2): {"_packed":{"reward":[1,2]}},
            (1,1): {"_packed":{"reward":[2,3]}},
            (1,2): {"_packed":{"reward":[2,3]}}
        }

        plotter = TestPlotter()
        result = Result({}, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners(labels=['a','b'])

        expected_lines = [
            ([1,2],[3/2,4/2],None,"#1f77b4",1,'a'),
            ([1,2],[3/2,4/2],None,"#ff7f0e",1,'b')
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_empty_results(self):
        plotter = TestPlotter()
        result = Result({}, {}, {})

        result.set_plotter(plotter)
        result.plot_learners()

        expected_lines = []

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

if __name__ == '__main__':
    unittest.main()