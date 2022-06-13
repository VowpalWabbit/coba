import unittest
import unittest.mock
import timeit
import importlib.util

from math import sqrt
from pathlib import Path

from coba.pipes import ListSink
from coba.contexts import CobaContext, IndentLogger
from coba.exceptions import CobaException, CobaExit
from coba.statistics import BinomialConfidenceInterval

from coba.experiments.results import TransactionIO, TransactionIO_V3, TransactionIO_V4
from coba.experiments.results import Result, Table
from coba.experiments.results import MatplotlibPlotter
from coba.experiments.results import moving_average, exponential_moving_average
from coba.experiments.results import FilterPlottingData, SmoothPlottingData, ContrastPlottingData, TransformToXYE

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
                MatplotlibPlotter().plot(None,None,None,None,None,None,None,None,None,None,None,None)

    def test_plot_lines_title_xlabel_ylabel(self):
        with unittest.mock.patch('matplotlib.pyplot.show') as show:
            with unittest.mock.patch('matplotlib.pyplot.savefig') as savefig:
                with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

                    mock_ax = plt_figure().add_subplot()

                    mock_ax.get_xticks.return_value = [1,2]
                    mock_ax.get_xlim.return_value   = [2,2]

                    lines = [
                        ([1,2], [5,6], None, "B", 1.00, 'L1', '-'),
                        ([3,4], [7,8], None, "R", 0.25, 'L2', '-')
                    ]

                    MatplotlibPlotter().plot(None,lines,"title","xlabel","ylabel",None,None,True,True,None,None,None)

                    plt_figure().add_subplot.assert_called_with(111)

                    self.assertEqual(([1,2],[5,6],'-'), mock_ax.plot.call_args_list[0][0])
                    self.assertEqual('B'              , mock_ax.plot.call_args_list[0][1]["color"])
                    self.assertEqual(1                , mock_ax.plot.call_args_list[0][1]["alpha"])
                    self.assertEqual('L1'             , mock_ax.plot.call_args_list[0][1]["label"])

                    self.assertEqual(([3,4],[7,8],'-'), mock_ax.plot.call_args_list[1][0])
                    self.assertEqual('R'              , mock_ax.plot.call_args_list[1][1]["color"])
                    self.assertEqual(.25              , mock_ax.plot.call_args_list[1][1]["alpha"])
                    self.assertEqual('L2'             , mock_ax.plot.call_args_list[1][1]["label"])

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
                        ([1,2], [5,6], [10,11], "B", 1.00, 'L1', '-'),
                        ([3,4], [7,8], [12,13], "R", 0.25, 'L2', '-')
                    ]

                    MatplotlibPlotter().plot(None, lines, "title", "xlabel", "ylabel", None, None, True, True, None, None, None,)

                    plt_figure().add_subplot.assert_called_with(111)

                    self.assertEqual(([1,2],[5,6],[10,11],None,'-'), mock_ax.errorbar.call_args_list[0][0])
                    self.assertEqual('B'                           , mock_ax.errorbar.call_args_list[0][1]["color"])
                    self.assertEqual(1                             , mock_ax.errorbar.call_args_list[0][1]["alpha"])
                    self.assertEqual('L1'                          , mock_ax.errorbar.call_args_list[0][1]["label"])

                    self.assertEqual(([3,4],[7,8],[12,13],None,'-'), mock_ax.errorbar.call_args_list[1][0])
                    self.assertEqual('R'                           , mock_ax.errorbar.call_args_list[1][1]["color"])
                    self.assertEqual(.25                           , mock_ax.errorbar.call_args_list[1][1]["alpha"])
                    self.assertEqual('L2'                          , mock_ax.errorbar.call_args_list[1][1]["label"])

                    mock_ax.set_xticks.called_once_with([2,2])
                    mock_ax.set_title.colled_once_with('title')
                    mock_ax.set_xlabel.called_once_with('xlabel')
                    mock_ax.set_ylabel.called_once_with('ylabel')

                    self.assertEqual(1, show.call_count)
                    self.assertEqual(0, savefig.call_count)

    def test_plot_xlim1(self):
        with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

            lines = [
                ([1,2], [5,6], None, "B", 1.00, 'L1', '-'),
                ([3,4], [7,8], None, "R", 0.25, 'L2', '-')
            ]

            mock_ax = plt_figure().add_subplot()
            mock_ax.get_xticks.return_value = [1,2]
            mock_ax.get_xlim.return_value   = [2,2]
            MatplotlibPlotter().plot(None,lines,"title","xlabel","ylabel",(2,3),None,True,True,None,None,None)
            self.assertEqual(([2],[6],'-'), mock_ax.plot.call_args_list[0][0])
            self.assertEqual(([3],[7],'-'), mock_ax.plot.call_args_list[1][0])

    def test_plot_xlim2(self):
        with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

            lines = [
                ([1,2], [5,6], [4,3], "B", 1.00, 'L1', '-'),
                ([3,4], [7,8], [2,1], "R", 0.25, 'L2', '-')
            ]

            mock_ax = plt_figure().add_subplot()
            mock_ax.get_xticks.return_value = [1,2]
            mock_ax.get_xlim.return_value   = [2,2]
            MatplotlibPlotter().plot(None,lines,"title","xlabel","ylabel",(3,4),None,True,True,None,None,None)
            self.assertEqual(([3,4],[7,8],[2,1],None,'-'), mock_ax.errorbar.call_args_list[0][0])

    def test_plot_ylim(self):
        with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

            lines = [
                ([1,2], [5,6], None, "B", 1.00, 'L1', '-'),
                ([3,4], [7,8], None, "R", 0.25, 'L2', '-')
            ]

            mock_ax = plt_figure().add_subplot()
            mock_ax.get_xticks.return_value = [1,2]
            mock_ax.get_xlim.return_value   = [2,2]
            MatplotlibPlotter().plot(None,lines,"title","xlabel","ylabel",None,(6,7),True,True,None,None,None)
            self.assertEqual(([2],[6],'-'), mock_ax.plot.call_args_list[0][0])
            self.assertEqual(([3],[7],'-'), mock_ax.plot.call_args_list[1][0])

    def test_plot_xrotation(self):
        with unittest.mock.patch('matplotlib.pyplot.xticks') as xticks:
            with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

                lines = [
                    ([1,2], [5,6], None, "B", 1.00, 'L1', '-'),
                    ([3,4], [7,8], None, "R", 0.25, 'L2', '-')
                ]

                mock_ax = plt_figure().add_subplot()
                mock_ax.get_xticks.return_value = [1,2]
                mock_ax.get_xlim.return_value   = [2,2]
                MatplotlibPlotter().plot(None,lines,"title","xlabel","ylabel",None,(6,7),True,True,90,None,None)
                self.assertEqual(([2],[6],'-'), mock_ax.plot.call_args_list[0][0])
                self.assertEqual(([3],[7],'-'), mock_ax.plot.call_args_list[1][0])

                self.assertEqual({'rotation':90},xticks.call_args[1])

    def test_plot_yrotation(self):
        with unittest.mock.patch('matplotlib.pyplot.yticks') as yticks:
            with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

                lines = [
                    ([1,2], [5,6], None, "B", 1.00, 'L1', '-'),
                    ([3,4], [7,8], None, "R", 0.25, 'L2', '-')
                ]

                mock_ax = plt_figure().add_subplot()
                mock_ax.get_xticks.return_value = [1,2]
                mock_ax.get_xlim.return_value   = [2,2]
                MatplotlibPlotter().plot(None,lines,"title","xlabel","ylabel",None,(6,7),True,True,None,90,None)
                self.assertEqual(([2],[6],'-'), mock_ax.plot.call_args_list[0][0])
                self.assertEqual(([3],[7],'-'), mock_ax.plot.call_args_list[1][0])

                self.assertEqual({'rotation':90},yticks.call_args[1])

    def test_plot_no_xticks(self):
        with unittest.mock.patch('matplotlib.pyplot.xticks') as xticks:
            with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

                lines = [
                    ([1,2], [5,6], None, "B", 1.00, 'L1', '-'),
                    ([3,4], [7,8], None, "R", 0.25, 'L2', '-')
                ]

                mock_ax = plt_figure().add_subplot()
                mock_ax.get_xticks.return_value = [1,2]
                mock_ax.get_xlim.return_value   = [2,2]
                MatplotlibPlotter().plot(None,lines,"title","xlabel","ylabel",None,(6,7),False,True,None,None,None)
                self.assertEqual(([2],[6],'-'), mock_ax.plot.call_args_list[0][0])
                self.assertEqual(([3],[7],'-'), mock_ax.plot.call_args_list[1][0])

                self.assertEqual([],xticks.call_args[0][0])

    def test_plot_no_yticks(self):
        with unittest.mock.patch('matplotlib.pyplot.yticks') as yticks:
            with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

                lines = [
                    ([1,2], [5,6], None, "B", 1.00, 'L1', '-'),
                    ([3,4], [7,8], None, "R", 0.25, 'L2', '-')
                ]

                mock_ax = plt_figure().add_subplot()
                mock_ax.get_xticks.return_value = [1,2]
                mock_ax.get_xlim.return_value   = [2,2]
                MatplotlibPlotter().plot(None,lines,"title","xlabel","ylabel",None,(6,7),True,False,None,None,None)
                self.assertEqual(([2],[6],'-'), mock_ax.plot.call_args_list[0][0])
                self.assertEqual(([3],[7],'-'), mock_ax.plot.call_args_list[1][0])

                self.assertEqual([],yticks.call_args[0][0])

    def test_plot_ax(self):
        with unittest.mock.patch('matplotlib.pyplot.show') as show:
            with unittest.mock.patch('matplotlib.pyplot.savefig') as savefig:
                with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

                    mock_ax = plt_figure().add_subplot()
                    self.assertEqual(1, plt_figure().add_subplot.call_count)

                    mock_ax.get_xticks.return_value = [1,2]
                    mock_ax.get_xlim.return_value   = [2,2]

                    lines = [
                        ([1,2], [5,6], None, "B", 1.00, 'L1', '-'),
                        ([3,4], [7,8], None, "R", 0.25, 'L2', '-')
                    ]

                    MatplotlibPlotter().plot(mock_ax,lines,"title","xlabel","ylabel",None,None,True,True,None,None,None)

                    self.assertEqual(1, plt_figure().add_subplot.call_count)

                    self.assertEqual(([1,2],[5,6],'-'), mock_ax.plot.call_args_list[0][0])
                    self.assertEqual('B'              , mock_ax.plot.call_args_list[0][1]["color"])
                    self.assertEqual(1                , mock_ax.plot.call_args_list[0][1]["alpha"])
                    self.assertEqual('L1'             , mock_ax.plot.call_args_list[0][1]["label"])

                    self.assertEqual(([3,4],[7,8],'-'), mock_ax.plot.call_args_list[1][0])
                    self.assertEqual('R'              , mock_ax.plot.call_args_list[1][1]["color"])
                    self.assertEqual(.25              , mock_ax.plot.call_args_list[1][1]["alpha"])
                    self.assertEqual('L2'             , mock_ax.plot.call_args_list[1][1]["label"])

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
        plotter.plot(None, [[]], 'abc', 'def', 'efg', (1,0), (0,1), True, True, None, None, None)

        expected_log = "The xlim end is less than the xlim start. Plotting is impossible."

        self.assertEqual(expected_log, CobaContext.logger.sink.items[0])

    def test_plot_bad_ylim(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        plotter = MatplotlibPlotter()
        plotter.plot(None, [[]], 'abc', 'def', 'efg', (0,1), (1,0), True, True, None, None, None)

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
                        ([1,2], [5,6], None, "B", 1.00, 'L1', '-'),
                        ([3,4], [7,8], None, "R", 0.25, 'L2', '-')
                    ]

                    MatplotlibPlotter().plot(None,lines,"title","xlabel","ylabel",None,None,True,True,None,None,"abc")

                    plt_figure().add_subplot.assert_called_with(111)

                    self.assertEqual(([1,2],[5,6],'-'), mock_ax.plot.call_args_list[0][0])
                    self.assertEqual('B'              , mock_ax.plot.call_args_list[0][1]["color"])
                    self.assertEqual(1                , mock_ax.plot.call_args_list[0][1]["alpha"])
                    self.assertEqual('L1'             , mock_ax.plot.call_args_list[0][1]["label"])

                    self.assertEqual(([3,4],[7,8],'-'), mock_ax.plot.call_args_list[1][0])
                    self.assertEqual('R'              , mock_ax.plot.call_args_list[1][1]["color"])
                    self.assertEqual(.25              , mock_ax.plot.call_args_list[1][1]["alpha"])
                    self.assertEqual('L2'             , mock_ax.plot.call_args_list[1][1]["label"])

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
            plotter.plot(None, [], 'abc', 'def', 'efg', None, None, True, True, None, None, None)

            self.assertEqual(0, plt_figure().add_subplot.call_count)
            self.assertEqual(["No data was found for plotting in the given results."], CobaContext.logger.sink.items)

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

    def test_plot_learners_bad_x_index(self):
        with self.assertRaises(CobaException):
            Result({}, {}, {}).plot_learners(x=['index','a'])

    def test_plot_learners_one_environment_all_default(self):

        lrns = {1:{'full_name':'learner_1'}, 2:{ 'full_name':'learner_2'} }
        ints = {(0,1): {"_packed":{"reward":[1,2]}},(0,2):{"_packed":{"reward":[1,2]}}}

        plotter = TestPlotter()
        result = Result({}, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners()

        expected_lines = [
            ((1,2),(1,1.5),(None,None),0,1,'learner_1','-'),
            ((1,2),(1,1.5),(None,None),1,1,'learner_2','-')
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
            ((1,2),(3/2,4/2),(None,None),0,1,'learner_1','-'),
            ((1,2),(3/2,4/2),(None,None),1,1,'learner_2','-')
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_one_environment_x_not_index(self):

        envs = {0:{}}
        lrns = {1:{'full_name':'learner_1'}, 2:{ 'full_name':'learner_2'} }
        ints = {(0,1): {"_packed":{"reward":[1,2]}},(0,2):{"_packed":{"reward":[1,2]}}}

        plotter = TestPlotter()
        result = Result(envs, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners(x=['environment_id'])

        expected_lines = [
            (('0',),(1.5,),(None,),0,1,'learner_1','.'),
            (('0',),(1.5,),(None,),1,1,'learner_2','.')
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
            ((1,2),(3/2,4/2),(sqrt(2)/2,sqrt(2)/2),0,1,'learner_1','-'),
            ((1,2),(3/2,4/2),(sqrt(2)/2,sqrt(2)/2),1,1,'learner_2','-')
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
            ((1,2),(2,5/2),(None,None),0,1,'learner_1','-'),
            ((1,2),(2,5/2),(None,None),1,1,'learner_2','-')
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

        expected_log = "This result contains environments of different lengths. The plot only includes interactions up to the shortest environment. To supress this warning in the future call <result>.filter_fin(n_interactions) before plotting."
        expected_lines = [
            ((1,2),(3/2,4/2),(None,None),0,1,'learner_1','-'),
            ((1,2),(3/2,4/2),(None,None),1,1,'learner_2','-')
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])
        self.assertEqual(expected_log, CobaContext.logger.sink.items[0])

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
        self.assertEqual("abc", plotter.plot_calls[0][11])

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
            ((1,2),(3/2,4/2),(None,None),0,1,'a','-'),
            ((1,2),(3/2,4/2),(None,None),1,1,'b','-')
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_empty_results(self):
        plotter = TestPlotter()
        result = Result({}, {}, {})

        result.set_plotter(plotter)
        
        with self.assertRaises(CobaException):
            result.plot_learners()

    def test_plot_contrast_bad_x_index(self):
        with self.assertRaises(CobaException):
            Result({}, {}, {}).plot_contrast(0, 1, x=['index','a'])

    def test_plot_contrast(self):
        with self.assertRaises(CobaException):
            Result({}, {}, {}).plot_contrast(0, 1, x=['index','a'])

    def test_plot_contrast_one_environment_all_default(self):

        lrns = {1:{'full_name':'learner_1'}, 2:{'full_name':'learner_2'} }
        ints = {(0,1): {"_packed":{"reward":[0,3,9]}},(0,2):{"_packed":{"reward":[1,2,6]}}}

        plotter = TestPlotter()
        result = Result({}, lrns, ints)

        result.set_plotter(plotter)
        result.plot_contrast(1,2,'index')

        expected_lines = [
            ((1,)  ,(-1,), (None,), 2     , 1, 'learner_2 (1)', '-'),
            ((2,)  ,( 0,), (None,), 1     , 1, 'Tie (1)'      , '-'),
            ((3,)  ,( 1,), (None,), 0     , 1, 'learner_1 (1)', '-'),
            ((1,3,),(0,0), None   , "#888", 1, None           , '-')
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines[0], plotter.plot_calls[0][1][0])
        self.assertEqual(expected_lines[1], plotter.plot_calls[0][1][1])
        self.assertEqual(expected_lines[2], plotter.plot_calls[0][1][2])
        self.assertEqual(expected_lines[3], plotter.plot_calls[0][1][3])

    def test_plot_contrast_one_environment_env_index(self):

        envs = {0:{'a':1}, 1:{'a':2}, 2:{'a':3}}
        lrns = {1:{'full_name':'learner_1'},2:{'full_name':'learner_2'}}
        ints = {
            (0,1): {"_packed":{"reward":[0,3,12]}},
            (0,2): {"_packed":{"reward":[1,2,6 ]}},
            (1,1): {"_packed":{"reward":[0,3,6 ]}},
            (1,2): {"_packed":{"reward":[1,2,6 ]}},
            (2,1): {"_packed":{"reward":[0,3,9 ]}},
            (2,2): {"_packed":{"reward":[1,2,6 ]}},
        }

        plotter = TestPlotter()
        result = Result(envs, lrns, ints)

        result.set_plotter(plotter)
        result.plot_contrast(1,2,'a')

        expected_lines = [
            (('2',)   , (0, ), (None,    ), 1     , 1, 'Tie (1)'      , '.'),
            (('3','1'), (1,2), (None,None), 0     , 1, 'learner_1 (2)', '.'),
            (('2','1'), (0,0), None       , "#888", 1, None           , '-')
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines[0], plotter.plot_calls[0][1][0])
        self.assertEqual(expected_lines[1], plotter.plot_calls[0][1][1])
        self.assertEqual(expected_lines[2], plotter.plot_calls[0][1][2])

class moving_average_Tests(unittest.TestCase):

    def test_sliding_windows(self):
        self.assertEqual([0,1/2,1/2,0/2,1/2], moving_average([0,1,0,0,1],span=2))
        self.assertEqual([0,1/2,1/3,1/3,1/3], moving_average([0,1,0,0,1],span=3))
        self.assertEqual([0,1/2,1/3,1/4,2/4], moving_average([0,1,0,0,1],span=4))

    def test_rolling_windows(self):
        self.assertEqual([0,1/2,1/3,1/4,2/5], moving_average([0,1,0,0,1],span=None))
        self.assertEqual([0,1/2,1/3,1/4,2/5], moving_average([0,1,0,0,1],span=5   ))
        self.assertEqual([0,1/2,1/3,1/4,2/5], moving_average([0,1,0,0,1],span=6   ))

    def test_no_window(self):
        self.assertEqual([0,1,0,0,1], moving_average([0,1,0,0,1],span=1))

class exponential_moving_average_Tets(unittest.TestCase):

    def test_span_2(self):
        self.assertEqual([1,1.75,2.62,3.55], [round(v,2) for v in exponential_moving_average([1,2,3,4],span=2)])

class FilterPlottingData_Tests(unittest.TestCase):

    def test_normal_use_case(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        rows = [
            {"environment_id":0, "learner_id":0, "reward":[]},
            {"environment_id":0, "learner_id":1, "reward":[]}
        ]

        expected_rows = rows
        actual_rows = FilterPlottingData().filter(rows, ['index'], "reward", None)

        self.assertEqual(expected_rows,actual_rows)
        self.assertEqual([], CobaContext.logger.sink.items)

    def test_learner_id_filter(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        rows = [
            {"environment_id":0, "learner_id":0, "reward":[]},
            {"environment_id":0, "learner_id":1, "reward":[]},
            {"environment_id":0, "learner_id":2, "reward":[]}
        ]

        expected_rows = [
            {"environment_id":0, "learner_id":0, "reward":[]},
            {"environment_id":0, "learner_id":1, "reward":[]},
        ]

        actual_rows = FilterPlottingData().filter(rows, ['index'], "reward", [0,1])

        self.assertEqual(expected_rows,actual_rows)
        self.assertEqual([], CobaContext.logger.sink.items)

    def test_all_env_finished_with_equal_lengths_and_x_index(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        rows = [
            {"environment_id":0, "learner_id":0, "reward":[]},
            {"environment_id":0, "learner_id":1, "reward":[]}
        ]

        expected_rows = rows
        actual_rows = FilterPlottingData().filter(rows, ['index'], "reward", None)

        self.assertEqual(expected_rows,actual_rows)
        self.assertEqual([], CobaContext.logger.sink.items)

    def test_all_env_finished_with_unequal_lengths_and_x_index(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        rows = [
            {"environment_id":0, "learner_id":0, "reward":[1]},
            {"environment_id":0, "learner_id":1, "reward":[1]},
            {"environment_id":1, "learner_id":0, "reward":[1,1]},
            {"environment_id":1, "learner_id":1, "reward":[1,1]}
        ]

        expected_rows = [
            {"environment_id":0, "learner_id":0, "reward":[1]},
            {"environment_id":0, "learner_id":1, "reward":[1]},
            {"environment_id":1, "learner_id":0, "reward":[1]},
            {"environment_id":1, "learner_id":1, "reward":[1]}
        ]
        actual_rows = FilterPlottingData().filter(rows, ['index'], "reward", None)

        self.assertEqual(expected_rows,actual_rows)
        self.assertEqual(["This result contains environments of different lengths. The plot only includes interactions up to the shortest environment. To supress this warning in the future call <result>.filter_fin(n_interactions) before plotting."], CobaContext.logger.sink.items)

    def test_not_all_env_finished_with_unequal_lengths_and_x_index(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        rows = [
            {"environment_id":0, "learner_id":0, "reward":[1]},
            {"environment_id":1, "learner_id":0, "reward":[1,1]},
            {"environment_id":1, "learner_id":1, "reward":[1,1]}
        ]

        expected_rows = [
            {"environment_id":1, "learner_id":0, "reward":[1]},
            {"environment_id":1, "learner_id":1, "reward":[1]}
        ]
        actual_rows = FilterPlottingData().filter(rows, ['index'], "reward", None)

        self.assertEqual(expected_rows,actual_rows)
        self.assertIn("This result contains environments not present for all learners. Environments not present for all learners have been excluded. To supress this warning in the future call <result>.filter_fin() before plotting.", CobaContext.logger.sink.items)
        self.assertIn("This result contains environments of different lengths. The plot only includes interactions up to the shortest environment. To supress this warning in the future call <result>.filter_fin(n_interactions) before plotting.", CobaContext.logger.sink.items)

    def test_no_env_finished_for_all_learners(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        rows = [
            {"environment_id":0, "learner_id":0, "reward":[1]},
            {"environment_id":1, "learner_id":1, "reward":[1,1]}
        ]

        with self.assertRaises(CobaException) as e:
            FilterPlottingData().filter(rows, ['index'], "reward", None)

        self.assertEqual(str(e.exception),"This result does not contain an environment which has been finished for every learner. Plotting has been stopped.")

    def test_no_data(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        rows = []

        with self.assertRaises(CobaException) as e:
            FilterPlottingData().filter(rows, ['index'], "reward", None)

        self.assertEqual(str(e.exception),"This result doesn't contain any evaluation data to plot.")

    def test_all_env_finished_with_unequal_lengths_and_not_x_index(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        rows = [
            {"environment_id":0, "learner_id":0, "reward":[1]},
            {"environment_id":0, "learner_id":1, "reward":[1]},
            {"environment_id":1, "learner_id":0, "reward":[1,1]},
            {"environment_id":1, "learner_id":1, "reward":[1,1]}
        ]

        expected_rows = [
            {"environment_id":0, "learner_id":0, "reward":[1  ]},
            {"environment_id":0, "learner_id":1, "reward":[1  ]},
            {"environment_id":1, "learner_id":0, "reward":[1,1]},
            {"environment_id":1, "learner_id":1, "reward":[1,1]}
        ]

        actual_rows = FilterPlottingData().filter(rows, ['openml_task'], "reward", None)
        self.assertEqual(expected_rows,actual_rows)

class SmoothPlottingData_Tests(unittest.TestCase):

    def test_full_span(self):

        rows = [
            {"environment_id":0, "learner_id":0, "reward":[0,2,4]},
            {"environment_id":0, "learner_id":1, "reward":[0,4,8]}
        ]

        expected_rows = [
            {"environment_id":0, "learner_id":0, "reward":[0,1,2]},
            {"environment_id":0, "learner_id":1, "reward":[0,2,4]}
        ]

        actual_rows = SmoothPlottingData().filter(rows, "reward", None)

        self.assertEqual(expected_rows,actual_rows)

    def test_part_span(self):

        rows = [
            {"environment_id":0, "learner_id":0, "reward":[0,2,2]},
            {"environment_id":0, "learner_id":1, "reward":[0,4,4]}
        ]

        expected_rows = [
            {"environment_id":0, "learner_id":0, "reward":[0,1,2]},
            {"environment_id":0, "learner_id":1, "reward":[0,2,4]}
        ]

        actual_rows = SmoothPlottingData().filter(rows, "reward", 2)

        self.assertEqual(expected_rows,actual_rows)

    def test_no_span(self):

        rows = [
            {"environment_id":0, "learner_id":0, "reward":[0,2,4]},
            {"environment_id":0, "learner_id":1, "reward":[0,4,8]}
        ]

        expected_rows = [
            {"environment_id":0, "learner_id":0, "reward":[0,2,4]},
            {"environment_id":0, "learner_id":1, "reward":[0,4,8]}
        ]

        actual_rows = SmoothPlottingData().filter(rows, "reward", 1)

        self.assertEqual(expected_rows,actual_rows)

class ContrastPlottingData_Tests(unittest.TestCase):

    def test_2_env_diff_0(self):

        rows = [
            {"environment_id":0, "learner_id":0, "reward":[0,2,4]},
            {"environment_id":0, "learner_id":1, "reward":[0,4,8]},
            {"environment_id":1, "learner_id":0, "reward":[0,0,0]},
            {"environment_id":1, "learner_id":1, "reward":[1,2,3]}
        ]

        expected_rows = [
            {"environment_id":0, "reward":[0,2,4]},
            {"environment_id":1, "reward":[1,2,3]}
        ]

        actual_rows = ContrastPlottingData().filter(rows, "reward", "diff", 1)

        self.assertEqual(expected_rows,actual_rows)

    def test_2_env_diff_1(self):

        rows = [
            {"environment_id":0, "learner_id":0, "reward":[0,2,4]},
            {"environment_id":0, "learner_id":1, "reward":[0,4,8]},
            {"environment_id":1, "learner_id":0, "reward":[0,0,0]},
            {"environment_id":1, "learner_id":1, "reward":[1,2,3]}
        ]

        expected_rows = [
            {"environment_id":0, "reward":[-0,-2,-4]},
            {"environment_id":1, "reward":[-1,-2,-3]}
        ]

        actual_rows = ContrastPlottingData().filter(rows, "reward", "diff", 0)

        self.assertEqual(expected_rows,actual_rows)

    def test_2_env_prob_0(self):

        rows = [
            {"environment_id":0, "learner_id":0, "reward":[0,2,4]},
            {"environment_id":0, "learner_id":1, "reward":[0,4,8]},
            {"environment_id":1, "learner_id":0, "reward":[0,0,0]},
            {"environment_id":1, "learner_id":1, "reward":[1,2,3]}
        ]

        expected_rows = [
            {"environment_id":0, "reward":[0,1,1]},
            {"environment_id":1, "reward":[1,1,1]}
        ]

        actual_rows = ContrastPlottingData().filter(rows, "reward", "prob", 1)
        
        self.assertEqual(expected_rows,actual_rows)

    def test_2_env_prob_1(self):

        rows = [
            {"environment_id":0, "learner_id":0, "reward":[0,2,4]},
            {"environment_id":0, "learner_id":1, "reward":[0,4,8]},
            {"environment_id":1, "learner_id":0, "reward":[0,0,0]},
            {"environment_id":1, "learner_id":1, "reward":[1,2,3]}
        ]

        expected_rows = [
            {"environment_id":0, "reward":[0,0,0]},
            {"environment_id":1, "reward":[0,0,0]}
        ]
        
        actual_rows = ContrastPlottingData().filter(rows, "reward", "prob", 0)
        
        self.assertEqual(expected_rows,actual_rows)

class TransformXYE_Tests(unittest.TestCase):

    def test_x_index_no_err(self):

        rows = [
            {"environment_id":0, "reward":[0,2,4]},
            {"environment_id":1, "reward":[0,4,8]}
        ]

        envs = {
            0: {'a':1,'b':2},
            1: {'a':3,'b':4},
        }

        expected_rows = [
            (1, 0, None),
            (2, 3, None),
            (3, 6, None),
        ]

        actual_rows = TransformToXYE().filter(rows, envs, ['index'], "reward", None)

        self.assertEqual(expected_rows,actual_rows)

    def test_x_index_sd_err(self):

        rows = [
            {"environment_id":0, "reward":[0,1]},
            {"environment_id":1, "reward":[0,3]}
        ]

        envs = {
            0: {'a':1,'b':2},
            1: {'a':3,'b':4},
        }

        expected_rows = [
            (1, 0, 0      ),
            (2, 2, sqrt(2)),
        ]

        actual_rows = TransformToXYE().filter(rows, envs, ['index'], "reward", 'sd')

        self.assertEqual(expected_rows,actual_rows)

    def test_x_index_se_err(self):

        rows = [
            {"environment_id":0, "reward":[0,1]},
            {"environment_id":1, "reward":[0,3]}
        ]

        envs = {
            0: {'a':1,'b':2},
            1: {'a':3,'b':4},
        }

        expected_rows = [
            (1, 0, (0   ,0   )),
            (2, 2, (1.96,1.96)),
        ]

        actual_rows = TransformToXYE().filter(rows, envs, ['index'], "reward", 'se')

        self.assertEqual(expected_rows,actual_rows)

    def test_x_index_bs_err(self):

        rows = [
            {"environment_id":0, "learner_id":0, "reward":[0,1]},
            {"environment_id":0, "learner_id":1, "reward":[0,0]}
        ]

        envs = {
            0: {'a':1,'b':2},
            1: {'a':3,'b':4},
        }

        expected_rows = [
            (1,   0, (0 , 0)),
            (2, 1/2, (.5,.5)),
        ]

        actual_rows = TransformToXYE().filter(rows, envs, ['index'], "reward", 'bs')
        self.assertEqual(expected_rows,actual_rows)

    def test_x_index_bi_err(self):

        rows = [
            {"environment_id":0, "reward":[0,1]},
            {"environment_id":1, "reward":[0,1]},
            {"environment_id":2, "reward":[0,1]},
            {"environment_id":3, "reward":[0,0]},
            {"environment_id":4, "reward":[0,0]},
            {"environment_id":5, "reward":[0,0]},
        ]

        envs = {
            0: {'a':1,'b':2},
            1: {'a':3,'b':4},
        }

        expected_rows = [
            (1,   0, (0         , 0.39033   )),
            (2, 1/2, (.5-0.18761, 0.81238-.5)),
        ]

        actual_rows = TransformToXYE().filter(rows, envs, ['index'], "reward", 'bi')

        self.assertEqual      (expected_rows[0][0:2] ,actual_rows[0][0:2]   )
        self.assertEqual      (expected_rows[0][2][0],actual_rows[0][2][0]  )
        self.assertAlmostEqual(expected_rows[0][2][1],actual_rows[0][2][1],4)

        self.assertEqual      (expected_rows[1][0:2] ,actual_rows[1][0:2]   )
        self.assertAlmostEqual(expected_rows[1][2][0],actual_rows[1][2][0],4)
        self.assertAlmostEqual(expected_rows[1][2][1],actual_rows[1][2][1],4)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "sklearn is not installed so we must skip this test.")
    def test_x_index_bi_exact_err(self):

        rows = [
            {"environment_id":0, "reward":[1]},
            {"environment_id":1, "reward":[1]},
            {"environment_id":2, "reward":[1]},
            {"environment_id":3, "reward":[0]},
            {"environment_id":4, "reward":[0]},
            {"environment_id":5, "reward":[0]},
        ]

        envs = {
            0: {'a':1,'b':2},
            1: {'a':3,'b':4},
        }

        expected_rows = [
            (1, 1/2, (.5-0.11811, 0.88188-.5)),
        ]

        actual_rows = TransformToXYE().filter(rows, envs, ['index'], "reward", BinomialConfidenceInterval('clopper-pearson'))

        self.assertEqual      (expected_rows[0][0:2] ,actual_rows[0][0:2]   )
        self.assertAlmostEqual(expected_rows[0][2][0],actual_rows[0][2][0],4)
        self.assertAlmostEqual(expected_rows[0][2][1],actual_rows[0][2][1],4)

    def test_one_env_index_bs_err(self):

        rows = [
            {"environment_id":0, "reward":[1,2]},
            {"environment_id":1, "reward":[2,4]},
            {"environment_id":2, "reward":[3,6]},
            {"environment_id":3, "reward":[4,8]},
        ]

        envs = {
            0: {'a':1,'b':2},
            1: {'a':3,'b':4},
            2: {'a':1,'b':2},
            3: {'a':3,'b':4},
        }

        expected_rows = [
            ('1', 4, (2,2)),
            ('3', 6, (2,2)),
        ]

        actual_rows = TransformToXYE().filter(rows, envs, 'a', "reward", 'bs')

        self.assertEqual(expected_rows,actual_rows)

    def test_two_env_index_bs_err(self):

        rows = [
            {"environment_id":0, "reward":[1,2]},
            {"environment_id":1, "reward":[2,4]},
            {"environment_id":2, "reward":[3,6]},
            {"environment_id":3, "reward":[4,8]},
        ]

        envs = {
            0: {'a':1,'b':2},
            1: {'a':3,'b':5},
            2: {'a':1,'b':2},
            3: {'a':3,'b':6},
        }

        expected_rows = [
            ('(1, 2)', 4, (2,2)),
            ('(3, 5)', 4, (0,0)),
            ('(3, 6)', 8, (0,0)),
        ]

        actual_rows = TransformToXYE().filter(rows, envs, ['a','b'], "reward", 'bs')

        self.assertEqual(expected_rows,actual_rows)

if __name__ == '__main__':
    unittest.main()
