import unittest
import unittest.mock
import importlib.util

from math import sqrt
from pathlib import Path

from coba.pipes import ListSink
from coba.contexts import CobaContext, IndentLogger
from coba.exceptions import CobaException, CobaExit
from coba.statistics import BinomialConfidenceInterval

from coba.experiments.results import TransactionIO, TransactionIO_V3, TransactionIO_V4
from coba.experiments.results import Result, Table, Count, Repeat, Compress
from coba.experiments.results import MatplotlibPlotter
from coba.experiments.results import moving_average, exponential_moving_average, old_to_new
from coba.experiments.results import FilterPlottingData, SmoothPlottingData, ContrastPlottingData, TransformToXYE

class TestPlotter:

    def __init__(self):
        self.plot_calls = []

    def plot(self, *args) -> None:
        self.plot_calls.append(args)

class Repeat_Tests(unittest.TestCase):

    def test_iter(self):
        repeat = Repeat(1,10)
        self.assertEqual(list(repeat),[1]*10)

    def test_len(self):
        repeat = Repeat(1,10)
        self.assertEqual(len(repeat),10)

    def test_eq(self):
        repeat = Repeat(1,10)
        self.assertEqual(repeat,Repeat(1,10))

class Count_Tests(unittest.TestCase):

    def test_iter(self):
        count = Count(1,10)
        self.assertEqual(list(count),list(range(1,10)))

    def test_len(self):
        count = Count(1,10)
        self.assertEqual(len(count),9)

    def test_eq(self):
        count = Count(1,10)
        self.assertEqual(count,Count(1,10))

class Compress_Tests(unittest.TestCase):

    def test_iter(self):
        compress = Compress([1,2,3],[True,False,True])
        self.assertEqual(list(compress),[1,3])

    def test_len(self):
        compress = Compress([1,2,3],[True,False,True])
        self.assertEqual(len(compress),2)
    
    def test_eq(self):
        compress = Compress([1,2,3],[True,False,True])
        self.assertEqual(compress,Compress([1,2,3],[True,False,True]))

class old_to_new_Tests(unittest.TestCase):

    def test_simple(self):
        envs,lrns,ints = old_to_new({}, {}, {(0,1): {"_packed":{"reward":[1,3]}},(0,2):{"_packed":{"reward":[1,4]}}})
        self.assertEqual(envs,Table(['environment_id']))
        self.assertEqual(lrns,Table(['learner_id']))
        self.assertEqual(ints,Table(['environment_id', 'learner_id', 'index', 'reward']).insert(rows=[(0,1,1,1),(0,1,2,3),(0,2,1,1),(0,2,2,4)]))

    def test_simple2(self):
        envs,lrns,ints = old_to_new({}, {}, {})
        self.assertEqual(envs,Table(['environment_id']))
        self.assertEqual(lrns,Table(['learner_id']))
        self.assertEqual(ints,Table(['environment_id', 'learner_id', 'index']))

    def test_simple3(self):
        envs,lrns,ints = old_to_new({}, {}, {(0,1): {"_packed":{"reward":[1,3]}},(0,2):{"_packed":{"z":[1,4]}}})
        self.assertEqual(envs,Table(['environment_id']))
        self.assertEqual(lrns,Table(['learner_id']))
        self.assertEqual(ints,Table(['environment_id', 'learner_id', 'index', 'reward','z']).insert(rows=[(0,1,1,1,None),(0,1,2,3,None),(0,2,1,None,1),(0,2,2,None,4)]))

class Table_Tests(unittest.TestCase):

    def test_table_str(self):
        self.assertEqual("{'Columns': ['id', 'col'], 'Rows': 2}",str(Table(['id','col']).insert(rows=[[1,2],[2,3]])))

    def test_ipython_display(self):
        with unittest.mock.patch("builtins.print") as mock:
            table = Table(['id','col']).insert(rows=[[1,2],[2,3]])
            table._ipython_display_()
            mock.assert_called_once_with(str(table))

    def test_insert_item(self):
        table = Table(['a','b']).insert(rows=[['a','B'],['A','B']])

        self.assertEqual(list(table), [('a','B'),('A','B')])
        self.assertEqual(table.col_names, ['a','b'])
        self.assertEqual(2, len(table))

    def test_filter_kwarg_str(self):
        table = Table(['a','b']).insert(rows=[['a','b'],['A','B']])

        filtered_table = table.filter(b="B")

        self.assertEqual(2, len(table))
        self.assertEqual([('a','b'),('A','B')],list(table))

        self.assertEqual(1, len(filtered_table))
        self.assertEqual([('A','B')], list(filtered_table))

    def test_filter_kwarg_int_1(self):
        table = Table(['a','b']).insert(rows=[['1','b'],['12','B']])

        filtered_table = table.filter(a=1,comparison='match')

        self.assertEqual(2, len(table))
        self.assertEqual([('1','b'),('12','B')], list(table))

        self.assertEqual(1, len(filtered_table))
        self.assertEqual([('1','b')], list(filtered_table))

    def test_filter_kwarg_int_2(self):
        table = Table(['a','b']).insert(rows=[[1,'b'],[12,'B']])

        filtered_table = table.filter(a=1)

        self.assertEqual(2, len(table))
        self.assertEqual([(1,'b'),(12,'B')], list(table))

        self.assertEqual(1, len(filtered_table))
        self.assertEqual([(1,'b')], list(filtered_table))

    def test_filter_kwarg_pred(self):
        table = Table(['a','b']).insert(rows=[['1','b'],['12','B']])

        filtered_table = table.filter(a= lambda a: a =='1')

        self.assertEqual(2, len(table))
        self.assertEqual([('1','b'),('12','B')], list(table))

        self.assertEqual(1, len(filtered_table))
        self.assertEqual([('1','b')], list(filtered_table))

    def test_filter_kwarg_multi(self):
        table = Table(['a','b','c']).insert(rows=[
            ['1', 'b', 'c'],
            ['2', 'b', 'C'],
            ['3', 'B', 'c'],
            ['4', 'B', 'C']
        ])

        filtered_table = table.filter(b="b", c="C")

        self.assertEqual(4, len(table))
        self.assertEqual(3, len(filtered_table))
        self.assertEqual([('1','b','c'),('2','b','C'),('4','B','C')], list(filtered_table))

    def test_filter_without_any(self):
        table = Table(['a','b','c']).insert(rows=[
            ['1', 'b', 'c'],
            ['2', 'b', 'C'],
        ])

        filtered_table = table.filter()

        self.assertEqual(2, len(table))
        self.assertEqual(2, len(filtered_table))
        self.assertEqual([('1','b','c'),('2','b','C')], list(filtered_table))

    def test_filter_pred(self):
        table = Table(['a','b']).insert(rows=[['A','B'],['a','b']])

        filtered_table = table.filter(lambda row: row[1]=="B")

        self.assertEqual(2, len(table))
        self.assertEqual([('A','B'),('a','b')], list(table))

        self.assertEqual(1, len(filtered_table))
        self.assertEqual([('A','B')], list(filtered_table))

    def test_filter_sequence_1(self):
        table = Table(['a','b']).insert(rows=[['a','b'], ['A','B'], ['1','C']])

        filtered_table = table.filter(a=['a','1'])

        self.assertEqual(3, len(table))
        self.assertEqual([('a','b'),('A','B'),('1','C')], list(table))

        self.assertEqual(2, len(filtered_table))
        self.assertEqual([('a','b'),('1','C')], list(filtered_table))

    def test_filter_sequence_2(self):
        table = Table(['a','b']).insert(rows=[['1','b'], ['2','B'], ['3','C']])

        filtered_table = table.filter(a=[1,2],comparison='match')

        self.assertEqual(3, len(table))
        self.assertEqual([('1','b'),('2','B'),('3','C')], list(table))

        self.assertEqual(2, len(filtered_table))
        self.assertEqual([('1','b'),('2','B')], list(filtered_table))

    def test_filter_sequence_3(self):
        table = Table(['a']).insert(rows=[[ ['1']], [ ['2']], [ ['3']]])

        filtered_table = table.filter(a=[['1']],comparison='in')

        self.assertEqual(3, len(table))
        self.assertEqual([(['1'],),(['2'],),(['3'],)], list(table))

        self.assertEqual(1, len(filtered_table))
        self.assertEqual([(['1'],)], list(filtered_table))

    def test_filter_repeat_less(self):
        table = Table(['a']).insert(cols=[Repeat(1,10)])
        table = table.insert(cols=[Repeat(2,10)])

        filtered_table = table.filter(a=1,comparison='<=')

        self.assertEqual(20, len(table))

        self.assertEqual(10, len(filtered_table))

    def test_filter_repeat_more(self):
        table = Table(['a']).insert(cols=[Repeat(1,10)])
        table = table.insert(cols=[Repeat(2,10)])

        filtered_table = table.filter(a=1,comparison='>')

        self.assertEqual(20, len(table))

        self.assertEqual(10, len(filtered_table))

    def test_filter_count_less(self):
        table = Table(['a','b']).insert(cols=[Count(1,11),Count(1,11)])
        table = table.insert(cols=[Count(1,11),Count(1,11)])

        filtered_table = table.filter(a=5,comparison='<=')

        self.assertEqual(20, len(table))

        self.assertEqual(10, len(filtered_table))

    def test_filter_count_more(self):
        table = Table(['a','b']).insert(cols=[Count(1,11),Count(1,11)])
        table = table.insert(cols=[Count(1,11),Count(1,11)])

        filtered_table = table.filter(a=5,comparison='>')

        self.assertEqual(20, len(table))

        self.assertEqual(10, len(filtered_table))

    def test_filter_count_much_less(self):
        table = Table(['a','b']).insert(cols=[Count(1,11),Count(1,11)])
        table = table.insert(cols=[Count(1,11),Count(1,11)])

        filtered_table = table.filter(a=600,comparison='<=')

        self.assertEqual(20, len(table))

        self.assertEqual(20, len(filtered_table))

    def test_filter_count_much_more_true(self):
        table = Table(['a','b']).insert(cols=[Count(1,11),Count(1,11)])
        table = table.insert(cols=[Count(1,11),Count(1,11)])

        filtered_table = table.filter(a=-600,comparison='>=')

        self.assertEqual(20, len(table))

        self.assertEqual(20, len(filtered_table))

    def test_filter_count_much_more_false(self):
        table = Table(['a','b']).insert(cols=[Count(1,11),Count(1,11)])
        table = table.insert(cols=[Count(1,11),Count(1,11)])

        filtered_table = table.filter(a=600,comparison='>=')

        self.assertEqual(20, len(table))

        self.assertEqual(0, len(filtered_table))

    def test_filter_repeat_equals(self):
        table = Table(['a']).insert(cols=[Repeat(1,2)]).insert(cols=[Repeat(2,2)])

        filtered_table = table.filter(a=1,comparison='=')

        self.assertEqual(4, len(table))

        self.assertEqual(2, len(filtered_table))

    def test_filter_match_number_number(self):
        table = Table(['a']).insert(cols=[Repeat(1,2)]).insert(cols=[Repeat(2,2)])

        filtered_table = table.filter(a=1,comparison='match')

        self.assertEqual(4, len(table))

        self.assertEqual(2, len(filtered_table))

    def test_filter_match_number_str(self):
        table = Table(['a']).insert(cols=[Repeat('1',2)]).insert(cols=[Repeat('2',2)])

        filtered_table = table.filter(a=1,comparison='match')

        self.assertEqual(4, len(table))

        self.assertEqual(2, len(filtered_table))

    def test_filter_match_str_str(self):
        table = Table(['a']).insert(cols=[Repeat('1',2)]).insert(cols=[Repeat('2',2)])

        filtered_table = table.filter(a='1',comparison='match')

        self.assertEqual(4, len(table))

        self.assertEqual(2, len(filtered_table))


@unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas is not installed so we must skip pandas tests")
class Table_Pandas_Tests(unittest.TestCase):

    def test_pandas(self):

        import pandas as pd #type: ignore
        import pandas.testing #type: ignore

        table = Table(['a','b','c','d','e']).insert(rows=[['A','B',1,'d',None],['B',None,None,None,'E']])

        expected_df = pd.DataFrame([
            dict(a='A',b='B',c=1,d='d'),
            dict(a='B',e='E')
        ])

        actual_df = table.to_pandas()

        pandas.testing.assert_frame_equal(expected_df,actual_df)

    def test_pandas_with_array_column(self):
        import pandas as pd   #type: ignore
        import pandas.testing #type: ignore

        table = Table(['a','b','c','d','e']).insert(rows=[['A','B',[1,2],'d',None],['B',None,None,None,'E']])

        expected_df = pd.DataFrame([
            dict(a='A',b='B',c=[1,2],d='d'),
            dict(a='B',e='E')
        ])

        actual_df = table.to_pandas()

        pandas.testing.assert_frame_equal(expected_df,actual_df)

    def test_pandas_with_dict_column(self):
        import pandas as pd   #type: ignore
        import pandas.testing #type: ignore

        table = Table(['a','b','c','d','e']).insert(rows=[['A','B',{'z':10},'d',None],['B',None,None,None,'E']])

        expected_df = pd.DataFrame([
            dict(a='A',b='B',c={'z':10},d='d'),
            dict(a='B',e='E')
        ])

        actual_df = table.to_pandas()

        pandas.testing.assert_frame_equal(expected_df,actual_df)

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
                        ([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                        ([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
                    ]

                    MatplotlibPlotter().plot(None,lines,"title","xlabel","ylabel",None,None,True,True,None,None,"screen")

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

                    mock_ax = plt_figure().add_subplot()

                    mock_ax.get_xticks.return_value = [1,2]
                    mock_ax.get_xlim.return_value   = [2,2]

                    lines = [
                        ([1,2], [5,6], None, [10,11], "B", 1.00, 'L1', '-', 1),
                        ([3,4], [7,8], None, [12,13], "R", 0.25, 'L2', '-', 1)
                    ]

                    MatplotlibPlotter().plot(None, lines, "title", "xlabel", "ylabel", None, None, True, True, None, None, "screen")

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
                ([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                ([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
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
                ([1,2], [5,6], None, [4,3], "B", 1.00, 'L1', '-', 1),
                ([3,4], [7,8], None, [2,1], "R", 0.25, 'L2', '-', 1)
            ]

            mock_ax = plt_figure().add_subplot()
            mock_ax.get_xticks.return_value = [1,2]
            mock_ax.get_xlim.return_value   = [2,2]
            MatplotlibPlotter().plot(None,lines,"title","xlabel","ylabel",(3,4),None,True,True,None,None,None)
            self.assertEqual(([3,4],[7,8],[2,1],None,'-'), mock_ax.errorbar.call_args_list[0][0])

    def test_plot_ylim(self):
        with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

            lines = [
                ([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                ([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
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
                    ([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                    ([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
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
                    ([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                    ([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
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
                    ([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                    ([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
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
                    ([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                    ([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
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
                        ([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                        ([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
                    ]

                    MatplotlibPlotter().plot(mock_ax,lines,"title","xlabel","ylabel",None,None,True,True,None,None,None)

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
                        ([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                        ([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
                    ]

                    MatplotlibPlotter().plot(None,lines,"title","xlabel","ylabel",None,None,True,True,None,None,"abc")

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
                        ([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                        ([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
                    ]

                    MatplotlibPlotter().plot(None,lines,"title","xlabel","ylabel",None,None,True,True,None,None,None)

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
                    ([1,2], [5,6], None, None, "B", 1.00, 'L1', '-', 1),
                    ([3,4], [7,8], None, None, "R", 0.25, 'L2', '-', 1)
                ]

                MatplotlibPlotter().plot(None,lines,"title","xlabel","ylabel",None,None,True,True,None,None,None)

                plt_figure.assert_called_with(num='coba')
                plt_figure().add_subplot.assert_called_with(111)

    def test_no_lines(self):
        with unittest.mock.patch('matplotlib.pyplot.figure') as plt_figure:

            CobaContext.logger = IndentLogger()
            CobaContext.logger.sink = ListSink()

            plotter = MatplotlibPlotter()
            plotter.plot(None, [], 'abc', 'def', 'efg', None, None, True, True, None, None, None)

            self.assertEqual(0, plt_figure().add_subplot.call_count)
            self.assertEqual(["No data was found for plotting."], CobaContext.logger.sink.items)

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

        transactions = [
            ["T0",1,2],
            ["T1",0,{"name":"lrn1"}],
            ["T2",1,{"source":"test"}],
            ["T3",[1,0], [{"reward":3},{"reward":4}]]
        ]

        io.write(transactions)

        result = io.read()

        self.assertEqual({**result.experiment, "n_learners":1, "n_environments":2}, result.experiment)
        self.assertEqual([{'learner_id':0,'name':"lrn1"}], result.learners.to_dicts())
        self.assertEqual([{'environment_id':1,'source':"test"}], result.environments.to_dicts())
        self.assertEqual([{'learner_id':0,'environment_id':1,'index':1,'reward':3},{'learner_id':0,'environment_id':1,'index':2,'reward':4}], result.interactions.to_dicts())

    def test_simple_resume(self):
        io = TransactionIO("coba/tests/.temp/transaction.log")

        transactions = [
            ["T0",1,2],
            ["T1",0,{"name":"lrn1"}],
            ["T2",1,{"source":"test"}],
            ["T3",[1,0], [{"reward":3},{"reward":4}]]
        ]

        io.write(transactions)

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

    def test_interaction_rows(self):
        result = Result(int_rows=[['environment_id','learner_id','index','reward'],(0,1,1,1),(0,1,2,3),(0,2,1,1),(0,2,2,4)])
        self.assertEqual("{'Learners': 0, 'Environments': 0, 'Interactions': 4}", str(result))
        self.assertEqual( [(0,1,1,1),(0,1,2,3),(0,2,1,1),(0,2,2,4)], list(result.interactions))

    def test_has_preamble(self):
        self.assertDictEqual(Result(exp_dict={"n_learners":1, "n_environments":2}).experiment, {"n_learners":1, "n_environments":2})

    def test_exception_when_no_file(self):
        with self.assertRaises(Exception):
            Result.from_file("abcd")

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
                return {"learner":{"family":"lrn1", "a":1}, "source":2, "logged":True, "scale":True, "batched": 2}
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

        expected_envs = [
            {"environment_id":0, "source":1, "logged":True, "scale":True, "batched": None},
            {"environment_id":1, "source":2, "logged":True, "scale":True, "batched": 2   },
        ]

        expected_lrns = [
            {"learner_id":0, "family":"lrn1", "a":1},
            {"learner_id":1, "family":"lrn2", "a":2},
        ]

        expected_ints = [
            {'environment_id': 0, 'learner_id': 0, 'index':1, 'reward': 1 },
            {'environment_id': 0, 'learner_id': 0, 'index':2, 'reward': 2 },
            {'environment_id': 0, 'learner_id': 1, 'index':1, 'reward': 5 },
            {'environment_id': 0, 'learner_id': 1, 'index':2, 'reward': 6 },
            {'environment_id': 1, 'learner_id': 0, 'index':1, 'reward': 3 },
            {'environment_id': 1, 'learner_id': 0, 'index':2, 'reward': 4 },
        ]

        res = Result.from_logged_envs([Logged1(),Logged2(),Logged3()])
        self.assertCountEqual(res.environments.to_dicts(), expected_envs)
        self.assertCountEqual(res.learners.to_dicts()    , expected_lrns)
        self.assertCountEqual(res.interactions.to_dicts(), expected_ints)

    def test_filter_fin_sans_n_interactions(self):
        envs = [['environment_id'             ],[1,],[2,]]
        lrns = [['learner_id'                 ],[1,],[2,]]
        ints = [['environment_id','learner_id'],[1,1],[1,2],[2,1]]

        original_result = Result(envs, lrns, ints)
        filtered_result = original_result.filter_fin()

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(1, len(filtered_result.environments))
        self.assertEqual(2, len(filtered_result.learners))
        self.assertEqual(2, len(filtered_result.interactions))

    def test_filter_fin_with_n_interactions(self):
        envs = [['environment_id'],[1,],[2,]]
        lrns = [['learner_id'    ],[1,],[2,]]
        ints = [['environment_id','learner_id','index','reward'],
            [1,1,1,1],[1,1,2,2],
            [1,2,1,1],[1,2,2,2],[1,2,3,3],
            [2,1,1,1],
            [2,2,1,1]
        ]

        original_result = Result(envs, lrns, ints)
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

        envs = [['environment_id'],[1,],[2,]]
        lrns = [['learner_id'    ],[1,],[2,]]
        ints = [['environment_id','learner_id'],[1,1],[2,1]]

        original_result = Result(envs, lrns, ints)
        filtered_result = original_result.filter_fin()

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(2, len(original_result.interactions))

        self.assertEqual(0, len(filtered_result.environments))
        self.assertEqual(0, len(filtered_result.learners))
        self.assertEqual(0, len(filtered_result.interactions))
        self.assertEqual(["There was no environment which was finished for every learner."], CobaContext.logger.sink.items)

    def test_filter_env(self):

        envs = [['environment_id'],[1,],[2,]]
        lrns = [['learner_id'    ],[1,],[2,]]
        ints = [['environment_id','learner_id'],[1,1],[2,1],[1,2]]

        original_result = Result(envs, lrns, ints)
        filtered_result = original_result.filter_env(environment_id=2)

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(1, len(filtered_result.environments))
        self.assertEqual(1, len(filtered_result.learners))
        self.assertEqual(1, len(filtered_result.interactions))

    def test_filter_env_no_match(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[1,],[2,]]
        lrns = [['learner_id'    ],[1,],[2,]]
        ints = [['environment_id','learner_id'],[1,1],[2,1],[1,2]]

        original_result = Result(envs, lrns, ints)
        filtered_result = original_result.filter_env(environment_id=3)

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(0, len(filtered_result.environments))
        self.assertEqual(0, len(filtered_result.learners))
        self.assertEqual(0, len(filtered_result.interactions))
        self.assertEqual(["No environments matched the given filter."], CobaContext.logger.sink.items)

    def test_filter_lrn_1(self):

        envs = [['environment_id'],[1,],[2,]]
        lrns = [['learner_id'    ],[1,],[2,]]
        ints = [['environment_id','learner_id'],[1,1],[2,1],[1,2]]

        original_result = Result(envs, lrns, ints)
        filtered_result = original_result.filter_lrn(learner_id=2)

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(1, len(filtered_result.environments))
        self.assertEqual(1, len(filtered_result.learners))
        self.assertEqual(1, len(filtered_result.interactions))

    def test_filter_lrn_2(self):

        envs = [['environment_id'],[1,],[2,]]
        lrns = [['learner_id'    ],[1,],[2,],[3,]]
        ints = [['environment_id','learner_id'],[1,1],[1,2],[2,3]]

        original_result = Result(envs, lrns, ints)
        filtered_result = original_result.filter_lrn(learner_id=[2,1])

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(3, len(original_result.learners))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(1, len(filtered_result.environments))
        self.assertEqual(2, len(filtered_result.learners))
        self.assertEqual(2, len(filtered_result.interactions))

    def test_filter_lrn_no_match(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[1,],[2,]]
        lrns = [['learner_id'    ],[1,],[2,],[3,]]
        ints = [['environment_id','learner_id'],[1,1],[1,2],[2,3]]

        original_result = Result(envs, lrns, ints)
        filtered_result = original_result.filter_lrn(learner_id=5)

        self.assertEqual(2, len(original_result.environments))
        self.assertEqual(3, len(original_result.learners))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(0, len(filtered_result.environments))
        self.assertEqual(0, len(filtered_result.learners))
        self.assertEqual(0, len(filtered_result.interactions))
        self.assertEqual(["No learners matched the given filter."], CobaContext.logger.sink.items)

    def test_copy(self):

        envs = [['environment_id'],[1,],[2,]]
        lrns = [['learner_id'    ],[1,],[2,],[3,]]
        ints = [['environment_id','learner_id'],[1,1],[1,2],[2,3]]

        result = Result(envs, lrns, ints)
        result_copy = result.copy()

        self.assertIsNot(result, result_copy)
        self.assertIsNot(result.learners, result_copy.learners)
        self.assertIsNot(result.environments, result_copy.environments)
        self.assertIsNot(result.interactions, result_copy.interactions)

        self.assertEqual(list(result.learners), list(result_copy.learners))
        self.assertEqual(list(result.environments), list(result_copy.environments))
        self.assertEqual(list(result.interactions), list(result_copy.interactions))

    def test_str(self):

        envs = [['environment_id'],[1,],[2,]]
        lrns = [['learner_id'    ],[1,],[2,],[3,]]
        ints = [['environment_id','learner_id'],[1,1],[1,2],[2,3]]

        self.assertEqual("{'Learners': 3, 'Environments': 2, 'Interactions': 3}", str(Result(envs, lrns, ints)))

    def test_ipython_display_(self):

        with unittest.mock.patch("builtins.print") as mock:

            envs = [['environment_id'],[1,],[2,]]
            lrns = [['learner_id'    ],[1,],[2,],[3,]]
            ints = [['environment_id','learner_id'],[1,1],[1,2],[2,3]]

            result = Result(envs, lrns, ints)
            result._ipython_display_()
            mock.assert_called_once_with(str(result))

    def test_plot_learners_bad_x_index(self):
        CobaContext.logger.sink = ListSink()
        Result().plot_learners(x=['index','a'])
        self.assertIn("The x-axis cannot contain", CobaContext.logger.sink.items[0])

    def test_plot_learners_one_environment_all_default(self):

        envs = [['environment_id'],[0]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        ints = [['environment_id','learner_id','index','reward'],[0,1,1,1],[0,1,2,2],[0,2,1,1],[0,2,2,2]]

        plotter = TestPlotter()
        result = Result(envs, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners()

        expected_lines = [
            ((1,2),(1,1.5),(None,None),(None,None),0,1,'learner_1','-', 1),
            ((1,2),(1,1.5),(None,None),(None,None),1,1,'learner_2','-', 1)
        ]

        self.assertEqual("Progressive Reward (1 Environments)", plotter.plot_calls[0][2])
        self.assertEqual("Interaction", plotter.plot_calls[0][3])

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_one_environment_lrn_params(self):

        envs = [['environment_id'],[0]]
        lrns = [['learner_id', 'family','i','j','t'],[1,'learner_1',1,2,None],[2,'learner_2',None,None,2]]
        ints = [['environment_id','learner_id','index','reward'],[0,1,1,1],[0,1,2,2],[0,2,1,1],[0,2,2,2]]

        plotter = TestPlotter()
        result = Result(envs, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners()

        expected_lines = [
            ((1,2),(1,1.5),(None,None),(None,None),0,1,'learner_1(i=1,j=2)','-', 1),
            ((1,2),(1,1.5),(None,None),(None,None),1,1,'learner_2(t=2)','-', 1)
        ]

        self.assertEqual("Progressive Reward (1 Environments)", plotter.plot_calls[0][2])
        self.assertEqual("Interaction", plotter.plot_calls[0][3])

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_two_environments_all_default(self):

        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        ints = [['environment_id','learner_id','index','reward'],
            [0,1,1,1],[0,1,2,2],
            [0,2,1,1],[0,2,2,2],
            [1,1,1,2],[1,1,2,3],
            [1,2,1,2],[1,2,2,3],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners()

        expected_lines = [
            ((1,2),(3/2,4/2),(None,None),(None,None),0,1,'learner_1','-', 1),
            ((1,2),(3/2,4/2),(None,None),(None,None),1,1,'learner_2','-', 1)
        ]

        self.assertEqual("Progressive Reward (2 Environments)", plotter.plot_calls[0][2])
        self.assertEqual("Interaction", plotter.plot_calls[0][3])
        
        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_one_environment_x_not_index(self):

        envs = [['environment_id'],[0]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        ints = [['environment_id','learner_id','index','reward'],
            [0,1,1,1],[0,1,2,2],
            [0,2,1,1],[0,2,2,2],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners(x=['environment_id'])

        expected_lines = [
            (('0',),(1.5,),(None,),(None,),0,1,'learner_1','.',1),
            (('0',),(1.5,),(None,),(None,),1,1,'learner_2','.',1)
        ]

        self.assertEqual("Final Progressive Reward (1 Environments)", plotter.plot_calls[0][2])
        self.assertEqual("environment_id", plotter.plot_calls[0][3])

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_two_environments_err_sd(self):
        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        ints = [['environment_id','learner_id','index','reward'],
            [0,1,1,1],[0,1,2,2],
            [0,2,1,1],[0,2,2,2],
            [1,1,1,2],[1,1,2,3],
            [1,2,1,2],[1,2,2,3],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners(err='sd')

        expected_lines = [
            ((1,2),(3/2,4/2),(None,None),(sqrt(2)/2,sqrt(2)/2),0,1,'learner_1','-', 1),
            ((1,2),(3/2,4/2),(None,None),(sqrt(2)/2,sqrt(2)/2),1,1,'learner_2','-', 1)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_mixed_env_count(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        ints = [['environment_id','learner_id','index','reward'],
            [0,1,1,1],[0,1,2,2],[0,1,3,3],
            [1,1,1,2],[1,1,2,3],
            [1,2,1,2],[1,2,2,3],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners()

        expected_log = "Environments not present for all learners have been excluded. To supress this call filter_fin() before plotting."
        expected_lines = [
            ((1,2),(2,5/2),(None,None),(None,None),0,1,'learner_1','-', 1),
            ((1,2),(2,5/2),(None,None),(None,None),1,1,'learner_2','-', 1)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])
        self.assertEqual(expected_log, CobaContext.logger.sink.items[0])

    def test_plot_learners_mixed_env_length(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        ints = [['environment_id','learner_id','index','reward'],
            [0,1,1,1],[0,1,2,2],[0,1,3,3],
            [0,2,1,1],[0,2,2,2],
            [1,1,1,2],[1,1,2,3],
            [1,2,1,2],[1,2,2,3],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners()

        expected_log = "This result contains environments of different lengths. The plot only includes interactions up to the shortest environment. To supress this warning in the future call <result>.filter_fin(n_interactions) before plotting."
        expected_lines = [
            ((1,2),(3/2,4/2),(None,None),(None,None),0,1,'learner_1','-', 1),
            ((1,2),(3/2,4/2),(None,None),(None,None),1,1,'learner_2','-', 1)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])
        self.assertEqual(expected_log, CobaContext.logger.sink.items[0])

    def test_plot_learners_filename(self):

        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        ints = [['environment_id','learner_id','index','reward'],
            [0,1,1,1],[0,1,2,2],
            [0,2,1,1],[0,2,2,2],
            [1,1,1,2],[1,1,2,3],
            [1,2,1,2],[1,2,2,3],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners(out="abc")

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual("abc", plotter.plot_calls[0][11])

    def test_plot_learners_ax(self):

        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        ints = [['environment_id','learner_id','index','reward'],
            [0,1,1,1],[0,1,2,2],
            [0,2,1,1],[0,2,2,2],
            [1,1,1,2],[1,1,2,3],
            [1,2,1,2],[1,2,2,3],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners(ax=1)

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(1, plotter.plot_calls[0][0])

    def test_plot_learners_xlim_ylim(self):

        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        ints = [['environment_id','learner_id','index','reward'],
            [0,1,1,1],[0,1,2,2],
            [0,2,1,1],[0,2,2,2],
            [1,1,1,2],[1,1,2,3],
            [1,2,1,2],[1,2,2,3],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners(xlim=(1,2), ylim=(2,3))

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual((1,2), plotter.plot_calls[0][5])
        self.assertEqual((2,3), plotter.plot_calls[0][6])

    def test_plot_learners_labels_int_colors(self):

        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        ints = [['environment_id','learner_id','index','reward'],
            [0,1,1,1],[0,1,2,2],
            [0,2,1,1],[0,2,2,2],
            [1,1,1,2],[1,1,2,3],
            [1,2,1,2],[1,2,2,3],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners(labels=['a','b'],colors=3)

        expected_lines = [
            ((1,2),(3/2,4/2),(None,None),(None,None),3,1,'a','-', 1),
            ((1,2),(3/2,4/2),(None,None),(None,None),4,1,'b','-', 1)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_labels_no_colors(self):
        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        ints = [['environment_id','learner_id','index','reward'],
            [0,1,1,1],[0,1,2,2],
            [0,2,1,1],[0,2,2,2],
            [1,1,1,2],[1,1,2,3],
            [1,2,1,2],[1,2,2,3],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners(labels=['a','b'])

        expected_lines = [
            ((1,2),(3/2,4/2),(None,None),(None,None),0,1,'a','-', 1),
            ((1,2),(3/2,4/2),(None,None),(None,None),1,1,'b','-', 1)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])
    
    def test_plot_learners_labels_one_color(self):
        
        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        ints = [['environment_id','learner_id','index','reward'],
            [0,1,1,1],[0,1,2,2],
            [0,2,1,1],[0,2,2,2],
            [1,1,1,2],[1,1,2,3],
            [1,2,1,2],[1,2,2,3],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners(labels=['a','b'],colors=1)

        expected_lines = [
            ((1,2),(3/2,4/2),(None,None),(None,None),1,1,'a','-', 1),
            ((1,2),(3/2,4/2),(None,None),(None,None),2,1,'b','-', 1)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_bad_labels(self):
        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        ints = [['environment_id','learner_id','index','reward'],
            [0,1,1,1],[0,1,2,2],
            [0,2,1,1],[0,2,2,2],
            [1,1,1,2],[1,1,2,3],
            [1,2,1,2],[1,2,2,3],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners(labels=['a'],colors=1)

        expected_lines = [
            ((1,2),(3/2,4/2),(None,None),(None,None),1,1,'a'        ,'-', 1),
            ((1,2),(3/2,4/2),(None,None),(None,None),2,1,'learner_2','-', 1)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_one_int_color(self):
        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        ints = [['environment_id','learner_id','index','reward'],
            [0,1,1,1],[0,1,2,2],
            [0,2,1,1],[0,2,2,2],
            [1,1,1,2],[1,1,2,3],
            [1,2,1,2],[1,2,2,3],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners(colors=2)

        expected_lines = [
            ((1,2),(3/2,4/2),(None,None),(None,None),2,1,'learner_1','-', 1),
            ((1,2),(3/2,4/2),(None,None),(None,None),3,1,'learner_2','-', 1)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_one_list_color(self):
        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        ints = [['environment_id','learner_id','index','reward'],
            [0,1,1,1],[0,1,2,2],
            [0,2,1,1],[0,2,2,2],
            [1,1,1,2],[1,1,2,3],
            [1,2,1,2],[1,2,2,3],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners(colors=[2])

        expected_lines = [
            ((1,2),(3/2,4/2),(None,None),(None,None),2,1,'learner_1','-', 1),
            ((1,2),(3/2,4/2),(None,None),(None,None),3,1,'learner_2','-', 1)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_top_n_positive(self):
        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2'],[3,'learner_3']]
        ints = [['environment_id','learner_id','index','reward'],
            [0,1,1,1],[0,1,2,2],
            [0,2,1,3],[0,2,2,4],
            [0,3,1,5],[0,3,2,6],
            [1,1,1,1],[1,1,2,2],
            [1,2,1,3],[1,2,2,4],
            [1,3,1,5],[1,3,2,6],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners(top_n=2)

        expected_lines = [
            ((1,2),(5,5.5),(None,None),(None,None),2,1,'learner_3','-', 1),
            ((1,2),(3,3.5),(None,None),(None,None),1,1,'learner_2','-', 1)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_top_n_negative(self):
        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2'],[3,'learner_3']]
        ints = [['environment_id','learner_id','index','reward'],
            [0,1,1,1],[0,1,2,2],
            [0,2,1,3],[0,2,2,4],
            [0,3,1,5],[0,3,2,6],
            [1,1,1,1],[1,1,2,2],
            [1,2,1,3],[1,2,2,4],
            [1,3,1,5],[1,3,2,6],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, ints)

        result.set_plotter(plotter)
        result.plot_learners(top_n=-2)

        expected_lines = [
            ((1,2),(3,3.5),(None,None),(None,None),1,1,'learner_2','-', 1),
            ((1,2),(1,1.5),(None,None),(None,None),0,1,'learner_1','-', 1)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertSequenceEqual(expected_lines, plotter.plot_calls[0][1])

    def test_plot_learners_empty_results(self):
        plotter = TestPlotter()
        result = Result()

        result.set_plotter(plotter)
        
        CobaContext.logger.sink = ListSink()
        result.plot_learners()
        self.assertIn("This result doesn't",CobaContext.logger.sink.items[0])

    def test_plot_contrast_bad_x_index(self):
        CobaContext.logger.sink = ListSink()
        Result().plot_contrast(0, 1, x=['index','a'])
        self.assertIn("The x-axis cannot",CobaContext.logger.sink.items[0])

    def test_plot_contrast_no_matches(self):
        CobaContext.logger.sink = ListSink()
        Result().plot_contrast(0, 1, x=['a'])
        self.assertIn("This result doesn't",CobaContext.logger.sink.items[0])

    def test_plot_contrast_index(self):
        envs = [['environment_id'],]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        ints = [['environment_id','learner_id','index','reward'],
            [0,1,1,0],[0,1,2,3],[0,1,3,9],
            [0,2,1,1],[0,2,2,2],[0,2,3,6],
            [1,1,1,1],[1,1,2,2],[1,1,3,6],
            [1,2,1,0],[1,2,2,3],[1,2,3,9],
        ]

        CobaContext.logger.sink = ListSink()
        Result(envs, lrns, ints).plot_contrast(0, 1, x='index')
        self.assertIn("plot_contrast does not currently", CobaContext.logger.sink.items[0])

    def test_plot_contrast_four_environment_all_default(self):

        envs = [['environment_id'],[0],[1],[2],[3]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        ints = [['environment_id','learner_id','index','reward'],
            [0,1,1,0],[0,1,2,3],[0,1,3,9],
            [0,2,1,1],[0,2,2,2],[0,2,3,6],
            [1,1,1,1],[1,1,2,2],[1,1,3,6],
            [1,2,1,0],[1,2,2,3],[1,2,3,9],
            [2,1,1,0],[2,1,2,0],[2,1,3,6],
            [2,2,1,0],[2,2,2,3],[2,2,3,9],
            [3,1,1,0],[3,1,2,3],[3,1,3,9],
            [3,2,1,0],[3,2,2,0],[3,2,3,6],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, ints)

        result.set_plotter(plotter)
        result.plot_contrast(1,2)

        expected_lines = [
            (('2','1'), (-2,-1), (None,None), (None,None), 0     , 1, 'learner_2 (2)', '.', 1.),
            (('0','3'), ( 1, 2), (None,None), (None,None), 2     , 1, 'learner_1 (2)', '.', 1.),
            (('2','3'), ( 0, 0),  None      ,  None      , "#888", 1, None           , '-', .5),
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines[0], plotter.plot_calls[0][1][0])
        self.assertEqual(expected_lines[1], plotter.plot_calls[0][1][1])
        self.assertEqual(expected_lines[2], plotter.plot_calls[0][1][2])
    
    def test_plot_contrast_four_environment_reverse(self):

        envs = [['environment_id'],[0],[1],[2],[3]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        ints = [['environment_id','learner_id','index','reward'],
            [0,1,1,0],[0,1,2,3],[0,1,3,9],
            [0,2,1,1],[0,2,2,2],[0,2,3,6],
            [1,1,1,1],[1,1,2,2],[1,1,3,6],
            [1,2,1,0],[1,2,2,3],[1,2,3,9],
            [2,1,1,0],[2,1,2,0],[2,1,3,6],
            [2,2,1,0],[2,2,2,3],[2,2,3,9],
            [3,1,1,0],[3,1,2,3],[3,1,3,9],
            [3,2,1,0],[3,2,2,0],[3,2,3,6],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, ints)

        result.set_plotter(plotter)
        result.plot_contrast(1,2,reverse=True)

        expected_lines = [
            (('3','0'), ( 2, 1), (None,None), (None,None), 2     , 1, 'learner_1 (2)', '.', 1.),
            (('1','2'), (-1,-2), (None,None), (None,None), 0     , 1, 'learner_2 (2)', '.', 1.),
            (('3','2'), ( 0, 0),  None      ,  None      , "#888", 1, None           , '-', .5),
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines[0], plotter.plot_calls[0][1][0])
        self.assertEqual(expected_lines[1], plotter.plot_calls[0][1][1])
        self.assertEqual(expected_lines[2], plotter.plot_calls[0][1][2])

    def test_plot_contrast_one_environment_env_not_index(self):

        envs = [['environment_id','a'],[0,1],[1,2],[2,3]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        ints = [['environment_id','learner_id','index','reward'],
            [0,1,1,0],[0,1,2,3],[0,1,3,12],
            [0,2,1,1],[0,2,2,2],[0,2,3,6],
            [1,1,1,0],[1,1,2,3],[1,1,3,6],
            [1,2,1,1],[1,2,2,2],[1,2,3,6],
            [2,1,1,0],[2,1,2,3],[2,1,3,9],
            [2,2,1,1],[2,2,2,2],[2,2,3,6],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, ints)

        result.set_plotter(plotter)
        result.plot_contrast(1,2,'a')

        expected_lines = [
            (('2',)   , (0, ), (None,    ), (None,    ), 1     , 1, 'Tie (1)'      , '.', 1.),
            (('3','1'), (1,2), (None,None), (None,None), 2     , 1, 'learner_1 (2)', '.', 1.),
            (('2','1'), (0,0), None       , None       , "#888", 1, None           , '-', .5)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines[0], plotter.plot_calls[0][1][0])
        self.assertEqual(expected_lines[1], plotter.plot_calls[0][1][1])
        self.assertEqual(expected_lines[2], plotter.plot_calls[0][1][2])

    def test_plot_scat_contrast_one_environment_env_index(self):

        envs = [['environment_id','a'],[0,1],[1,2],[2,3]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        ints = [['environment_id','learner_id','index','reward'],
            [0,1,1,0],[0,1,2,3],[0,1,3,12],
            [0,2,1,1],[0,2,2,2],[0,2,3,6],
            [1,1,1,0],[1,1,2,3],[1,1,3,6],
            [1,2,1,1],[1,2,2,2],[1,2,3,6],
            [2,1,1,0],[2,1,2,3],[2,1,3,9],
            [2,2,1,1],[2,2,2,2],[2,2,3,6],
        ]

        plotter = TestPlotter()
        result = Result(envs, lrns, ints)

        result.set_plotter(plotter)
        result.plot_contrast(1,2,'a',mode='scat')

        expected_lines = [
            ((3,  ), (3,  ), (None,    ), (None,    ), 1     , 1, 'Tie (1)'      , '.', 1.),
            ((5, 4), (3, 3), (None,None), (None,None), 2     , 1, 'learner_1 (2)', '.', 1.),
            ((0, 5), (0, 5), None       , None       , "#888", 1, None           , '-', .5)
        ]

        self.assertEqual(1, len(plotter.plot_calls))
        self.assertEqual(expected_lines[0], plotter.plot_calls[0][1][0])
        self.assertEqual(expected_lines[1], plotter.plot_calls[0][1][1])
        self.assertEqual(expected_lines[2], plotter.plot_calls[0][1][2])

class moving_average_Tests(unittest.TestCase):

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

class exponential_moving_average_Tets(unittest.TestCase):

    def test_span_2(self):
        self.assertEqual([1,1.75,2.62,3.55], [round(v,2) for v in list(exponential_moving_average([1,2,3,4],span=2))])

class FilterPlottingData_Tests(unittest.TestCase):

    def test_all_env_finished_with_equal_lengths_and_x_index(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[0]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        ints = [['environment_id','learner_id','index','reward'], [0,0,1,1], [0,1,1,1],]
        result = Result(envs, lrns, ints)

        expected_rows = result.interactions
        actual_rows = FilterPlottingData().filter(result, ['index'], "reward")

        self.assertEqual(expected_rows,actual_rows)
        self.assertEqual([], CobaContext.logger.sink.items)

    def test_all_env_finished_with_unequal_lengths_and_x_index(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        ints = [['environment_id','learner_id','index','reward'], [0,0,1,1],[0,1,1,1],[1,0,1,1],[1,0,2,1],[1,1,1,1],[1,1,2,1]]
        result = Result(envs, lrns, ints)

        expected_rows = [(0,0,1,1),(0,1,1,1),(1,0,1,1),(1,1,1,1)]
        actual_rows = FilterPlottingData().filter(result, ['index'], "reward")

        self.assertEqual(expected_rows,list(actual_rows))
        self.assertEqual(["This result contains environments of different lengths. The plot only includes interactions up to the shortest environment. To supress this warning in the future call <result>.filter_fin(n_interactions) before plotting."], CobaContext.logger.sink.items)

    def test_not_all_env_finished_with_unequal_lengths_and_x_index(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        ints = [['environment_id','learner_id','index','reward'], [0,0,1,1],[1,0,1,1],[1,0,2,1],[1,1,1,1],[1,1,2,1]]
        result = Result(envs, lrns, ints)

        expected_rows = [(1,0,1,1),(1,1,1,1)]
        actual_rows = FilterPlottingData().filter(result, ['index'], "reward")

        self.assertEqual(expected_rows,list(actual_rows))
        self.assertIn("Environments not present for all learners have been excluded. To supress this call filter_fin() before plotting.", CobaContext.logger.sink.items)
        self.assertIn("This result contains environments of different lengths. The plot only includes interactions up to the shortest environment. To supress this warning in the future call <result>.filter_fin(n_interactions) before plotting.", CobaContext.logger.sink.items)

    def test_no_env_finished_for_all_learners(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        ints = [['environment_id','learner_id','index','reward'],[0,0,1,1],[1,1,1,1],[1,1,2,1]]
        result = Result(envs, lrns, ints)

        with self.assertRaises(CobaException) as e:
            FilterPlottingData().filter(result, ['index'], "reward")

        self.assertEqual(str(e.exception),"This result does not contain an environment which has been finished for every learner. Plotting has been stopped.")

    def test_no_data(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        with self.assertRaises(CobaException) as e:
            FilterPlottingData().filter(Result(), ['index'], "reward")

        self.assertEqual(str(e.exception),"This result doesn't contain any evaluation data to plot.")

    def test_all_env_finished_with_unequal_lengths_and_not_x_index(self):

        CobaContext.logger = IndentLogger()
        CobaContext.logger.sink = ListSink()

        envs = [['environment_id'],[0],[1]]
        lrns = [['learner_id', 'family'],[1,'learner_1'],[2,'learner_2']]
        ints = [['environment_id','learner_id','index','reward'],[0,0,1,1],[0,1,1,1],[1,0,1,1],[1,0,2,1],[1,1,1,1],[1,1,2,1]]
        result = Result(envs, lrns, ints)

        expected_rows = [
           (0,0,1,1),
           (0,1,1,1),
           (1,0,1,1),(1,0,2,1),
           (1,1,1,1),(1,1,2,1)
        ]

        actual_rows = FilterPlottingData().filter(result, ['openml_task'], "reward")
        self.assertEqual(expected_rows,list(actual_rows))

class SmoothPlottingData_Tests(unittest.TestCase):

    def test_full_span(self):

        rows = Table(['environment_id','learner_id','index','reward']).insert(rows=[
            [0,0,1,0],[0,0,2,2],[0,0,3,4],
            [0,1,1,0],[0,1,2,4],[0,1,3,8],
        ])

        expected_rows = [
            {"environment_id":0, "learner_id":0, "reward":[0,1,2]},
            {"environment_id":0, "learner_id":1, "reward":[0,2,4]}
        ]

        actual_rows = SmoothPlottingData().filter(rows, "reward", None)
        actual_rows[0]['reward'] = list(actual_rows[0]['reward'])
        actual_rows[1]['reward'] = list(actual_rows[1]['reward'])

        self.assertEqual(expected_rows,actual_rows)

    def test_part_span(self):

        rows = Table(['environment_id','learner_id','index','reward']).insert(rows=[
            [0,0,1,0],[0,0,2,2],[0,0,3,2],
            [0,1,1,0],[0,1,2,4],[0,1,3,4],
        ])

        expected_rows = [
            {"environment_id":0, "learner_id":0, "reward":[0,1,2]},
            {"environment_id":0, "learner_id":1, "reward":[0,2,4]}
        ]

        actual_rows = SmoothPlottingData().filter(rows, "reward", 2)
        actual_rows[0]['reward'] = list(actual_rows[0]['reward'])
        actual_rows[1]['reward'] = list(actual_rows[1]['reward'])

        self.assertEqual(expected_rows,actual_rows)

    def test_no_span(self):

        rows = Table(['environment_id','learner_id','index','reward']).insert(rows=[
            [0,0,1,0],[0,0,2,2],[0,0,3,4],
            [0,1,1,0],[0,1,2,4],[0,1,3,8],
        ])

        expected_rows = [
            {"environment_id":0, "learner_id":0, "reward":[0,2,4]},
            {"environment_id":0, "learner_id":1, "reward":[0,4,8]}
        ]

        actual_rows = SmoothPlottingData().filter(rows, "reward", 1)
        actual_rows[0]['reward'] = list(actual_rows[0]['reward'])
        actual_rows[1]['reward'] = list(actual_rows[1]['reward'])

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

    def test_2_env_scat_1(self):

        rows = [
            {"environment_id":0, "learner_id":0, "reward":[0,2,4]},
            {"environment_id":0, "learner_id":1, "reward":[0,4,8]},
            {"environment_id":1, "learner_id":0, "reward":[0,0,0]},
            {"environment_id":1, "learner_id":1, "reward":[1,2,3]}
        ]

        expected_rows = [
            {"environment_id":0, "reward":[(0,0),(2,4),(4,8)]},
            {"environment_id":1, "reward":[(0,1),(0,2),(0,3)]}
        ]

        actual_rows = ContrastPlottingData().filter(rows, "reward", "scat", 0)

        self.assertEqual(expected_rows,actual_rows)

    def test_2_env_callable(self):

        rows = [
            {"environment_id":0, "learner_id":0, "reward":[0,2,4]},
            {"environment_id":0, "learner_id":1, "reward":[0,4,8]},
            {"environment_id":1, "learner_id":0, "reward":[0,0,0]},
            {"environment_id":1, "learner_id":1, "reward":[1,2,3]}
        ]

        expected_rows = [
            {"environment_id":0, "reward":[0,-2,-4]},
            {"environment_id":1, "reward":[-1,-2,-3]}
        ]

        actual_rows = ContrastPlottingData().filter(rows, "reward", lambda x,y: y-x, 1)
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
            (1, 0, None, None),
            (2, 3, None, None),
            (3, 6, None, None),
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
            (1, 0, None, 0      ),
            (2, 2, None, sqrt(2)),
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
            (1, 0, None, (0   ,0   )),
            (2, 2, None, (1.96,1.96)),
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
            (1,   0, None, (0 , 0)),
            (2, 1/2, None, (.5,.5)),
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
            (1,   0, None, (0         , 0.39033   )),
            (2, 1/2, None, (.5-0.18761, 0.81238-.5)),
        ]

        actual_rows = TransformToXYE().filter(rows, envs, ['index'], "reward", 'bi')

        self.assertEqual      (expected_rows[0][0:3] ,actual_rows[0][0:3]   )
        self.assertEqual      (expected_rows[0][3][0],actual_rows[0][3][0]  )
        self.assertAlmostEqual(expected_rows[0][3][1],actual_rows[0][3][1],4)

        self.assertEqual      (expected_rows[1][0:3] ,actual_rows[1][0:3]   )
        self.assertAlmostEqual(expected_rows[1][3][0],actual_rows[1][3][0],4)
        self.assertAlmostEqual(expected_rows[1][3][1],actual_rows[1][3][1],4)

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
            (1, 1/2, None, (.5-0.11811, 0.88188-.5)),
        ]

        actual_rows = TransformToXYE().filter(rows, envs, ['index'], "reward", BinomialConfidenceInterval('clopper-pearson'))

        self.assertEqual      (expected_rows[0][0:3] ,actual_rows[0][0:3]   )
        self.assertAlmostEqual(expected_rows[0][3][0],actual_rows[0][3][0],4)
        self.assertAlmostEqual(expected_rows[0][3][1],actual_rows[0][3][1],4)

    def test_one_env_bs_err(self):

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
            ('1', 4, None, (2,2)),
            ('3', 6, None, (2,2)),
        ]

        actual_rows = TransformToXYE().filter(rows, envs, 'a', "reward", 'bs')

        self.assertEqual(expected_rows,actual_rows)

    def test_two_env_bs_err(self):

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
            ('(1, 2)', 4, None, (2,2)),
            ('(3, 5)', 4, None, (0,0)),
            ('(3, 6)', 8, None, (0,0)),
        ]

        actual_rows = TransformToXYE().filter(rows, envs, ['a','b'], "reward", 'bs')

        self.assertEqual(expected_rows,actual_rows)
    
    def test_index_pairwise_bs_err(self):

        rows = [
            {"environment_id":0, "reward":[(1,2),(3,4)]},
            {"environment_id":1, "reward":[(3,6),(3,6)]},
        ]

        envs = {
            0: {'a':1,'b':2},
            1: {'a':3,'b':5},
        }

        expected_rows = [
            (2, 4, (1,1), (2,2)),
            (3, 5, (0,0), (1,1)),
        ]

        actual_rows = TransformToXYE().filter(rows, envs, ['index'], "reward", 'bs')

        self.assertEqual(expected_rows,actual_rows)

    def test_two_env_pairwise_bs_err(self):

        rows = [
            {"environment_id":0, "reward":[(1,2)]},
            {"environment_id":1, "reward":[(2,4)]},
            {"environment_id":2, "reward":[(3,6)]},
            {"environment_id":3, "reward":[(4,8)]},
        ]

        envs = {
            0: {'a':1,'b':2},
            1: {'a':3,'b':5},
            2: {'a':1,'b':2},
            3: {'a':3,'b':6},
        }

        expected_rows = [
            (2, 4, (1,1), (2,2)),
            (2, 4, (0,0), (0,0)),
            (4, 8, (0,0), (0,0)),
        ]

        actual_rows = TransformToXYE().filter(rows, envs, ['a','b'], "reward", 'bs')

        self.assertEqual(expected_rows,actual_rows)

if __name__ == '__main__':
    unittest.main()
