import math
import unittest
import timeit

from coba.benchmarks.transactions import Transaction
from coba.benchmarks.results import Result, Table, InteractionsTable

class Table_Tests(unittest.TestCase):

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

        expected = [('A', 'B', float('nan')), ('a', float('nan'), 'C')]
        actual   = table.to_tuples()

        for tuple1, tuple2 in zip(expected,actual):
            for val1, val2 in zip(tuple1,tuple2):
                if isinstance(val1,float) and math.isnan(val1): 
                    self.assertTrue(math.isnan(val2))
                else:
                    self.assertEqual(val1,val2)

    def test_tuples_with_array_column(self):

        table = Table("test", ['a'], [dict(a='A', b='B', c=[1,2], d='d'), dict(a='B', e='E') ])

        expected_tuples = [ ('A', 'B', [1,2], 'd', float('nan')), ('B', float('nan'), float('nan'), float('nan'), 'E') ]
        actual_tuples = table.to_tuples()

        for expected_tuple, actual_tuple in zip(expected_tuples,actual_tuples):
            self.assertTrue(all([ v1==v2 or (math.isnan(v1) and math.isnan(v2)) for v1,v2 in zip(expected_tuple,actual_tuple) ]))

    def test_tuples_with_dict_column(self):

        table = Table("test", ['a'], [dict(a='A',b='B',c={'z':5},d='d'), dict(a='B',e='E')])

        expected_tuples = [ ('A', 'B', {'z':5}, 'd', float('nan')), ('B', float('nan'), float('nan'), float('nan'), 'E') ]
        actual_tuples = table.to_tuples()

        for expected_tuple, actual_tuple in zip(expected_tuples,actual_tuples):
            self.assertTrue(all([ v1==v2 or (math.isnan(v1) and math.isnan(v2)) for v1,v2 in zip(expected_tuple,actual_tuple) ]))

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
            dict(a='A',index=1,b=1.,c=[1,2],d='d',z=[1,2]),
            dict(a='A',index=2,b=1.,c=[1,2],d='d',z=[3,4]),
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

    def test_two_packed_items(self):
        table = Table("test", ['a'], [dict(a='A', c=1, _packed=dict(b=['B','b'],d=['D','d']))])

        self.assertTrue('A' in table)

        self.assertEqual(table['A'], {'a':'A', 'index':[1,2], 'b':['B','b'], 'c':1, 'd':['D','d']})

        self.assertEqual(2, len(table))

        self.assertEqual([('A', 1, 'B', 1, 'D'), ('A', 2, 'b', 1, 'd')], list(table.to_tuples()))

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

        rows  = [dict(simulation_id=i,learner_id=2,C=5,A=5,N=1,_packed=dict(reward=[2]*9000)) for i in range(2) ]
        table = Table("test", ['simulation_id', 'learner_id'], rows)
        time = min(timeit.repeat(lambda:table.to_pandas(), repeat=6, number=1))
        
        #best time on my laptop was 0.15
        self.assertLess(time,1)

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

    def test_filter_in(self):
        table = Table("test", ['a'], [{'a':'a', 'b':'b'}, {'a':'A', 'b':'B'}])

        filtered_table = table.filter(lambda row: row["b"]=="B")

        self.assertEqual(2, len(table))
        self.assertEqual([('A', 'B'),('a', 'b')], list(table.to_tuples()))

        self.assertNotIn("a", filtered_table)

class InteractionTable_Tests(unittest.TestCase):
    def test_simple_each_span_none(self):
        table = InteractionsTable("ABC", ["simulation_id", "learner_id"], rows=[
            {"simulation_id":0, "learner_id":0, "_packed": {"reward":[1,2,3]}},
            {"simulation_id":1, "learner_id":0, "_packed": {"reward":[3,6,9]}},
            {"simulation_id":0, "learner_id":1, "_packed": {"reward":[2,4,6]}}
        ])

        expected = [[0,0,1,1.5,2], [0,1,3,4.5,6], [1,0,2,3,4]]
        actual   = table.to_progressive_list(each=True)

        self.assertCountEqual(expected,actual)

    def test_simple_not_each_span_none(self):
        table = InteractionsTable("ABC", ["simulation_id", "learner_id"], rows=[
            {"simulation_id":0, "learner_id":0, "_packed": {"reward":[1,2,3]}},
            {"simulation_id":1, "learner_id":0, "_packed": {"reward":[3,6,9]}},
            {"simulation_id":0, "learner_id":1, "_packed": {"reward":[2,4,6]}}
        ])

        expected = [[0,2,3,4], [1,2,3,4]]
        actual   = table.to_progressive_list(each=False)

        self.assertCountEqual(expected,actual)

    def test_simple_each_span_one(self):
        table = InteractionsTable("ABC", ["simulation_id", "learner_id"], rows=[
            {"simulation_id":0, "learner_id":0, "_packed": {"reward":[1,2,3]}},
            {"simulation_id":1, "learner_id":0, "_packed": {"reward":[3,6,9]}},
            {"simulation_id":0, "learner_id":1, "_packed": {"reward":[2,4,6]}}
        ])

        expected = [[0,0,1,2,3], [0,1,3,6,9], [1,0,2,4,6]]
        actual   = table.to_progressive_list(each=True,span=1)

        self.assertCountEqual(expected,actual)

    def test_simple_not_each_span_one(self):
        table = InteractionsTable("ABC", ["simulation_id", "learner_id"], rows=[
            {"simulation_id":0, "learner_id":0, "_packed": {"reward":[1,2,3]}},
            {"simulation_id":1, "learner_id":0, "_packed": {"reward":[3,6,9]}},
            {"simulation_id":0, "learner_id":1, "_packed": {"reward":[2,4,6]}}
        ])

        expected = [[0,2,4,6], [1,2,4,6]]
        actual   = table.to_progressive_list(each=False,span=1)

        self.assertCountEqual(expected,actual)

    def test_simple_each_span_two(self):
        table = InteractionsTable("ABC", ["simulation_id", "learner_id"], rows=[
            {"simulation_id":0, "learner_id":0, "_packed": {"reward":[1,2,3]}},
            {"simulation_id":1, "learner_id":0, "_packed": {"reward":[2,4,6]}},
        ])
        expected = [[0,0,1,1.75,2.6152],[0,1,2,3.5,5.2307]]
        actual   = table.to_progressive_list(each=True,span=2)

        self.assertEqual(len(expected), len(actual))

        for E,A in zip(expected, actual):
            for e,a in zip(E,A):
                self.assertAlmostEqual(e,a,places=3)

class Result_Tests(unittest.TestCase):

    def test_has_interactions_key(self):
        result = Result.from_transactions([
            Transaction.interactions(0, 1, a='A', _packed=dict(reward=[1,1])),
            Transaction.interactions(0, 2, b='B', _packed=dict(reward=[1,1]))
        ])

        self.assertEqual("{'Learners': 0, 'Simulations': 0, 'Interactions': 4}", str(result))

        self.assertTrue( (0,1) in result._interactions)
        self.assertTrue( (0,2) in result._interactions)

        self.assertEqual(len(result._interactions), 4)

    def test_has_version(self):
        result = Result.from_transactions([Transaction.version(1)])
        self.assertEqual(result.version, 1)

    def test_exception_when_no_file(self):
        with self.assertRaises(Exception):
            Result.from_file("abcd")

    def test_filter_fin(self):

        sims = [{"simulation_id":1},{"simulation_id":2}]
        lrns = [{"learner_id":1}, {"learner_id":2}]
        ints = [{"simulation_id":1, "learner_id":1},{"simulation_id":1,"learner_id":2}, {"simulation_id":2,"learner_id":1}]

        original_result = Result(1, {}, sims, lrns, ints)
        filtered_result = original_result.filter_fin()

        self.assertEqual(2, len(original_result.simulations))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(1, len(filtered_result.simulations))
        self.assertEqual(2, len(filtered_result.learners))
        self.assertEqual(2, len(filtered_result.interactions))

    def test_filter_sim(self):

        sims = [{"simulation_id":1},{"simulation_id":2}]
        lrns = [{"learner_id":1}, {"learner_id":2}]
        ints = [{"simulation_id":1, "learner_id":1},{"simulation_id":1,"learner_id":2}, {"simulation_id":2,"learner_id":1}]

        original_result = Result(1, {}, sims, lrns, ints)
        filtered_result = original_result.filter_sim(simulation_id=2)

        self.assertEqual(2, len(original_result.simulations))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(1, len(filtered_result.simulations))
        self.assertEqual(2, len(filtered_result.learners))
        self.assertEqual(1, len(filtered_result.interactions))

    def test_filter_lrn_1(self):

        sims = [{"simulation_id":1},{"simulation_id":2}]
        lrns = [{"learner_id":1}, {"learner_id":2}]
        ints = [{"simulation_id":1, "learner_id":1},{"simulation_id":1,"learner_id":2}, {"simulation_id":2,"learner_id":1}]

        original_result = Result(1, {}, sims, lrns, ints)
        filtered_result = original_result.filter_lrn(learner_id=2)

        self.assertEqual(2, len(original_result.simulations))
        self.assertEqual(2, len(original_result.learners))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(2, len(filtered_result.simulations))
        self.assertEqual(1, len(filtered_result.learners))
        self.assertEqual(1, len(filtered_result.interactions))

    def test_filter_lrn_2(self):

        sims = [{"simulation_id":1},{"simulation_id":2}]
        lrns = [{"learner_id":1}, {"learner_id":2}, {"learner_id":3}]
        ints = [{"simulation_id":1, "learner_id":1},{"simulation_id":1,"learner_id":2}, {"simulation_id":2,"learner_id":3}]

        original_result = Result(1, {}, sims, lrns, ints)
        filtered_result = original_result.filter_lrn(learner_id=[2,1])

        self.assertEqual(2, len(original_result.simulations))
        self.assertEqual(3, len(original_result.learners))
        self.assertEqual(3, len(original_result.interactions))

        self.assertEqual(2, len(filtered_result.simulations))
        self.assertEqual(2, len(filtered_result.learners))
        self.assertEqual(2, len(filtered_result.interactions))

if __name__ == '__main__':
    unittest.main()