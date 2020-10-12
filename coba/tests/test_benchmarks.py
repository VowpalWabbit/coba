import unittest

from typing import Tuple, Hashable, cast
from pathlib import Path

from coba.simulations import LambdaSimulation, LazySimulation
from coba.learners import LambdaLearner
from coba.execution import ExecutionContext, NoneLogger
from coba.statistics import BatchMeanEstimator, StatisticalEstimate
from coba.json import CobaJsonEncoder, CobaJsonDecoder
from coba.benchmarks import (
    CountBatcher, SizeBatcher, SizesBatcher, Batcher, ResultDiskWriter, 
    ResultMemoryWriter, ResultWriter, Table, UniversalBenchmark, Result
)

ExecutionContext.Logger = NoneLogger()

class Table_Tests(unittest.TestCase):

    def test_to_from_json(self):
        expected_table = Table[Hashable]("test", 0)
        expected_table.add_row(0, a='A')
        expected_table.add_row((0,1,2), b='B')
        expected_table.add_row('a', c='C')

        json_txt = CobaJsonEncoder().encode(expected_table)

        actual_table = CobaJsonDecoder().decode(json_txt, [Table])

        self.assertEqual(actual_table.to_tuples(), expected_table.to_tuples())

    def test_table_contains(self):
        table = Table("test",0)
        table.add_row(0, a='A')
        table.add_row((0,1,2), b='B')
        table.add_row('a', c='C')

        self.assertTrue(0 in table)
        self.assertTrue((0,1,2) in table)
        self.assertTrue('a' in table)
        self.assertFalse('b' in table)

class Result_Tests(unittest.TestCase):

    def test_has_batch_key(self):
        writer = ResultMemoryWriter(0)
        writer.write_batch(0,1,None,2, a='A')
        result = Result.from_result_writer(writer)

        self.assertTrue(result.has_batch(0,1,None,2))

    def test_to_from_json(self):
        writer = ResultMemoryWriter(0)
        writer.write_learner(0,a='A')
        writer.write_simulation(0,b='B')
        writer.write_batch(0,1,1,2,mean=BatchMeanEstimator([1,2,3]))
        
        expected_result = Result.from_result_writer(writer)

        json_txt = CobaJsonEncoder().encode(expected_result)

        actual_result = CobaJsonDecoder().decode(json_txt, [Result, Table, StatisticalEstimate])

        self.assertEqual(actual_result.to_tuples(), expected_result.to_tuples())

    def test_to_from_json_file(self):
        writer = ResultMemoryWriter(0)
        writer.write_learner(0,a='A')
        writer.write_simulation(0,b='B')
        writer.write_batch(0,1,None,2,mean=BatchMeanEstimator([1,2,3]))

        expected_result = Result.from_result_writer(writer)

        try:
            expected_result.to_json_file('.test/test.json')
            actual_result = Result.from_json_file('.test/test.json')
        finally:
            if Path('.test/test.json').exists(): Path('.test/test.json').unlink()

        self.assertEqual(actual_result.to_tuples(), expected_result.to_tuples())

    def test_to_from_transaction_file_once(self):

        def write_result(writer: ResultWriter)-> None:
            writer.write_learner(0,a='A')
            writer.write_simulation(0,b='B')
            writer.write_batch(0,1,None,2,mean=BatchMeanEstimator([1,2,3]))

        try:
            disk_writer = ResultDiskWriter(".test/transactions.log")
            memory_writer = ResultMemoryWriter(0)
            
            write_result(disk_writer)
            write_result(memory_writer)

            expected_result = Result.from_result_writer(memory_writer,0)
            actual_result = Result.from_result_writer(disk_writer,0)
        finally:
            if Path('.test/transactions.log').exists(): Path('.test/transactions.log').unlink()

        self.assertEqual(actual_result.to_tuples(), expected_result.to_tuples())

    def test_to_from_transaction_file_twice(self):

        def write_first_result(writer: ResultWriter)-> None:
            writer.write_learner(0,a='A')
            writer.write_simulation(0,b='B')
            writer.write_batch(0,1,None,2,mean=BatchMeanEstimator([1,2,3]))

        def write_second_result(writer: ResultWriter)-> None:
            writer.write_learner(0,a='z')
            writer.write_simulation(0,b='q')
            writer.write_batch(1,1,None,0,mean=BatchMeanEstimator([1,2,3,4,5]))

        try:
            disk_writer1 = ResultDiskWriter(".test/transactions.log")
            disk_writer2 = ResultDiskWriter(".test/transactions.log")
            memory_writer = ResultMemoryWriter(0)
            
            write_first_result(disk_writer1)
            write_second_result(disk_writer2)

            write_first_result(memory_writer)
            write_second_result(memory_writer)

            actual_result = Result.from_transaction_file(".test/transactions.log",0)
            expected_result = Result.from_result_writer(memory_writer)
        finally:
            if Path('.test/transactions.log').exists(): Path('.test/transactions.log').unlink()

        self.assertEqual(actual_result.to_tuples(), expected_result.to_tuples())

class UniversalBenchmark_Tests(unittest.TestCase):

    def test_from_json(self):
        json = """{
            "batches"     : {"count":1},
            "ignore_first": false,
            "shuffle"     : [1283],
            "simulations" : [
                {"type":"classification","from":{"format":"openml","id":1116}}
            ]
        }"""

        benchmark = UniversalBenchmark.from_json(json)

        self.assertFalse(benchmark._ignore_first)
        self.assertEqual(len(benchmark._simulations),1)
        self.assertIsInstance(benchmark._simulations[0],LazySimulation)

    def test_evaluate_sims(self):
        sim1            = LambdaSimulation(5, lambda i: (i,i), lambda s: [0,1,2], lambda s,a: a)
        sim2            = LambdaSimulation(4, lambda i: (i,i), lambda s: [3,4,5], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[Tuple[int,int],int](lambda s,A: s[0]%3, family="0")
        benchmark       = UniversalBenchmark([sim1,sim2], CountBatcher(1), ignore_first=False, ignore_raise=False)

        actual_learners,actual_simulations,actual_batches = benchmark.evaluate([learner_factory]).to_tuples()

        expected_learners    = [(0,"0","0")]
        expected_simulations = [(0, 5, 1, 1, 2, 3), (1, 4, 1, 1, 2, 3)]
        expected_batches     = [(0, 0, None, 0, 5, BatchMeanEstimator([0,1,2,0,1])), (0, 1, None, 0, 4, BatchMeanEstimator([3,4,5,3]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_batches, expected_batches)

    def test_evaluate_seeds(self):
        sim1            = LambdaSimulation(5, lambda i: (i,i), lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[Tuple[int,int],int](lambda s,A: s[0]%3, family="0")
        benchmark       = UniversalBenchmark([sim1], SizesBatcher([2]), False, False, [1,4])

        actual_learners,actual_simulations,actual_batches = benchmark.evaluate([learner_factory]).to_tuples()

        expected_learners    = [(0,"0","0")]
        expected_simulations = [(0, 2, 1, 2, 2, 3)]
        expected_batches     = [(0, 0, 1, 0, 2, BatchMeanEstimator([1,0])), (0, 0, 4, 0, 2, BatchMeanEstimator([2,0]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_batches, expected_batches)

    def test_evaluate_sims_small(self):
        sim1            = LambdaSimulation(5, lambda i: (i,i), lambda s: [0,1,2], lambda s,a: a)
        sim2            = LambdaSimulation(4, lambda i: (i,i), lambda s: [3,4,5], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[Tuple[int,int],int](lambda s,A: s[0]%3, family="0")
        benchmark       = UniversalBenchmark([sim1,sim2], CountBatcher(1,5), ignore_first=False, ignore_raise=False)

        actual_learners,actual_simulations,actual_batches = benchmark.evaluate([learner_factory]).to_tuples()

        expected_learners    = [(0,"0","0")]
        expected_simulations = [(0, 5, 1, 1, 2, 3), (1, 0, 0, 1, 2, 3)]
        expected_batches     = [(0, 0, None, 0, 5, BatchMeanEstimator([0,1,2,0,1]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_batches, expected_batches)

    def test_evaluate_ignore_first(self):
        sim             = LambdaSimulation(50, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: s%3, family="0")
        benchmark       = UniversalBenchmark([sim], SizesBatcher([1, 2]), ignore_first=True, ignore_raise=False)

        actual_learners, actual_simulations, actual_batches = benchmark.evaluate([learner_factory]).to_tuples()

        expected_learners    = [(0,"0","0")]
        expected_simulations = [(0, 2, 1, 1, 1, 3)]
        expected_batches     = [(0, 0, None, 0, 2, BatchMeanEstimator([1,2]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_batches, expected_batches)

    def test_evaluate_lazy_sim(self):
        sim1            = LazySimulation[int,int](lambda:LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s, a: a))
        benchmark       = UniversalBenchmark([sim1], CountBatcher(1), ignore_first=False, ignore_raise=False)
        learner_factory = lambda: LambdaLearner[int,int](lambda s, A: s%3, family="0")
        
        actual_learners,actual_simulations,actual_batches = benchmark.evaluate([learner_factory]).to_tuples()

        expected_learners    = [(0,"0","0")]
        expected_simulations = [(0, 5, 1, 1, 1, 3)]
        expected_batches     = [(0, 0, None, 0, 5, BatchMeanEstimator([0,1,2,0,1]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_batches, expected_batches)

    def test_evalute_learners(self):
        sim              = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory1 = lambda: LambdaLearner[int,int](lambda s,A: A[s%3], family="0")
        learner_factory2 = lambda: LambdaLearner[int,int](lambda s,A: A[s%3], family="1")
        benchmark        = UniversalBenchmark([sim], CountBatcher(1), ignore_first=False, ignore_raise=False)

        actual_results = benchmark.evaluate([learner_factory1, learner_factory2])
        actual_learners,actual_simulations,actual_batches = actual_results.to_tuples()

        expected_learners     = [(0,"0","0"), (1,"1","1")]
        expected_simulations  = [(0, 5, 1, 1, 1, 3)]
        expected_batches = [
            (0, 0, None, 0, 5, BatchMeanEstimator([0,1,2,0,1])), 
            (1, 0, None, 0, 5, BatchMeanEstimator([0,1,2,0,1]))
        ]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_batches, expected_batches)

    def test_transaction_resume_1(self):
        sim             = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: A[s%3], family="0")
        broken_factory  = lambda: LambdaLearner[int,int](lambda s,A: A[500], family="0")
        benchmark       = UniversalBenchmark([sim], CountBatcher(1), ignore_first=False)

        #the second time the broken_factory() shouldn't ever be used for learning or choosing
        #because it already worked the first time and we are "resuming" benchmark from transaction.log
        try:
            first_results  = benchmark.evaluate([learner_factory], ".test/transactions.log")
            second_results = benchmark.evaluate([broken_factory], ".test/transactions.log")

            actual_learners,actual_simulations,actual_batches = second_results.to_tuples()

            expected_learners    = [(0,"0","0")]
            expected_simulations = [(0, 5, 1, 1, 1, 3)]
            expected_batches     = [(0, 0, None, 0, 5, BatchMeanEstimator([0,1,2,0,1]))]
        finally:
            if Path('.test/transactions.log').exists(): Path('.test/transactions.log').unlink()            

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_batches, expected_batches)

    def test_transaction_resume_ignore_first(self):
        sim             = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: A[s%3], family="0")
        broken_factory  = lambda: LambdaLearner[int,int](lambda s,A: A[500], family="0")
        benchmark       = UniversalBenchmark([sim], CountBatcher(2), ignore_first=True)

        #the second time the broken_factory() shouldn't ever be used for learning or choosing
        #because it already worked the first time and we are "resuming" benchmark from transaction.log

        try:
            first_results  = benchmark.evaluate([learner_factory], ".test/transactions.log")
            second_results = benchmark.evaluate([broken_factory], ".test/transactions.log")

            actual_learners,actual_simulations,actual_performances = second_results.to_tuples()

            expected_learners     = [(0,"0","0")]
            expected_simulations  = [(0, 2, 1, 1, 1, 3)]
            expected_performances = [ (0, 0, None, 0, 2, BatchMeanEstimator([0,1]))]
        finally:
            if Path('.test/transactions.log').exists(): Path('.test/transactions.log').unlink()            

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_performances, expected_performances)

class CountBatcher_Tests(unittest.TestCase):

    def test_from_json_1(self):
        batcher = CountBatcher.from_json('{"count":3, "min":9, "max":12}')
        self.assertEqual(batcher._batch_count, 3)
        self.assertEqual(batcher._min_interactions, 9)
        self.assertEqual(batcher._max_interactions, 12)
    
    def test_from_json_2(self):
        batcher = cast(CountBatcher,Batcher.from_json('{"count":3, "min":9, "max":12}'))
        self.assertIsInstance(batcher, CountBatcher)
        self.assertEqual(batcher._batch_count, 3)
        self.assertEqual(batcher._min_interactions, 9)
        self.assertEqual(batcher._max_interactions, 12)

    def test_batch_sizes_sans_min_max(self):
        self.assertEqual(CountBatcher(3).batch_sizes(0), [])
        self.assertEqual(CountBatcher(3).batch_sizes(1), [1,0,0])
        self.assertEqual(CountBatcher(3).batch_sizes(2), [1,1,0])
        self.assertEqual(CountBatcher(3).batch_sizes(3), [1,1,1])
        self.assertEqual(CountBatcher(3).batch_sizes(4), [2,1,1])
        self.assertEqual(CountBatcher(3).batch_sizes(5), [2,2,1])
        self.assertEqual(CountBatcher(3).batch_sizes(6), [2,2,2])

    def test_batch_sizes_with_min_sans_max(self):
        self.assertEqual(CountBatcher(3,9).batch_sizes(8 ), [])
        self.assertEqual(CountBatcher(3,9).batch_sizes(9 ), [3,3,3])
        self.assertEqual(CountBatcher(3,9).batch_sizes(10), [4,3,3])
        self.assertEqual(CountBatcher(3,9).batch_sizes(11), [4,4,3])
        self.assertEqual(CountBatcher(3,9).batch_sizes(12), [4,4,4])

    def test_batch_sizes_with_min_max(self):
        self.assertEqual(CountBatcher(3,9,11).batch_sizes(8 ), [])
        self.assertEqual(CountBatcher(3,9,11).batch_sizes(9 ), [3,3,3])
        self.assertEqual(CountBatcher(3,9,11).batch_sizes(10), [4,3,3])
        self.assertEqual(CountBatcher(3,9,11).batch_sizes(11), [4,4,3])
        self.assertEqual(CountBatcher(3,9,11).batch_sizes(12), [4,4,3])

class SizeBatcher_Tests(unittest.TestCase):

    def test_from_json_1(self):
        batcher = SizeBatcher.from_json('{"size":3, "min":9, "max":12}')
        self.assertEqual(batcher._batch_size, 3)
        self.assertEqual(batcher._min_interactions, 9)
        self.assertEqual(batcher._max_interactions, 12)

    def test_from_json_2(self):
        batcher = cast(SizeBatcher,Batcher.from_json('{"size":3, "min":9, "max":12}'))
        self.assertIsInstance(batcher, SizeBatcher)
        self.assertEqual(batcher._batch_size, 3)
        self.assertEqual(batcher._min_interactions, 9)
        self.assertEqual(batcher._max_interactions, 12)

    def test_batch_sizes_sans_min_max(self):
        self.assertEqual(SizeBatcher(1).batch_sizes(0), [])
        self.assertEqual(SizeBatcher(1).batch_sizes(1), [1])
        self.assertEqual(SizeBatcher(1).batch_sizes(2), [1,1])
        self.assertEqual(SizeBatcher(1).batch_sizes(3), [1,1,1])
        self.assertEqual(SizeBatcher(2).batch_sizes(4), [2,2])
        self.assertEqual(SizeBatcher(2).batch_sizes(5), [2,2])
        self.assertEqual(SizeBatcher(2).batch_sizes(6), [2,2,2])

    def test_batch_sizes_with_min_sans_max(self):
        self.assertEqual(SizeBatcher(3,9).batch_sizes(8 ), [])
        self.assertEqual(SizeBatcher(3,9).batch_sizes(9 ), [3,3,3])
        self.assertEqual(SizeBatcher(3,9).batch_sizes(10), [3,3,3])
        self.assertEqual(SizeBatcher(3,9).batch_sizes(11), [3,3,3])
        self.assertEqual(SizeBatcher(3,9).batch_sizes(12), [3,3,3,3])

    def test_batch_sizes_with_min_max(self):
        self.assertEqual(SizeBatcher(3,9,11).batch_sizes(8 ), [])
        self.assertEqual(SizeBatcher(3,9,11).batch_sizes(9 ), [3,3,3])
        self.assertEqual(SizeBatcher(3,9,11).batch_sizes(10), [3,3,3])
        self.assertEqual(SizeBatcher(3,9,11).batch_sizes(11), [3,3,3])
        self.assertEqual(SizeBatcher(3,9,11).batch_sizes(12), [3,3,3])

class SizesBatcher_Tests(unittest.TestCase):

    def test_from_json_1(self):
        batcher = SizesBatcher.from_json('{"sizes":[1,1,1]}')
        self.assertEqual(batcher._batch_sizes, [1,1,1])

    def test_from_json_2(self):
        batcher = cast(SizesBatcher,Batcher.from_json('{"sizes":[1,1,1]}'))
        self.assertIsInstance(batcher, SizesBatcher)
        self.assertEqual(batcher._batch_sizes, [1,1,1])

    def test_batch_sizes(self):
        self.assertEqual(SizesBatcher([1,1,1]).batch_sizes(0), [])
        self.assertEqual(SizesBatcher([1,1,1]).batch_sizes(1), [])
        self.assertEqual(SizesBatcher([1,1,1]).batch_sizes(2), [])
        self.assertEqual(SizesBatcher([1,1,1]).batch_sizes(3), [1,1,1])
        self.assertEqual(SizesBatcher([1,1,1]).batch_sizes(4), [1,1,1])
        self.assertEqual(SizesBatcher([1,1,1]).batch_sizes(5), [1,1,1])
        self.assertEqual(SizesBatcher([1,1,1]).batch_sizes(6), [1,1,1])

if __name__ == '__main__':
    unittest.main()