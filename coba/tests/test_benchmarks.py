import unittest

from typing import Tuple, Hashable
from pathlib import Path

from coba.simulations import LambdaSimulation, LazySimulation
from coba.learners import LambdaLearner
from coba.benchmarks import Table, UniversalBenchmark, Result
from coba.execution import ExecutionContext, NoneLogger
from coba.statistics import BatchMeanEstimator, StatisticalEstimate
from coba.json import CobaJsonEncoder, CobaJsonDecoder

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

    def test_save_transactions(self):
        table = Table[int]("test", save_transactions=True)

        table.add_row(2, **{'A':1, 'B':2})
        table.add_row(3, **{'A':1, 'C':2})
        table.add_row(4, **{'A':1, 'D':StatisticalEstimate(1,2)})

        actual_transactions = table.pop_transactions()
        expected_transactions = [
            (2, {'A':1, 'B':2}),
            (3, {'A':1, 'C':2}),
            (4, {'A':1, 'D':StatisticalEstimate(1,2)})
        ]

        self.assertSequenceEqual(actual_transactions,expected_transactions)

    def test_no_save_transactions(self):
        table = Table[int]("test", save_transactions=False)

        table.add_row(2, **{'A':1, 'B':2})
        table.add_row(3, **{'A':1, 'C':2})
        table.add_row(4, **{'A':1, 'D':StatisticalEstimate(1,2)})

        actual_transactions = table.pop_transactions()
        expected_transactions = []

        self.assertSequenceEqual(actual_transactions,expected_transactions)

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
        result = Result(0)
        result.add_batch(0,1,2, a='A')

        self.assertTrue(result.has_batch(0,1,2))

    def test_to_from_json(self):
        expected_result = Result(0)
        expected_result.add_learner(0,a='A')
        expected_result.add_simulation(0,b='B')
        expected_result.add_batch(0,0,0,mean=BatchMeanEstimator([1,2,3]))

        json_txt = CobaJsonEncoder().encode(expected_result)

        actual_result = CobaJsonDecoder().decode(json_txt, [Result, Table, StatisticalEstimate])

        self.assertEqual(actual_result.to_tuples(), expected_result.to_tuples())

    def test_to_from_json_file(self):
        expected_result = Result(0)
        expected_result.add_learner(0,a='A')
        expected_result.add_simulation(0,b='B')
        expected_result.add_batch(0,0,0,mean=BatchMeanEstimator([1,2,3]))

        try:
            expected_result.to_json_file('test.json')
            actual_result = Result.from_json_file('test.json')
        finally:
            if Path('test.json').exists(): Path('test.json').unlink()

        self.assertEqual(actual_result.to_tuples(), expected_result.to_tuples())

    def test_to_from_transaction_file_once(self):

        try:
            expected_result = Result(0, "transactions.log")
            expected_result.add_learner(0,a='A')
            expected_result.add_simulation(0,b='B')
            expected_result.add_batch(0,1,2,mean=BatchMeanEstimator([1,2,3]))

            actual_result = Result(0, "transactions.log")
        finally:
            if Path('transactions.log').exists(): Path('transactions.log').unlink()

        self.assertEqual(actual_result.to_tuples(), expected_result.to_tuples())

    def test_to_from_transaction_file_twice(self):

        try:
            expected_result = Result(0, "transactions.log")
            expected_result.add_learner(0,a='A')
            expected_result.add_simulation(0,b='B')
            expected_result.add_batch(0,1,2,mean=BatchMeanEstimator([1,2,3]))

            expected_result = Result(0, "transactions.log")
            expected_result.add_learner(1,a='z')
            expected_result.add_simulation(1,b='q')
            expected_result.add_batch(1,1,0,mean=BatchMeanEstimator([1,2,3,4,5]))

            actual_result = Result(0, "transactions.log")
        finally:
            if Path('transactions.log').exists(): Path('transactions.log').unlink()

        self.assertEqual(actual_result.to_tuples(), expected_result.to_tuples())

class UniversalBenchmark_Tests(unittest.TestCase):

    def test_from_json(self):
        json = """{
            "batches": {"count":1},
            "ignore_first": false,
            "simulations": [
                {"seed":1283,"type":"classification","from":{"format":"openml","id":1116}},
                {"seed":1283,"type":"classification","from":{"format":"openml","id":1116}}
            ]
        }"""

        benchmark = UniversalBenchmark.from_json(json)

        self.assertFalse(benchmark._ignore_first)
        self.assertEqual(len(benchmark._simulations),2)
        self.assertIsInstance(benchmark._simulations[0],LazySimulation)

    def test_batch_count_1(self):
        sim             = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: A[s%3], family="0")
        benchmark       = UniversalBenchmark([sim], batch_count=1, ignore_first=False)

        actual_results = benchmark.evaluate([learner_factory])
        actual_learners,actual_simulations,actual_performances = actual_results.to_tuples()

        expected_learners     = [(0,"0", "0")]
        expected_simulations  = [(0, 5, 1, 1, 3)]
        expected_performances = [ (0, 0, 0, 5, BatchMeanEstimator([0,1,2,0,1]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_performances, expected_performances)

    def test_batch_count_2(self):
        sim             = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: A[s%3], family="0")
        benchmark       = UniversalBenchmark([sim], batch_count=2, ignore_first=False)

        actual_results = benchmark.evaluate([learner_factory])
        actual_learners,actual_simulations,actual_performances = actual_results.to_tuples()

        expected_learners     = [(0,"0","0")]
        expected_simulations  = [(0, 5, 2, 1, 3)]
        expected_performances = [ (0, 0, 0, 3, BatchMeanEstimator([0,1,2])), (0, 0, 1, 2, BatchMeanEstimator([0,1]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_performances, expected_performances)

    def test_batch_size_1(self):
        sim             = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: s%3, family="0")
        benchmark       = UniversalBenchmark([sim], batch_size=2, ignore_first=False)

        actual_results = benchmark.evaluate([learner_factory])
        actual_learners,actual_simulations,actual_performances = actual_results.to_tuples()

        expected_learners     = [(0,"0","0")]
        expected_simulations  = [(0, 4, 2, 1, 3)]
        expected_performances = [ (0, 0, 0, 2, BatchMeanEstimator([0,1])), (0, 0, 1, 2, BatchMeanEstimator([2,0]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_performances, expected_performances)

    def test_batch_size_2(self):
        sim             = LambdaSimulation(50, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: s%3, family="0")
        benchmark       = UniversalBenchmark([sim], batch_size=[1, 2, 4, 1], ignore_first=False)

        actual_results = benchmark.evaluate([learner_factory])
        actual_learners,actual_simulations,actual_performances = actual_results.to_tuples()

        expected_learners     = [(0,"0","0")]
        expected_simulations  = [(0, 8, 4, 1, 3)]
        expected_performances = [
            (0, 0, 0, 1, BatchMeanEstimator([0])), 
            (0, 0, 1, 2, BatchMeanEstimator([1,2])),
            (0, 0, 2, 4, BatchMeanEstimator([0,1,2,0])),
            (0, 0, 3, 1, BatchMeanEstimator([1]))
        ]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_performances, expected_performances)

    def test_ignore_first(self):
        sim             = LambdaSimulation(50, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: s%3, family="0")
        benchmark       = UniversalBenchmark([sim], batch_size=[1, 2, 4, 1], ignore_first=True)

        actual_results = benchmark.evaluate([learner_factory])
        actual_learners,actual_simulations,actual_performances = actual_results.to_tuples()

        expected_learners     = [(0,"0","0")]
        expected_simulations  = [(0, 8, 3, 1, 3)]
        expected_performances = [
            (0, 0, 0, 2, BatchMeanEstimator([1,2])),
            (0, 0, 1, 4, BatchMeanEstimator([0,1,2,0])),
            (0, 0, 2, 1, BatchMeanEstimator([1]))
        ]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_performances, expected_performances)

    def test_sims(self):
        sim1            = LambdaSimulation(5, lambda i: (i,i), lambda s: [0,1,2], lambda s,a: a)
        sim2            = LambdaSimulation(4, lambda i: (i,i), lambda s: [3,4,5], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[Tuple[int,int],int](lambda s,A: s[0]%3, family="0")
        benchmark       = UniversalBenchmark([sim1,sim2], batch_count=1, ignore_first=False)

        actual_results = benchmark.evaluate([learner_factory])
        actual_learners,actual_simulations,actual_performances = actual_results.to_tuples()

        expected_learners     = [(0,"0","0")]
        expected_simulations  = [(0, 5, 1, 2, 3), (1, 4, 1, 2, 3)]
        expected_performances = [(0, 0, 0, 5, BatchMeanEstimator([0,1,2,0,1])), (0, 1, 0, 4, BatchMeanEstimator([3,4,5,3]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_performances, expected_performances)

    def test_lazy_sim(self):
        sim1            = LazySimulation[int,int](lambda:LambdaSimulation(50, lambda i: i, lambda s: [0,1,2], lambda s, a: a))
        benchmark       = UniversalBenchmark([sim1], batch_size=[4,2], ignore_first=False)
        learner_factory = lambda: LambdaLearner[int,int](lambda s, A: s%3, family="0")
        
        actual_results = benchmark.evaluate([learner_factory])

        actual_learners,actual_simulations,actual_performances = actual_results.to_tuples()

        expected_learners     = [(0,"0","0")]
        expected_simulations  = [(0, 6, 2, 1, 3)]
        expected_performances = [(0, 0, 0, 4, BatchMeanEstimator([0,1,2,0])), (0, 0, 1, 2, BatchMeanEstimator([1,2]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_performances, expected_performances)

    def test_learners(self):
        sim              = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory1 = lambda: LambdaLearner[int,int](lambda s,A: A[s%3], family="0")
        learner_factory2 = lambda: LambdaLearner[int,int](lambda s,A: A[s%3], family="1")
        benchmark        = UniversalBenchmark([sim], batch_count=1, ignore_first=False)

        actual_results = benchmark.evaluate([learner_factory1, learner_factory2])

        actual_learners,actual_simulations,actual_performances = actual_results.to_tuples()

        expected_learners     = [(0,"0","0"), (1,"1","1")]
        expected_simulations  = [(0, 5, 1, 1, 3)]
        expected_performances = [(0, 0, 0, 5, BatchMeanEstimator([0,1,2,0,1])), (1, 0, 0, 5, BatchMeanEstimator([0,1,2,0,1]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_performances, expected_performances)

    def test_tranaction_resume_1(self):
        sim             = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: A[s%3], family="0")
        broken_factory  = lambda: LambdaLearner[int,int](lambda s,A: A[500], family="0")
        benchmark       = UniversalBenchmark([sim], batch_count=1, ignore_first=False)

        #the second time the broken_factory() shouldn't ever be used for learning or choosing
        #because it already worked the first time and we are "resuming" benchmark from transaction.log

        try:
            first_results  = benchmark.evaluate([learner_factory], transaction_file="transactions.log")
            second_results = benchmark.evaluate([broken_factory], transaction_file="transactions.log")

            actual_learners,actual_simulations,actual_performances = second_results.to_tuples()

            expected_learners     = [(0,"0","0")]
            expected_simulations  = [(0, 5, 1, 1, 3)]
            expected_performances = [ (0, 0, 0, 5, BatchMeanEstimator([0,1,2,0,1]))]
        finally:
            if Path('transactions.log').exists(): Path('transactions.log').unlink()            

            self.assertSequenceEqual(actual_learners, expected_learners)
            self.assertSequenceEqual(actual_simulations, expected_simulations)
            self.assertSequenceEqual(actual_performances, expected_performances)

    def test_tranaction_resume_2(self):
        sim             = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: A[s%3], family="0")
        broken_factory  = lambda: LambdaLearner[int,int](lambda s,A: A[500], family="0")
        benchmark       = UniversalBenchmark([sim], batch_count=2, ignore_first=True)

        #the second time the broken_factory() shouldn't ever be used for learning or choosing
        #because it already worked the first time and we are "resuming" benchmark from transaction.log

        try:
            first_results  = benchmark.evaluate([learner_factory], transaction_file="transactions.log")
            second_results = benchmark.evaluate([broken_factory], transaction_file="transactions.log")

            actual_learners,actual_simulations,actual_performances = second_results.to_tuples()

            expected_learners     = [(0,"0","0")]
            expected_simulations  = [(0, 5, 1, 1, 3)]
            expected_performances = [ (0, 0, 0, 2, BatchMeanEstimator([0,1]))]
        finally:
            if Path('transactions.log').exists(): Path('transactions.log').unlink()            

            self.assertSequenceEqual(actual_learners, expected_learners)
            self.assertSequenceEqual(actual_simulations, expected_simulations)
            self.assertSequenceEqual(actual_performances, expected_performances)

if __name__ == '__main__':
    unittest.main()