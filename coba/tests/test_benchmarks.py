import unittest

from pathlib import Path
from statistics import mean
from typing import Tuple, cast

from coba.simulations import LambdaSimulation, LazySimulation, JsonSimulation
from coba.learners import LambdaLearner
from coba.execution import ExecutionContext, NoneLogger
from coba.benchmarks import (
    UniversalBenchmark, Result, LearnerFactory, 
    Transaction, TransactionIsNew, CountBatcher, 
    SizeBatcher, SizesBatcher, Batcher
)

ExecutionContext.Logger = NoneLogger()

class TransactionIsNew_Test(unittest.TestCase):
    
    def test_duplicates_are_dropped(self):
        existing = Result.from_transactions([
            Transaction.learner(0, a='A'),
            Transaction.simulation(0, b='B'),
            Transaction.batch(0, 1, None, 2, reward=mean([1,2,3]))
        ])

        filter = TransactionIsNew(existing)

        transactions = list(filter.filter([
            Transaction.learner(0, a='A'), 
            Transaction.simulation(0, b='B'), 
            Transaction.batch(0, 1, None, 2, reward=mean([1,2,3]))]
        ))

        self.assertEqual(len(transactions), 0)

    def test_non_duplicates_are_kept(self):
        existing = Result.from_transactions([
            Transaction.learner(0, a='A'),
            Transaction.simulation(0, b='B'),
            Transaction.batch(0, 1, None, 2, reward=mean([1,2,3]))
        ])

        filter = TransactionIsNew(existing)

        transactions = list(filter.filter([
            Transaction.learner(1, a='A'), 
            Transaction.simulation(1, b='B'), 
            Transaction.batch(1, 1, None, 2, reward=mean([1,2,3]))]
        ))

        self.assertEqual(len(transactions), 3)

class Result_Tests(unittest.TestCase):

    def test_has_batches_key(self):
        result = Result.from_transactions([
            Transaction.batch(0, 1, None, 2, a='A'),
            Transaction.batch(0, 2, None, 2, b='B')
        ])
        
        self.assertTrue( (0,1,None,2) in result.batches)
        self.assertTrue( (0,2,None,2) in result.batches)

        self.assertEqual(len(result.batches), 2)

    def test_has_version(self):
        result = Result.from_transactions([Transaction.version(1)])
        self.assertEqual(result.version, 1)

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
        self.assertEqual(benchmark._seeds, [1283])
        self.assertEqual(len(benchmark._simulations),1)
        self.assertIsInstance(benchmark._simulations[0],JsonSimulation)

    def test_evaluate_sims(self):
        sim1            = LambdaSimulation(5, lambda i: (i,i), lambda s: [0,1,2], lambda s,a: a)
        sim2            = LambdaSimulation(4, lambda i: (i,i), lambda s: [3,4,5], lambda s,a: a)
        learner_factory = LearnerFactory[Tuple[int,int],int](LambdaLearner, lambda s,A: s[0]%3, family="0")
        benchmark       = UniversalBenchmark([sim1,sim2], batch_count=1, ignore_first=False, ignore_raise=False, processes=1)

        actual_learners,actual_simulations,actual_batches = benchmark.evaluate([learner_factory]).to_tuples()

        expected_learners    = [(0,"0","0")]
        expected_simulations = [(0, 5, 1, 2, 3), (1, 4, 1, 2, 3)]
        expected_batches     = [(0, 0, None, 0, 5, mean([0,1,2,0,1])), (0, 1, None, 0, 4, mean([3,4,5,3]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_batches, expected_batches)

    def test_evaluate_seeds(self):
        sim1            = LambdaSimulation(5, lambda i: (i,i), lambda s: [0,1,2], lambda s,a: a)
        learner_factory = LearnerFactory[Tuple[int,int],int](LambdaLearner,lambda s,A: s[0]%3, family="0")
        benchmark       = UniversalBenchmark([sim1], batch_sizes=[2], ignore_first=False, ignore_raise=False, shuffle_seeds=[1,4], processes=1)

        actual_learners,actual_simulations,actual_batches = benchmark.evaluate([learner_factory]).to_tuples()

        expected_learners    = [(0,"0","0")]
        expected_simulations = [(0, 2, 1, 2, 3)]
        expected_batches     = [(0, 0, 1, 0, 2, mean([1,0])), (0, 0, 4, 0, 2, mean([2,0]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_batches, expected_batches)

    def test_evaluate_sims_small(self):
        sim1            = LambdaSimulation(5, lambda i: (i,i), lambda s: [0,1,2], lambda s,a: a)
        sim2            = LambdaSimulation(4, lambda i: (i,i), lambda s: [3,4,5], lambda s,a: a)
        learner_factory = LearnerFactory[Tuple[int,int],int](LambdaLearner,lambda s,A: s[0]%3, family="0")
        benchmark       = UniversalBenchmark([sim1,sim2], batch_count=1, min_interactions=5, ignore_first=False, ignore_raise=False, processes=1)

        actual_learners,actual_simulations,actual_batches = benchmark.evaluate([learner_factory]).to_tuples()

        expected_learners    = [(0,"0","0")]
        expected_simulations = [(0, 5, 1, 2, 3), (1, 0, 0, 2, 3)]
        expected_batches     = [(0, 0, None, 0, 5, mean([0,1,2,0,1]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_batches, expected_batches)

    def test_evaluate_ignore_first(self):
        sim             = LambdaSimulation(50, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = LearnerFactory[int,int](LambdaLearner,lambda s,A: s%3, family="0")
        benchmark       = UniversalBenchmark([sim], batch_sizes=[1, 2], ignore_first=True, ignore_raise=False, processes=1)

        actual_learners, actual_simulations, actual_batches = benchmark.evaluate([learner_factory]).to_tuples()

        expected_learners    = [(0,"0","0")]
        expected_simulations = [(0, 2, 1, 1, 3)]
        expected_batches     = [(0, 0, None, 0, 2, mean([1,2]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_batches, expected_batches)

    def test_evaluate_lazy_sim(self):
        sim1            = LazySimulation(lambda:LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s, a: a))
        benchmark       = UniversalBenchmark([sim1], batch_count=1, ignore_first=False, ignore_raise=False, processes=1)
        learner_factory = LearnerFactory(LambdaLearner,lambda s, A: s%3, family="0")
        
        actual_learners,actual_simulations,actual_batches = benchmark.evaluate([learner_factory]).to_tuples()

        expected_learners    = [(0,"0","0")]
        expected_simulations = [(0, 5, 1, 1, 3)]
        expected_batches     = [(0, 0, None, 0, 5, mean([0,1,2,0,1]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_batches, expected_batches)

    def test_evalute_learners(self):
        sim              = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory1 = LearnerFactory[int,int](LambdaLearner, lambda s,A: A[s%3], family="0")
        learner_factory2 = LearnerFactory[int,int](LambdaLearner, lambda s,A: A[s%3], family="1")
        benchmark        = UniversalBenchmark([sim], batch_count=1, ignore_first=False, ignore_raise=False, processes=1)

        actual_results = benchmark.evaluate([learner_factory1, learner_factory2])
        actual_learners,actual_simulations,actual_batches = actual_results.to_tuples()

        expected_learners     = [(0,"0","0"), (1,"1","1")]
        expected_simulations  = [(0, 5, 1, 1, 3)]
        expected_batches = [
            (0, 0, None, 0, 5, mean([0,1,2,0,1])), 
            (1, 0, None, 0, 5, mean([0,1,2,0,1]))
        ]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_batches, expected_batches)

    def test_transaction_resume_1(self):
        sim             = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = LearnerFactory[int,int](LambdaLearner, lambda s,A: A[s%3], family="0")
        broken_factory  = LearnerFactory[int,int](LambdaLearner, lambda s,A: A[500], family="0")
        benchmark       = UniversalBenchmark([sim], batch_count=1, ignore_first=False, processes=1)

        #the second time the broken_factory() shouldn't ever be used for learning or choosing
        #because it already worked the first time and we are "resuming" benchmark from transaction.log
        try:
            first_results  = benchmark.evaluate([learner_factory], ".test/transactions.log")
            second_results = benchmark.evaluate([broken_factory], ".test/transactions.log")

            actual_learners,actual_simulations,actual_batches = second_results.to_tuples()

            expected_learners    = [(0,"0","0")]
            expected_simulations = [(0, 5, 1, 1, 3)]
            expected_batches     = [(0, 0, None, 0, 5, mean([0,1,2,0,1]))]
        finally:
            if Path('.test/transactions.log').exists(): Path('.test/transactions.log').unlink()

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_batches, expected_batches)

    def test_transaction_resume_ignore_first(self):
        sim             = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = LearnerFactory[int,int](LambdaLearner, lambda s,A: A[s%3], family="0")
        broken_factory  = LearnerFactory[int,int](LambdaLearner, lambda s,A: A[500], family="0")
        benchmark       = UniversalBenchmark([sim], batch_count=2, ignore_first=True, processes=1)

        #the second time the broken_factory() shouldn't ever be used for learning or choosing
        #because it already worked the first time and we are "resuming" benchmark from transaction.log

        try:
            first_results  = benchmark.evaluate([learner_factory], ".test/transactions.log")
            second_results = benchmark.evaluate([broken_factory], ".test/transactions.log")

            actual_learners,actual_simulations,actual_performances = second_results.to_tuples()

            expected_learners     = [(0,"0","0")]
            expected_simulations  = [(0, 2, 1, 1, 3)]
            expected_performances = [ (0, 0, None, 0, 2, mean([0,1]))]
        finally:
            if Path('.test/transactions.log').exists(): Path('.test/transactions.log').unlink()            

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_performances, expected_performances)

if __name__ == '__main__':
    unittest.main()