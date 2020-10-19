import unittest

from pathlib import Path
from statistics import mean

from coba.simulations import LambdaSimulation, LazySimulation, JsonSimulation
from coba.learners import LambdaLearner
from coba.execution import ExecutionContext, NoneLogger
from coba.json import CobaJsonEncoder, CobaJsonDecoder
from coba.preprocessing import CountBatcher, SizesBatcher
from coba.benchmarks import TransactionReadWrite, UniversalBenchmark, Result
from coba.data import DiskReadWrite, MemoryReadWrite

ExecutionContext.Logger = NoneLogger()

class Result_Tests(unittest.TestCase):

    def test_has_batch_key(self):
        writer = TransactionReadWrite(MemoryReadWrite())
        writer.write_batch(0,1,None,2, a='A')
        result = Result.from_transactions(writer.read())

        self.assertTrue(result.has_batch(0,1,None,2))

    def test_to_from_transaction_log_once(self):

        def write_result(writer: TransactionReadWrite)-> None:
            writer.write_learner(0, a='A')
            writer.write_simulation(0, b='B')
            writer.write_batch(0, 1, None, 2, reward=mean([1,2,3]))

        try:
            disk   = TransactionReadWrite(DiskReadWrite(".test/transactions.log"))
            memory = TransactionReadWrite(MemoryReadWrite()) 
            
            write_result(disk)
            write_result(memory)

            expected_result = Result.from_transactions(memory.read())
            actual_result   = Result.from_transactions(disk.read())
        finally:
            if Path('.test/transactions.log').exists(): Path('.test/transactions.log').unlink()

        self.assertEqual(actual_result.to_tuples(), expected_result.to_tuples())

    def test_to_from_transaction_log_twice(self):

        def write_first_result(writer: TransactionReadWrite)-> None:
            writer.write_learner(0,a='A')
            writer.write_simulation(0,b='B')
            writer.write_batch(0,1,None,2,reward=mean([1,2,3]))

        def write_second_result(writer: TransactionReadWrite)-> None:
            writer.write_learner(0,a='z')
            writer.write_simulation(0,b='q')
            writer.write_batch(1,1,None,0,reward=mean([1,2,3,4,5]))

        try:
            disk1   = TransactionReadWrite(DiskReadWrite(".test/transactions.log"))
            disk2   = TransactionReadWrite(DiskReadWrite(".test/transactions.log"))
            memory = TransactionReadWrite(MemoryReadWrite()) 

            write_first_result(disk1)
            write_second_result(disk2)
                    
            
            write_first_result(memory)
            write_second_result(memory)

            actual_result   = Result.from_transaction_log(".test/transactions.log")
            expected_result = Result.from_transactions(memory.read())
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
        self.assertEqual(benchmark._seeds, [1283])
        self.assertEqual(len(benchmark._simulations),1)
        self.assertIsInstance(benchmark._simulations[0],JsonSimulation)

    def test_evaluate_sims(self):
        sim1            = LambdaSimulation(5, lambda i: (i,i), lambda s: [0,1,2], lambda s,a: a)
        sim2            = LambdaSimulation(4, lambda i: (i,i), lambda s: [3,4,5], lambda s,a: a)
        learner_factory = lambda: LambdaLearner(lambda s,A: s[0]%3, family="0") #type: ignore
        benchmark       = UniversalBenchmark([sim1,sim2], CountBatcher(1), ignore_first=False, ignore_raise=False, max_processes=1)

        actual_learners,actual_simulations,actual_batches = benchmark.evaluate([learner_factory]).to_tuples()

        expected_learners    = [(0,"0","0")]
        expected_simulations = [(0, 5, 1, 1, 2, 3), (1, 4, 1, 1, 2, 3)]
        expected_batches     = [(0, 0, None, 0, 5, mean([0,1,2,0,1])), (0, 1, None, 0, 4, mean([3,4,5,3]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_batches, expected_batches)

    def test_evaluate_seeds(self):
        sim1            = LambdaSimulation(5, lambda i: (i,i), lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner(lambda s,A: s[0]%3, family="0") #type: ignore
        benchmark       = UniversalBenchmark([sim1], SizesBatcher([2]), False, False, [1,4], max_processes=1)

        actual_learners,actual_simulations,actual_batches = benchmark.evaluate([learner_factory]).to_tuples()

        expected_learners    = [(0,"0","0")]
        expected_simulations = [(0, 2, 1, 2, 2, 3)]
        expected_batches     = [(0, 0, 1, 0, 2, mean([1,0])), (0, 0, 4, 0, 2, mean([2,0]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_batches, expected_batches)

    def test_evaluate_sims_small(self):
        sim1            = LambdaSimulation(5, lambda i: (i,i), lambda s: [0,1,2], lambda s,a: a)
        sim2            = LambdaSimulation(4, lambda i: (i,i), lambda s: [3,4,5], lambda s,a: a)
        learner_factory = lambda: LambdaLearner(lambda s,A: s[0]%3, family="0") #type: ignore
        benchmark       = UniversalBenchmark([sim1,sim2], CountBatcher(1,5), ignore_first=False, ignore_raise=False, max_processes=1)

        actual_learners,actual_simulations,actual_batches = benchmark.evaluate([learner_factory]).to_tuples()

        expected_learners    = [(0,"0","0")]
        expected_simulations = [(0, 5, 1, 1, 2, 3), (1, 0, 0, 1, 2, 3)]
        expected_batches     = [(0, 0, None, 0, 5, mean([0,1,2,0,1]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_batches, expected_batches)

    def test_evaluate_ignore_first(self):
        sim             = LambdaSimulation(50, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner(lambda s,A: s%3, family="0") #type: ignore
        benchmark       = UniversalBenchmark([sim], SizesBatcher([1, 2]), ignore_first=True, ignore_raise=False, max_processes=1)

        actual_learners, actual_simulations, actual_batches = benchmark.evaluate([learner_factory]).to_tuples()

        expected_learners    = [(0,"0","0")]
        expected_simulations = [(0, 2, 1, 1, 1, 3)]
        expected_batches     = [(0, 0, None, 0, 2, mean([1,2]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_batches, expected_batches)

    def test_evaluate_lazy_sim(self):
        sim1            = LazySimulation(lambda:LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s, a: a))
        benchmark       = UniversalBenchmark([sim1], CountBatcher(1), ignore_first=False, ignore_raise=False, max_processes=1)
        learner_factory = lambda: LambdaLearner(lambda s, A: s%3, family="0") #type: ignore
        
        actual_learners,actual_simulations,actual_batches = benchmark.evaluate([learner_factory]).to_tuples()

        expected_learners    = [(0,"0","0")]
        expected_simulations = [(0, 5, 1, 1, 1, 3)]
        expected_batches     = [(0, 0, None, 0, 5, mean([0,1,2,0,1]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_batches, expected_batches)

    def test_evalute_learners(self):
        sim              = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory1 = lambda: LambdaLearner(lambda s,A: A[s%3], family="0") #type: ignore
        learner_factory2 = lambda: LambdaLearner(lambda s,A: A[s%3], family="1") #type: ignore
        benchmark        = UniversalBenchmark([sim], CountBatcher(1), ignore_first=False, ignore_raise=False, max_processes=1)

        actual_results = benchmark.evaluate([learner_factory1, learner_factory2])
        actual_learners,actual_simulations,actual_batches = actual_results.to_tuples()

        expected_learners     = [(0,"0","0"), (1,"1","1")]
        expected_simulations  = [(0, 5, 1, 1, 1, 3)]
        expected_batches = [
            (0, 0, None, 0, 5, mean([0,1,2,0,1])), 
            (1, 0, None, 0, 5, mean([0,1,2,0,1]))
        ]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_batches, expected_batches)

    def test_transaction_resume_1(self):
        sim             = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner(lambda s,A: A[s%3], family="0") #type: ignore
        broken_factory  = lambda: LambdaLearner(lambda s,A: A[500], family="0") #type: ignore
        benchmark       = UniversalBenchmark([sim], CountBatcher(1), ignore_first=False, max_processes=1)

        #the second time the broken_factory() shouldn't ever be used for learning or choosing
        #because it already worked the first time and we are "resuming" benchmark from transaction.log
        try:
            first_results  = benchmark.evaluate([learner_factory], ".test/transactions.log")
            second_results = benchmark.evaluate([broken_factory], ".test/transactions.log")

            actual_learners,actual_simulations,actual_batches = second_results.to_tuples()

            expected_learners    = [(0,"0","0")]
            expected_simulations = [(0, 5, 1, 1, 1, 3)]
            expected_batches     = [(0, 0, None, 0, 5, mean([0,1,2,0,1]))]
        finally:
            if Path('.test/transactions.log').exists(): Path('.test/transactions.log').unlink()

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_batches, expected_batches)

    def test_transaction_resume_ignore_first(self):
        sim             = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner(lambda s,A: A[s%3], family="0") #type: ignore
        broken_factory  = lambda: LambdaLearner(lambda s,A: A[500], family="0") #type: ignore
        benchmark       = UniversalBenchmark([sim], CountBatcher(2), ignore_first=True, max_processes=1)

        #the second time the broken_factory() shouldn't ever be used for learning or choosing
        #because it already worked the first time and we are "resuming" benchmark from transaction.log

        try:
            first_results  = benchmark.evaluate([learner_factory], ".test/transactions.log")
            second_results = benchmark.evaluate([broken_factory], ".test/transactions.log")

            actual_learners,actual_simulations,actual_performances = second_results.to_tuples()

            expected_learners     = [(0,"0","0")]
            expected_simulations  = [(0, 2, 1, 1, 1, 3)]
            expected_performances = [ (0, 0, None, 0, 2, mean([0,1]))]
        finally:
            if Path('.test/transactions.log').exists(): Path('.test/transactions.log').unlink()            

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_performances, expected_performances)

if __name__ == '__main__':
    unittest.main()