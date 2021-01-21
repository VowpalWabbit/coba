
import unittest

from pathlib import Path
from statistics import mean

from coba.simulations import LambdaSimulation
from coba.execution import ExecutionContext, NoneLogger
from coba.learners import Learner
from coba.benchmarks import Benchmark, Result, Transaction, TransactionIsNew

#for testing purposes
class ModuloLearner(Learner[int,int]):
    def __init__(self, family="0"):
        self._family = family

    @property
    def family(self):
        return self._family

    @property
    def params(self):
        return {}

    def predict(self, key, context, actions):
        return [ int(i == actions.index(actions[context%len(actions)])) for i in range(len(actions)) ]

    def learn(self, key, context, action, reward, probability):
        pass

class BrokenLearner(Learner[int,int]):
    @property
    def family(self):
        return "0"

    @property
    def params(self):
        return {}

    def predict(self, key, context, actions):
        raise Exception()

    def learn(self, key, context, action, reward, probability):
        pass

class NotPicklableLearner(Learner[int,int]):
    @property
    def family(self):
        return "0"

    @property
    def params(self):
        return {}

    def __init__(self):
        self._val = lambda x: 1

    def predict(self, key, context, actions):
        return 0

    def learn(self, key, context, action, reward, probability):
        pass

class TransactionIsNew_Test(unittest.TestCase):
    
    def test_duplicates_are_dropped(self):
        existing = Result.from_transactions([
            Transaction.learner(0, a='A'),
            Transaction.simulation(0, b='B'),
            Transaction.batch(0, 1, reward=mean([1,2,3]))
        ])

        filter = TransactionIsNew(existing)

        transactions = list(filter.filter([
            Transaction.learner(0, a='A'), 
            Transaction.simulation(0, b='B'), 
            Transaction.batch(0, 1, reward=mean([1,2,3]))]
        ))

        self.assertEqual(len(transactions), 0)

    def test_non_duplicates_are_kept(self):
        existing = Result.from_transactions([
            Transaction.learner(0, a='A'),
            Transaction.simulation(0, b='B'),
            Transaction.batch(0, 1, reward=mean([1,2,3]))
        ])

        filter = TransactionIsNew(existing)

        transactions = list(filter.filter([
            Transaction.learner(1, a='A'), 
            Transaction.simulation(1, b='B'), 
            Transaction.batch(1, 1, reward=mean([1,2,3]))]
        ))

        self.assertEqual(len(transactions), 3)

class Result_Tests(unittest.TestCase):

    def test_has_batches_key(self):
        result = Result.from_transactions([
            Transaction.batch(0, 1, a='A'),
            Transaction.batch(0, 2, b='B')
        ])

        self.assertTrue( (0,1) in result.batches)
        self.assertTrue( (0,2) in result.batches)

        self.assertEqual(len(result.batches), 2)

    def test_has_version(self):
        result = Result.from_transactions([Transaction.version(1)])
        self.assertEqual(result.version, 1)

class Benchmark_Single_Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        ExecutionContext.Logger = NoneLogger()
        ExecutionContext.Config.processes = 1

    def test_from_json(self):
        json = """{
            "batches"     : {"count":1},
            "ignore_first": false,
            "shuffle"     : [1283],
            "simulations" : [
                {"type":"classification","from":{"format":"openml","id":1116}}
            ]
        }"""

        benchmark = Benchmark.from_json(json)

        self.assertEqual(1, len(benchmark._simulation_pipes))

    def test_sims(self):
        sim1            = LambdaSimulation(5, lambda t: t, lambda t: [0,1,2], lambda c,a: a)
        sim2            = LambdaSimulation(4, lambda t: t, lambda t: [3,4,5], lambda c,a: a)
        learner_factory = ModuloLearner()
        benchmark       = Benchmark([sim1,sim2], batch_count=1, ignore_raise=False)

        actual_learners,actual_simulations,actual_batches = benchmark.evaluate([learner_factory]).to_tuples()

        expected_learners    = [(0,"0","0")]
        expected_simulations = [(0, '0', ['{"Batch":[None, 1, None]}'], 5, 1, 1, 3), (1, '1', ['{"Batch":[None, 1, None]}'], 4, 1, 1, 3)]
        expected_batches     = [(0, 0, [5], [mean([0,1,2,0,1])]), (1, 0, [4], [mean([3,4,5,3])])]

        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_simulations, expected_simulations)
        self.assertCountEqual(actual_batches, expected_batches)

    def test_seeds(self):
        sim1      = LambdaSimulation(5, lambda t: t, lambda t: [0,1,2], lambda c,a: a)
        learner   = ModuloLearner()
        benchmark = Benchmark([sim1], batch_sizes=[2], ignore_raise=False, seeds=[1,4])

        actual_learners,actual_simulations,actual_batches = benchmark.evaluate([learner]).to_tuples()

        expected_learners    = [(0,"0","0")]
        expected_simulations = [(0, '0', ['{"Shuffle":1}', '{"Batch":[None, None, [2]]}'], 2, 1, 1, 3), (1, '0', ['{"Shuffle":4}', '{"Batch":[None, None, [2]]}'], 2, 1, 1, 3)]
        expected_batches     = [(0, 0, [2], [mean([1,0])]), (1, 0, [2], [mean([2,0])])]

        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_simulations, expected_simulations)
        self.assertCountEqual(actual_batches, expected_batches)

    def test_take(self):
        sim1      = LambdaSimulation(5, lambda t: t, lambda t: [0,1,2], lambda c,a: a)
        sim2      = LambdaSimulation(4, lambda t: t, lambda t: [3,4,5], lambda c,a: a)
        learner   = ModuloLearner()
        benchmark = Benchmark([sim1,sim2], batch_count=1, take=5, ignore_raise=False)

        actual_learners,actual_simulations,actual_batches = benchmark.evaluate([learner]).to_tuples()

        expected_learners    = [(0,"0","0")]
        expected_simulations = [(0, '0', ['{"Take":5}', '{"Batch":[None, 1, None]}'], 5, 1, 1, 3), (1, '1', ['{"Take":5}', '{"Batch":[None, 1, None]}'], 0, 0, 0, 0)]
        expected_batches     = [(0, 0, [5], [mean([0,1,2,0,1])])]

        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_simulations, expected_simulations)
        self.assertCountEqual(actual_batches, expected_batches)

    def test_learners(self):
        sim       = LambdaSimulation(5, lambda t: t, lambda t: [0,1,2], lambda c,a: a)
        learner1  = ModuloLearner("0") #type: ignore
        learner2  = ModuloLearner("1") #type: ignore
        benchmark = Benchmark([sim], batch_count=1, ignore_raise=False)

        actual_results = benchmark.evaluate([learner1, learner2])
        actual_learners,actual_simulations,actual_batches = actual_results.to_tuples()

        expected_learners     = [(0,"0","0"), (1,"1","1")]
        expected_simulations  = [(0, '0', ['{"Batch":[None, 1, None]}'], 5, 1, 1, 3)]
        expected_batches      = [(0, 0, [5], [mean([0,1,2,0,1])]), (0, 1, [5], [mean([0,1,2,0,1])]) ]

        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_simulations, expected_simulations)
        self.assertCountEqual(actual_batches, expected_batches)

    def test_transaction_resume_1(self):
        sim             = LambdaSimulation(5, lambda t: t, lambda t: [0,1,2], lambda c,a: a)
        working_learner = ModuloLearner()
        broken_learner  = BrokenLearner()
        benchmark       = Benchmark([sim], batch_count=1)

        #the second time the broken_factory() shouldn't ever be used for learning or choosing
        #because it already worked the first time and we are "resuming" benchmark from transaction.log
        try:
            first_results  = benchmark.evaluate([working_learner], "coba/tests/.temp/transactions.log")
            second_results = benchmark.evaluate([broken_learner], "coba/tests/.temp/transactions.log")

            actual_learners,actual_simulations,actual_batches = second_results.to_tuples()

            expected_learners    = [(0,"0","0")]
            expected_simulations = [(0, '0', ['{"Batch":[None, 1, None]}'], 5, 1, 1, 3)]
            expected_batches     = [(0, 0, [5], [mean([0,1,2,0,1])])]
        finally:
            if Path('coba/tests/.temp/transactions.log').exists(): Path('coba/tests/.temp/transactions.log').unlink()

        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_simulations, expected_simulations)
        self.assertCountEqual(actual_batches, expected_batches)

class Benchmark_Multi_Tests(Benchmark_Single_Tests):
    
    @classmethod
    def setUpClass(cls) -> None:
        ExecutionContext.Logger = NoneLogger()
        ExecutionContext.Config.processes = 2

    def test_not_picklable_learner(self):
        sim1      = LambdaSimulation(5, lambda t: t, lambda t: [0,1,2], lambda c,a: a)
        learner   = NotPicklableLearner()
        benchmark = Benchmark([sim1], batch_sizes=[2], ignore_raise=False, seeds=[1,4])

        with self.assertRaises(Exception) as cm:
            benchmark.evaluate([learner])

        self.assertTrue("Learners are required to be picklable" in str(cm.exception))

if __name__ == '__main__':
    unittest.main()