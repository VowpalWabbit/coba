import unittest

from typing import Tuple

from coba.simulations import LambdaSimulation, LazySimulation
from coba.learners import LambdaLearner
from coba.benchmarks import UniversalBenchmark
from coba.execution import ExecutionContext, NoneLogger
from coba.statistics import BatchMeanEstimator

ExecutionContext.Logger = NoneLogger()

class UniversalBenchmark_Tests(unittest.TestCase):

    def test_from_json(self):
        json = """{
            "batches": {"count":1},
            "simulations": [
                {"seed":1283,"type":"classification","from":{"format":"openml","id":1116}},
                {"seed":1283,"type":"classification","from":{"format":"openml","id":1116}}
            ]
        }"""

        benchmark = UniversalBenchmark.from_json(json)

        self.assertEqual(len(benchmark._simulations),2)
        self.assertIsInstance(benchmark._simulations[0],LazySimulation)

    def test_batch_count_1(self):
        sim             = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: A[s%3], name="0")
        benchmark       = UniversalBenchmark([sim], batch_count=1)

        actual_results = benchmark.evaluate([learner_factory])
        actual_learners,actual_simulations,actual_performances = actual_results.to_tuples()

        expected_learners     = [(0,"0")]
        expected_simulations  = [(0, 5, 1, 3)]
        expected_performances = [ (0, 0, 0, 5, BatchMeanEstimator([0,1,2,0,1]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_performances, expected_performances)

    def test_batch_count_2(self):
        sim             = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: A[s%3], name="0")
        benchmark       = UniversalBenchmark([sim], batch_count=2)

        actual_results = benchmark.evaluate([learner_factory])
        actual_learners,actual_simulations,actual_performances = actual_results.to_tuples()

        expected_learners     = [(0,"0")]
        expected_simulations  = [(0, 5, 1, 3)]
        expected_performances = [ (0, 0, 0, 3, BatchMeanEstimator([0,1,2])), (0, 0, 1, 2, BatchMeanEstimator([0,1]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_performances, expected_performances)

    def test_batch_size_1(self):
        sim             = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: s%3, name="0")
        benchmark       = UniversalBenchmark([sim], batch_size=2)

        actual_results = benchmark.evaluate([learner_factory])
        actual_learners,actual_simulations,actual_performances = actual_results.to_tuples()

        expected_learners     = [(0,"0")]
        expected_simulations  = [(0, 4, 1, 3)]
        expected_performances = [ (0, 0, 0, 2, BatchMeanEstimator([0,1])), (0, 0, 1, 2, BatchMeanEstimator([2,0]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_performances, expected_performances)

    def test_batch_size_2(self):
        sim             = LambdaSimulation(50, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: s%3, name="0")
        benchmark       = UniversalBenchmark([sim], batch_size=[1, 2, 4, 1])

        actual_results = benchmark.evaluate([learner_factory])
        actual_learners,actual_simulations,actual_performances = actual_results.to_tuples()

        expected_learners     = [(0,"0")]
        expected_simulations  = [(0, 8, 1, 3)]
        expected_performances = [
            (0, 0, 0, 1, BatchMeanEstimator([0])), 
            (0, 0, 1, 2, BatchMeanEstimator([1,2])),
            (0, 0, 2, 4, BatchMeanEstimator([0,1,2,0])),
            (0, 0, 3, 1, BatchMeanEstimator([1]))
        ]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_performances, expected_performances)

    def test_sims(self):
        sim1            = LambdaSimulation(5, lambda i: (i,i), lambda s: [0,1,2], lambda s,a: a)
        sim2            = LambdaSimulation(4, lambda i: (i,i), lambda s: [3,4,5], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[Tuple[int,int],int](lambda s,A: s[0]%3, name="0")
        benchmark       = UniversalBenchmark([sim1,sim2], batch_count=1)

        actual_results = benchmark.evaluate([learner_factory])
        actual_learners,actual_simulations,actual_performances = actual_results.to_tuples()

        expected_learners     = [(0,"0")]
        expected_simulations  = [(0, 5, 2, 3), (1, 4, 2, 3)]
        expected_performances = [(0, 0, 0, 5, BatchMeanEstimator([0,1,2,0,1])), (0, 1, 0, 4, BatchMeanEstimator([3,4,5,3]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_performances, expected_performances)
    
    def test_lazy_sim(self):
        sim1            = LazySimulation[int,int](lambda:LambdaSimulation(50, lambda i: i, lambda s: [0,1,2], lambda s, a: a))
        benchmark       = UniversalBenchmark([sim1], batch_size=[4,2])
        learner_factory = lambda: LambdaLearner[int,int](lambda s, A: s%3, name="0")
        
        actual_results = benchmark.evaluate([learner_factory])

        actual_learners,actual_simulations,actual_performances = actual_results.to_tuples()

        expected_learners     = [(0,"0")]
        expected_simulations  = [(0, 6, 1, 3)]
        expected_performances = [(0, 0, 0, 4, BatchMeanEstimator([0,1,2,0])), (0, 0, 1, 2, BatchMeanEstimator([1,2]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_performances, expected_performances)

    def test_learners(self):
        sim              = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory1 = lambda: LambdaLearner[int,int](lambda s,A: A[s%3], name="0")
        learner_factory2 = lambda: LambdaLearner[int,int](lambda s,A: A[s%3], name="1")
        benchmark        = UniversalBenchmark([sim], batch_count=1)

        actual_results = benchmark.evaluate([learner_factory1, learner_factory2])

        actual_learners,actual_simulations,actual_performances = actual_results.to_tuples()

        expected_learners     = [(0,"0"), (1,"1")]
        expected_simulations  = [(0, 5, 1, 3)]
        expected_performances = [(0, 0, 0, 5, BatchMeanEstimator([0,1,2,0,1])), (1, 0, 0, 5, BatchMeanEstimator([0,1,2,0,1]))]

        self.assertSequenceEqual(actual_learners, expected_learners)
        self.assertSequenceEqual(actual_simulations, expected_simulations)
        self.assertSequenceEqual(actual_performances, expected_performances)

if __name__ == '__main__':
    unittest.main()