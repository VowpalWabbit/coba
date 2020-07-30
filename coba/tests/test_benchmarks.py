import unittest

import itertools

from coba.simulations import LambdaSimulation, LazySimulation
from coba.learners import LambdaLearner
from coba.benchmarks import UniversalBenchmark, Result
from coba.statistics import SummaryStats

class UniversalBenchmark_Tests(unittest.TestCase):

    def _verify_result_from_expected_obs(self, actual_results, expected_obs):

        expected_results = []
        key = lambda o: o[0:3]
        for group in itertools.groupby(sorted(expected_obs, key=key), key):
            expected_results.append(Result(*group[0], SummaryStats().add_observations([i[3] for i in group[1]])))

        self.assertEqual(len(actual_results), len(expected_results))

        for actual_result, expected_result in zip(actual_results, expected_results):
            self.assertEqual(actual_result.learner_name, expected_result.learner_name)
            self.assertEqual(actual_result.simulation_index, expected_result.simulation_index)
            self.assertEqual(actual_result.batch_index, expected_result.batch_index)
            self.assertEqual(actual_result.stats.mean, expected_result.stats.mean)

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

    def test_one_sim_batch_size_five_ones(self):
        sim             = LambdaSimulation[int,int](50, lambda i: i, lambda s: [0,1,2], lambda s, a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s, A: s%3, name="0")
        benchmark       = UniversalBenchmark[int,int]([sim], batch_size=[1]*5)

        results = benchmark.evaluate([learner_factory])

        expected_observations = [
            ("0",0,0,0),("0",0,1,1),("0",0,2,2),("0",0,3,0),("0",0,4,1)
        ]

        self._verify_result_from_expected_obs(results, expected_observations)

    def test_one_sim_batch_count_one(self):
        sim             = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: A[s%3], name="0")
        benchmark       = UniversalBenchmark([sim], batch_count=1)

        results = benchmark.evaluate([learner_factory])

        expected_observations = [
            ("0",0,0,0),("0",0,0,1),("0",0,0,2),("0",0,0,0),("0",0,0,1)
        ]

        self._verify_result_from_expected_obs(results, expected_observations)

    def test_one_sim_batch_count_two(self):
        sim             = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: A[s%3], name="0")
        benchmark       = UniversalBenchmark([sim], batch_count=2)

        results = benchmark.evaluate([learner_factory])

        expected_observations = [
            ("0",0,0,0),("0",0,0,1),("0",0,0,2),("0",0,1,0),("0",0,1,1)
        ]

        self._verify_result_from_expected_obs(results, expected_observations)

    def test_one_sim_batch_size_three_threes(self):
        sim             = LambdaSimulation(50, lambda i: i, lambda s: [0,1,2], lambda s, a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s, A: s%3,name="0")
        benchmark       = UniversalBenchmark([sim], batch_size=[3,3,3])

        results = benchmark.evaluate([learner_factory])

        expected_observations = [
            ("0",0,0,0),("0",0,0,1),("0",0,0,2),("0",0,1,0),("0",0,1,1),("0",0,1,2),("0",0,2,0),("0",0,2,1),("0",0,2,2)
        ]

        self._verify_result_from_expected_obs(results, expected_observations)

    def test_one_sim_batch_size_four_and_two(self):
        sim            = LambdaSimulation(50, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s, A: s%3, name="0")
        benchmark       = UniversalBenchmark([sim], batch_size=[4,2])

        results = benchmark.evaluate([learner_factory])

        expected_observations = [
            ("0",0,0,0),("0",0,0,1),("0",0,0,2),("0",0,0,0),("0",0,1,1),("0",0,1,2)
        ]

        self._verify_result_from_expected_obs(results, expected_observations)

    def test_one_sim_batch_size_sequence(self):
        sim            = LambdaSimulation(50, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: s%3, name="0")
        benchmark       = UniversalBenchmark([sim], batch_size=[1, 2, 4, 1])

        results = benchmark.evaluate([learner_factory])

        expected_observations = [
            ("0",0,0,0),("0",0,1,1),("0",0,1,2),("0",0,2,0),("0",0,2,1),("0",0,2,2),("0",0,2,0),("0",0,3,1)
        ]

        self._verify_result_from_expected_obs(results, expected_observations)

    def test_two_sims_batch_size_five_ones(self):
        sim1            = LambdaSimulation(50, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        sim2            = LambdaSimulation(50, lambda i: i, lambda s: [3,4,5], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: s%3, name="0")
        benchmark       = UniversalBenchmark([sim1,sim2], batch_size=[1]*5)

        results = benchmark.evaluate([learner_factory])

        expected_observations = [
            ("0",0,0,0),("0",0,1,1),("0",0,2,2),("0",0,3,0),("0",0,4,1),
            ("0",1,0,3),("0",1,1,4),("0",1,2,5),("0",1,3,3),("0",1,4,4)
        ]

        self._verify_result_from_expected_obs(results, expected_observations)

    def test_two_sims_batch_count_one(self):
        sim1            = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        sim2            = LambdaSimulation(4, lambda i: i, lambda s: [3,4,5], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: s%3, name="0")
        benchmark       = UniversalBenchmark([sim1,sim2], batch_count=1)

        results = benchmark.evaluate([learner_factory])

        expected_observations = [
            ("0",0,0,0),("0",0,0,1),("0",0,0,2),("0",0,0,0),("0",0,0,1),
            ("0",1,0,3),("0",1,0,4),("0",1,0,5),("0",1,0,3)
        ]

        self._verify_result_from_expected_obs(results, expected_observations)

    def test_two_sims_batch_count_two(self):
        sim1            = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        sim2            = LambdaSimulation(4, lambda i: i, lambda s: [3,4,5], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: s%3, name="0")
        benchmark       = UniversalBenchmark([sim1,sim2], batch_count=2)

        results = benchmark.evaluate([learner_factory])

        expected_observations = [
            ("0",0,0,0),("0",0,0,1),("0",0,0,2),("0",0,1,0),("0",0,1,1),
            ("0",1,0,3),("0",1,0,4),("0",1,1,5),("0",1,1,3)
        ]

        self._verify_result_from_expected_obs(results, expected_observations)

    def test_two_sims_batch_size_three_threes(self):
        sim1            = LambdaSimulation(50, lambda i: i, lambda s: [0,1,2], lambda s, a: a)
        sim2            = LambdaSimulation(50, lambda i: i, lambda s: [3,4,5], lambda s, a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s, A: s%3, name="0")
        benchmark       = UniversalBenchmark[int,int]([sim1,sim2], batch_size= [3,3,3])

        results = benchmark.evaluate([learner_factory])

        expected_observations = [
            ("0",0,0,0),("0",0,0,1),("0",0,0,2),("0",0,1,0),("0",0,1,1),("0",0,1,2),("0",0,2,0),("0",0,2,1),("0",0,2,2),
            ("0",1,0,3),("0",1,0,4),("0",1,0,5),("0",1,1,3),("0",1,1,4),("0",1,1,5),("0",1,2,3),("0",1,2,4),("0",1,2,5)
        ]

        self._verify_result_from_expected_obs(results, expected_observations)

    def test_two_sims_batch_size_four_and_two(self):
        sim1            = LambdaSimulation(50, lambda i: i, lambda s: [0,1,2], lambda s, a: a)
        sim2            = LambdaSimulation(50, lambda i: i, lambda s: [3,4,5], lambda s, a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s, A: s%3, name="0")
        benchmark       = UniversalBenchmark([sim1,sim2], batch_size = [4,2])

        results = benchmark.evaluate([learner_factory])

        expected_observations = [
            ("0",0,0,0),("0",0,0,1),("0",0,0,2),("0",0,0,0),("0",0,1,1),("0",0,1,2),
            ("0",1,0,3),("0",1,0,4),("0",1,0,5),("0",1,0,3),("0",1,1,4),("0",1,1,5)
        ]

        self._verify_result_from_expected_obs(results, expected_observations)

    def test_lazy_sim_two_batches(self):
        sim1            = LazySimulation[int,int](lambda:LambdaSimulation(50, lambda i: i, lambda s: [0,1,2], lambda s, a: a))
        benchmark       = UniversalBenchmark([sim1], batch_size=[4,2])
        learner_factory = lambda: LambdaLearner[int,int](lambda s, A: s%3, name="0")
        
        results = benchmark.evaluate([learner_factory])

        expected_observations = [
            ("0",0,0,0),("0",0,0,1),("0",0,0,2),("0",0,0,0),("0",0,1,1),("0",0,1,2)
        ]

        self._verify_result_from_expected_obs(results, expected_observations)

class Result_Tests(unittest.TestCase):

    def test_to_dict_works(self):
        pass

if __name__ == '__main__':
    unittest.main()