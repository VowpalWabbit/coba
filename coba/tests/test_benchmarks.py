import unittest

from math import isnan

from coba.simulations import LambdaSimulation, LazySimulation
from coba.learners import LambdaLearner
from coba.benchmarks import Stats, Result, UniversalBenchmark

class Stats_Tests(unittest.TestCase):
    def test_from_values_multi_mean_is_correct_1(self):
        stats = Stats.from_observations([1,1,3,3])
        self.assertEqual(2,stats.mean)

    def test_from_values_multi_mean_is_correct_2(self):
        stats = Stats.from_observations([1,1,1,1])
        self.assertEqual(1,stats.mean)

    def test_from_values_single_mean_is_correct(self):
        stats = Stats.from_observations([3])
        self.assertEqual(3,stats.mean)

    def test_from_values_empty_mean_is_correct(self):
        stats = Stats.from_observations([])
        self.assertTrue(isnan(stats.mean))

class Result_Tests(unittest.TestCase):

    def test_batch_means1(self):
        result = Result.from_observations([(0,0,3), (0,0,4), (0,1,5), (0,1,5), (0,2,6)], False)

        self.assertEqual(len(result.batch_stats),3)

        self.assertAlmostEqual((3.+4.)/2., result.batch_stats[0].mean)
        self.assertAlmostEqual((5+5  )/2., result.batch_stats[1].mean)
        self.assertAlmostEqual((6    )/1., result.batch_stats[2].mean)

    def test_batch_means2(self):
        result = Result.from_observations([(0,0,3), (0,0,4), (0,1,5), (0,1,5), (0,2,6), (1,0,3)], False)

        self.assertEqual(len(result.batch_stats),3)

        self.assertAlmostEqual((3.5+3)/2., result.batch_stats[0].mean)
        self.assertAlmostEqual((5+5  )/2., result.batch_stats[1].mean)
        self.assertAlmostEqual((6    )/1., result.batch_stats[2].mean)

    def test_cumulative_batch_means1(self):
        result = Result.from_observations([(0,0,3), (0,0,4), (0,1,5), (0,1,5), (0,2,6)], False)

        self.assertEqual(len(result.cumulative_batch_stats),3)

        self.assertAlmostEqual((3+4    )/2., result.cumulative_batch_stats[0].mean)
        self.assertAlmostEqual((3.5+5  )/2., result.cumulative_batch_stats[1].mean)
        self.assertAlmostEqual((3.5+5+6)/3., result.cumulative_batch_stats[2].mean)

    def test_cumulative_batch_means2(self):
        result = Result.from_observations([(0,0,3), (0,0,4), (0,1,5), (0,1,5), (0,2,6), (1,0,3)], False)

        self.assertEqual(len(result.cumulative_batch_stats),3)

        self.assertAlmostEqual((3.5+3   )/2., result.cumulative_batch_stats[0].mean)
        self.assertAlmostEqual((3.25+5  )/2., result.cumulative_batch_stats[1].mean)
        self.assertAlmostEqual((3.25+5+6)/3., result.cumulative_batch_stats[2].mean)

class UniversalBenchmark_Tests(unittest.TestCase):

    def _verify_result_from_expected_obs(self, actual_result, expected_obs):
        expected_result = Result.from_observations(expected_obs)

        self.assertEqual(len(actual_result.batch_stats), len(expected_result.batch_stats))
        self.assertEqual(len(actual_result.sim_stats), len(expected_result.sim_stats))

        for actual_stat, expected_stat in zip(actual_result.batch_stats, expected_result.batch_stats):
            self.assertEqual(actual_stat.mean, expected_stat.mean)

        for actual_stat, expected_stat in zip(actual_result.sim_stats, expected_result.sim_stats):
            self.assertEqual(actual_stat.mean, expected_stat.mean)

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
        learner_factory = lambda: LambdaLearner[int,int](lambda s, A: int(s%3))
        benchmark       = UniversalBenchmark[int,int]([sim], batch_size=[1]*5)

        result = benchmark.evaluate([learner_factory])[0]

        expected_observations = [
            (0,0,0),(0,1,1),(0,2,2),(0,3,0),(0,4,1)
        ]

        self._verify_result_from_expected_obs(result, expected_observations)

    def test_one_sim_batch_count_one(self):
        sim             = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: A[int(s%3)])
        benchmark       = UniversalBenchmark([sim], batch_count=1)

        result = benchmark.evaluate([learner_factory])[0]

        expected_observations = [
            (0,0,0),(0,0,1),(0,0,2),(0,0,0),(0,0,1)
        ]

        self._verify_result_from_expected_obs(result, expected_observations)

    def test_one_sim_batch_count_two(self):
        sim             = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: A[int(s%3)])
        benchmark       = UniversalBenchmark([sim], batch_count=2)

        result = benchmark.evaluate([learner_factory])[0]

        expected_observations = [
            (0,0,0),(0,0,1),(0,0,2),(0,1,0),(0,1,1)
        ]

        self._verify_result_from_expected_obs(result, expected_observations)

    def test_one_sim_batch_size_three_threes(self):
        sim             = LambdaSimulation(50, lambda i: i, lambda s: [0,1,2], lambda s, a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s, A: int(s%3))
        benchmark       = UniversalBenchmark([sim], batch_size=[3,3,3])

        result = benchmark.evaluate([learner_factory])[0]

        expected_observations = [
            (0,0,0),(0,0,1),(0,0,2),(0,1,0),(0,1,1),(0,1,2),(0,2,0),(0,2,1),(0,2,2)
        ]

        self._verify_result_from_expected_obs(result, expected_observations)

    def test_one_sim_batch_size_four_and_two(self):
        sim            = LambdaSimulation(50, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s, A: s%3)
        benchmark       = UniversalBenchmark([sim], batch_size=[4,2])

        result = benchmark.evaluate([learner_factory])[0]

        expected_observations = [
            (0,0,0),(0,0,1),(0,0,2),(0,0,0),(0,1,1),(0,1,2)
        ]

        self._verify_result_from_expected_obs(result, expected_observations)

    def test_one_sim_batch_size_sequence(self):
        sim            = LambdaSimulation(50, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: int(s%3))
        benchmark       = UniversalBenchmark([sim], batch_size=[1, 2, 4, 1])

        result = benchmark.evaluate([learner_factory])[0]

        expected_observations = [
            (0,0,0),(0,1,1),(0,1,2),(0,2,0),(0,2,1),(0,2,2),(0,2,0),(0,3,1)
        ]

        self._verify_result_from_expected_obs(result, expected_observations)

    def test_two_sims_batch_size_five_ones(self):
        sim1            = LambdaSimulation(50, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        sim2            = LambdaSimulation(50, lambda i: i, lambda s: [3,4,5], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: int(s%3))
        benchmark       = UniversalBenchmark([sim1,sim2], batch_size=[1]*5)

        result = benchmark.evaluate([learner_factory])[0]

        expected_observations = [
            (0,0,0),(0,1,1),(0,2,2),(0,3,0),(0,4,1),
            (1,0,3),(1,1,4),(1,2,5),(1,3,3),(1,4,4)
        ]

        self._verify_result_from_expected_obs(result, expected_observations)

    def test_two_sims_batch_count_one(self):
        sim1            = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        sim2            = LambdaSimulation(4, lambda i: i, lambda s: [3,4,5], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: int(s%3))
        benchmark       = UniversalBenchmark([sim1,sim2], batch_count=1)

        result = benchmark.evaluate([learner_factory])[0]

        expected_observations = [
            (0,0,0),(0,0,1),(0,0,2),(0,0,0),(0,0,1),
            (1,0,3),(1,0,4),(1,0,5),(1,0,3)
        ]

        self._verify_result_from_expected_obs(result, expected_observations)

    def test_two_sims_batch_count_two(self):
        sim1            = LambdaSimulation(5, lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        sim2            = LambdaSimulation(4, lambda i: i, lambda s: [3,4,5], lambda s,a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s,A: int(s%3))
        benchmark       = UniversalBenchmark([sim1,sim2], batch_count=2)

        result = benchmark.evaluate([learner_factory])[0]

        expected_observations = [
            (0,0,0),(0,0,1),(0,0,2),(0,1,0),(0,1,1),
            (1,0,3),(1,0,4),(1,1,5),(1,1,3)
        ]

        self._verify_result_from_expected_obs(result, expected_observations)

    def test_two_sims_batch_size_three_threes(self):
        sim1            = LambdaSimulation(50, lambda i: i, lambda s: [0,1,2], lambda s, a: a)
        sim2            = LambdaSimulation(50, lambda i: i, lambda s: [3,4,5], lambda s, a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s, A: int(s%3))

        benchmark       = UniversalBenchmark[int,int]([sim1,sim2], batch_size= [3,3,3])

        result = benchmark.evaluate([learner_factory])[0]

        expected_observations = [
            (0,0,0),(0,0,1),(0,0,2),(0,1,0),(0,1,1),(0,1,2),(0,2,0),(0,2,1),(0,2,2),
            (1,0,3),(1,0,4),(1,0,5),(1,1,3),(1,1,4),(1,1,5),(1,2,3),(1,2,4),(1,2,5)
        ]

        self._verify_result_from_expected_obs(result, expected_observations)

    def test_two_sims_batch_size_four_and_two(self):
        sim1            = LambdaSimulation(50, lambda i: i, lambda s: [0,1,2], lambda s, a: a)
        sim2            = LambdaSimulation(50, lambda i: i, lambda s: [3,4,5], lambda s, a: a)
        learner_factory = lambda: LambdaLearner[int,int](lambda s, A: int(s%3))
        benchmark       = UniversalBenchmark([sim1,sim2], batch_size = [4,2])

        result = benchmark.evaluate([learner_factory])[0]

        expected_observations = [
            (0,0,0),(0,0,1),(0,0,2),(0,0,0),(0,1,1),(0,1,2),
            (1,0,3),(1,0,4),(1,0,5),(1,0,3),(1,1,4),(1,1,5)
        ]

        self._verify_result_from_expected_obs(result, expected_observations)

    def test_lazy_sim_two_batches(self):
        sim1            = LazySimulation[int,int](lambda:LambdaSimulation(50, lambda i: i, lambda s: [0,1,2], lambda s, a: a))
        benchmark       = UniversalBenchmark([sim1], batch_size=[4,2])
        learner_factory = lambda: LambdaLearner[int,int](lambda s, A: int(s%3))
        result = benchmark.evaluate([learner_factory])[0]

        expected_observations = [
            (0,0,0),(0,0,1),(0,0,2),(0,0,0),(0,1,1),(0,1,2)
        ]

        self._verify_result_from_expected_obs(result, expected_observations)

if __name__ == '__main__':
    unittest.main()