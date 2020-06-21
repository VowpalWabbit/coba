import unittest

from math import sqrt
from itertools import cycle, islice, repeat
from typing import cast

from bbench.games import LambdaGame, Round
from bbench.solvers import LambdaSolver
from bbench.benchmarks import Stats, Result, UniversalBenchmark

class Test_Stats_Instance(unittest.TestCase):
    def test_from_values_multi_mean_is_correct_1(self):
        stats = Stats.from_values([1,1,3,3])
        self.assertEqual(2,stats.mean)

    def test_from_values_multi_mean_is_correct_2(self):
        stats = Stats.from_values([1,1,1,1])
        self.assertEqual(1,stats.mean)

    def test_from_values_single_mean_is_correct(self):
        stats = Stats.from_values([3])
        self.assertEqual(3,stats.mean)

    def test_from_values_empty_mean_is_correct(self):
        stats = Stats.from_values([])
        self.assertIsNone(stats.mean)

class Test_Result_Instance(unittest.TestCase):
    def test_batch_means(self):
        result = Result([(1,1,3), (1,1,4), (1,2,5), (1,2,5), (1,3,6)])

        self.assertAlmostEqual(3.5, result.batch_stats[0].mean)
        self.assertAlmostEqual(5  , result.batch_stats[1].mean)
        self.assertAlmostEqual(6  , result.batch_stats[2].mean)

    def test_sweep_means(self):
        result = Result([(1,1,3), (1,1,4), (1,2,5), (1,2,5), (1,3,6)])

        self.assertAlmostEqual(3.5 , result.sweep_stats[0].mean)
        self.assertAlmostEqual(4.25, result.sweep_stats[1].mean)
        self.assertAlmostEqual(4.6 , result.sweep_stats[2].mean)

class Test_UniversalBenchmark(unittest.TestCase):

    def test_one_game_five_rounds_batch_size_one(self):
        game           = LambdaGame(lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        solver_factory = lambda: LambdaSolver(lambda s,A: A[s%3])
        benchmark      = UniversalBenchmark([game], 5, 1)

        result = benchmark.evaluate(solver_factory)

        expected_observations = [
            (0,0,0),(0,1,1),(0,2,2),(0,3,0),(0,4,1)
        ]

        self.assertEqual(result.observations, expected_observations)

    def test_one_game_five_rounds_batch_size_five(self):
        game           = LambdaGame(lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        solver_factory = lambda: LambdaSolver(lambda s,A: A[s%3])
        benchmark      = UniversalBenchmark([game], 5, 5)

        result = benchmark.evaluate(solver_factory)

        expected_observations = [
            (0,0,0),(0,0,1),(0,0,2),(0,0,0),(0,0,1)
        ]

        self.assertEqual(result.observations, expected_observations)

    def test_one_game_nine_rounds_batch_size_three(self):
        game           = LambdaGame(lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        solver_factory = lambda: LambdaSolver(lambda s,A: A[s%3])
        benchmark      = UniversalBenchmark([game], 9, 3)

        result = benchmark.evaluate(solver_factory)

        expected_observations = [
            (0,0,0),(0,0,1),(0,0,2),(0,1,0),(0,1,1),(0,1,2),(0,2,0),(0,2,1),(0,2,2)
        ]

        self.assertEqual(result.observations, expected_observations)

    def test_one_game_six_rounds_batch_size_four(self):
        game1          = LambdaGame(lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        solver_factory = lambda: LambdaSolver(lambda s,A: s%3)
        benchmark      = UniversalBenchmark([game1], 6, 4)

        result = benchmark.evaluate(solver_factory)

        expected_observations = [
            (0,0,0),(0,0,1),(0,0,2),(0,0,0),(0,1,1),(0,1,2)
        ]

        self.assertEqual(result.observations, expected_observations)

    def test_one_game_six_rounds_batch_size_power_of_two(self):
        game1          = LambdaGame(lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        solver_factory = lambda: LambdaSolver(lambda s,A: s%3)
        benchmark      = UniversalBenchmark([game1], 8, lambda i: 2**i)

        result = benchmark.evaluate(solver_factory)

        expected_observations = [
            (0,0,0),(0,1,1),(0,1,2),(0,2,0),(0,2,1),(0,2,2),(0,2,0),(0,3,1)
        ]

        self.assertEqual(result.observations, expected_observations)


    def test_two_games_five_rounds_batch_size_one(self):
        game1          = LambdaGame(lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        game2          = LambdaGame(lambda i: i, lambda s: [3,4,5], lambda s,a: a)
        solver_factory = lambda: LambdaSolver(lambda s,A: s%3)
        benchmark      = UniversalBenchmark([game1,game2], 5, 1)

        result = benchmark.evaluate(solver_factory)

        expected_observations = [
            (0,0,0),(0,1,1),(0,2,2),(0,3,0),(0,4,1),
            (1,0,3),(1,1,4),(1,2,5),(1,3,3),(1,4,4)
        ]

        self.assertEqual(result.observations, expected_observations)

    def test_two_games_five_rounds_batch_size_five(self):
        game1          = LambdaGame(lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        game2          = LambdaGame(lambda i: i, lambda s: [3,4,5], lambda s,a: a)
        solver_factory = lambda: LambdaSolver(lambda s,A: s%3)
        benchmark      = UniversalBenchmark([game1,game2], 5, 5)

        result = benchmark.evaluate(solver_factory)

        expected_observations = [
            (0,0,0),(0,0,1),(0,0,2),(0,0,0),(0,0,1),
            (1,0,3),(1,0,4),(1,0,5),(1,0,3),(1,0,4)
        ]

        self.assertEqual(result.observations, expected_observations)

    def test_two_games_nine_rounds_batch_size_three(self):
        game1          = LambdaGame(lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        game2          = LambdaGame(lambda i: i, lambda s: [3,4,5], lambda s,a: a)
        solver_factory = lambda: LambdaSolver(lambda s,A: s%3)
        benchmark      = UniversalBenchmark([game1,game2], 9, 3)

        result = benchmark.evaluate(solver_factory)

        expected_observations = [
            (0,0,0),(0,0,1),(0,0,2),(0,1,0),(0,1,1),(0,1,2),(0,2,0),(0,2,1),(0,2,2),
            (1,0,3),(1,0,4),(1,0,5),(1,1,3),(1,1,4),(1,1,5),(1,2,3),(1,2,4),(1,2,5)
        ]

        self.assertEqual(result.observations, expected_observations)

    def test_two_games_six_rounds_batch_size_four(self):
        game1          = LambdaGame(lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        game2          = LambdaGame(lambda i: i, lambda s: [3,4,5], lambda s,a: a)
        solver_factory = lambda: LambdaSolver(lambda s,A: s%3)
        benchmark      = UniversalBenchmark([game1,game2], 6, 4)

        result = benchmark.evaluate(solver_factory)

        expected_observations = [
            (0,0,0),(0,0,1),(0,0,2),(0,0,0),(0,1,1),(0,1,2),
            (1,0,3),(1,0,4),(1,0,5),(1,0,3),(1,1,4),(1,1,5)
        ]

        self.assertEqual(result.observations, expected_observations)

if __name__ == '__main__':
    unittest.main()