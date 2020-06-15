import unittest

from math import sqrt
from itertools import cycle, islice, repeat
from typing import cast

from bbench.games import LambdaGame, Round
from bbench.solvers import LambdaSolver
from bbench.benchmarks import Stats, Result, UniversalBenchmark

class Test_Stats_Instance(unittest.TestCase):
    def test_multi_mean_is_correct_1(self):
        stats = Stats([1,1,3,3])
        self.assertEqual(2,stats.mean)

    def test_multi_mean_is_correct_2(self):
        stats = Stats([1,1,1,1])
        self.assertEqual(1,stats.mean)

    def test_single_mean_is_correct(self):
        stats = Stats([3])
        self.assertEqual(3,stats.mean)

    def test_empty_mean_is_correct(self):
        stats = Stats([])
        self.assertIsNone(stats.mean)

class Test_Result_Instance(unittest.TestCase):
    def test_iteration_means(self):
        result = Result([(1,1,3), (1,1,4), (1,2,5), (1,2,5), (1,3,6)])

        self.assertEqual(3.5, result.iteration_stats[0].mean)
        self.assertEqual(5  , result.iteration_stats[1].mean)
        self.assertEqual(6  , result.iteration_stats[2].mean)

    def test_progressive_means(self):
        result = Result([(1,1,3), (1,1,4), (1,2,5), (1,2,5), (1,3,6)])

        self.assertEqual(3.5 , result.progressive_stats[0].mean)
        self.assertEqual(4.25, result.progressive_stats[1].mean)
        self.assertEqual(4.6 , result.progressive_stats[2].mean)

class Test_UniversalBenchmark(unittest.TestCase):

    def test_one_game_one_round_five_iterations(self):
        game           = LambdaGame(lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        solver_factory = lambda: LambdaSolver(lambda s,A: A[s%3])
        benchmark      = UniversalBenchmark([game], lambda i: 1, 5)

        result = benchmark.evaluate(solver_factory)

        expected_observations = [
            (0,0,0),(0,1,1),(0,2,2),(0,3,0),(0,4,1)
        ]

        self.assertEqual(expected_observations, result.observations)

    def test_one_game_five_rounds_one_iteration(self):
        game           = LambdaGame(lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        solver_factory = lambda: LambdaSolver(lambda s,A: A[s%3])
        benchmark      = UniversalBenchmark([game], lambda i: 5, 1)

        result = benchmark.evaluate(solver_factory)

        expected_observations = [
            (0,0,0),(0,0,1),(0,0,2),(0,0,0),(0,0,1)
        ]

        self.assertEqual(expected_observations, result.observations)

    def test_one_game_three_rounds_three_iterations(self):
        game           = LambdaGame(lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        solver_factory = lambda: LambdaSolver(lambda s,A: A[s%3])
        benchmark      = UniversalBenchmark([game], lambda i: 3, 3)

        result = benchmark.evaluate(solver_factory)

        expected_observations = [
            (0,0,0),(0,0,1),(0,0,2),(0,1,0),(0,1,1),(0,1,2),(0,2,0),(0,2,1),(0,2,2)
        ]

        self.assertEqual(expected_observations, result.observations)

    def test_two_games_one_round_five_iterations(self):
        game1          = LambdaGame(lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        game2          = LambdaGame(lambda i: i, lambda s: [3,4,5], lambda s,a: a)
        solver_factory = lambda: LambdaSolver(lambda s,A: s%3)
        benchmark      = UniversalBenchmark([game1,game2], lambda i: 1, 5)

        result = benchmark.evaluate(solver_factory)

        expected_observations = [
            (0,0,0),(0,1,1),(0,2,2),(0,3,0),(0,4,1),
            (1,0,3),(1,1,4),(1,2,5),(1,3,3),(1,4,4)
        ]

        self.assertEqual(expected_observations, result.observations)

    def test_two_games_five_rounds_one_iteration(self):
        game1          = LambdaGame(lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        game2          = LambdaGame(lambda i: i, lambda s: [3,4,5], lambda s,a: a)
        solver_factory = lambda: LambdaSolver(lambda s,A: s%3)
        benchmark      = UniversalBenchmark([game1,game2], lambda i: 5, 1)

        result = benchmark.evaluate(solver_factory)

        expected_observations = [
            (0,0,0),(0,0,1),(0,0,2),(0,0,0),(0,0,1),
            (1,0,3),(1,0,4),(1,0,5),(1,0,3),(1,0,4)
        ]

        self.assertEqual(expected_observations, result.observations)

    def test_two_games_three_rounds_three_iterations(self):
        game1          = LambdaGame(lambda i: i, lambda s: [0,1,2], lambda s,a: a)
        game2          = LambdaGame(lambda i: i, lambda s: [3,4,5], lambda s,a: a)
        solver_factory = lambda: LambdaSolver(lambda s,A: s%3)
        benchmark      = UniversalBenchmark([game1,game2], lambda i: 3, 3)

        result = benchmark.evaluate(solver_factory)

        expected_observations = [
            (0,0,0),(0,0,1),(0,0,2),(0,1,0),(0,1,1),(0,1,2),(0,2,0),(0,2,1),(0,2,2),
            (1,0,3),(1,0,4),(1,0,5),(1,1,3),(1,1,4),(1,1,5),(1,2,3),(1,2,4),(1,2,5)
        ]

        self.assertEqual(expected_observations, result.observations)

if __name__ == '__main__':
    unittest.main()