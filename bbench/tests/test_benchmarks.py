import unittest

from math import sqrt
from itertools import cycle, islice, repeat
from typing import cast

from bbench.games import LambdaGame, Round
from bbench.solvers import LambdaSolver
from bbench.benchmarks import Stats, Result, Result2, ProgressiveBenchmark, TraditionalBenchmark

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

class Test_Result2_Instance(unittest.TestCase):
    def test_iteration_means(self):
        result = Result2([(1,1,3), (1,1,4), (1,2,5), (1,2,5), (1,3,6)])

        self.assertEqual(3.5, result.iteration_stats[0].mean)
        self.assertEqual(5  , result.iteration_stats[1].mean)
        self.assertEqual(6  , result.iteration_stats[2].mean)

    def test_progressive_means(self):
        result = Result2([(1,1,3), (1,1,4), (1,2,5), (1,2,5), (1,3,6)])

        self.assertEqual(3.5 , result.progressive_stats[0].mean)
        self.assertEqual(4.25, result.progressive_stats[1].mean)
        self.assertEqual(4.6 , result.progressive_stats[2].mean)

class Test_Result_Instance(unittest.TestCase):

    def test_result_points_correct_for_samples_of_1(self) -> None:
        result = Result.from_samples([10,20,30,40])

        self.assertEqual(result.values[0], 10)
        self.assertEqual(result.values[1], 20)
        self.assertEqual(result.values[2], 30)
        self.assertEqual(result.values[3], 40)

    def test_result_points_correct_for_samples_of_2(self) -> None:
        result = Result.from_samples([[10,20],[20,30],[30,40],[40,50]])

        self.assertEqual(result.values[0], 15)
        self.assertEqual(result.values[1], 25)
        self.assertEqual(result.values[2], 35)
        self.assertEqual(result.values[3], 45)

    def test_result_errors_correct_for_samples_of_1(self) -> None:
        result = Result.from_samples([10,20,30,40])

        self.assertIsNone(result.errors[0])
        self.assertIsNone(result.errors[1])
        self.assertIsNone(result.errors[2])
        self.assertIsNone(result.errors[3])

    def test_result_errors_correct_for_samples_of_2(self) -> None:
        result = Result.from_samples([[10,20],[20,40],[30,60],[40,80]])

        #these are here to make mypy happy
        assert isinstance(result.errors[0], float)
        assert isinstance(result.errors[1], float)
        assert isinstance(result.errors[2], float)
        assert isinstance(result.errors[3], float)

        self.assertAlmostEqual(result.errors[0], 5/sqrt(2))
        self.assertAlmostEqual(result.errors[1], 10/sqrt(2))
        self.assertAlmostEqual(result.errors[2], 15/sqrt(2))
        self.assertAlmostEqual(result.errors[3], 20/sqrt(2))

class Test_ProgressiveBenchmark(unittest.TestCase):
    def test_single_game(self) -> None:
        self.assert_progessivebenchmark_for_reward_sets([[1,3]])

    def test_multi_game(self) -> None:
        self.assert_progessivebenchmark_for_reward_sets([[1,3],[5,6]])

    def assert_progessivebenchmark_for_reward_sets(self, rewards) -> None:     
        
        S = lambda i: [0,1][i%2]
        A = lambda s: [0,1]
        Rs = map(lambda r: (lambda s,a: r[a]), rewards)
        C = lambda s,a: s

        games   = [LambdaGame(S, A, R) for R in Rs]
        solver  = lambda: LambdaSolver(C)

        result = ProgressiveBenchmark(games).evaluate(solver)

        for n, (value, error) in enumerate(zip(result.values,result.errors)):
            expected_rewards = [list(islice(cycle(r),(n+1))) for r in rewards ]
            expected_values  = [sum(r)/len(r) for r in expected_rewards]
            
            expected_mean    = sum(expected_values)/len(expected_values)
            expected_error = sqrt(sum((v-expected_mean)**2 for v in expected_values))/len(expected_values)

            self.assertAlmostEqual(value, expected_mean)
            
            if(len(expected_values) == 1):
                self.assertIsNone(error)
            else:
                self.assertAlmostEqual(cast(float,error), expected_error)

class Test_TraditionalBenchmark(unittest.TestCase):
    def test_single_game_one_round_one_iteration(self) -> None:
        self.assert_traditionalbenchmark_for_reward_sets([[1,3]], 1, 1)
    
    def test_single_game_one_round_ten_iteration(self) -> None:
        self.assert_traditionalbenchmark_for_reward_sets([[1,3]], 1, 10)
    
    def test_single_game_ten_round_one_iteration(self) -> None:
        self.assert_traditionalbenchmark_for_reward_sets([[1,3]], 10, 1)
    
    def test_single_game_ten_round_ten_iteration(self) -> None:
        self.assert_traditionalbenchmark_for_reward_sets([[1,3]], 10, 10)
    
    def test_multi_game_one_round_one_iteration(self) -> None:
        self.assert_traditionalbenchmark_for_reward_sets([[1,3],[5,6]], 1, 1)
    
    def test_multi_game_one_round_ten_iteration(self) -> None:
        self.assert_traditionalbenchmark_for_reward_sets([[1,3],[5,6]], 1, 10)
    
    def test_multi_game_ten_round_one_iteration(self) -> None:
        self.assert_traditionalbenchmark_for_reward_sets([[1,3],[5,6]], 10, 1)
    
    def test_multi_game_ten_round_ten_iteration(self) -> None:
        self.assert_traditionalbenchmark_for_reward_sets([[1,3],[5,6]], 9, 10)

    def assert_traditionalbenchmark_for_reward_sets(self, rewards, n_rounds, n_iterations) -> None:
        
        S = lambda i: [0,1][i%2]
        A = lambda s: [0,1]
        Rs = map(lambda r: (lambda s,a: r[a]), rewards)
        C = lambda s,a: s

        games   = [LambdaGame(S, A, R) for R in Rs]
        solver  = lambda: LambdaSolver(C)

        result = TraditionalBenchmark(games, n_rounds, n_iterations).evaluate(solver)

        reward_cycles = [cycle(r) for r in rewards]

        for n, (value, error) in enumerate(zip(result.values,result.errors)):
            expected_rwds = [r for rc in reward_cycles for r in islice(rc,n_rounds) ]
            expected_mean = sum(expected_rwds)/len(expected_rwds)
            expected_error = sqrt(sum((v-expected_mean)**2 for v in expected_rwds))/len(expected_rwds)

            self.assertAlmostEqual(value, expected_mean)

            if(len(expected_rwds) == 1):
                self.assertIsNone(error)
            else:
                self.assertAlmostEqual(cast(float,error), expected_error)

        self.assertEqual(n+1, n_iterations)
if __name__ == '__main__':
    unittest.main()
