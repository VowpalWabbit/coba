import unittest

from math import sqrt
from itertools import cycle, islice
from typing import cast

from bbench.games import Game, Round
from bbench.solvers import LambdaSolver
from bbench.benchmarks import Result, ProgressiveBenchmark, TraditionalBenchmark

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
        actions = lambda s: [0,1]
        games   = [Game.from_iterable(cycle([0,1]), actions, lambda s,a,r=r:r[a]) for r in rewards] #type: ignore
        solver  = lambda: LambdaSolver(lambda s,a: cast(int,s))

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
                self.assertAlmostEqual(error, expected_error)

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
        self.assert_traditionalbenchmark_for_reward_sets([[1,3],[5,6]], 10, 10)

    def assert_traditionalbenchmark_for_reward_sets(self, rewards, n_rounds, n_iterations) -> None:
        actions = lambda s: [0,1]
        games   = [Game.from_iterable(cycle([0,1]), actions, lambda s,a,r=r:r[a]) for r in rewards] #type: ignore
        solver  = lambda: LambdaSolver(lambda s,a: cast(int,s))

        result = TraditionalBenchmark(games, n_rounds, n_iterations).evaluate(solver)

        reward_cycles = [cycle(r) for r in rewards]

        for n, (value, error) in enumerate(zip(result.values,result.errors)):
            expected_rwds = [e for r in reward_cycles for e in islice(r,n_rounds) ]
            expected_mean = sum(expected_rwds)/len(expected_rwds)
            expected_error = sqrt(sum((v-expected_mean)**2 for v in expected_rwds))/len(expected_rwds)

            self.assertAlmostEqual(value, expected_mean)

            if(len(expected_rwds) == 1):
                self.assertIsNone(error)
            else:
                self.assertAlmostEqual(error, expected_error)

        self.assertEqual(n+1, n_iterations)
if __name__ == '__main__':
    unittest.main()
