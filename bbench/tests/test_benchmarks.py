import unittest

from math import sqrt
from itertools import cycle, islice
from typing import cast

from bbench.games import Game, Round
from bbench.solvers import LambdaSolver
from bbench.benchmarks import Result, ProgressiveBenchmark

class Test_Result_Instance(unittest.TestCase):

    def test_result_points_correct_for_samples_of_1(self) -> None:
        result = Result([10,20,30,40])

        self.assertEqual(result.points[0], (1,10))
        self.assertEqual(result.points[1], (2,20))
        self.assertEqual(result.points[2], (3,30))
        self.assertEqual(result.points[3], (4,40))

    def test_result_points_correct_for_samples_of_2(self) -> None:
        result = Result([[10,20],[20,30],[30,40],[40,50]])

        self.assertEqual(result.points[0], (1,15))
        self.assertEqual(result.points[1], (2,25))
        self.assertEqual(result.points[2], (3,35))
        self.assertEqual(result.points[3], (4,45))

    def test_result_errors_correct_for_samples_of_1(self) -> None:
        result = Result([10,20,30,40])

        self.assertIsNone(result.errors[0])
        self.assertIsNone(result.errors[1])
        self.assertIsNone(result.errors[2])
        self.assertIsNone(result.errors[3])

    def test_result_errors_correct_for_samples_of_2(self) -> None:
        result = Result([[10,20],[20,40],[30,60],[40,80]])

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

        rewards = [1,3]
        game    = Game.from_iterable(cycle([0,1]), lambda s: [0,1], lambda s,a:rewards[cast(int,a)])
        solver  = lambda: LambdaSolver(lambda s,a: cast(int,s))

        result = ProgressiveBenchmark([game]).evaluate(solver)

        for n, (point, error) in enumerate(zip(result.points,result.errors)):
            expected_rewards = list(islice(cycle(rewards),(n+1)))
            expected_progressive = sum(expected_rewards)/ len(expected_rewards)

            self.assertEqual(point[0], n+1)
            self.assertAlmostEqual(point[1], expected_progressive)
            self.assertIsNone(error)

    def test_multi_game(self) -> None:

        rewards = [[1,3], [5,6]]
        actions = lambda s: [0,1]
        games   = [Game.from_iterable(cycle([0,1]), actions, lambda s,a,r=r:r[a]) for r in rewards] #type: ignore
        solver  = lambda: LambdaSolver(lambda s,a: cast(int,s))

        result = ProgressiveBenchmark(games).evaluate(solver)

        for n, (point, error) in enumerate(zip(result.points,result.errors)):
            expected_rewards = [list(islice(cycle(r),(n+1))) for r in rewards ]
            expected_values  = [sum(r)/len(r) for r in expected_rewards]
            
            expected_mean    = sum(expected_values)/len(expected_values)
            expected_error   = sqrt(sum((v-expected_mean)**2 for v in expected_values))/len(expected_values)

            self.assertEqual(point[0], n+1)
            self.assertAlmostEqual(point[1], expected_mean)
            self.assertAlmostEqual(error, expected_error)

if __name__ == '__main__':
    unittest.main()
