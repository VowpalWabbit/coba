import unittest
import math

from bbench.benchmarks import Result

class Test_Result_Instance(unittest.TestCase):

    def test_result_points_correct_for_samples_of_1(self):
        result = Result([10,20,30,40])

        self.assertEqual(result.points[0], (1,10))
        self.assertEqual(result.points[1], (2,20))
        self.assertEqual(result.points[2], (3,30))
        self.assertEqual(result.points[3], (4,40))

    def test_result_points_correct_for_samples_of_2(self):
        result = Result([[10,20],[20,30],[30,40],[40,50]])

        self.assertEqual(result.points[0], (1,15))
        self.assertEqual(result.points[1], (2,25))
        self.assertEqual(result.points[2], (3,35))
        self.assertEqual(result.points[3], (4,45))

    def test_result_errors_correct_for_samples_of_1(self):
        result = Result([10,20,30,40])

        self.assertIsNone(result.errors[0])
        self.assertIsNone(result.errors[1])
        self.assertIsNone(result.errors[2])
        self.assertIsNone(result.errors[3])

    def test_result_errors_correct_for_samples_of_2(self):
        result = Result([[10,20],[20,40],[30,60],[40,80]])

        self.assertAlmostEqual(result.errors[0], 5/math.sqrt(2))
        self.assertAlmostEqual(result.errors[1], 10/math.sqrt(2))
        self.assertAlmostEqual(result.errors[2], 15/math.sqrt(2))
        self.assertAlmostEqual(result.errors[3], 20/math.sqrt(2))

if __name__ == '__main__':
    unittest.main()
