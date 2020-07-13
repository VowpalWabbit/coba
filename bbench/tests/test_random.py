import time
import unittest

import bbench.random

class Random_Tests(unittest.TestCase):

    def test_speed_of_randoms(self):
        start = time.time()
        numbers = bbench.random.randoms(500000)
        secs = time.time() - start

        self.assertEqual(len(numbers), 500000)

        self.assertLess(secs,1)

    def test_value_of_randoms(self):

        numbers = bbench.random.randoms(500000)

        self.assertEqual(len(numbers), 500000)

        for n in numbers:
            self.assertLessEqual(n, 1)
            self.assertGreaterEqual(n, 0)

    def test_speed_of_shuffle(self):

        start = time.time()
        bbench.random.shuffle(list(range(500000)))
        secs = time.time() - start

        print(secs)

        self.assertLess(secs,2)

    def test_value_of_shuffle(self):

        numbers = bbench.random.shuffle(list(range(500000)))

        self.assertEqual(len(numbers), 500000)
        self.assertNotEqual(numbers, list(range(500000)))

    def test_random_repetability(self):

        bbench.random.seed(10)

        actual_random_numbers = bbench.random.randoms(5)
        expected_random_numbers = [
            0.08635475773956139,
            0.18061295168531402,
            0.16033128757060625,
            0.9826165502729048,
            0.14168352553777724,
        ]

        for actual,expected in zip(actual_random_numbers,expected_random_numbers):
            self.assertAlmostEqual(actual,expected)

if __name__ == '__main__':
    unittest.main()
