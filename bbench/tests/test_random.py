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

        self.assertLess(secs,1)

    def test_value_of_shuffle(self):

        numbers = bbench.random.shuffle(list(range(500000)))

        self.assertEqual(len(numbers), 500000)
        self.assertNotEqual(numbers, list(range(500000)))

if __name__ == '__main__':
    unittest.main()
