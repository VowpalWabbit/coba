import unittest
import pickle

from collections import Counter

from coba.utilities import PackageChecker
import coba.random

class CobaRandom_Tests(unittest.TestCase):
    def test_seed(self):
        rng = coba.random.CobaRandom(2)
        self.assertEqual(2, rng.seed)
        rng.random()
        self.assertEqual(2, rng.seed)

    def test_random(self):
        for _ in range(10000):
            n = coba.random.random()
            self.assertLessEqual(n,1)
            self.assertGreaterEqual(n,0)

    def test_randoms_zero(self):
        self.assertEqual([],coba.random.randoms(0))

    def test_randoms_n_neg_1(self):
        with self.assertRaises(ValueError) as e:
            coba.random.CobaRandom().randoms(-1)
        self.assertIn("n must be an integer greater than or equal to 0", str(e.exception))

    def test_randoms_n_2(self):
        cr1 = coba.random.CobaRandom(seed=1)
        cr2 = coba.random.CobaRandom(seed=1)
        cr2._m_is_power_of_2 = False
        self.assertEqual( cr1.randoms(3), cr2.randoms(3) )

    def test_randoms_n(self):
        numbers = coba.random.randoms(500000)
        self.assertEqual(len(numbers), 500000)
        for n in numbers:
            self.assertLessEqual(n, 1)
            self.assertGreaterEqual(n, 0)

        numbers = coba.random.randoms(500000,0,10)
        self.assertEqual(len(numbers), 500000)
        for n in numbers:
            self.assertLessEqual(n, 10)
            self.assertGreaterEqual(n, 0)

    def test_randoms_unchanged(self):
        coba.random.seed(10)
        actual_random_numbers = [ round(n,2) for n in coba.random.randoms(5) ]
        self.assertEqual([0.09, 0.18, 0.16, 0.98, 0.14], actual_random_numbers)

    def test_randoms_repeatable(self):
        coba.random.seed(10)
        actual_random_numbers_1 = coba.random.randoms(5)
        coba.random.seed(10)
        actual_random_numbers_2 = coba.random.randoms(5)
        self.assertSequenceEqual(actual_random_numbers_1, actual_random_numbers_2)

    @unittest.skipUnless(PackageChecker.scipy(strict=False), "scipy is not installed so we must skip statistical tests")
    def test_randoms_uniform(self):
        import numpy as np
        from scipy.stats import chisquare
        frequencies = Counter(np.digitize(coba.random.randoms(50000), bins=[i/50 for i in range(50)]))
        self.assertLess(0.00001, chisquare(list(frequencies.values())).pvalue)

    def test_shuffle_not_in_place(self):
        input  = list(range(500000))
        output = coba.random.shuffle(input)
        self.assertIsNot(input,output)
        self.assertEqual(set(input), set(output))
        self.assertNotEqual(input, output)

    def test_shuffle_in_place(self):
        input  = list(range(500000))
        output = coba.random.shuffle(input,inplace=True)
        self.assertIs(input,output)
        self.assertEqual(set(range(500000)), set(output))
        self.assertNotEqual(list(range(500000)), output)

    def test_shuffle_iterable(self):
        output = coba.random.shuffle(range(500000),inplace=False)
        self.assertEqual(set(range(500000)), set(output))
        self.assertNotEqual(list(range(500000)), output)

    def test_shuffle_empty(self):
        self.assertEqual([], coba.random.shuffle([]))

    def test_shuffle_repeatable(self):
        coba.random.seed(10)
        shuffle_1 = coba.random.shuffle([1,2,3,4,5])
        coba.random.seed(10)
        shuffle_2 = coba.random.shuffle([1,2,3,4,5])
        self.assertEqual(shuffle_1, shuffle_2)

    def test_shuffle_unchaged(self):
        coba.random.seed(10)
        shuffle = coba.random.shuffle(list(range(20)))
        self.assertEqual([1, 4, 0, 19, 6, 15, 3, 16, 11, 10, 7, 17, 13, 8, 9, 14, 18, 12, 5, 2],shuffle)

    @unittest.skipUnless(PackageChecker.scipy(strict=False), "scipy is not installed so we must skip statistical tests")
    def test_shuffle_is_unbiased(self):
        from scipy.stats import chisquare
        base = list(range(5))
        frequencies = Counter([tuple(coba.random.shuffle(base)) for _ in range(100000)])
        self.assertLess(0.00001, chisquare(list(frequencies.values())).pvalue)

    def test_randint_is_bound_correctly_1(self):
        observed_ints = set([coba.random.randint(0,2) for _ in range(100)])
        self.assertEqual(3, len(set(observed_ints)))
        self.assertIn(0, observed_ints)
        self.assertIn(1, observed_ints)
        self.assertIn(2, observed_ints)

    def test_randint_is_bound_correctly_2(self):
        observed_ints = set([coba.random.randint(-3,-1) for _ in range(100)])
        self.assertEqual(3, len(set(observed_ints)))
        self.assertIn(-3, observed_ints)
        self.assertIn(-2, observed_ints)
        self.assertIn(-1, observed_ints)

    def test_randint_unchaged(self):
        coba.random.seed(10)
        self.assertEqual(1,coba.random.randint(1,10))
        self.assertEqual(2,coba.random.randint(1,10))
        self.assertEqual(2,coba.random.randint(1,10))
        self.assertEqual(10,coba.random.randint(1,10))
        self.assertEqual(2,coba.random.randint(1,10))
        self.assertEqual(7,coba.random.randint(1,10))
        coba.random.seed(10)
        self.assertEqual(1,coba.random.randint(1,10))
        self.assertEqual(2,coba.random.randint(1,10))
        self.assertEqual(2,coba.random.randint(1,10))
        self.assertEqual(10,coba.random.randint(1,10))
        self.assertEqual(2,coba.random.randint(1,10))
        self.assertEqual(7,coba.random.randint(1,10))

    @unittest.skipUnless(PackageChecker.scipy(strict=False), "scipy is not installed so we must skip statistical tests")
    def test_randint_uniform(self):
        from scipy.stats import chisquare
        frequencies = Counter([coba.random.randint(1,6) for _ in range(50000)])
        self.assertLess(0.00001, chisquare(list(frequencies.values())).pvalue)

    def test_randints(self):
        observed_ints = sum((coba.random.randints(10,0,2) for _ in range(100)),[])
        self.assertEqual(3, len(set(observed_ints)))
        self.assertIn(0, observed_ints)
        self.assertIn(1, observed_ints)
        self.assertIn(2, observed_ints)

    def test_randints_unchaged(self):
        coba.random.seed(10)
        self.assertEqual([1, 2, 2, 10, 2, 7, 10, 8, 3, 2],coba.random.randints(10,1,10))
        self.assertEqual([6, 7, 2, 6, 4, 4, 7, 0, 5, 1],coba.random.randints(10,0,10))

    def test_choice1(self):
        choice = coba.random.choice([(0,1),(1,0)])
        self.assertIsInstance(choice, tuple)

    def test_choice2(self):
        choice = coba.random.choice([(0,1),(1,0)],[0.5,0.5])
        self.assertIsInstance(choice, tuple)

    def test_choicew(self):
        counts = Counter([coba.random.choicew([0,1],[0.25,0.75]) for _ in range(100000)])
        self.assertEqual(len(counts),2)
        self.assertAlmostEqual(counts[(0,.25)]/counts.total(), .25, places=2)
        self.assertAlmostEqual(counts[(1,.75)]/counts.total(), .75, places=2)
        counts = Counter([coba.random.choicew([0,1]) for _ in range(100000)])
        self.assertEqual(len(counts),2)
        self.assertAlmostEqual(counts[(0,.5)]/counts.total(), .5, places=2)
        self.assertAlmostEqual(counts[(1,.5)]/counts.total(), .5, places=2)

    def test_choice_exception(self):
        with self.assertRaises(ValueError) as e:
            coba.random.CobaRandom().choice([1,2,3],[0,0,0])
        self.assertIn("The sum of weights cannot be zero", str(e.exception))

    def test_choice_unchanged(self):
        coba.random.seed(10)
        choice = coba.random.choice(list(range(1000)), [1/1000]*1000)
        self.assertEqual(86, choice)

    def test_choice_repeatable(self):
        coba.random.seed(10)
        choice_1 = coba.random.choice(list(range(1000)), [1/1000]*1000)
        coba.random.seed(10)
        choice_2 = coba.random.choice(list(range(1000)), [1/1000]*1000)
        self.assertEqual(choice_1, choice_2)

    def test_gauss(self):
        expected = 0.626
        coba.random.seed(1)
        self.assertEqual(expected, round(coba.random.CobaRandom(seed=1).gauss(0,1),3))
        self.assertEqual(expected, round(coba.random.gauss(0,1),3))

    def test_gausses(self):
        expected = [0.626, -2.012]
        coba.random.seed(1)
        self.assertEqual(expected, [round(r,3) for r in coba.random.CobaRandom(seed=1).gausses(2,0,1)])
        self.assertEqual(expected, [round(r,3) for r in coba.random.gausses(2,0,1)])

    @unittest.skipUnless(PackageChecker.scipy(strict=False), "scipy is not installed so we must skip statistical tests")
    def test_gauss_normal(self):
        from scipy.stats import shapiro
        self.assertLess(0.00001, shapiro(coba.random.gausses(1000,0,1)).pvalue)

    def test_pickle(self):
        cr = pickle.loads(pickle.dumps(coba.random.CobaRandom(seed=5)))
        self.assertEqual(5, cr._seed)

if __name__ == '__main__':
    unittest.main()
