import unittest
import pickle

from coba.primitives import Categorical

class Categorical_Tests(unittest.TestCase):
    def test_value(self):
        self.assertEqual("A", Categorical("A",["A","B"]))
    
    def test_levels(self):
        self.assertEqual(["A","B"], Categorical("A",["A","B"]).levels)

    def test_eq(self):
        self.assertEqual(Categorical("A",["A","B"]), Categorical("A",["A","B"]))

    def test_ne(self):
        self.assertNotEqual(1, Categorical("A",["A","B"]))

    def test_str(self):
        self.assertEqual("A", str(Categorical("A",["A","B"])))

    def test_repr(self):
        self.assertEqual("Categorical('A',['A', 'B'])", repr(Categorical("A",["A","B"])))

    def test_pickle(self):
        out = pickle.loads(pickle.dumps(Categorical("A",["A","B"])))

        self.assertIsInstance(out,Categorical)
        self.assertEqual(out.levels, ['A',"B"])

if __name__ == '__main__':
    unittest.main()
