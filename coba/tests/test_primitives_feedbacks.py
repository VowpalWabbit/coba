import unittest

from coba.primitives import SequenceFeedback

class SequenceFeedback_Tests(unittest.TestCase):
    def test_sequence(self):
        fb = SequenceFeedback([4,5,6])

        self.assertEqual(3,len(fb))
        self.assertEqual([4,5,6],fb)
        self.assertEqual(4,fb[0])
        self.assertEqual(4,fb.eval(0))
        self.assertEqual(5,fb.eval(1))
        self.assertEqual(6,fb.eval(2))

if __name__ == '__main__':
    unittest.main()