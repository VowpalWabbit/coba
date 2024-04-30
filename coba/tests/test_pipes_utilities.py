
import unittest
from coba.exceptions import CobaException
from coba.pipes.utilities import resolve_params

class Pipe1:
    def params(self):
        return {'p1':1}

class Pipe2:
    @property
    def params(self):
        return {'p2':2}

class Pipe3:
    @property
    def params(self):
        return {'p3':3}

class Pipe4:
    @property
    def params(self):
        return 1

class Pipe5:
    pass

class resolve_params_Tests(unittest.TestCase):
    def test_one_good_pipe(self):
        self.assertEqual(resolve_params([Pipe2()]), {'p2':2})

    def test_two_good_pipe(self):
        self.assertEqual(resolve_params([Pipe2(),Pipe3()]), {'p2':2,'p3':3})

    def test_three_good_pipe(self):
        self.assertEqual(resolve_params([Pipe5(),Pipe2(),Pipe3()]), {'p2':2,'p3':3})

    def test_callable_params(self):
        with self.assertRaises(CobaException) as r:
            resolve_params([Pipe2(),Pipe1()])
        self.assertEqual(str(r.exception), "The params on Pipe1 is not decorated with @property")

    def test_not_dict_params(self):
        with self.assertRaises(CobaException) as r:
            resolve_params([Pipe2(),Pipe4()])
        self.assertEqual(str(r.exception), "The params on Pipe4 is not a dict type")

    def test_no_params(self):
        self.assertEqual(resolve_params([Pipe5()]),{})

if __name__ == '__main__':
    unittest.main()
