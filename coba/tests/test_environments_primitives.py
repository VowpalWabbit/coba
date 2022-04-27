import unittest

from coba.environments import SafeEnvironment
from coba.pipes import Pipes, Identity, Shuffle

class DummyEnvironment:

    def read(self):
        return []

class SafeEnvironment_Tests(unittest.TestCase):

    def test_params(self):
        self.assertEqual({'type': 'DummyEnvironment'}, SafeEnvironment(DummyEnvironment()).params)

    def test_read(self):
        self.assertEqual([], SafeEnvironment(DummyEnvironment()).read())

    def test_str(self):
        self.assertEqual('DummyEnvironment(shuffle=1)', str(SafeEnvironment(Pipes.join(DummyEnvironment(), Shuffle(1)))))
        self.assertEqual('DummyEnvironment', str(SafeEnvironment(DummyEnvironment())))

    def test_with_nesting(self):
        self.assertIsInstance(SafeEnvironment(SafeEnvironment(DummyEnvironment()))._environment, DummyEnvironment)

    def test_with_pipes(self):
        self.assertEqual({'type': 'DummyEnvironment', "shuffle":1}, SafeEnvironment(Pipes.join(DummyEnvironment(), Shuffle(1))) .params)

if __name__ == '__main__':
    unittest.main()