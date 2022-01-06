import unittest

from coba.pipes import DiskIO, MemoryIO, NullIO
from coba.environments import ReaderEnvironment

class ReaderSimulation_Tests(unittest.TestCase):

    def test_params(self):
        self.assertEqual({'source': 'abc'}   , ReaderEnvironment(DiskIO("abc"), None, None).params)
        self.assertEqual({'source': 'memory'}, ReaderEnvironment(MemoryIO(), None, None).params)
        self.assertEqual({'source': 'NullIO'}, ReaderEnvironment(NullIO(), None, None).params)
