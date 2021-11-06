import unittest

from pathlib import Path

from coba.pipes import DiskIO, MemoryIO, QueueIO
from coba.config import NullLogger, CobaConfig

CobaConfig.logger = NullLogger()

class DiskIO_Tests(unittest.TestCase):
    def test_simple_sans_gz(self):

        filepath = "coba/tests/.temp/test.log"

        try:
            io = DiskIO(filepath)

            io.write(["a","b","c"])
            out = list(io.read())

            self.assertEqual(["a","b","c"], out)

        finally:
            if Path(filepath).exists(): Path(filepath).unlink()

    def test_simple_with_gz(self):

        filepath = "coba/tests/.temp/test.log.gz"

        try:
            io = DiskIO(filepath)

            io.write(["a","b","c"])
            out = list(io.read())

            self.assertEqual(["a","b","c"], out)

        finally:
            if Path(filepath).exists(): Path(filepath).unlink()

class MemoryIO_Tests(unittest.TestCase):
    def test_simple(self):

        io = MemoryIO()

        io.write(["a","b","c"])
        out = list(io.read())

        self.assertEqual(["a","b","c"], out)

    def test_string(self):

        io = MemoryIO()

        io.write("abc")
        out = list(io.read())

        self.assertEqual(["abc"], out)

class QueueIO_Tests(unittest.TestCase):
    def test_simple_no_blocking(self):

        io = QueueIO(blocking_get=False)

        io.write(["a","b","c"])
        out = list(io.read())

        self.assertEqual(["a","b","c"], out)

    def test_simple_blocking(self):

        io = QueueIO()

        io.write(["a","b","c", None])
        out = list(io.read())

        self.assertEqual(["a","b","c"], out)

    def test_string(self):

        io = QueueIO(blocking_get=False)
        io.write("abc")
        out = list(io.read())
        self.assertEqual(["abc"], out)

if __name__ == '__main__':
    unittest.main()