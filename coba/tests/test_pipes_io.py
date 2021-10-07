import unittest

from pathlib import Path

from coba.pipes import DiskSource, DiskSink
from coba.encodings import NumericEncoder, OneHotEncoder
from coba.config import NoneLogger, CobaConfig

CobaConfig.Logger = NoneLogger()

class DiskIO_Tests(unittest.TestCase):
    def test_simple_sans_gz(self):

        filepath = "coba/tests/.temp/test.log"

        try:
            sink   = DiskSink(filepath)
            source = DiskSource(filepath)

            sink.write(["a","b","c"])
            out = list(source.read())

            self.assertEqual(["a","b","c"], out)

        finally:
            if Path(filepath).exists(): Path(filepath).unlink()

    def test_simple_with_gz(self):

        filepath = "coba/tests/.temp/test.log.gz"

        try:
            sink   = DiskSink(filepath)
            source = DiskSource(filepath)

            sink.write(["a","b","c"])
            out = list(source.read())

            self.assertEqual(["a","b","c"], out)

        finally:
            if Path(filepath).exists(): Path(filepath).unlink()

if __name__ == '__main__':
    unittest.main()