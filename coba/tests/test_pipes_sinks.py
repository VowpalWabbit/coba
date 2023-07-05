import unittest
import unittest.mock
import pickle
import gzip

from pathlib import Path

from coba.pipes import DiskSink, ListSink, QueueSink, NullSink, ConsoleSink, LambdaSink
from coba.contexts import NullLogger, CobaContext

CobaContext.logger = NullLogger()

class BrokenQueue:

    def __init__(self, exception):
        self._exception = exception

    def get(self):
        raise self._exception

    def put(self,item):
        raise self._exception

class NullSink_Tests(unittest.TestCase):

    def test_write(self):
        NullSink().write([1,2,3])

class ConsoleSink_Tests(unittest.TestCase):

    def test_write(self):
        with unittest.mock.patch("builtins.print") as mock:
            ConsoleSink().write("abc")
            mock.assert_called_with("abc")

class DiskSink_Tests(unittest.TestCase):

    def setUp(self) -> None:
        if Path("coba/tests/.temp/test.log").exists(): Path("coba/tests/.temp/test.log").unlink()
        if Path("coba/tests/.temp/test.gz").exists(): Path("coba/tests/.temp/test.gz").unlink()

    def tearDown(self) -> None:
        if Path("coba/tests/.temp/test.log").exists(): Path("coba/tests/.temp/test.log").unlink()
        if Path("coba/tests/.temp/test.gz").exists(): Path("coba/tests/.temp/test.gz").unlink()

    def test_simple_sans_gz(self):

        sink = DiskSink("coba/tests/.temp/test.log")

        sink.write("a")
        sink.write("b")
        sink.write("c")

        self.assertEqual(["a","b","c"], Path("coba/tests/.temp/test.log").read_text().splitlines())

    def test_simple_with_gz(self):

        sink = DiskSink("coba/tests/.temp/test.gz")

        sink.write("a")
        sink.write("b")
        sink.write("c")

        lines = gzip.decompress(Path("coba/tests/.temp/test.gz").read_bytes()).decode('utf-8').splitlines()

        self.assertEqual(["a","b","c"], lines)

    def test_is_picklable(self):
        pickle.dumps(DiskSink("coba/tests/.temp/test.gz"))

class ListSink_Tests(unittest.TestCase):

    def test_simple(self):
        sink = ListSink()

        sink.write("a")
        sink.write("b")
        sink.write("c")

        self.assertEqual(["a","b","c"], sink.items)

class QueueSink_Tests(unittest.TestCase):

    def test_write(self):

        sink = QueueSink()

        sink.write("a")
        sink.write("b")
        sink.write("c")

        self.assertEqual("a", sink._queue.get())
        self.assertEqual("b", sink._queue.get())
        self.assertEqual("c", sink._queue.get())

    def test_write_exception(self):

        with self.assertRaises(Exception):
            QueueSink(BrokenQueue(Exception())).write(1)

        QueueSink(BrokenQueue(EOFError())).write(1)
        QueueSink(BrokenQueue(BrokenPipeError())).write(1)

class LambdaSink_Tests(unittest.TestCase):

    def test_write(self):
        items = []
        sink = LambdaSink(items.append)
        sink.write("a")
        sink.write("b")
        self.assertEqual(items,["a","b"])

if __name__ == '__main__':
    unittest.main()
