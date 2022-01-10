import unittest
import unittest.mock
import requests.exceptions

from pathlib import Path

from coba.pipes import DiskIO, ListIO, QueueIO, NullIO, ConsoleIO, HttpIO, IdentityIO
from coba.contexts import NullLogger, CobaContext

CobaContext.logger = NullLogger()

class BrokenQueue:

    def __init__(self, exception):
        self._exception = exception

    def get(self):
        raise self._exception

    def put(self,item):
        raise self._exception

class NullIO_Tests(unittest.TestCase):
    
    def test_read(self):
        self.assertEqual(0, len(NullIO().read()))

    def test_write(self):
        NullIO().write([1,2,3])

class ConsoleIO_Tests(unittest.TestCase):

    def test_read(self):
        with unittest.mock.patch("builtins.input", return_value="abc"):
            self.assertEqual('abc', ConsoleIO().read())

    def test_write(self):
        with unittest.mock.patch("builtins.print") as mock:
            ConsoleIO().write("abc")
            mock.assert_called_with("abc")

class DiskIO_Tests(unittest.TestCase):

    def setUp(self) -> None:
        if Path("coba/tests/.temp/test.log").exists(): Path("coba/tests/.temp/test.log").unlink()
        if Path("coba/tests/.temp/test.gz").exists(): Path("coba/tests/.temp/test.gz").unlink()

    def tearDown(self) -> None:
        if Path("coba/tests/.temp/test.log").exists(): Path("coba/tests/.temp/test.log").unlink()
        if Path("coba/tests/.temp/test.gz").exists(): Path("coba/tests/.temp/test.gz").unlink()

    def test_simple_sans_gz(self):

        io = DiskIO("coba/tests/.temp/test.log")

        io.write("a")
        io.write("b")
        io.write("c")
        
        self.assertEqual(["a","b","c"], list(io.read()))

    def test_simple_with_gz(self):

        io = DiskIO("coba/tests/.temp/test.gz")

        io.write("a")
        io.write("b")
        io.write("c")
        
        self.assertEqual(["a","b","c"], list(io.read()))

class ListIO_Tests(unittest.TestCase):
    
    def test_simple(self):
        io = ListIO()
        io.write("a")
        io.write("b")
        io.write("c")
        self.assertEqual(["a","b","c"], io.read())

    def test_string(self):

        io = ListIO()
        io.write("abc")
        self.assertEqual(["abc"], io.read())

class QueueIO_Tests(unittest.TestCase):
    
    def test_read_write_sans_blocking(self):

        io = QueueIO(block=False)

        io.write("a")
        io.write("b")
        io.write("c")

        self.assertEqual(["a","b","c"], list(io.read()))

    def test_read_write_with_blocking(self):

        io = QueueIO()

        io.write("a")
        io.write("b")
        io.write("c")
        io.write(None)

        self.assertEqual(["a","b","c"], list(io.read()))

    def test_read_exception(self):

        with self.assertRaises(Exception):
            list(QueueIO(BrokenQueue(Exception())).read())

        list(QueueIO(BrokenQueue(EOFError())).read())
        list(QueueIO(BrokenQueue(BrokenPipeError())).read())

    def test_write_exception(self):

        with self.assertRaises(Exception):
            QueueIO(BrokenQueue(Exception())).write(1)

        QueueIO(BrokenQueue(EOFError())).write(1)
        QueueIO(BrokenQueue(BrokenPipeError())).write(1)

class HttpIO_Tests(unittest.TestCase):

    def test_read(self):
        try:
            with HttpIO("http://www.google.com").read() as response:
                self.assertIn(b"google", response.content)
        except requests.exceptions.ConnectionError as e:
            pass

    def test_write(self):
        with self.assertRaises(NotImplementedError):
            HttpIO("www.google.com").write("abc")

class IdentityIO_Tests(unittest.TestCase):
    
    def test_init_read(self):

        io = IdentityIO('a')
        self.assertEqual("a", io.read())

    def test_write_read(self):
        io = IdentityIO()
        io.write("a")
        self.assertEqual("a", io.read())

if __name__ == '__main__':
    unittest.main()
