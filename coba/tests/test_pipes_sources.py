import unittest
import unittest.mock
import requests.exceptions
import pickle
import gzip

from queue import Queue
from pathlib import Path

from coba.utilities import PackageChecker
from coba.exceptions import CobaException
from coba.context import NullLogger, CobaContext
from coba.pipes.sources import IdentitySource, DiskSource, QueueSource, NullSource, UrlSource, SourceFilters
from coba.pipes.sources import HttpSource, LambdaSource, IterableSource, DataFrameSource, NextSource

CobaContext.logger = NullLogger()

class BrokenQueue:

    def __init__(self, exception):
        self._exception = exception

    def get(self):
        raise self._exception

    def put(self,item):
        raise self._exception

class ReprSource:
    def __init__(self,items):
        self.items = items
    def __str__(self):
        return "ReprSource"
    def read(self):
        return self.items

class ParamsSource:
    @property
    def params(self):
        return {'source':"ParamsSource"}

    def read(self):
        return 1

class NoParamsSource:
    def read(self):
        return 1

class ReprFilter:
    def __init__(self,id=""):
        self._id = id

    def __str__(self):
        return f"ReprFilter{self._id}"

    def filter(self, item):
        return item

class ParamsFilter:
    @property
    def params(self):
        return {'filter':"ParamsFilter"}

    def filter(self,item):
        return item

class NoParamsFilter:
    def filter(self,item):
        return item

class SourceFilters_Tests(unittest.TestCase):
    def test_init_source_filters(self):
        source = SourceFilters(ReprSource([1,2]), ReprFilter(), ReprFilter())
        self.assertEqual(3, len(source))
        self.assertIsInstance(source[0], ReprSource)
        self.assertIsInstance(source[1], ReprFilter)
        self.assertIsInstance(source[2], ReprFilter)
        self.assertEqual("ReprSource,ReprFilter,ReprFilter", str(source))

    def test_sourcefilter_filters(self):
        source = SourceFilters(SourceFilters(ReprSource([1,2]), ReprFilter()), ReprFilter())
        self.assertEqual(3, len(source))
        self.assertIsInstance(source[0], ReprSource)
        self.assertIsInstance(source[1], ReprFilter)
        self.assertIsInstance(source[2], ReprFilter)
        self.assertEqual("ReprSource,ReprFilter,ReprFilter", str(source))

    def test_read1(self):
        source = SourceFilters(ReprSource([1,2]), ReprFilter(), ReprFilter())
        self.assertEqual([1,2], list(source.read()))

    def test_read2(self):
        source = SourceFilters(SourceFilters(ReprSource([1,2]), ReprFilter()), ReprFilter())
        self.assertEqual([1,2], list(source.read()))

    def test_params(self):
        source = SourceFilters(NoParamsSource(), NoParamsFilter())
        self.assertEqual({}, source.params)
        source = SourceFilters(NoParamsSource(), ParamsFilter())
        self.assertEqual({'filter':'ParamsFilter'}, source.params)

        source = SourceFilters(NoParamsSource(), NoParamsFilter(), ParamsFilter())
        self.assertEqual({'filter':'ParamsFilter'}, source.params)

        source = SourceFilters(ParamsSource(), NoParamsFilter(), ParamsFilter())
        self.assertEqual({'source':'ParamsSource','filter':'ParamsFilter'}, source.params)

        source = SourceFilters(ParamsSource(), ParamsFilter(), ParamsFilter())
        self.assertEqual({'source':'ParamsSource','filter1':'ParamsFilter','filter2':'ParamsFilter'}, source.params)

    def test_len(self):
        self.assertEqual(3, len(SourceFilters(ReprSource([1,2]), ReprFilter(), ReprFilter())))

    def test_getitem(self):
        source  = ReprSource([1,2])
        filter1 = ReprFilter()
        filter2 = ReprFilter()

        pipe = SourceFilters(source, filter1, filter2)

        self.assertIs(pipe[0],source)
        self.assertIs(pipe[1],filter1)
        self.assertIs(pipe[2],filter2)

        self.assertIs(pipe[-1],filter2)
        self.assertIs(pipe[-2],filter1)
        self.assertIs(pipe[-3],source)

    def test_iter(self):
        source  = ReprSource([1,2])
        filter1 = ReprFilter()
        filter2 = ReprFilter()
        pipes = list(SourceFilters(source, filter1, filter2))
        self.assertEqual(pipes, [source,filter1,filter2])

class NullSource_Tests(unittest.TestCase):
    def test_read(self):
        self.assertEqual(0, len(NullSource().read()))

class IdentitySource_Tests(unittest.TestCase):
    def test_read(self):
        item = [1,2,3]
        self.assertIs(item, IdentitySource(item).read())

    def test_params(self):
        self.assertEqual({}, IdentitySource(None).params)
        self.assertEqual({'a','b'}, IdentitySource(None,params={'a','b'}).params)

class DiskSource_Tests(unittest.TestCase):
    def setUp(self) -> None:
        if Path("coba/tests/.temp/test.log").exists(): Path("coba/tests/.temp/test.log").unlink()
        if Path("coba/tests/.temp/test.gz").exists(): Path("coba/tests/.temp/test.gz").unlink()

    def tearDown(self) -> None:
        if Path("coba/tests/.temp/test.log").exists(): Path("coba/tests/.temp/test.log").unlink()
        if Path("coba/tests/.temp/test.gz").exists(): Path("coba/tests/.temp/test.gz").unlink()

    def test_simple_sans_gz(self):
        Path("coba/tests/.temp/test.log").write_text("a\nb\nc")
        self.assertEqual(["a","b","c"], list(DiskSource("coba/tests/.temp/test.log").read()))

    def test_simple_with_gz(self):
        Path("coba/tests/.temp/test.gz").write_bytes(gzip.compress(b'a\nb\nc'))
        self.assertEqual(["a","b","c"], list(DiskSource("coba/tests/.temp/test.gz").read()))

    def test_is_picklable(self):
        pickle.dumps(DiskSource("coba/tests/.temp/test.gz"))

    def test_start_and_include_loc(self):
        Path("coba/tests/.temp/test.log").write_text("a\nb\nc")

        for (loc,val) in DiskSource("coba/tests/.temp/test.log", include_loc=True).read():
            self.assertEqual(next(DiskSource("coba/tests/.temp/test.log", start_loc=loc).read()),val)

        Path("coba/tests/.temp/test.gz").write_bytes(gzip.compress(b'a\nb\nc'))
        for (loc,val) in DiskSource("coba/tests/.temp/test.gz", include_loc=True).read():
            self.assertEqual(next(DiskSource("coba/tests/.temp/test.gz", start_loc=loc).read()),val)

class QueueSource_Tests(unittest.TestCase):
    def test_read_sans_blocking(self):

        queue = Queue()
        queue.put('a')
        queue.put('b')
        queue.put('c')

        source = QueueSource(queue, block=False)
        self.assertEqual(["a","b","c"], list(source.read()))

    def test_read_with_blocking(self):

        queue = Queue()

        queue.put('a')
        queue.put('b')
        queue.put('c')
        queue.put(None)

        source = QueueSource(queue, block=True)
        self.assertEqual(["a","b","c"], list(source.read()))

    def test_read_exception(self):

        with self.assertRaises(Exception):
            list(QueueSource(BrokenQueue(Exception())).read())

        list(QueueSource(BrokenQueue(EOFError())).read())
        list(QueueSource(BrokenQueue(BrokenPipeError())).read())

class HttpSource_Tests(unittest.TestCase):
    def test_read(self):
        try:
            with HttpSource("http://www.google.com").read() as response:
                self.assertIn(b"google", response.content)
        except requests.exceptions.ConnectionError as e:
            pass

class ListSource_Tests(unittest.TestCase):
    def test_read_1(self):
        io = IterableSource(['a','b'])
        self.assertEqual(["a",'b'], list(io.read()))

    def test_read_2(self):
        io = IterableSource()
        self.assertEqual([], list(io.read()))

class LambdaSource_Tests(unittest.TestCase):
    def test_read(self):
        io = LambdaSource(lambda:"a")
        self.assertEqual("a",io.read())
        self.assertEqual("a",io.read())

class UrlSource_Tests(unittest.TestCase):
    def test_http_scheme(self):
        url = "http://www.google.com"
        self.assertIsInstance(UrlSource(url)._source, HttpSource)
        self.assertEqual(url, UrlSource(url)._source._url)

    def test_https_scheme(self):
        url = "https://www.google.com"
        self.assertIsInstance(UrlSource(url)._source, HttpSource)
        self.assertEqual(url, UrlSource(url)._source._url)

    def test_file_scheme(self):
        url = "file://c:/users"
        self.assertIsInstance(UrlSource(url)._source, DiskSource)
        self.assertEqual(url[7:], UrlSource(url)._source._path)

    def test_no_scheme(self):
        url = "c:/users"
        self.assertIsInstance(UrlSource(url)._source, DiskSource)
        self.assertEqual(url, UrlSource(url)._source._path)

    def test_unknown_scheme(self):
        with self.assertRaises(CobaException):
            UrlSource("irc://fail")

class NextSource_Tests(unittest.TestCase):

    def test_generator(self):
        class TestSource:
            def read(self):
                yield 1
                yield 2
        self.assertEqual(NextSource(TestSource()).read(),1)

    def test_list(self):
        class TestSource:
            def read(self):
                return [1,2]
        self.assertEqual(NextSource(TestSource()).read(),1)

    def test_generator_close(self):
        class TestSource:
            def __init__(self) -> None:
                self.closed=False
            def read(self):
                try:
                    yield 1
                    yield 2
                except GeneratorExit:
                    self.closed=True
        inner_source = TestSource()
        outer_source = NextSource(inner_source)
        self.assertFalse(inner_source.closed)
        self.assertEqual(outer_source.read(),1)
        self.assertTrue(inner_source.closed)

    def test_list_close(self):
        class TestSource:
            def read(self):
                return [1,2]
        inner_source = TestSource()
        outer_source = NextSource(inner_source)
        self.assertEqual(outer_source.read(),1)

@unittest.skipUnless(PackageChecker.pandas(strict=False), "pandas is not installed so we must skip pandas tests")
class DataFrameSource_Tests(unittest.TestCase):
    def test_simple(self):
        import pandas as pd
        source = DataFrameSource(pd.DataFrame({'a':[1,2],'b':[3,4]}))
        expected = [{'a':1,'b':3},{'a':2,'b':4}]
        self.assertEqual(list(source.read()),expected)

if __name__ == '__main__':
    unittest.main()
