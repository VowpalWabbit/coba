import unittest
import unittest.mock
import pickle
import gzip

from pathlib import Path

from coba.context import NullLogger, CobaContext

from coba.pipes.sinks import DiskSink, ListSink, QueueSink, NullSink, ConsoleSink, LambdaSink, FiltersSink

CobaContext.logger = NullLogger()

class BrokenQueue:
    def __init__(self, exception):
        self._exception = exception

    def get(self):
        raise self._exception

    def put(self,item):
        raise self._exception

class ReprSink:
    def __init__(self) -> None:
        self.items = []
    def __str__(self):
        return "ReprSink"
    def write(self,item):
        self.items.append(item)

class ReprFilter:
    def __init__(self,id=""):
        self._id = id
    def __str__(self):
        return f"ReprFilter{self._id}"
    def filter(self, item):
        return item

class NoParamsFilter:
    def filter(self,item):
        return item

class ParamsSink:
    @property
    def params(self):
        return {'sink':"ParamsSink"}
    def write(self, item):
        pass

class NoParamsSink:
    def write(self, item):
        pass

class ParamsFilter:
    @property
    def params(self):
        return {'filter':"ParamsFilter"}

    def filter(self,item):
        return item

class FiltersSink_Tests(unittest.TestCase):
    def test_init_filters_sink(self):
        filter = FiltersSink(ReprFilter(), ReprFilter(), ReprSink())
        self.assertEqual(3, len(filter))
        self.assertIsInstance(filter[0], ReprFilter)
        self.assertIsInstance(filter[1], ReprFilter)
        self.assertIsInstance(filter[2], ReprSink)
        self.assertEqual("ReprFilter | ReprFilter | ReprSink", str(filter))

    def test_init_filters_filterssink(self):
        filter = FiltersSink(ReprFilter(), FiltersSink(ReprFilter(), ReprSink()))
        self.assertEqual(3, len(filter))
        self.assertIsInstance(filter[0], ReprFilter)
        self.assertIsInstance(filter[1], ReprFilter)
        self.assertIsInstance(filter[2], ReprSink)
        self.assertEqual("ReprFilter | ReprFilter | ReprSink", str(filter))

    def test_write1(self):
        sink = FiltersSink(ReprFilter(), ReprFilter(), ReprSink())
        sink.write(1)
        sink.write(2)
        self.assertEqual([1,2], sink[-1].items)

    def test_write2(self):
        sink = FiltersSink(ReprFilter(), FiltersSink(ReprFilter(), ReprSink()))
        sink.write(1)
        sink.write(2)

        self.assertEqual([1,2], sink[-1].items)

    def test_params(self):
        sink = FiltersSink(NoParamsFilter(), NoParamsSink())
        self.assertEqual({}, sink.params)

        sink = FiltersSink(ParamsFilter(), NoParamsSink())
        self.assertEqual({'filter':'ParamsFilter'}, sink.params)

        sink = FiltersSink(NoParamsFilter(), ParamsFilter(), NoParamsSink())
        self.assertEqual({'filter':'ParamsFilter'}, sink.params)

        sink = FiltersSink(NoParamsFilter(), ParamsFilter(), ParamsSink())
        self.assertEqual({'sink':'ParamsSink','filter':'ParamsFilter'}, sink.params)

        source = FiltersSink(ParamsFilter(), ParamsFilter(), ParamsSink())
        self.assertEqual({'filter1':'ParamsFilter','filter2':'ParamsFilter','sink':'ParamsSink'}, source.params)

    def test_len(self):
        self.assertEqual(3,len(FiltersSink(ReprFilter(), ReprFilter(), ReprSink())))

    def test_getitem(self):
        filter1 = ReprFilter()
        filter2 = ReprFilter()
        sink    = ReprSink()

        pipe = FiltersSink(filter1, filter2, sink)

        self.assertIs(filter1,pipe[0])
        self.assertIs(filter2,pipe[1])
        self.assertIs(sink   ,pipe[2])

        self.assertIs(sink   ,pipe[-1])
        self.assertIs(filter2,pipe[-2])
        self.assertIs(filter1,pipe[-3])

    def test_iter(self):
        filter1 = ReprFilter()
        filter2 = ReprFilter()
        sink    = ReprSink()

        pipes = list(FiltersSink(filter1, filter2, sink))
        self.assertEqual(pipes, [filter1,filter2,sink])

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
