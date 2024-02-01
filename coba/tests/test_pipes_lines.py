import unittest
import unittest.mock
import multiprocessing as mp
import threading as mt

from typing import Iterable, Any

from coba.primitives import Filter
from coba.pipes      import QueueSink, QueueSource, Foreach

from coba.pipes.lines import SourceSink, ProcessLine, ThreadLine

spawn_context = mp.get_context("spawn")

class ReprSink:
    def __init__(self) -> None:
        self.items = []
    def __str__(self):
        return "ReprSink"
    def write(self,item):
        self.items.append(item)

class ReprSource:
    def __init__(self,items):
        self.items = items
    def __str__(self):
        return "ReprSource"
    def read(self):
        return self.items

class ReprFilter(Filter):
    def __init__(self,id=""):
        self._id = id
    def __str__(self):
        return f"ReprFilter{self._id}"
    def filter(self, item: Any) -> Any:
        return item

class ExceptionFilter(Filter):
    def __init__(self, exc = Exception("Exception Filter")):
        self._exc = exc
    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        raise self._exc

class ParamsSource:
    @property
    def params(self):
        return {'source':"ParamsSource"}

    def read(self):
        return 1

class NoParamsSource:
    def read(self):
        return 1

class ParamsFilter:
    @property
    def params(self):
        return {'filter':"ParamsFilter"}
    def filter(self,item):
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

class SourceSink_Tests(unittest.TestCase):

    def test_run_source_sink(self):
        source = ReprSource([1,2])
        sink   = ReprSink()

        pipeline = SourceSink(source, Foreach(sink))
        pipeline.run()

        self.assertEqual([1,2], sink.items)
        self.assertEqual("ReprSource | ReprSink", str(pipeline))
        self.assertEqual({}, pipeline.params)

    def test_run_source_filters_sink(self):
        source = ReprSource([1,2])
        sink   = ReprSink()

        pipeline = SourceSink(source, ReprFilter(), ReprFilter(), Foreach(sink))
        pipeline.run()

        self.assertEqual([1,2], sink.items)
        self.assertEqual("ReprSource | ReprFilter | ReprFilter | ReprSink", str(pipeline))
        self.assertEqual({}, pipeline.params)

    def test_params(self):
        line = SourceSink(NoParamsSource(), NoParamsFilter(), NoParamsSink())
        self.assertEqual({}, line.params)

        line = SourceSink(NoParamsSource(), ParamsFilter(), NoParamsSink())
        self.assertEqual({'filter':'ParamsFilter'}, line.params)

        line = SourceSink(NoParamsSource(), NoParamsFilter(), ParamsFilter(), NoParamsSink())
        self.assertEqual({'filter':'ParamsFilter'}, line.params)

        line = SourceSink(ParamsSource(), ParamsFilter(), ParamsSink())
        self.assertEqual({'source':'ParamsSource','sink':'ParamsSink','filter':'ParamsFilter'}, line.params)

        line = SourceSink(ParamsSource(), ParamsFilter(), ParamsSink())
        self.assertEqual({'source':'ParamsSource','sink':'ParamsSink','filter':'ParamsFilter'}, line.params)

        line = SourceSink(ParamsSource(), ParamsFilter(), ParamsFilter(), ParamsSink())
        self.assertEqual({'source':'ParamsSource','sink':'ParamsSink','filter1':'ParamsFilter','filter2':'ParamsFilter'}, line.params)

    def test_len(self):
        self.assertEqual(2, len(SourceSink(ReprSource([1,2]), Foreach(ReprSink()))))

    def test_getitem(self):
        source = ReprSource([1,2])
        sink   = Foreach(ReprSink())

        self.assertIs(SourceSink(source, sink)[0],source)
        self.assertIs(SourceSink(source, sink)[1],sink)

class ProcessLine_Tests(unittest.TestCase):

    def setUp(self) -> None:
        self.mode = ProcessLine

    def test_run_async_process_no_callback(self):
        queue1    = spawn_context.Queue()
        queue2    = spawn_context.Queue()

        queue1.put(1)
        queue1.put(2)
        queue1.put(None)

        line = SourceSink(QueueSource(queue1), QueueSink(queue2,True))
        proc = self.mode(line)

        proc.start()
        proc.join()

        self.assertEqual([1,2], [queue2.get(True),queue2.get(True)])
        self.assertIsNone(proc.exception)
        self.assertIsNone(proc.traceback)
        self.assertTrue(proc.poisoned)
        self.assertIs(proc.pipeline, line)

    def test_run_async_process_with_callback(self):
        queue  = spawn_context.Queue()
        event  = mt.Event()
        holder = []

        def callback(item):
            holder.append(item.exception)
            event.set()

        line = SourceSink(ReprSource([1,2]), QueueSink(queue,True))
        proc = self.mode(line,callback=callback)

        proc.start()
        proc.join()
        event.wait()

        self.assertEqual([1,2], [queue.get(True),queue.get(True)])
        self.assertEqual(None, holder[0])
        self.assertIsNone(proc.exception)
        self.assertIsNone(proc.traceback)
        self.assertFalse(proc.poisoned)

    def test_run_async_process_with_exception(self):
        line = SourceSink(ReprSource([1,2]), ExceptionFilter(), ReprSink())
        proc = self.mode(line)

        proc.start()
        proc.join()

        self.assertEqual(str(proc.exception),"Exception Filter")
        self.assertTrue(proc.traceback)
        self.assertFalse(proc.poisoned)

    def test_run_async_process_with_exception_and_callback(self):
        holder = []
        event  = mt.Event()

        def callback(item):
            holder.append(item.exception)
            event.set()

        line = SourceSink(ReprSource([1,2]), ExceptionFilter(), ReprSink())
        proc = self.mode(line,callback)

        proc.start()
        proc.join()

        self.assertEqual(str(proc.exception),"Exception Filter")
        event.wait()
        self.assertEqual(str(holder[0]),"Exception Filter")

    def test_run_async_process_with_exception_and_callback(self):
        holder = []
        event  = mt.Event()

        def callback(item):
            holder.append(item.exception)
            event.set()

        line = SourceSink(ReprSource([1,2]), ExceptionFilter(), ReprSink())
        proc = self.mode(line,callback)

        proc.start()
        proc.join()

        self.assertEqual(str(proc.exception),"Exception Filter")
        event.wait()
        self.assertEqual(str(holder[0]),"Exception Filter")

class ThreadLine_Tests(ProcessLine_Tests):
    def setUp(self) -> None:
        self.mode = ThreadLine

if __name__ == '__main__':
    unittest.main()