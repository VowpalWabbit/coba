
import unittest

from multiprocessing import current_process
from typing import Iterable, Any

from coba.tools import CobaConfig, BasicLogger
from coba.data.filters import Filter, IdentityFilter
from coba.data.sinks import MemorySink
from coba.data.sources import MemorySource
from coba.data.pipes import Pipe

class Pipe_Tests(unittest.TestCase):

    class ReprSource(MemorySource):
        def __repr__(self):
            return "ReprSource"

    class ReprFilter(IdentityFilter):
        def __repr__(self):
            return "ReprFilter"

    class ReprSink(MemorySink):
        def __repr__(self) -> str:
            return "ReprSink"

    class ProcessNameFilter(Filter):
        def filter(self, items: Iterable[Any]) -> Iterable[Any]:

            process_name = current_process().name

            for _ in items:
                CobaConfig.Logger.log(process_name)
                yield process_name

    class ExceptionFilter(Filter):
        def filter(self, items: Iterable[Any]) -> Iterable[Any]:
            raise Exception("Exception Filter")

    def test_single_process_multitask(self):
        source = MemorySource(list(range(10)))
        sink   = MemorySink()

        Pipe.join(source, [Pipe_Tests.ProcessNameFilter()], sink).run()

        self.assertEqual(sink.items, ['MainProcess']*10)

    def test_singleprocess_singletask(self):
        source = MemorySource(list(range(4)))
        sink   = MemorySink[Iterable[int]]()

        Pipe.join(source, [Pipe_Tests.ProcessNameFilter()], sink).run(1,1)

        self.assertEqual(len(set(sink.items)), 4)

    def test_multiprocess_multitask(self):
        source = MemorySource(list(range(40)))
        sink   = MemorySink()

        Pipe.join(source, [Pipe_Tests.ProcessNameFilter()], sink).run(2)

        self.assertEqual(len(set(sink.items)), 2)

    def test_multiprocess_singletask(self):
        source = MemorySource(list(range(4)))
        sink   = MemorySink()

        Pipe.join(source, [Pipe_Tests.ProcessNameFilter()], sink).run(2,1)

        self.assertEqual(len(set(sink.items)), 4)

    def test_exception_multiprocess(self):
        source = MemorySource(list(range(4)))
        sink   = MemorySink()

        with self.assertRaises(Exception):
            Pipe.join(source, [Pipe_Tests.ExceptionFilter()], sink).run(2,1)

    def test_exception_singleprocess(self):
        source = MemorySource(list(range(4)))
        sink   = MemorySink()

        with self.assertRaises(Exception):
            Pipe.join(source, [Pipe_Tests.ExceptionFilter()], sink).run()

    def test_logging(self):
        
        #this is an important example. Even if we set the main logger's
        #with_stamp to false it doesn't propogate to the processes.

        sink   = MemorySink()
        logger = BasicLogger(sink, with_stamp=False, with_name=True)
        logs   = sink.items

        CobaConfig.Logger = logger

        source = MemorySource(list(range(4)))
        sink   = MemorySink()

        Pipe.join(source, [Pipe_Tests.ProcessNameFilter()], sink).run(2,1)

        self.assertEqual(len(logs), 4)
        self.assertEqual(sink.items, [ l.split(' ')[3] for l in logs ] )
        self.assertEqual(sink.items, [ l.split(' ')[5] for l in logs ] )

    def test_repr1(self):

        source  = Pipe_Tests.ReprSource([0,1,2])
        filters = [Pipe_Tests.ReprFilter(), Pipe_Tests.ReprFilter()]
        sink    = Pipe_Tests.ReprSink()

        expected_repr = "ReprSource,ReprFilter,ReprFilter,ReprSink"
        self.assertEqual(expected_repr, str(Pipe.join(source, filters, sink)))
    
    def test_repr2(self):

        source  = Pipe_Tests.ReprSource([0,1,2])
        filters = [Pipe_Tests.ReprFilter(), Pipe_Tests.ReprFilter()]

        expected_repr = "ReprSource,ReprFilter,ReprFilter"
        self.assertEqual(expected_repr, str(Pipe.join(source, filters)))

    def test_repr3(self):

        filters = [Pipe_Tests.ReprFilter(), Pipe_Tests.ReprFilter()]
        sink    = Pipe_Tests.ReprSink()

        expected_repr = "ReprFilter,ReprFilter,ReprSink"
        self.assertEqual(expected_repr, str(Pipe.join(filters, sink)))

    def test_repr4(self):

        filter = Pipe.FiltersFilter([Pipe_Tests.ReprFilter(), Pipe_Tests.ReprFilter()])

        expected_repr = "ReprFilter,ReprFilter"
        self.assertEqual(expected_repr, str(filter))

if __name__ == '__main__':
    unittest.main()