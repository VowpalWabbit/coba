
import unittest

from multiprocessing import current_process
from typing import Iterable, Any

from coba.utilities import CobaConfig
from coba.pipes import Pipe, Filter, IdentityFilter, MemorySink, MemorySource

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

    def test_run(self):
        source = MemorySource(list(range(10)))
        sink   = MemorySink()

        Pipe.join(source, [Pipe_Tests.ProcessNameFilter()], sink).run()

        self.assertEqual(sink.items, ['MainProcess']*10)

    def test_exception(self):
        source = MemorySource(list(range(4)))
        sink   = MemorySink()

        with self.assertRaises(Exception):
            Pipe.join(source, [Pipe_Tests.ExceptionFilter()], sink).run()

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