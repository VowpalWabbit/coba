
import unittest

from multiprocessing import current_process
from typing import Iterable, Any

from coba.tools import CobaConfig, UniversalLog
from coba.data.filters import Filter
from coba.data.sinks import MemorySink
from coba.data.sources import MemorySource
from coba.data.pipes import Pipe

class Pipe_Tests(unittest.TestCase):

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
        
        actual_logs = []

        CobaConfig.Logger = UniversalLog(lambda msg,end: actual_logs.append((msg,end)))

        source = MemorySource(list(range(4)))
        sink   = MemorySink()

        Pipe.join(source, [Pipe_Tests.ProcessNameFilter()], sink).run(2,1)

        self.assertEqual(len(actual_logs), 4)
        self.assertEqual(sink.items, [ l[0][20:] for l in actual_logs ] )

if __name__ == '__main__':
    unittest.main()