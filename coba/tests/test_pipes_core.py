
import unittest

from multiprocessing import current_process
from typing import Iterable, Any

from coba.pipes import Pipe, Filter, Identity, MemoryIO

class Pipe_Tests(unittest.TestCase):


    class ReprIO(MemoryIO):
        def __repr__(self):
            return "ReprIO"

    class ReprFilter(Identity):
        def __repr__(self):
            return "ReprFilter"

    class ProcessNameFilter(Filter):
        def filter(self, items: Iterable[Any]) -> Iterable[Any]:

            process_name = current_process().name

            for _ in items:
                yield process_name

    class ExceptionFilter(Filter):
        def filter(self, items: Iterable[Any]) -> Iterable[Any]:
            raise Exception("Exception Filter")

    def test_run(self):
        memoryIO = MemoryIO(list(range(10)))

        Pipe.join(memoryIO, [Pipe_Tests.ProcessNameFilter()], memoryIO).run()

        self.assertEqual(memoryIO.items, ['MainProcess']*10)

    def test_exception(self):
        memoryIO = MemoryIO(list(range(4)))

        with self.assertRaises(Exception):
            Pipe.join(memoryIO, [Pipe_Tests.ExceptionFilter()], memoryIO).run()

    def test_repr1(self):

        source  = Pipe_Tests.ReprIO([0,1,2])
        filters = [Pipe_Tests.ReprFilter(), Pipe_Tests.ReprFilter()]
        sink    = Pipe_Tests.ReprIO()

        expected_repr = "ReprIO,ReprFilter,ReprFilter,ReprIO"
        self.assertEqual(expected_repr, str(Pipe.join(source, filters, sink)))
    
    def test_repr2(self):

        source  = Pipe_Tests.ReprIO([0,1,2])
        filters = [Pipe_Tests.ReprFilter(), Pipe_Tests.ReprFilter()]

        expected_repr = "ReprIO,ReprFilter,ReprFilter"
        self.assertEqual(expected_repr, str(Pipe.join(source, filters)))

    def test_repr3(self):

        filters = [Pipe_Tests.ReprFilter(), Pipe_Tests.ReprFilter()]
        sink    = Pipe_Tests.ReprIO()

        expected_repr = "ReprFilter,ReprFilter,ReprIO"
        self.assertEqual(expected_repr, str(Pipe.join(filters, sink)))

    def test_repr4(self):

        filter = Pipe.FiltersFilter([Pipe_Tests.ReprFilter(), Pipe_Tests.ReprFilter()])

        expected_repr = "ReprFilter,ReprFilter"
        self.assertEqual(expected_repr, str(filter))

    def test_join_flattens_filters(self):

        filter1 = Pipe.join([Identity(), Identity()])
        filter2 = Pipe.join([filter1, Identity()])
        filter3 = Pipe.join([filter2, filter2])

        self.assertEqual(6, len(filter3._filters))

if __name__ == '__main__':
    unittest.main()