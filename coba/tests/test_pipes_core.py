
import unittest

from multiprocessing import current_process
from typing import Iterable, Any

from coba.exceptions import CobaException
from coba.pipes import Filter, ListIO
from coba.pipes.core import Pipe, Foreach, SourceFilters, FiltersFilter, FiltersSink, Pipeline

class SingleItemIdentity:
    def filter(self,item):
        return item

class ReprIO(ListIO):
    def __str__(self):
        return "ReprIO"

class ReprFilter(Filter):
    def __str__(self):
        return "ReprFilter"
    
    def filter(self, item: Any) -> Any:
        return item

class ProcessNameFilter(Filter):
    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        for _ in items:
            yield current_process().name

class ExceptionFilter(Filter):
    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        raise Exception("Exception Filter")

class SourceFilters_Tests(unittest.TestCase):

    def test_init_source_filters(self):

        io = ReprIO()

        io.write(1)
        io.write(2)

        filter = SourceFilters(io, [ReprFilter(), ReprFilter()])

        self.assertIsInstance(filter._source, ReprIO)
        self.assertEqual(2, len(filter._filter._filters))
        self.assertIsInstance(filter._filter._filters[0], ReprFilter)
        self.assertIsInstance(filter._filter._filters[1], ReprFilter)
        self.assertEqual("ReprIO,ReprFilter,ReprFilter", str(filter))

    def test_sourcefilter_filters(self):

        io = ReprIO()

        io.write(1)
        io.write(2)

        source = SourceFilters(io, [ReprFilter()])
        filter = SourceFilters(source, [ReprFilter()])

        self.assertIsInstance(filter._source, ReprIO)
        self.assertEqual(2, len(filter._filter._filters))
        self.assertIsInstance(filter._filter._filters[0], ReprFilter)
        self.assertIsInstance(filter._filter._filters[1], ReprFilter)
        self.assertEqual("ReprIO,ReprFilter,ReprFilter", str(filter))

    def test_read1(self):

        io = ReprIO()

        io.write(1)
        io.write(2)

        filter = SourceFilters(io, [ReprFilter(), ReprFilter()])

        self.assertEqual([1,2], list(filter.read()))

    def test_read2(self):

        io = ReprIO()

        io.write(1)
        io.write(2)

        source = SourceFilters(io, [ReprFilter()])
        filter = SourceFilters(source, [ReprFilter()])

        self.assertEqual([1,2], list(filter.read()))

class FiltersFilter_Tests(unittest.TestCase):
    
    def test_init_filters(self):

        filter = FiltersFilter([ReprFilter(), ReprFilter()])

        self.assertEqual(2, len(filter._filters))
        self.assertIsInstance(filter._filters[0], ReprFilter)
        self.assertIsInstance(filter._filters[1], ReprFilter)
        self.assertEqual("ReprFilter,ReprFilter", str(filter))

    def test_init_filtersfilter(self):

        filter = FiltersFilter([FiltersFilter([ReprFilter(), ReprFilter()]), ReprFilter()])

        self.assertEqual(3, len(filter._filters))
        self.assertIsInstance(filter._filters[0], ReprFilter)
        self.assertIsInstance(filter._filters[1], ReprFilter)
        self.assertIsInstance(filter._filters[2], ReprFilter)
        self.assertEqual("ReprFilter,ReprFilter,ReprFilter", str(filter))

    def test_read1(self):

        self.assertEqual([0,1,2], list(FiltersFilter([ReprFilter(), ReprFilter()]).filter(range(3))))

    def test_read2(self):

        self.assertEqual([0,1,2], list(FiltersFilter([FiltersFilter([ReprFilter(), ReprFilter()]), ReprFilter()]).filter(range(3))))

class FiltersSink_Tests(unittest.TestCase):
    
    def test_init_filters_sink(self):

        filter = FiltersSink([ReprFilter(), ReprFilter()], ReprIO())

        self.assertEqual(2, len(filter._filter._filters))
        self.assertIsInstance(filter._filter._filters[0], ReprFilter)
        self.assertIsInstance(filter._filter._filters[1], ReprFilter)
        self.assertIsInstance(filter._sink, ReprIO)
        self.assertEqual("ReprFilter,ReprFilter,ReprIO", str(filter))

    def test_init_filters_filterssink(self):

        sink   = FiltersSink([ReprFilter()], ReprIO())
        filter = FiltersSink([ReprFilter()], sink)

        self.assertEqual(2, len(filter._filter._filters))
        self.assertIsInstance(filter._filter._filters[0], ReprFilter)
        self.assertIsInstance(filter._filter._filters[1], ReprFilter)
        self.assertIsInstance(filter._sink, ReprIO)
        self.assertEqual("ReprFilter,ReprFilter,ReprIO", str(filter))

    def test_write1(self):

        io = ReprIO()
        sink = FiltersSink([ReprFilter(), ReprFilter()], io)

        sink.write(1)
        sink.write(2)

        self.assertEqual([1,2], list(io.read()))

    def test_read2(self):
        
        io    = ReprIO()
        sink  = FiltersSink([ReprFilter()], io)
        sink2 = FiltersSink([ReprFilter()], sink)
        
        
        sink2.write(1)
        sink2.write(2)
        
        self.assertEqual([1,2], list(io.read()))

class Pipeline_Tests(unittest.TestCase):

    def test_init_source_sink(self):
        source = ReprIO()
        sink = ReprIO()

        source.write(1)
        source.write(2)

        pipeline = Pipeline(source, [], Foreach(sink))
        pipeline.run()

        self.assertEqual([1,2], list(sink.read()))
        self.assertEqual("ReprIO,ReprIO", str(pipeline))

    def test_init_filters_source_sink(self):
        source = ReprIO()
        sink = ReprIO()

        source.write(1)
        source.write(2)

        pipeline = Pipeline(source, [ReprFilter(), ReprFilter()], Foreach(sink))
        pipeline.run()

        self.assertEqual([1,2], list(sink.read()))
        self.assertEqual("ReprIO,ReprFilter,ReprFilter,ReprIO", str(pipeline))

class Foreach_Tests(unittest.TestCase):
    
    def test_filter(self):
        self.assertEqual([0,1,2],  list(Foreach(SingleItemIdentity()).filter(range(3))))

    def test_write(self):
        io = ListIO()
        Foreach(io).write(range(3))
        self.assertEqual([0,1,2], list(io.read()))
    
    def test_str(self):

        self.assertEqual("ReprIO", str(Foreach(ReprIO())))

class Pipe_Tests(unittest.TestCase):

    def test_run(self):
        memoryIO_in  = ListIO(list(range(10)))
        memoryIO_out = ListIO(list(range(10)))

        Pipe.join(memoryIO_in, [ProcessNameFilter()], memoryIO_out).run()

        self.assertEqual(memoryIO_out.items[10], ['MainProcess']*10)

    def test_exception(self):
        memoryIO = ListIO(list(range(4)))

        with self.assertRaises(Exception):
            Pipe.join(memoryIO, [ExceptionFilter()], memoryIO).run()

    def test_join_source_filters_sink_repr(self):

        source  = ReprIO()
        filters = [ReprFilter(), ReprFilter()]
        sink    = ReprIO()

        self.assertEqual("ReprIO,ReprFilter,ReprFilter,ReprIO", str(Pipe.join(source, filters, sink)))
    
    def test_join_source_filters_repr(self):

        source  = ReprIO()
        filters = [ReprFilter(), ReprFilter()]

        self.assertEqual("ReprIO,ReprFilter,ReprFilter", str(Pipe.join(source, filters)))

    def test_join_filters_sink_repr(self):

        filters = [ReprFilter(), ReprFilter()]
        sink    = ReprIO()

        self.assertEqual("ReprFilter,ReprFilter,ReprIO", str(Pipe.join(filters, sink)))

    def test_join_filters_repr(self):
        self.assertEqual("ReprFilter,ReprFilter", str(Pipe.join([ReprFilter(), ReprFilter()])))

    def test_join_sourc_sink_repr(self):

        source  = ReprIO()
        sink    = ReprIO()

        self.assertEqual("ReprIO,ReprIO", str(Pipe.join(source, sink)))

    def test_join_flattens_filters(self):

        filter1 = Pipe.join([ReprFilter()])
        filter2 = Pipe.join([filter1, ReprFilter()])
        filter3 = Pipe.join([filter2, filter2])

        self.assertEqual(4, len(filter3._filters))

    def test_bad_exception(self):

        with self.assertRaises(CobaException):
            Pipe.join()

if __name__ == '__main__':
    unittest.main()