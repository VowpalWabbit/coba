
import unittest

from multiprocessing import current_process
from typing import Iterable, Any

from coba.exceptions import CobaException
from coba.pipes import Filter, ListSink, ListSource

from coba.pipes.core import Pipes, Foreach, SourceFilters, FiltersFilter, FiltersSink
from coba.pipes.core import CsvSource, ArffSource, ManikSource, LibsvmSource

class SingleItemIdentity:
    def filter(self,item):
        return item

class ReprSink(ListSink):
    def __init__(self, params={}) -> None:
        self._params = params
        super().__init__()

    def __str__(self):
        return "ReprSink"
    
    @property
    def params(self):
        return self._params


class ReprSource(ListSource):
    def __str__(self):
        return "ReprSource"

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

        filter = SourceFilters(ReprSource([1,2]), ReprFilter(), ReprFilter())

        self.assertIsInstance(filter._source, ReprSource)
        self.assertEqual(2, len(filter._filter._filters))
        self.assertIsInstance(filter._filter._filters[0], ReprFilter)
        self.assertIsInstance(filter._filter._filters[1], ReprFilter)
        self.assertEqual("ReprSource,ReprFilter,ReprFilter", str(filter))

    def test_sourcefilter_filters(self):

        source = SourceFilters(ReprSource([1,2]), ReprFilter())
        filter = SourceFilters(source, ReprFilter())

        self.assertIsInstance(filter._source, ReprSource)
        self.assertEqual(2, len(filter._filter._filters))
        self.assertIsInstance(filter._filter._filters[0], ReprFilter)
        self.assertIsInstance(filter._filter._filters[1], ReprFilter)
        self.assertEqual("ReprSource,ReprFilter,ReprFilter", str(filter))

    def test_read1(self):
        filter = SourceFilters(ReprSource([1,2]), ReprFilter(), ReprFilter())
        self.assertEqual([1,2], list(filter.read()))

    def test_read2(self):
        source = SourceFilters(ReprSource([1,2]), ReprFilter())
        filter = SourceFilters(source, ReprFilter())

        self.assertEqual([1,2], list(filter.read()))

class FiltersFilter_Tests(unittest.TestCase):
    
    def test_init_filters(self):

        filter = FiltersFilter(ReprFilter(), ReprFilter())

        self.assertEqual(2, len(filter._filters))
        self.assertIsInstance(filter._filters[0], ReprFilter)
        self.assertIsInstance(filter._filters[1], ReprFilter)
        self.assertEqual("ReprFilter,ReprFilter", str(filter))

    def test_init_filtersfilter(self):

        filter = FiltersFilter(FiltersFilter(ReprFilter(), ReprFilter()), ReprFilter())

        self.assertEqual(3, len(filter._filters))
        self.assertIsInstance(filter._filters[0], ReprFilter)
        self.assertIsInstance(filter._filters[1], ReprFilter)
        self.assertIsInstance(filter._filters[2], ReprFilter)
        self.assertEqual("ReprFilter,ReprFilter,ReprFilter", str(filter))

    def test_read1(self):

        self.assertEqual([0,1,2], list(FiltersFilter(ReprFilter(), ReprFilter()).filter(range(3))))

    def test_read2(self):

        self.assertEqual([0,1,2], list(FiltersFilter(FiltersFilter(ReprFilter(), ReprFilter()), ReprFilter()).filter(range(3))))

class FiltersSink_Tests(unittest.TestCase):
    
    def test_init_filters_sink(self):

        filter = FiltersSink(ReprFilter(), ReprFilter(), ReprSink())

        self.assertEqual(2, len(filter._filter._filters))
        self.assertIsInstance(filter._filter._filters[0], ReprFilter)
        self.assertIsInstance(filter._filter._filters[1], ReprFilter)
        self.assertIsInstance(filter._sink, ReprSink)
        self.assertEqual("ReprFilter,ReprFilter,ReprSink", str(filter))

    def test_init_filters_filterssink(self):

        sink   = FiltersSink(ReprFilter(), ReprSink())
        filter = FiltersSink(ReprFilter(), sink)

        self.assertEqual(2, len(filter._filter._filters))
        self.assertIsInstance(filter._filter._filters[0], ReprFilter)
        self.assertIsInstance(filter._filter._filters[1], ReprFilter)
        self.assertIsInstance(filter._sink, ReprSink)
        self.assertEqual("ReprFilter,ReprFilter,ReprSink", str(filter))

    def test_write1(self):

        sink = FiltersSink(ReprFilter(), ReprFilter(), ReprSink())
        sink.write(1)
        sink.write(2)
        self.assertEqual([1,2], sink._sink.items)

    def test_read2(self):
        
        sink  = FiltersSink(ReprFilter(), ReprSink())
        sink2 = FiltersSink(ReprFilter(), sink)        
        sink2.write(1)
        sink2.write(2)
        
        self.assertEqual([1,2], sink._sink.items)

class PipesLine_Tests(unittest.TestCase):

    def test_init_source_sink(self):
        source = ReprSource([1,2])
        sink   = ReprSink()

        pipeline = Pipes.Line(source, Foreach(sink))
        pipeline.run()

        self.assertEqual([1,2], sink.items)
        self.assertEqual("ReprSource,ReprSink", str(pipeline))
        self.assertEqual({}, pipeline.params)

    def test_init_source_filters_sink(self):
        source = ReprSource([1,2])
        sink   = ReprSink()

        pipeline = Pipes.Line(source, ReprFilter(), ReprFilter(), Foreach(sink))
        pipeline.run()

        self.assertEqual([1,2], sink.items)
        self.assertEqual("ReprSource,ReprFilter,ReprFilter,ReprSink", str(pipeline))
        self.assertEqual({}, pipeline.params)

class Foreach_Tests(unittest.TestCase):
    
    def test_filter(self):
        self.assertEqual([0,1,2],  list(Foreach(SingleItemIdentity()).filter(range(3))))

    def test_write(self):
        sink = ListSink()
        Foreach(sink).write(range(3))
        self.assertEqual([0,1,2], sink.items)
    
    def test_str(self):
        self.assertEqual("ReprSink", str(Foreach(ReprSink())))

    def test_params(self):
        self.assertEqual({'a':1}, Foreach(ReprSink(params={'a':1})).params)

class Pipes_Tests(unittest.TestCase):

    def test_run(self):
        sink   = ListSource(list(range(10)))
        source = ListSink()

        Pipes.join(sink, ProcessNameFilter(), source).run()

        self.assertEqual(source.items[0], ['MainProcess']*10)

    def test_exception(self):
        with self.assertRaises(Exception):
            Pipes.join(ListSource(list(range(4))), ExceptionFilter(), ListSink()).run()

    def test_join_source_filters_sink_repr(self):

        source  = ReprSource()
        filters = [ReprFilter(), ReprFilter()]
        sink    = ReprSink()

        self.assertEqual("ReprSource,ReprFilter,ReprFilter,ReprSink", str(Pipes.join(source, *filters, sink)))
    
    def test_join_source_filters_repr(self):

        source  = ReprSource()
        filters = [ReprFilter(), ReprFilter()]

        self.assertEqual("ReprSource,ReprFilter,ReprFilter", str(Pipes.join(source, *filters)))

    def test_join_source_foreach_filter(self):
        filter = SourceFilters(ReprSource([1,2]), Foreach(ReprFilter()))
        self.assertIsInstance(filter, SourceFilters)

    def test_join_filters_sink_repr(self):

        filters = [ReprFilter(),ReprFilter()]
        sink    = ReprSink()

        self.assertEqual("ReprFilter,ReprFilter,ReprSink", str(Pipes.join(*filters, sink)))

    def test_join_filters_repr(self):
        self.assertEqual("ReprFilter,ReprFilter", str(Pipes.join(ReprFilter(), ReprFilter())))

    def test_join_source_sink_repr(self):

        source  = ReprSource()
        sink    = ReprSink()

        self.assertEqual("ReprSource,ReprSink", str(Pipes.join(source, sink)))

    def test_join_flattens_filters(self):

        filter1 = Pipes.join(ReprFilter())
        filter2 = Pipes.join(filter1, ReprFilter())
        filter3 = Pipes.join(filter2, filter2)

        self.assertEqual(4, len(filter3._filters))

    def test_bad_exception(self):

        with self.assertRaises(CobaException):
            Pipes.join()

        with self.assertRaises(CobaException):
            Pipes.join(object())

class CsvSource_Tests(unittest.TestCase):

    def test_simple(self):
        self.assertEqual([["1","2","3"]], list(CsvSource(ListSource(["1,2,3"])).read()))

class ArffSource_Tests(unittest.TestCase):

    def test_simple(self):
        lines = [
            "@relation news20",
            "@attribute a numeric",
            "@attribute B numeric",
            "@data",
            "1,  2",
            "2,  3",
        ]

        expected = [
            [1, 2],
            [2, 3]
        ]

        self.assertEqual(expected, list(ArffSource(ListSource(lines)).read()))

class LibsvmSource_Tests(unittest.TestCase):

    def test_simple(self):
        lines = [
            "0 1:2 2:3",
            "1 1:1 2:1",
            "2 2:1",
            "1 1:1",
        ]

        expected = [
            ({1:2, 2:3} ,['0']),
            ({1:1, 2:1}, ['1']),
            ({     2:1}, ['2']),
            ({1:1     }, ['1'])
        ]

        self.assertEqual(expected, list(LibsvmSource(ListSource(lines)).read()))

class ManikSource_Tests(unittest.TestCase):

    def test_simple(self):
        lines = [
            "meta line",
            "0 1:2 2:3",
            "1 1:1 2:1",
            "2 2:1",
            "1 1:1",
        ]

        expected = [
            ({1:2, 2:3} ,['0']),
            ({1:1, 2:1}, ['1']),
            ({     2:1}, ['2']),
            ({1:1     }, ['1'])
        ]

        self.assertEqual(expected, list(ManikSource(ListSource(lines)).read()))

if __name__ == '__main__':
    unittest.main()