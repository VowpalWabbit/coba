
import unittest
import time

import multiprocessing as mp

from typing import Iterable, Any

from coba.exceptions import CobaException
from coba.pipes import Filter, ListSink, IterableSource, Foreach, SourceFilters

from coba.pipes.core import Pipes

class ReprSink(ListSink):
    def __init__(self, params={}) -> None:
        self._params = params
        super().__init__()

    def __str__(self):
        return "ReprSink"

    @property
    def params(self):
        return self._params

class ReprSource(IterableSource):
    def __str__(self):
        return "ReprSource"

class ReprFilter(Filter):
    def __init__(self,id=""):
        self._id = id

    def __str__(self):
        return f"ReprFilter{self._id}"

    def filter(self, item: Any) -> Any:
        return item

class ProcessNameFilter(Filter):
    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        for _ in items:
            yield mp.current_process().name

class ExceptionFilter(Filter):
    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        raise Exception("Exception Filter")

class Pipes_Tests(unittest.TestCase):

    def test_run(self):
        source = IterableSource(list(range(10)))
        sink   = ListSink()

        Pipes.join(source, ProcessNameFilter(), sink).run()

        self.assertEqual(sink.items[0], ['MainProcess']*10)

    def test_filter_order(self):

        class AddFilter:
            def filter(self,items):
                for item in items:
                    yield item+1

        class MultFilter:
            def filter(self,items):
                for item in items:
                    yield item*2

        self.assertEqual([4,6], list(Pipes.join(AddFilter(), MultFilter()).filter([1,2])))

    def test_exception(self):
        with self.assertRaises(Exception):
            Pipes.join(IterableSource(list(range(4))), ExceptionFilter(), ListSink()).run()

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
        filter = Pipes.join(ReprSource([1,2]), Foreach(ReprFilter()))
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

if __name__ == '__main__':
    unittest.main()