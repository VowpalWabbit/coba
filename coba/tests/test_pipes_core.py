
import unittest

import multiprocessing as mp
from typing import Iterable, Any

from coba.exceptions import CobaException
from coba.pipes import ListSink, IterableSource, Foreach

from coba.pipes.core import Pipes, join

class SingleItemIdentity:
    def filter(self,item):
        return item

class ReprSink:
    def __init__(self,params={}) -> None:
        self.items = []
        self.params = params
    def __str__(self):
        return "ReprSink"
    def write(self,item):
        self.items.append(item)

class ReprSource:
    def __init__(self,items=[]):
        self.items = items
    def __str__(self):
        return "ReprSource"
    def read(self):
        return self.items

class ReprFilter:
    def __init__(self,id=""):
        self._id = id

    def __str__(self):
        return f"ReprFilter{self._id}"

    def filter(self, item: Any) -> Any:
        return item

class ProcessNameFilter:
    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        for _ in items:
            yield mp.current_process().name

class ExceptionFilter:
    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        raise Exception("Exception Filter")

class join_tests(unittest.TestCase):

    def test_join(self):
        self.assertEqual("ReprSource | ReprSink", str(join(ReprSource(), ReprSink())))

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
        self.assertEqual("ReprSource | ReprFilter | ReprFilter | ReprSink", str(Pipes.join(source, *filters, sink)))

    def test_join_source_filters_repr(self):
        source  = ReprSource()
        filters = [ReprFilter(), ReprFilter()]
        self.assertEqual("ReprSource | ReprFilter | ReprFilter", str(Pipes.join(source, *filters)))

    def test_join_source_foreach_filter(self):
        filter = Pipes.join(ReprSource([1,2]), Foreach(ReprFilter()))
        self.assertEqual(list(filter.read()), [1,2])

    def test_join_filters_sink_repr(self):
        self.assertEqual("ReprFilter | ReprFilter | ReprSink", str(Pipes.join(ReprFilter(),ReprFilter(),ReprSink())))

    def test_join_filters_repr(self):
        self.assertEqual("ReprFilter | ReprFilter", str(Pipes.join(ReprFilter(), ReprFilter())))

    def test_join_source_sink_repr(self):
        self.assertEqual("ReprSource | ReprSink", str(Pipes.join(ReprSource(), ReprSink())))

    def test_join_flattens_filters(self):
        filter = Pipes.join(ReprFilter())
        filter = Pipes.join(filter, ReprFilter())
        filter = Pipes.join(filter, filter)
        self.assertEqual(4, len(filter))

    def test_bad_exception(self):
        with self.assertRaises(CobaException):
            Pipes.join()
        with self.assertRaises(CobaException):
            Pipes.join(object())

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

if __name__ == '__main__':
    unittest.main()