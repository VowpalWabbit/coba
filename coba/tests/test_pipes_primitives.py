
import unittest

from typing import Any

from coba.pipes import Filter, ListSink, IterableSource

from coba.pipes.primitives import Foreach, SourceFilters, FiltersFilter, FiltersSink, SourceSink

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

    def test_params(self):
        source = SourceFilters(NoParamsSource(), NoParamsFilter())
        self.assertEqual({}, source.params)

        source = SourceFilters(NoParamsSource(), ParamsFilter())
        self.assertEqual({'filter':'ParamsFilter'}, source.params)

        source = SourceFilters(NoParamsSource(), NoParamsFilter(), ParamsFilter())
        self.assertEqual({'filter':'ParamsFilter'}, source.params)

        source = SourceFilters(ParamsSource(), NoParamsFilter(), ParamsFilter())
        self.assertEqual({'source':'ParamsSource','filter':'ParamsFilter'}, source.params)

        source = FiltersSink(ParamsSource(), ParamsFilter(), ParamsFilter())
        self.assertEqual({'source':'ParamsSource','filter1':'ParamsFilter','filter2':'ParamsFilter'}, source.params)

    def test_len(self):
        self.assertEqual(3, len(SourceFilters(ReprSource([1,2]), ReprFilter(), ReprFilter())))

    def test_getitem(self):
        source  = ReprSource([1,2])
        filter1 = Foreach(ReprSink())
        filter2 = Foreach(ReprSink())

        pipe = SourceFilters(source, filter1, filter2)

        self.assertIs(pipe[0],source)
        self.assertIs(pipe[1],filter1)
        self.assertIs(pipe[2],filter2)

        self.assertIs(pipe[-1],filter2)
        self.assertIs(pipe[-2],filter1)
        self.assertIs(pipe[-3],source)

    def test_iter(self):
        source  = ReprSource([1,2])
        filter1 = Foreach(ReprSink())
        filter2 = Foreach(ReprSink())
        pipes = list(SourceFilters(source, filter1, filter2))
        self.assertEqual(pipes, [source,filter1,filter2])

class FiltersFilter_Tests(unittest.TestCase):

    def test_init_filters(self):

        filter = FiltersFilter(ReprFilter("1"), ReprFilter("2"))

        self.assertEqual(2, len(filter._filters))
        self.assertIsInstance(filter._filters[0], ReprFilter)
        self.assertIsInstance(filter._filters[1], ReprFilter)
        self.assertEqual("ReprFilter1,ReprFilter2", str(filter))

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

    def test_params(self):
        source = FiltersFilter(NoParamsFilter(), NoParamsFilter())
        self.assertEqual({}, source.params)

        source = FiltersFilter(NoParamsFilter(), ParamsFilter())
        self.assertEqual({'filter':'ParamsFilter'}, source.params)

        source = FiltersFilter(ParamsFilter(), NoParamsFilter())
        self.assertEqual({'filter':'ParamsFilter'}, source.params)

        source = FiltersFilter(ParamsFilter(), ParamsFilter())
        self.assertEqual({'filter1':'ParamsFilter','filter2':'ParamsFilter'}, source.params)

    def test_len(self):
        self.assertEqual(2,len(FiltersFilter(ReprFilter("1"), ReprFilter("2"))))

    def test_getitem(self):
        filter1 = ReprFilter("1")
        filter2 = ReprFilter("2")

        pipe = FiltersFilter(filter1, filter2)

        self.assertIs(pipe[0],filter1)
        self.assertIs(pipe[1],filter2)

        self.assertIs(pipe[-1],filter2)
        self.assertIs(pipe[-2],filter1)

    def test_iter(self):
        filter1 = ReprFilter("1")
        filter2 = ReprFilter("2")

        pipes = list(FiltersFilter(filter1, filter2))
        self.assertEqual(pipes, [filter1,filter2])

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

class SourceSink_Tests(unittest.TestCase):

    def test_run_source_sink(self):
        source = ReprSource([1,2])
        sink   = ReprSink()

        pipeline = SourceSink(source, Foreach(sink))
        pipeline.run()

        self.assertEqual([1,2], sink.items)
        self.assertEqual("ReprSource,ReprSink", str(pipeline))
        self.assertEqual({}, pipeline.params)

    def test_run_source_filters_sink(self):
        source = ReprSource([1,2])
        sink   = ReprSink()

        pipeline = SourceSink(source, ReprFilter(), ReprFilter(), Foreach(sink))
        pipeline.run()

        self.assertEqual([1,2], sink.items)
        self.assertEqual("ReprSource,ReprFilter,ReprFilter,ReprSink", str(pipeline))
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