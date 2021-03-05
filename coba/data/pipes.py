"""The data.pipes module contains core classes for creating data pipelines.

TODO: Add docstrings for Pipe
"""

import collections

from multiprocessing import Manager, Pool
from threading       import Thread
from typing          import Sequence, Iterable, Any, overload, Union

from coba.data.sources import Source, QueueSource
from coba.data.filters import Filter
from coba.data.sinks   import Sink, QueueSink
from coba.tools        import CobaConfig, IndentLogger

class StopPipe(Exception):
    pass

class Pipe:

    class FiltersFilter(Filter):
        def __init__(self, filters: Sequence[Filter]):
            self._filters = filters

        def filter(self, items: Any) -> Any:
            for filter in self._filters:
                items = filter.filter(items)
            return items

        def __repr__(self) -> str:
            return ",".join(map(str,self._filters))

    class SourceFilters(Source):
        def __init__(self, source: Source, filters: Sequence[Filter]) -> None:
            self._source = source
            self._filter = Pipe.FiltersFilter(filters)

        def read(self) -> Any:
            return self._filter.filter(self._source.read())

        def __repr__(self) -> str:
            return ",".join(map(str,[self._source, self._filter]))

    class FiltersSink(Sink):
        def __init__(self, filters: Sequence[Filter], sink: Sink) -> None:
            self._filter = Pipe.FiltersFilter(filters)
            self._sink   = sink

        def final_sink(self) -> Sink:
            if isinstance(self._sink, Pipe.FiltersSink):
                return self._sink.final_sink()
            else:
                return self._sink

        def write(self, items: Iterable[Any]):
            self._sink.write(self._filter.filter(items))

        def __repr__(self) -> str:
            return ",".join(map(str,[self._filter, self._sink]))

    @overload
    @staticmethod
    def join(source: Source, filters: Sequence[Filter]) -> Source:
        ...
    
    @overload
    @staticmethod
    def join(filters: Sequence[Filter], sink: Sink) -> Sink:
        ...

    @overload
    @staticmethod
    def join(source: Source, sink: Sink) -> 'Pipe':
        ...

    @overload
    @staticmethod
    def join(source: Source, filters: Sequence[Filter], sink: Sink) -> 'Pipe':
        ...

    @overload
    @staticmethod
    def join(filters: Sequence[Filter]) -> Filter:
        ...

    @staticmethod #type: ignore
    def join(*args) -> Union[Source, Sink, 'Pipe', Filter]:

        if len(args) == 3:
            return Pipe(*args)

        if len(args) == 2:
            if isinstance(args[1], collections.Sequence):
                return Pipe.SourceFilters(args[0], args[1])
            elif isinstance(args[0], collections.Sequence):
                return Pipe.FiltersSink(args[0], args[1])
            else:
                return Pipe(args[0], [], args[1])
        
        if len(args) == 1:
            return Pipe.FiltersFilter(args[0])

        raise Exception("An unknown pipe was joined.")

    def __init__(self, source: Source, filters: Sequence[Filter], sink: Sink) -> None:
        self._source  = source
        self._filters = filters
        self._sink    = sink

    def run(self, processes: int = 1, maxtasksperchild=None) -> None:
        try:
            if processes == 1 and maxtasksperchild is None:
                filter = Pipe.join(self._filters)
            else:
                filter = MultiProcessFilter(self._filters, processes, maxtasksperchild)

            self._sink.write(filter.filter(self._source.read()))
        except StopPipe:
            pass

    def __repr__(self) -> str:
        return ",".join(map(str,[self._source, *self._filters, self._sink]))

class MultiProcessFilter(Filter[Iterable[Any], Iterable[Any]]):

    class Processor:

        def __init__(self, filters: Sequence[Filter], stdout: Sink, stderr: Sink, stdlog:Sink) -> None:
            self._filter = Pipe.join(filters)
            self._stdout = stdout
            self._stderr = stderr
            self._stdlog = stdlog

        def process(self, item) -> None:
            
            #One problem with this is that the settings on the main thread's logger 
            #aren't propogated to this logger. For example, with_stamp and with_name.
            #A possible solution is to deep copy the CobaConfig.Logger, set its `sink`
            #property to the `stdlog` and then pass it to `Processor.__init__`.
            CobaConfig.Logger = IndentLogger(self._stdlog, with_name=True)

            try:
                self._stdout.write(self._filter.filter([item]))

            except Exception as e:
                self._stderr.write([e])
                raise

            except KeyboardInterrupt:
                # if you are here because keyboard interrupt isn't working for multiprocessing
                # or you want to improve it in some way I can only say good luck. After many hours
                # of spelunking through stackoverflow and the python stdlib I still don't understand
                # why it works the way it does. I arrived at this solution not based on understanding
                # but based on experimentation. This seemed to fix my problem. As best I can tell 
                # KeyboardInterrupt is a very special kind of exception that propogates up everywhere
                # and all we have to do in our child processes is make sure they don't become zombified.
                pass

    def __init__(self, filters: Sequence[Filter], processes=1, maxtasksperchild=None) -> None:
        self._filters          = filters
        self._processes        = processes
        self._maxtasksperchild = maxtasksperchild

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:

        # It does seem like this could potentially be made faster...
        # I'm not sure how or why, but my original thread implementation
        # within Pool seemed to complete the full job about a minute and
        # a half faster... See commit 7fb3653 for that implementation.
        # My best guess is that 7fb3653 doesn't rely on a generator.

        if len(self._filters) == 0:
            return items

        with Pool(self._processes, maxtasksperchild=self._maxtasksperchild) as pool, Manager() as manager:

            std_queue = manager.Queue() #type: ignore
            err_queue = manager.Queue() #type: ignore
            log_queue = manager.Queue() #type: ignore

            stdout_writer, stdout_reader = QueueSink(std_queue), QueueSource(std_queue)
            stderr_writer, stderr_reader = QueueSink(err_queue), QueueSource(err_queue)
            stdlog_writer, stdlog_reader = QueueSink(log_queue), QueueSource(log_queue)

            log_thread = Thread(target=Pipe.join(stdlog_reader, [], CobaConfig.Logger.sink).run)
            processor  = MultiProcessFilter.Processor(self._filters, stdout_writer, stderr_writer, stdlog_writer)

            def finished_callback(result):
                std_queue.put(None)
                err_queue.put(None)
                log_queue.put(None)

            def error_callback(error):
                std_queue.put(None) #not perfect but I'm struggling to think of a better way. May result in some lost work.
                err_queue.put(None) #not perfect but I'm struggling to think of a better way. May result in some lost work.
                log_queue.put(None) #not perfect but I'm struggling to think of a better way. May result in some lost work.

            log_thread.start()

            result = pool.map_async(processor.process, items, callback=finished_callback, error_callback=error_callback) 

            # When items is empty finished_callback will not be called and we'll get stuck waiting for the poison pill.
            # When items is empty ready() will be true immediately and this check will place the poison pill into the queues.
            if result.ready():
                std_queue.put(None)
                err_queue.put(None)
                log_queue.put(None)

            #this structure is necessary to make sure we don't exit the context before we're done
            for item in stdout_reader.read():
                yield item

            # if an error occured within map_async this will cause it to re-throw 
            # in the main thread allowing us to capture it and handle it appropriately 
            try:
                result.get()
            except Exception as e:
                if "Can't pickle" in str(e) or "Pickling" in str(e):
                    message = (
                            "Learners are required to be picklable in order to evaluate a Benchmark in multiple processes. "
                            "To help with this learner's have an optional `def init(self) -> None` that is only called "
                            "after pickling has occured. Any non-picklable objects can be created within `init()` safely.")
                    raise Exception(message) from e

            # in the case where an exception occurred in one of our processes
            # we will have poisoned the std and err queue even though the pool
            # isn't finished yet, so we need to kill it here. We are unable to
            # kill it in the error_callback because that will cause a hang.
            pool.terminate()
            log_thread.join()

            for err in stderr_reader.read():
                if not isinstance(err, StopPipe):
                    raise err