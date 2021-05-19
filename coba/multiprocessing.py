from multiprocessing import Manager, Pool
from threading       import Thread
from typing          import Sequence, Iterable, Any

from coba.utilities  import CobaConfig, IndentLogger
from coba.pipes      import Filter, Sink, Pipe, StopPipe, QueueSource, QueueSink

class MultiprocessFilter(Filter[Iterable[Any], Iterable[Any]]):

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
            processor  = MultiprocessFilter.Processor(self._filters, stdout_writer, stderr_writer, stdlog_writer)

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
