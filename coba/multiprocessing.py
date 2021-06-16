from multiprocessing import Manager, Pool
from threading       import Thread
from typing          import Sequence, Iterable, Any

from coba.config import CobaConfig, IndentLogger
from coba.pipes  import Filter, Sink, Pipe, StopPipe, QueueSource, QueueSink

class MultiprocessFilter(Filter[Iterable[Any], Iterable[Any]]):

    class Processor:

        def __init__(self, filters: Sequence[Filter], stdout: Sink, stderr: Sink, stdlog:Sink, n_proc:int) -> None:
            self._filter = Pipe.join(filters)
            self._stdout = stdout
            self._stderr = stderr
            self._stdlog = stdlog
            self._n_proc = n_proc

        def process(self, item) -> None:
            
            #One problem with this is that the settings on the main thread's logger 
            #aren't propogated to this logger. For example, with_stamp and with_name.
            #A possible solution is to deep copy the CobaConfig.Logger, set its `sink`
            #property to the `stdlog` and then pass it to `Processor.__init__`.
            CobaConfig.Logger = IndentLogger(self._stdlog, with_name=self._n_proc > 1)

            try:
                self._stdout.write(self._filter.filter([item]))

            except Exception as e:
                self._stderr.write([e])

            except KeyboardInterrupt:
                #When ctrl-c is pressed on the keyboard KeyboardInterrupt is raised in each
                #process. We need to handle this here because Processor is always ran in a
                #background process and receives this. We can ignore this because the exception will
                #also be raised in our main process. Therefore we simply ignore and trust the main to
                #handle the keyboard interrupt gracefully.
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

            stdout_queue = manager.Queue() #type: ignore
            stderr_queue = manager.Queue() #type: ignore
            stdlog_queue = manager.Queue() #type: ignore

            stdout_writer, stdout_reader = QueueSink(stdout_queue), QueueSource(stdout_queue)
            stderr_writer, stderr_reader = QueueSink(stderr_queue), QueueSource(stderr_queue)
            stdlog_writer, stdlog_reader = QueueSink(stdlog_queue), QueueSource(stdlog_queue)

            # handle not picklable (this is handled by done_or_failed)
            # handle empty list (this is done by checking result.ready())
            # handle exceptions in process (unhandled exceptions can cause children to hang so we pass them to stderr)
            # handle ctrl-c without hanging 
            #   > don't call result.get when KeyboardInterrupt has been hit
            #   > handle EOFError,BrokenPipeError errors with queue since ctr-c kills manager

            def done_or_failed(a=None):

                if isinstance(a, Exception):
                    stderr_writer.write([a])

                stdout_writer.write([None])
                stdlog_writer.write([None])
                stderr_writer.write([None])

            log_thread = Thread(target=Pipe.join(stdlog_reader, [], CobaConfig.Logger.sink).run)
            log_thread.daemon = True
            log_thread.start()

            processor = MultiprocessFilter.Processor(self._filters, stdout_writer, stderr_writer, stdlog_writer, self._processes)
            result    = pool.map_async(processor.process, items, callback=done_or_failed, error_callback=done_or_failed, chunksize=1)

            # When items is empty finished_callback will not be called and we'll get stuck waiting for the poison pill.
            # When items is empty ready() will be true immediately and this check will place the poison pill into the queues.
            if result.ready(): done_or_failed()

            try:
                for item in stdout_reader.read():
                    yield item
            except KeyboardInterrupt:
                try:
                    pool.terminate()
                except:
                    pass
                log_thread.join()
                raise
            else:

                pool.close()
                pool.join()
                log_thread.join()

                for err in stderr_reader.read():

                    if isinstance(err, StopPipe):
                        continue

                    elif "Can't pickle" in str(err) or "Pickling" in str(err):
                        message = (
                            str(err) + ". Learners must be picklable to evaluate a Learner on a Benchmark in multiple processes. "
                            "To make a currently unpiclable learner picklable it should implement `__reduce(self)__`. "
                            "The simplest return from reduce is `return (<the learner class>, (<tuple of constructor arguments>))`. "
                            "For more information see https://docs.python.org/3/library/pickle.html#object.__reduce.")
                        raise Exception(message)

                    else:
                        raise err