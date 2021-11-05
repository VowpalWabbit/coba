import time
import sys
import multiprocessing.pool
import traceback

from multiprocessing import Manager, current_process
from threading       import Thread
from typing          import Sequence, Iterable, Any

from coba.pipes.core    import Pipe, StopPipe
from coba.pipes.filters import Filter
from coba.pipes.io      import Sink, QueueIO, NullIO

super_worker = multiprocessing.pool.worker #type: ignore

def worker(inqueue, outqueue, initializer=None, initargs=(), maxtasks=None, wrap_exception=False):
        try:
            super_worker(inqueue, outqueue, initializer, initargs, maxtasks, wrap_exception)
        except KeyboardInterrupt:
            #we handle this exception because otherwise it is thrown and written to console
            #by handling it ourself we can prevent it from being written to console
            sys.exit(2000)
        except AttributeError:
            #we handle this exception because otherwise it is thrown and written to console
            #by handling it ourself we can prevent it from being written to console
            sys.exit(1000) #this is the exitcode we use to indicate when we're exiting due to import errors

multiprocessing.pool.worker = worker #type: ignore

class MultiprocessFilter(Filter[Iterable[Any], Iterable[Any]]):

    class Processor:

        def __init__(self, filters: Sequence[Filter], stdout: Sink, stderr: Sink) -> None:
            self._filter = Pipe.join(filters)
            self._stdout = stdout
            self._stderr = stderr

        def process(self, item) -> None:

            try:
                self._stdout.write(self._filter.filter([item]))

            except StopPipe:
                pass

            except Exception as e:
                #this will scrub the exception of the traceback... A more advanced
                #method needs to be developed in order to make sure traceback is 
                self._stderr.write([(time.time(), current_process().name, e, traceback.format_tb(e.__traceback__))])

            except KeyboardInterrupt:
                #When ctrl-c is pressed on the keyboard KeyboardInterrupt is raised in each
                #process. We need to handle this here because Processor is always ran in a
                #background process and receives this. We can ignore this because the exception will
                #also be raised in our main process. Therefore we simply ignore and trust the main to
                #handle the keyboard interrupt gracefully.
                pass

    def __init__(self, filters: Sequence[Filter], processes: int = 1, maxtasksperchild: int = None, stderr: Sink = NullIO()) -> None:
        self._filters          = filters
        self._processes        = processes
        self._maxtasksperchild = maxtasksperchild
        self._stderr           = stderr or NullIO()

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:

        if len(self._filters) == 0:
            return items

        with Manager() as manager:

            stdout_IO = QueueIO(manager.Queue()) #type: ignore
            stderr_IO = QueueIO(manager.Queue()) #type: ignore

            class MyPool(multiprocessing.pool.Pool):

                _missing_error_definition_error_is_new = True

                def _join_exited_workers(self):

                    for worker in self._pool:
                        if worker.exitcode == 1000 and MyPool._missing_error_definition_error_is_new:

                                #this is a hack... This only works so long as we just 
                                #process one pipe at a time... This is true in our case.
                                #this is necessary because multiprocessing can get stuck 
                                #waiting for failed workers and that is frustrating for users.
                                MyPool._missing_error_definition_error_is_new = False

                                message = (
                                    "We attempted to evaluate your code in multiple processes but we were unable to find all the code "
                                    "definitions needed to pass the tasks to the processes. The two most common causes of this error are: "
                                    "1) a learner or simulation is defined in a Jupyter Notebook cell or 2) a necessary class definition "
                                    "exists inside the `__name__=='__main__'` code block in the main execution script. In either case "
                                    "you can choose one of two simple solutions: 1) evaluate your code in a single process with no limit "
                                    "child tasks or 2) define all necessary classes in a separate file and include the classes via import "
                                    "statements."                                    
                                )

                                stderr_IO.write(message)

                        if worker.exitcode is not None and worker.exitcode != 0:
                            #A worker exited in an uncontrolled manner and was unable to clean its job
                            #up. We therefore mark one of the jobs as "finished" but failed in order to 
                            #prevent waiting forever on a failed job that is actually no longer running.
                            list(self._cache.values())[0]._set(None, (False, None))

                    return super()._join_exited_workers()

            with MyPool(self._processes, maxtasksperchild=self._maxtasksperchild) as pool:

                # handle not picklable (this is handled by done_or_failed)    (TESTED)
                # handle empty list (this is done by checking result.ready()) (TESTED)
                # handle exceptions in process (unhandled exceptions can cause children to hang so we pass them to stderr) (TESTED)
                # handle ctrl-c without hanging 
                #   > don't call result.get when KeyboardInterrupt has been hit
                #   > handle EOFError,BrokenPipeError errors with queue since ctr-c kills manager
                # handle AttributeErrors. These occur when... (this is handled by shadowing several pool methods) (TESTED)
                #   > a class that is defined in a Jupyter Notebook cell is pickled
                #   > a class that is defined inside the __name__=='__main__' block is pickeled
                # handle Benchmark.evaluate not being called inside of __name__=='__main__' (this is handled by a big try/catch)

                def done_or_failed(results_or_exception=None):
                    #This method is called one time at the completion of map_async
                    #in the case that one of our jobs threw an exception the argument
                    #will contain an exception otherwise it will be the returned results
                    #of all the jobs. This method is executed on a thread in the Main context.

                    if isinstance(results_or_exception, Exception):
                        if "Can't pickle" in str(results_or_exception) or "Pickling" in str(results_or_exception):

                            message = (
                                str(results_or_exception) + ". We attempted to process your code on multiple processes and "
                                "the named class was not able to be pickled. This problem can be fixed in one of two ways: 1) "
                                "evaluate the benchmark in question on a single process with no limit on the tasks per child or 2) "
                                "modify the named class to be picklable. The easiest way to make the given class picklable is to "
                                "add `def __reduce__ (self) return (<the class in question>, (<tuple of constructor arguments>))` to "
                                "the class. For more information see https://docs.python.org/3/library/pickle.html#object.__reduce__."
                            )
                            stderr_IO.write(message)
                        else:
                            stderr_IO.write(results_or_exception)

                    stdout_IO.write(None)
                    stderr_IO.write(None)

                log_thread = Thread(target=Pipe.join(stderr_IO, [], self._stderr).run)
                log_thread.daemon = True
                log_thread.start()

                processor = MultiprocessFilter.Processor(self._filters, stdout_IO, stderr_IO)
                result    = pool.map_async(processor.process, items, callback=done_or_failed, error_callback=done_or_failed, chunksize=1)

                # When items is empty finished_callback will not be called and we'll get stuck waiting for the poison pill.
                # When items is empty ready() will be true immediately and this check will place the poison pill into the queues.
                if result.ready(): done_or_failed()

                try:
                    for item in stdout_IO.read():
                        yield item
                    pool.close()
                except (KeyboardInterrupt, Exception):
                    try:
                        pool.terminate()
                    except:
                        pass
                    raise
                finally:
                    pool.join()