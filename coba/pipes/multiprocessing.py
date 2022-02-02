
import time
import traceback
import pickle
import inspect
import collections.abc

from itertools import islice
from multiprocessing import current_process, Process, Queue
from threading import Thread
from typing import Iterable, Any, List, Optional
from coba.exceptions import CobaException

from coba.pipes.core       import Pipe, Foreach
from coba.pipes.primitives import Filter, Source
from coba.pipes.io         import Sink, QueueIO, ConsoleIO

# handle not picklable (this is handled by explicitly pickling)    (TESTED)
# handle empty list (this is done by PipesPool naturally) (TESTED)
# handle exceptions in process (wrap worker executing code in an exception handler) (TESTED)
# handle ctrl-c without hanging 
#   > This is done by making PipesPool terminate inside its ContextManager.__exit__
#   > This is also done by handling EOFError,BrokenPipeError in QueueIO since ctr-c kills multiprocessing.Pipe
# handle AttributeErrors. This occurs when... (this is handled PipePools.worker ) (TESTED)
#   > a class that is defined in a Jupyter Notebook cell is pickled
#   > a class that is defined inside the __name__=='__main__' block is pickled
# handle Experiment.evaluate not being called inside of __name__=='__main__' (this is handled by a big try/catch)

class PipesPool:
    # Writing our own multiprocessing pool probably seems a little silly.
    # However, Python's multiprocessing.Pool does a very poor job handling errors
    # and it often gets stuck in unrecoverable states. Given that this package is
    # meant to be used by a general audience who will likely have errors that need
    # to be debugged as they learn how to use coba this was unacepptable. Therefore,
    # after countless attempts to make multiprocessing.Pool work the decision was made
    # to write our own so we could add our own helpful error messages.
 
    def __enter__(self) -> 'PipesPool':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.terminate() 

    def __init__(self, n_processes: int, maxtasksperchild: Optional[int], stderr: Sink):

        self._n_processes = n_processes
        self._maxtasksperchild = maxtasksperchild or None
        self._given_stderr = stderr

        self._stdin  = None
        self._stderr = None
        self._stdout = None

    def map(self, filter: Filter[Any, Any], items:Iterable[Any]) -> Iterable[Any]:

        self._stdin  = QueueIO(Queue(maxsize=self._n_processes))
        self._stdout = QueueIO(Queue())
        self._stderr = QueueIO(Queue())

        # Without this multiprocessing.Queue() will output an ugly error message if a user ever hits ctrl-c.
        # By setting _ignore_epipe we prevent Queue() from displaying its message and we show our own friendly
        # message instead. In future versions of Python this could break but for now this works for 3.6-3.10.
        self._stdin ._queue._ignore_epipe = True
        self._stdout._queue._ignore_epipe = True
        self._stderr._queue._ignore_epipe = True

        self._threads = []

        self._completed = False
        self._terminate = False

        self._pool: List[Process] = []

        self._no_more_items = False

        def maintain_pool():

            finished = lambda: self._completed and (len(self._stdin) == 0 or self._terminate)

            while not finished():

                if self._terminate: 
                    break

                self._pool = [p for p in self._pool if p.is_alive()]

                for _ in range(self._n_processes-len(self._pool)):
                    args = (filter, self._stdin, self._stdout, self._stderr, self._maxtasksperchild)
                    process = Process(target=PipesPool.worker, args=args)
                    process.start()
                    self._pool.append(process)

                #I don't like this but it seems to be 
                #the fastest/simplest way out of all my tests...
                time.sleep(0.1) 

            if not self._terminate:
                for _ in self._pool: self._stdin.write(None)
            else:
                for p in self._pool: p.terminate()

            for p in self._pool: p.join()
            self._stderr.write(None)
            self._stdout.write(None)

        def populate_tasks():

            try:
                for item in items:

                    if self._terminate: break

                    try:

                        self._stdin.write(pickle.dumps(item))

                    except Exception as e:
                        if "pickle" in str(e) or "Pickling" in str(e):

                            message = str(e) if isinstance(e,CobaException) else (
                                str(e) + ". We attempted to process your code on multiple processes but the named class could not "
                                "be pickled. This problem can be fixed in one of two ways: 1) evaluate the experiment in question "
                                "on a single process with no limit on the tasks per child or 2) modify the named class to be "
                                "picklable. The easiest way to make a given class picklable is to add `def __reduce__ (self) return "
                                "(<the class in question>, (<tuple of constructor arguments>))` to the class. For more information "
                                "see https://docs.python.org/3/library/pickle.html#object.__reduce__."
                            )

                            self._stderr.write(message)

                            # I'm not sure what I think about this...
                            # It means pipes stops after a pickle error...
                            # This is how it has worked for a long time
                            # So we're leaving it as is for now...
                            break
                        else: #pragma: no cover
                            self._stderr.write((time.time(), current_process().name, e, traceback.format_tb(e.__traceback__)))
            except Exception as e:
                self._stderr.write((time.time(), current_process().name, e, traceback.format_tb(e.__traceback__)))

            self._completed = True

        log_thread = Thread(target=Pipe.join(self._stderr, [], Foreach(self._given_stderr)).run)
        log_thread.daemon = True
        log_thread.start()

        pool_thread = Thread(target=maintain_pool)
        pool_thread.daemon = True
        pool_thread.start()

        tasks_thread = Thread(target=populate_tasks)
        tasks_thread.daemon = True
        tasks_thread.start()

        self._threads.append(log_thread)
        self._threads.append(pool_thread)
        self._threads.append(tasks_thread)

        for item in self._stdout.read():
            yield item

    def close(self):

        while self._threads:
            self._threads.pop().join()

        if self._stdin : self._stdin._queue .close()
        if self._stdout: self._stdout._queue.close()
        if self._stderr: self._stderr._queue.close()

    def terminate(self):
        self._terminate = True

        if len(self._threads) > 2:
            self._threads[1].join()

    @property
    def is_terminated(self) -> bool:
        return self._terminate

    @staticmethod
    def worker(filter: Filter[Any,Any], stdin: Source, stdout: Sink, stderr: Sink, maxtasksperchild: Optional[int]):
        try:

            for item in islice(map(pickle.loads,stdin.read()),maxtasksperchild):
                result = filter.filter(item)

                #This is a bit of a hack primarily put in place to deal with
                #CobaMultiprocessing that performs coba logging of exceptions.
                #An alternative solution would be to raise a coba exception 
                #full logging decorators in the exception message. 
                if result is None: continue

                if inspect.isgenerator(result) or isinstance(result, collections.abc.Iterator):
                    stdout.write(list(result))
                else:
                    stdout.write(result)

        except Exception as e:

                if str(e).startswith("Can't get attribute"):

                    message = (
                        "We attempted to evaluate your code in multiple processes but we were unable to find all the code "
                        "definitions needed to pass the tasks to the processes. The two most common causes of this error are: "
                        "1) a learner or simulation is defined in a Jupyter Notebook cell or 2) a necessary class definition "
                        "exists inside the `__name__=='__main__'` code block in the main execution script. In either case "
                        "you can choose one of two simple solutions: 1) evaluate your code in a single process with no limit "
                        "child tasks or 2) define all necessary classes in a separate file and include the classes via import "
                        "statements."                                    
                    )

                    stderr.write(message)

                else:
                    #WARNING: this will scrub e of its traceback which is why the traceback is also sent as a string
                    stderr.write((time.time(), current_process().name, e, traceback.format_tb(e.__traceback__)))

        except KeyboardInterrupt:
            #When ctrl-c is pressed on the keyboard KeyboardInterrupt is raised in each
            #process. We need to handle this here because Processor is always run in a
            #background process and receives this. We can ignore this because the exception will
            #also be raised in our main process. Therefore we simply ignore and trust the main to
            #handle the keyboard interrupt correctly.
            pass

class PipeMultiprocessor(Filter[Iterable[Any], Iterable[Any]]):

    def __init__(self,
        filter: Filter[Any, Any],
        n_processes: int = 1,
        maxtasksperchild: int = 0,
        stderr: Sink = ConsoleIO(),
        chunked: bool = True) -> None:

        self._filter           = filter
        self._n_processes      = n_processes
        self._maxtasksperchild = maxtasksperchild
        self._stderr           = stderr
        self._chunked          = chunked

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:

        with PipesPool(self._n_processes, self._maxtasksperchild, self._stderr) as pool:
            for item in pool.map(self._filter, items):
                if self._chunked:
                    for inner_item in item: 
                        yield inner_item
                else:
                    yield item