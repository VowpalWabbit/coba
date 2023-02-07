import pickle
import threading as mt
import multiprocessing as mp

from itertools import islice, chain
from traceback import format_tb
from typing import Iterable, Mapping, Callable, Optional, Union, Sequence, Any
from coba.backports import Literal

from coba.utilities import peek_first
from coba.exceptions import CobaException
from coba.pipes.primitives import Filter, Line, SourceSink
from coba.pipes.filters import Slice
from coba.pipes.sources import IterableSource, QueueSource
from coba.pipes.sinks import QueueSink

# handle not picklable (this is handled by explicitly pickling) (UNITTESTED)
# handle empty list (this is done  naturally) (UNITTESTED)
# handle exceptions in process (wrap worker executing code in an exception handler) (UNITTESTED)
# handle ctrl-c without hanging
#   > This is done by making PipesPool terminate inside its ContextManager.__exit__
#   > This is also done by handling EOFError,BrokenPipeError in QueueIO since ctr-c kills multiprocessing.Pipe
# handle AttributeErrors. This occurs when... (this is handled PipePools.worker ) (UNITTESTED)
#   > a class that is defined in a Jupyter Notebook cell is pickled
#   > a class that is defined inside the __name__=='__main__' block is pickled
# handle Experiment.evaluate not being called inside of __name__=='__main__' (this is handled by a big try/catch)

#There are three potential contexts -- spawn, fork, and forkserver. On windows and mac spawn is the only option.
#On Linux the default option is fork. Fork creates processes faster and can share memory but also doesn't play
#well with threads. Therefore, to make behavior consistent across Linux, Windows, and Mac and avoid potential bugs
#we force our multiprocessors to always use spawn.
spawn_context = mp.get_context("spawn")

class MultiException(Exception):
    def __init__(self, exceptions: Sequence[Exception]):
        self.exceptions = exceptions

class ProcessLine(spawn_context.Process):

    ### We create a lock so that we can safely receive any possible exceptions. Empirical
    ### tests showed that creating a Pipe and Lock doesn't seem to slow us down too much.

    def __init__(self, line: Line, callback: Callable = None):

        self._line      = line
        self._callback  = callback
        self._exception = None
        self._traceback = None
        self._poisoned  = False

        super().__init__(daemon=True)

    def start(self) -> None:
        callback = self._callback

        self._callback         = None
        self._recv, self._send = spawn_context.Pipe(False)
        self._lock             = spawn_context.Lock()

        super().start()

        if callback:
            def join_and_call(self=self):
                self.join()
                callback(self._line, self._exception,self._traceback, self._poisoned)
            mt.Thread(target=join_and_call,daemon=True).start()

    def run(self):#pragma: no cover (coverage can't be tracked for code that runs on background prcesses)
        try:
            self._line.run()
        except Exception as e:
            if str(e).startswith("Can't get attribute"):

                message = (
                    "We attempted to evaluate your code in multiple processes but we were unable to find all the code "
                    "definitions needed to pass the tasks to the processes. The two most common causes of this error are: "
                    "1) a learner or simulation is defined in a Jupyter Notebook cell or 2) a necessary class definition "
                    "exists inside the `__name__=='__main__'` code block in the main execution script. In either case "
                    "you can choose one of three simple solutions: 1) evaluate your code on a single process with no limit on "
                    "child tasks, 2) if in Jupyter notebook define all necessary classes in a separate file and include the "
                    "classes via import statements, or 3) move your class definitions outside the `__name__=='__main__'` check."
                )

                ex,tb = CobaException(message),None
            else:
                ex,tb = e,format_tb(e.__traceback__)
        except KeyboardInterrupt as e:
            ex,tb = e,None
        else:
            ex,tb = None,None
        
        self._send.send((ex, tb, hasattr(self._line[0],'_poisoned') and self._line[0]._poisoned))

    def join(self) -> None:
        super().join()
        self._get_ex()

    @property
    def exception(self) -> Optional[Exception]:
        return self._exception

    def _get_ex(self):
        with self._lock:
            if not self._recv.closed:
                if self._recv.poll():
                    ex,tb,po = self._recv.recv()
                    self._exception = ex
                    self._traceback = tb
                    self._poisoned  = po
                self._send.close()
                self._recv.close()

class ThreadLine(mt.Thread):

    def __init__(self, line: Line, callback: Callable = None) -> None:
        self._line      = line
        self._callback  = callback
        self._exception = None
        self._traceback = None
        self._poisoned  = False
        super().__init__(daemon=True)

    def start(self) -> None:
        
        super().start()

        if self._callback:
            #we start a thread and join so that the callback
            #is not called until the thread is no longer alive
            def join_and_call():
                self.join()
                self._callback(self._line, self._exception, self._traceback, self._poisoned)
            mt.Thread(target=join_and_call,daemon=True).start()

    def run(self) -> None:
        try:
            self._line.run()
        except Exception as e:
            self._exception = e
            self._traceback = format_tb(e.__traceback__)
        self._poisoned  = hasattr(self._line[0],'_poisoned') and self._line[0]._poisoned

    @property
    def exception(self) -> Optional[Exception]:
        return self._exception

class AsyncableLine(SourceSink):

    def run_async(self, 
        callback:Callable[['AsyncableLine',Optional[Exception],Optional[str]],None]=None,
        mode:Literal['process','thread']='process') -> Union[ThreadLine,ProcessLine]:
        """Run the pipeline asynchronously."""
        
        mode = mode.lower()

        if mode == "process":
            worker = ProcessLine(line=self, callback=callback)
        elif mode == "thread":
            worker = ThreadLine(line=self, callback=callback)
        else:
            raise CobaException(f"Unrecognized pipe async mode {mode}. Valid values are 'process' and 'thread'.")

        worker.start()
        return worker

class Unchunker:
    def __init__(self, chunked: bool) -> None:
        self._chunked = chunked
    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        if not self._chunked: items = [items]
        for item in items: yield from item

class Pickler:

    def filter(self, items) -> Iterable[bytes]:
        try:
            yield from map(pickle.dumps,items)
        except Exception as e:
            if "pickle" in str(e) or "Pickling" in str(e):
                message = (
                    f"We attempted to process your code on multiple processes but were unable to do so due to a pickle "
                    f"error. The exact error received was '{str(e)}'. Errors this kind can often be fixed in one of two "
                    f"ways: 1) evaluate the experiment in question on a single process with no limit on the tasks per child "
                    f"or 2) modify the named class to be picklable. The easiest way to make a given class picklable is to "
                    f"add `def __reduce__(self): return (<the class in question>, (<tuple of constructor arguments>))` to "
                    f"the class. For more information see https://docs.python.org/3/library/pickle.html#object.__reduce__."
                )
                raise CobaException(message)
            else: #pragma: no cover
                raise

class Unpickler:

    def filter(self, items):
        yield from map(pickle.loads,items)

class Multiprocessor(Filter[Iterable[Any], Iterable[Any]]):
    """Create multiple processes to filter given items."""

    def __init__(self,
            filter: Filter[Iterable[Any], Iterable[Any]],
            n_processes: int = 1, 
            maxtasksperchild: int = 0,
            chunked: bool = False,) -> None:
        """Instantiate a Multiprocessor.

        Args:
            filter: The inner pipe that will be executed on multiple processes.
            n_processes: The number of processes that should be created to filter items.
            maxtasksperchild: The number of items/chunks a process should filter before restarting.
            chunked: Indicates that the given items have been chunked.
        """
        self._filter           = filter
        self._chunked          = chunked
        self._n_processes      = n_processes
        self._maxtasksperchild = maxtasksperchild or None

    @property
    def params(self) -> Mapping[str,Any]:
        return self._filter.params

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:

        items = peek_first(items)[1]

        if not items: return []

        if self._n_processes == 1 and self._maxtasksperchild is None:
            yield from self._filter.filter(Unchunker(self._chunked).filter(items))

        else:
            initial_items = list(islice(items,self._n_processes))
            n_procs       = min(len(initial_items), self._n_processes)
            items         = chain(initial_items, items)

            in_queue  = spawn_context.Queue(maxsize=n_procs)
            out_queue = spawn_context.Queue()
            in_put    = QueueSink(in_queue,foreach=True)
            in_get    = QueueSource(in_queue)
            out_put   = QueueSink(out_queue,foreach=True)
            out_get   = QueueSource(out_queue)
            pickler   = Pickler()
            unpickler = Unpickler() 
            get_max   = Slice(None,self._maxtasksperchild)
            unchunk   = Unchunker(self._chunked)

            load_line   = SourceSink(IterableSource(items), pickler, in_put)
            filter_line = SourceSink(in_get, unpickler, get_max, unchunk, self._filter, out_put)

            self._n_running  = n_procs
            self._exceptions = []
            self._poison     = None

            def load_finished_or_failed(line,exception,traceback,poisoned):
                if exception: self._exceptions.append(exception)
                in_put.write([self._poison]*self._n_running)

            def filt_maxed_poisoned_or_failed(line,exception,traceback,poisoned):
                if exception: self._exceptions.append(exception)

                if not poisoned and not self._exceptions:
                    ProcessLine(line,callback=filt_maxed_poisoned_or_failed).start()
                else:
                    self._n_running -=1
                    if self._n_running == 0: 
                        out_put.write([self._poison])

            load_thread = ThreadLine(load_line, callback=load_finished_or_failed)
            filt_procs  = [ProcessLine(filter_line, callback=filt_maxed_poisoned_or_failed) for _ in range(n_procs) ]

            load_thread.start()
            for p in filt_procs: p.start()

            yield from out_get.read()

            in_queue.close()
            out_queue.close()

            self._exceptions = [e for e in self._exceptions if not isinstance(e, KeyboardInterrupt)]

            if self._exceptions:
                raise MultiException(self._exceptions) if len(self._exceptions) > 1 else self._exceptions[0]
