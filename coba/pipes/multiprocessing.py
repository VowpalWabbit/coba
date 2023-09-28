import pickle
import importlib.util
import threading as mt
import multiprocessing as mp

from collections.abc import Iterator
from queue import Empty
from traceback import format_tb
from typing import Iterable, Mapping, Callable, Optional, Union, Sequence, Any, Literal

from coba.utilities import peek_first
from coba.exceptions import CobaException
from coba.pipes.primitives import Filter, Line, SourceSink
from coba.pipes.filters import Slice
from coba.pipes.sources import IterableSource, QueueSource
from coba.pipes.sinks import QueueSink

# handle not picklable (this is handled by explicitly pickling) (UNITTESTED)
# handle empty list (this is done  naturally) (UNITTESTED)
# handle exceptions in process (wrap worker executing code in an exception handler) (UNITTESTED)
# handle AttributeErrors. This occurs when... (handled by Pickler) (UNITTESTED)

# handle ctrl-c without hanging (manually tested)
#   > This is super hard... I've been trying to do this for years... here are things I've done
#   > I make all threads and processes daemons to protect against them not closing
#   > When I'm done I shutdown in-queue loading and then empty both the in-queue and out-queue
#   > I use try-finally idioms to close all queues that we create
#   > I assert that all processes are in fact closed when passed to the callback

# handle Experiment.evaluate not being called inside of __name__=='__main__' (manually tested)
#   > Handled by checking exitcode on processes and using an event synchronizer on first process start

# There are three potential contexts -- spawn, fork, and forkserver. On Windows and Mac spawn is the only option.
# On Linux the default option is fork. Fork creates processes faster and can share memory but also doesn't play
# well with threads. Therefore, to make behavior consistent across Linux, Windows, and Mac -- and to avoid
# potential bugs -- we force our multiprocessors to always use spawn regardless of the host OS.
spawn_context = mp.get_context("spawn")

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
        callback               = self._callback
        self._callback         = None
        self._recv, self._send = spawn_context.Pipe(False)
        self._lock             = spawn_context.Lock()

        super().start()

        if callback:
            def join_and_call(self=self):
                self.join()
                callback(self)
            mt.Thread(target=join_and_call,daemon=True).start()

    def run(self): #pragma: no cover (coverage can't be tracked for code that runs on background prcesses)
        try:
            self._line.run()
        except Exception as e:
            if str(e).startswith("Can't get attribute"):
                ex,tb = CobaException(
                    "We attempted to evaluate your code in multiple processes but we were unable to find all the code "
                    "definitions needed to pass the tasks to the processes. The two most common causes of this error are: "
                    "1) a learner or simulation is defined in a Jupyter Notebook cell or 2) a necessary class definition "
                    "exists inside the `__name__=='__main__'` code block in the main execution script. In either case you "
                    "can choose one of four simple solutions: 1) pip install cloudpickle into your python environment, 2) "
                    "evaluate your code on a single process, 3) if in Jupyter notebook define all necessary classes in a "
                    "separate file and include the classes via import statements, or 4) move your class definitions outside "
                    "the `__name__ == '__main__'` check."
                ),None
            else:
                ex,tb = e,format_tb(e.__traceback__)
        except KeyboardInterrupt as e:
            ex,tb = e,None
        else:
            ex,tb = None,None

        self._send.send((ex, tb, hasattr(self._line[0],'_poisoned') and self._line[0]._poisoned))

    def join(self) -> None:
        super().join()
        self._get_result()

    @property
    def pipeline(self) -> Line:
        return self._line

    @property
    def traceback(self) -> Sequence[str]:
        return self._traceback

    @property
    def exception(self) -> Optional[Exception]:
        return self._exception

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def _get_result(self):
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
                self._callback(self)
            mt.Thread(target=join_and_call,daemon=True).start()

    def run(self) -> None:
        try:
            self._line.run()
        except Exception as e:
            self._exception = e
            self._traceback = format_tb(e.__traceback__)
        self._poisoned  = hasattr(self._line[0],'_poisoned') and self._line[0]._poisoned

    @property
    def pipeline(self) -> Line:
        return self._line

    @property
    def traceback(self) -> Sequence[str]:
        return self._traceback

    @property
    def exception(self) -> Optional[Exception]:
        return self._exception

    @property
    def poisoned(self) -> bool:
        return self._poisoned

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

class Safe:
    def __init__(self, filter: Filter):
        if importlib.util.find_spec('cloudpickle'):
            import cloudpickle
            try:
                self._filter = cloudpickle.dumps(filter)
            except:
                self._filter = filter
        else:
            self._filter = filter

    def filter(self,items):
        if importlib.util.find_spec('cloudpickle') and isinstance(self._filter,bytes):
            import cloudpickle
            filter = cloudpickle.loads(self._filter)
        else:
            filter = self._filter

        yield from filter.filter(items)

class Foreach:
    def __init__(self,pipe:Filter) -> None:
        self._pipe = pipe

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        for item in items:
            out = self._pipe.filter(item)
            out = out if isinstance(out,Iterator) else [out]
            yield from out

class Pickler:
    def filter(self, items) -> Iterable[bytes]:
        try:
            if importlib.util.find_spec('cloudpickle'):
                import cloudpickle
                yield from map(cloudpickle.dumps,items)
            else:
                yield from map(pickle.dumps,items)
        except Exception as e:
            if "pickle" in str(e) or "Pickling" in str(e):
                message = (
                    f"We attempted to process your code on multiple processes but were unable to do so due to a pickle "
                    f"error. The exact error received was '{str(e)}'. Errors of this kind can often be fixed in one of three "
                    f"ways: 1) pip install cloudpickle in your conda environment, 2) evaluate the experiment in question on "
                    f"a single process with no limit on the tasks per child or 3) modify the named class to be picklable. "
                    f"The easiest way to make a given class picklable is to add `def __reduce__(self): return (<the class "
                    f"in question>, (<tuple of constructor arguments>))` to the class. For more information see "
                    f"https://docs.python.org/3/library/pickle.html#object.__reduce__."
                )
                raise CobaException(message)
            else: #pragma: no cover
                raise

class Unpickler:
    def filter(self, items):
        if importlib.util.find_spec('cloudpickle'):
            import cloudpickle
            yield from map(cloudpickle.loads,items)
        else:
            yield from map(pickle.loads,items)

class EventSetter:
    def __init__(self, event: mt.Event) -> None:
        self._event = event
    def filter(self, items):
        self._event.set()
        return items

class Stopper:
    def __init__(self) -> None:
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def filter(self, items):
        for item in items:
            if self._stop: break
            yield item

class Multiprocessor(Filter[Iterable[Any], Iterable[Any]]):
    """Create multiple processes to filter given items."""

    def __init__(self,
            filter: Filter[Iterable[Any], Iterable[Any]],
            n_processes: int = 1,
            maxtasksperchild: int = 0) -> None:
        """Instantiate a Multiprocessor.

        Args:
            filter: The inner pipe that will be executed on multiple processes.
            n_processes: The number of processes that should be created to filter items.
            maxtasksperchild: The number of items/chunks a process should filter before restarting.
        """
        self._filter           = filter
        self._max_processes    = n_processes
        self._maxtasksperchild = maxtasksperchild or None

    @property
    def params(self) -> Mapping[str,Any]:
        return self._filter.params

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:

        items = peek_first(items)[1]
        if not items: return []

        if self._max_processes == 1 and self._maxtasksperchild is None:
            yield from Foreach(self._filter).filter(items)

        else:
            event = spawn_context.Event()

            #for some reason if this mp queue get too big we can't keyboradinterrupt
            #therefore, we slightly limit its size and then empty it before closing.
            in_queue  = spawn_context.Queue(maxsize=self._max_processes*2)
            out_queue = spawn_context.Queue()
            in_put    = QueueSink(in_queue,foreach=True)
            in_get    = QueueSource(in_queue)
            out_put   = QueueSink(out_queue,foreach=True)
            out_get   = QueueSource(out_queue)
            pickler   = Pickler()
            unpickler = Unpickler()
            get_max   = Slice(None,self._maxtasksperchild)
            setter    = EventSetter(event)

            self._n_procs      = self._max_processes
            self._exceptions   = []
            self._poison       = None
            self._main_err     = False
            self._load_stopper = Stopper() #this works because the loader is a thread which means we have shared memory

            load_line   = SourceSink(IterableSource(items), self._load_stopper, pickler, in_put)
            filter_line = SourceSink(in_get, setter, unpickler, get_max, Safe(Foreach(self._filter)), out_put)

            def loader_finished_or_failed(worker: Union[ThreadLine,ProcessLine]):
                if worker.exception: self._exceptions.append(worker.exception)
                in_put.write(self._load_stopper.filter([self._poison]*self._n_procs))

            def filter_finished_or_failed(worker: Union[ThreadLine,ProcessLine]):
                if worker.exception: self._exceptions.append(worker.exception)

                assert not worker.is_alive()

                #only known cause of exitcode != 0 is a missing `if __name__ == '__main__'``.
                if worker.exitcode != 0: #pragma: no cover
                    #exitcode -15 is keyboard interrupt...
                    if worker.exitcode != -15:
                        print(worker.exitcode)
                    self._main_err = True
                    event.set()

                #we have to stop on exception since, depending on where the exception occurred,
                #we may not have actually read anything from the input queue. If we didn't then
                #the input queue will never empty and we'll be stuck starting processes forever.
                if not worker.poisoned and not self._exceptions and worker.exitcode == 0:
                    ProcessLine(worker.pipeline,callback=filter_finished_or_failed).start()
                else:
                    self._n_procs -= 1
                    if self._n_procs == 0:
                        try:
                            out_put.write([self._poison])
                        except ValueError: #pragma: no cover
                            pass

            load_thread = ThreadLine(load_line, callback=loader_finished_or_failed)
            filt_procs  = [ProcessLine(filter_line, callback=filter_finished_or_failed) for _ in range(self._n_procs)]

            try:
                load_thread.start()
                filt_procs.pop().start()

                #by waiting we can avoid throwing multiple exceptions
                #when there is a problem with starting a new process
                event.wait()
                if not self._main_err:
                    for p in filt_procs: p.start()
                    yield from out_get.read()

            finally:

                #stop loading into the input queue
                self._load_stopper.stop()

                #empty the input queue and then close it
                #if we don't empty first then we can easily
                #lock during a keyboard interrupt
                try:
                    while True: in_queue.get_nowait()
                except Empty:
                    pass
                    #closing can cause exceptions
                    #and doesn't seem to help anything
                    #in_queue.close()

                #empty the input queue and then close it
                #if we don't empty first then we can easily
                #lock during a keyboard interrupt
                try:
                    while True: out_queue.get_nowait()
                except Empty:
                    pass
                    #closing can cause exceptions
                    #and doesn't seem to help anything
                    #out_queue.close()

            if self._exceptions:
                raise self._exceptions[0]
