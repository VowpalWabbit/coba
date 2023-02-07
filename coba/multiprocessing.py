import multiprocessing as mp
from ctypes import c_short
from typing import Iterable, Any, Dict

from coba.utilities import coba_exit
from coba.contexts  import CobaContext, ConcurrentCacher, Logger, Cacher
from coba.pipes     import Pipes, Filter, Sink, Multiprocessor, Foreach, QueueSink, QueueSource, MultiException

class CobaMultiprocessor(Filter[Iterable[Any], Iterable[Any]]):

    class ProcessFilter:

        def __init__(self, filter: Filter, logger: Logger, cacher: Cacher, store: Dict[str,Any], logger_sink: Sink) -> None:

            self._filter      = filter
            self._logger      = logger
            self._cacher      = cacher
            self._store       = store
            self._logger_sink = logger_sink

        def filter(self, item: Any) -> Any:

            #this is set inside the process
            CobaContext.logger = self._logger
            CobaContext.cacher = self._cacher
            CobaContext.store  = self._store

            #at this point logger has been marshalled so we can
            #modify it without affecting the base process logger
            CobaContext.logger.sink = self._logger_sink

            yield from self._filter.filter(item)

    def __init__(self, filter: Filter, processes:int=1, maxtasksperchild:int=0, chunked:bool=False) -> None:
        self._filter           = filter
        self._processes        = processes
        self._maxtasksperchild = maxtasksperchild
        self._chunked          = chunked

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:

        try:
            if self._maxtasksperchild != 0 or self._processes > 1:
                spawn_context = mp.get_context("spawn")
                stdlog        = spawn_context.Queue()
                array         = spawn_context.RawArray(c_short,[0]*2**16)
                lock          = spawn_context.Lock()
                read_stdlog   = QueueSource(stdlog)
                write_stdlog  = QueueSink(stdlog)
                
                Pipes.join(read_stdlog,Foreach(CobaContext.logger.sink)).run_async(mode="thread")
                
                logger = CobaContext.logger
                cacher = ConcurrentCacher(CobaContext.cacher,array,lock)
                store  = { "openml_semaphore": mp.Semaphore(3) }
                
                filter = CobaMultiprocessor.ProcessFilter(self._filter, logger, cacher, store, write_stdlog)
            else:
                filter = self._filter

            try:
                yield from Multiprocessor(filter, self._processes, self._maxtasksperchild, self._chunked).filter(items)    
                
                # If the error was due to an uncaught exception in the given filter it could be the case that the user 
                # is expecting it therefore we don't want to supress it. On the other hand, if the error is due to the
                # act of multiprocessing then we know the user is not expecting it and we log it in a friendly way.
            except MultiException as e: #pragma: no cover
                err_due_to_multi = [ e for e in e.exceptions if     self._is_err_due_to_multiprocessing(e) ]
                err_due_to_logic = [ e for e in e.exceptions if not self._is_err_due_to_multiprocessing(e) ]
                if err_due_to_multi:
                    for e in err_due_to_multi: CobaContext.logger.log(e)
                if err_due_to_logic:
                    raise MultiException(err_due_to_logic)
            except Exception as e:
                if self._is_err_due_to_multiprocessing(e):
                    CobaContext.logger.log(e)
                else:
                    raise

            if self._maxtasksperchild != 0 or self._processes > 1:
                write_stdlog.write(None) #attempt to shutdown the logging process gracefully by sending the poison pill

        except RuntimeError as e: #pragma: no cover
            #This happens when importing main causes this code to run again
            coba_exit(str(e))

    def _is_err_due_to_multiprocessing(self, exception: Exception) -> bool:
        return 'pickle' in str(exception) or 'unable to find' in str(exception)