from threading import Thread
from multiprocessing import Manager, Queue, Lock, Condition, Semaphore
from typing import Iterable, Any, Dict

from coba.utilities import coba_exit
from coba.contexts  import CobaContext, ConcurrentCacher, Logger, Cacher
from coba.pipes     import Pipe, Filter, Sink, QueueIO, PipeMultiprocessor, Foreach

class CobaMultiprocessor(Filter[Iterable[Any], Iterable[Any]]):

    class PipeStderr(Sink[Any]):
        def write(self, item: Any) -> None:
            if isinstance(item,tuple):
                CobaContext.logger.log(item[2])
            else:
                CobaContext.logger.log(item)

    class ProcessFilter:

        def __init__(self, filter: Filter, logger: Logger, cacher: Cacher, store: Dict[str,Any], logger_sink: Sink) -> None:

            self._filter      = filter
            self._logger      = logger
            self._cacher      = cacher
            self._store       = store
            self._logger_sink = logger_sink

        def filter(self, item: Any) -> Any:

            #placing this here means this is set inside the process 
            CobaContext.logger = self._logger
            CobaContext.cacher = self._cacher
            CobaContext.store  = self._store

            #at this point logger has been marshalled so we can
            #modify it without affecting the base process logger
            CobaContext.logger.sink = self._logger_sink

            try:
                return self._filter.filter(item)
            except Exception as e:
                CobaContext.logger.log(e)

    def __init__(self, filter: Filter, processes=1, maxtasksperchild=0) -> None:
        self._filter           = filter
        self._processes        = processes
        self._maxtasksperchild = maxtasksperchild

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:

        try:

            with Manager() as manager:

                stdlog = QueueIO(Queue())
                stderr = CobaMultiprocessor.PipeStderr()                
                
                log_thread = Thread(target=Pipe.join(stdlog,[],Foreach(CobaContext.logger.sink)).run)
                log_thread.daemon = True
                log_thread.start()

                logger = CobaContext.logger
                cacher = ConcurrentCacher(CobaContext.cacher, manager.dict(), Lock(), Condition())
                store  = { "srcsema":  Semaphore(2) }

                filter = CobaMultiprocessor.ProcessFilter(self._filter, logger, cacher, store, stdlog)

                for item in PipeMultiprocessor(filter, self._processes, self._maxtasksperchild, stderr).filter(items):
                    yield item

                stdlog.write(None) #attempt to shutdown the logging process gracefully by sending the poison pill

        except RuntimeError as e: #pragma: no cover
            #This happens when importing main causes this code to run again
            coba_exit(str(e))
