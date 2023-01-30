from threading import Thread
from multiprocessing import Manager, Queue, Lock, Condition, Semaphore
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

            #placing this here means this is set inside the process
            CobaContext.logger = self._logger
            CobaContext.cacher = self._cacher
            CobaContext.store  = self._store

            #at this point logger has been marshalled so we can
            #modify it without affecting the base process logger
            CobaContext.logger.sink = self._logger_sink

            try:
                yield from self._filter.filter(item)
            except Exception as e:
                CobaContext.logger.log(e)

    def __init__(self, filter: Filter, processes:int=1, maxtasksperchild:int=0, chunked:bool=False) -> None:
        self._filter           = filter
        self._processes        = processes
        self._maxtasksperchild = maxtasksperchild
        self._chunked          = chunked

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:

        try:

            with Manager() as manager:
                stdlog     = Queue()
                get_stdlog = QueueSource(stdlog)
                put_stdlog = QueueSink(stdlog)

                log_thread = Thread(target=Pipes.join(get_stdlog,Foreach(CobaContext.logger.sink)).run, daemon=True)
                log_thread.start()

                logger = CobaContext.logger
                cacher = ConcurrentCacher(CobaContext.cacher, manager.dict(), Lock(), Condition())
                store  = { "openml_semaphore": Semaphore(3) }

                filter = CobaMultiprocessor.ProcessFilter(self._filter, logger, cacher, store, put_stdlog)

                try:
                    for item in Multiprocessor(filter, self._processes, self._maxtasksperchild, self._chunked).filter(items):
                        yield item
                except MultiException as e: #pragma: no cover
                    for e in e.exceptions: CobaContext.logger.log(e)
                except Exception as e:
                    CobaContext.logger.log(e)

                put_stdlog.write(None) #attempt to shutdown the logging process gracefully by sending the poison pill

        except RuntimeError as e: #pragma: no cover
            #This happens when importing main causes this code to run again
            coba_exit(str(e))
