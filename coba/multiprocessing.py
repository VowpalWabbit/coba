from multiprocessing import Manager
from threading import Thread
from typing import Iterable, Any, Dict

from coba.utilities import coba_exit
from coba.config    import CobaConfig, ConcurrentCacher, Logger, Cacher
from coba.pipes     import Filter, Sink, QueueIO, PipeMultiprocessor

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
            CobaConfig.logger = self._logger
            CobaConfig.cacher = self._cacher
            CobaConfig.store  = self._store
            
            #at this point logger has been marshalled so we can
            #modify it without affecting the base process logger
            CobaConfig.logger.sink = self._logger_sink

            try:
                return self._filter.filter(item)
            except Exception as e:
                CobaConfig.logger.log(e)

    def __init__(self, filter: Filter, processes=1, maxtasksperchild=0) -> None:
        self._filter           = filter
        self._processes        = processes
        self._maxtasksperchild = maxtasksperchild

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:

        try:

            with Manager() as manager:

                stderr = QueueIO(manager.Queue())

                def log_stderr():
                    for err in stderr.read():
                        if isinstance(err,str):
                            CobaConfig.logger.sink.write(err)
                        elif isinstance(err,tuple):
                            CobaConfig.logger.log(err[2])
                        elif isinstance(err,Exception):
                            CobaConfig.logger.log(err)

                log_thread = Thread(target=log_stderr)
                log_thread.daemon = True
                log_thread.start()

                logger = CobaConfig.logger
                cacher = ConcurrentCacher(CobaConfig.cacher, manager.dict(), manager.Lock(), manager.Condition())
                store  = { "srcsema":  manager.Semaphore(2) }

                filter = CobaMultiprocessor.ProcessFilter(self._filter, logger, cacher, store, stderr)

                for item in PipeMultiprocessor(filter, self._processes, self._maxtasksperchild, stderr).filter(items):
                    yield item

                stderr.write(None) #attempt to shutdown the logging process gracefully by sending the poison pill

        except RuntimeError as e:
            #This happens when importing main causes this code to run again
            coba_exit(str(e))
