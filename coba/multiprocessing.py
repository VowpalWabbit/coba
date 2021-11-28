from copy import deepcopy
from multiprocessing import Manager
from multiprocessing.managers import SyncManager
from threading import Thread
from typing import Iterable, Any

from coba.exceptions import CobaFatal
from coba.config     import CobaConfig, BasicLogger, IndentLogger
from coba.pipes      import Filter, Sink, QueueIO, MultiprocessFilter

class CobaMultiprocessFilter(Filter[Iterable[Any], Iterable[Any]]):

    class MarshalableFilter:

        def __init__(self, filter: Filter, logger_sink: Sink, with_name:bool, manager: SyncManager) -> None:

            self._logger    = deepcopy(CobaConfig.logger)
            self._cacher    = deepcopy(CobaConfig.cacher)
            self._srcsema   = manager.Semaphore(2)
            self._cachelock = manager.Lock()

            if isinstance(self._logger, IndentLogger):
                self._logger._with_name = with_name
                self._logger._sink      = logger_sink

            if isinstance(self._logger, BasicLogger):
                self._logger._with_name = with_name
                self._logger._sink      = logger_sink

            self._filter = filter

        def filter(self, item: Any) -> Any:

            #placing this here means this is only set inside the process 
            CobaConfig.logger            = self._logger
            CobaConfig.cacher            = self._cacher
            CobaConfig.store["srcsema"]  = self._srcsema
            CobaConfig.store["cachelck"] = self._cachelock

            result = self._filter.filter(item)

            try:
                try:
                    if isinstance(result,str):
                        return result
                    else:
                        for item in result: yield item
                except TypeError as e:
                    if "not iterable" in str(e):
                        return result
                    else:
                        raise 
                except (EOFError,BrokenPipeError):
                    pass
            except Exception as e:
                self._logger.log(e)

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

                filter = CobaMultiprocessFilter.MarshalableFilter(self._filter, stderr, self._processes>1, manager)

                for item in MultiprocessFilter(filter, self._processes, self._maxtasksperchild, stderr).filter(items):
                    yield item

        except RuntimeError as e:
            #This happens when importing main causes this code to run again
            raise CobaFatal(str(e))