from copy import deepcopy
from multiprocessing.synchronize import Lock, Semaphore
from multiprocessing import Manager
from threading import Thread
from typing import Sequence, Iterable, Any

from coba.exceptions import CobaFatal
from coba.config     import CobaConfig, BasicLogger, IndentLogger
from coba.pipes      import Filter, Sink, Pipe, QueueIO, MultiprocessFilter

class CobaMultiprocessFilter(Filter[Iterable[Any], Iterable[Any]]):

    class ConfiguredFilter:

        def __init__(self, filters: Sequence[Filter], logger_sink: Sink, with_name:bool, manager) -> None:

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

            self._filters = filters

        def filter(self, item: Iterable[Any]) -> Iterable[Any]:

            #placing this here means this is only set inside the process 
            CobaConfig.logger            = self._logger
            CobaConfig.cacher            = self._cacher
            CobaConfig.store["srcsema"]  = self._srcsema
            CobaConfig.store["cachelck"] = self._cachelock

            return Pipe.join(self._filters).filter(item)

    def __init__(self, filters: Sequence[Filter], processes=1, maxtasksperchild=0) -> None:
        self._filters          = filters
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
                        elif isinstance(err,Exception):
                            CobaConfig.logger.log_exception(err)
                        else:
                            CobaConfig.logger.log_exception(err[2])
                            #err[3] contains the stack trace... 
                            #I'm not sure what to do with it at this point, but here it is if any wants it in the future
                            #When we pass our exception back to the original thread it loses its stacktrace so if we want
                            #to report stack trace we'll need to turn it into a string and pass it with the exception???
                            #print("".join(err[3]))

                log_thread = Thread(target=log_stderr)
                log_thread.daemon = True
                log_thread.start()

                filter = CobaMultiprocessFilter.ConfiguredFilter(self._filters, stderr, self._processes>1, manager)

                for item in MultiprocessFilter([filter], self._processes, self._maxtasksperchild, stderr).filter(items):
                    yield item

        except RuntimeError as e:
            #This happens when importing main causes this code to run again
            raise CobaFatal(str(e))