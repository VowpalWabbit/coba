from copy import deepcopy
from multiprocessing.synchronize import Lock
from multiprocessing import Manager
from threading import Thread
from typing import Sequence, Iterable, Any

from coba.config import CobaConfig, CobaFatal, Cacher, Logger, BasicLogger, IndentLogger
from coba.pipes  import Filter, Sink, Pipe, StopPipe, QueueIO, MultiprocessFilter, MemoryIO

class CobaMultiprocessFilter(Filter[Iterable[Any], Iterable[Any]]):

    class ConfiguredFilter:

        def __init__(self, filters: Sequence[Filter], logger_sink: Sink, with_name:bool, source_lock: Lock) -> None:

            self._source_lock = source_lock
            self._logger      = deepcopy(CobaConfig.Logger)
            self._cacher      = deepcopy(CobaConfig.Cacher)

            if isinstance(self._logger, IndentLogger):
                self._logger._with_name = with_name
                self._logger._sink      = logger_sink
            
            if isinstance(self._logger, BasicLogger):
                self._logger._with_name = with_name
                self._logger._sink      = logger_sink

            self._filters = filters

        def filter(self, item: Iterable[Any]) -> Iterable[Any]:

            #placing this here means this is only set inside the process 
            CobaConfig.Logger = self._logger
            CobaConfig.Cacher = self._cacher

            return Pipe.join(self._filters).filter(item)

    def __init__(self, filters: Sequence[Filter], processes=1, maxtasksperchild=None) -> None:
        self._filters          = filters
        self._processes        = processes
        self._maxtasksperchild = maxtasksperchild

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:

        try:

            with Manager() as manager:
                
                stderr = QueueIO(manager.Queue())
                source_lock = manager.Lock()
                
                def log_stderr():
                    for err in stderr.read():
                        if isinstance(err,str):
                            CobaConfig.Logger.sink.write(err)
                        elif isinstance(err,Exception):
                            CobaConfig.Logger.log_exception(err)
                        else:
                            CobaConfig.Logger.log_exception(err[2])
                            print("".join(err[3]))

                log_thread = Thread(target=log_stderr)
                log_thread.daemon = True
                log_thread.start()

                filter = CobaMultiprocessFilter.ConfiguredFilter(self._filters, stderr, self._processes>1, source_lock)

                for item in MultiprocessFilter([filter], self._processes, self._maxtasksperchild, stderr).filter(items):
                    yield item

        except RuntimeError as e:
            #This happens when importing main causes this code to run again
            raise CobaFatal(str(e))