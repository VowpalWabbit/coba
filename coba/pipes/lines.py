import threading as mt
import multiprocessing as mp

from traceback import format_tb
from typing import Union, Callable, Sequence, Optional, Mapping, Iterator, Any

from coba.utilities  import try_else
from coba.exceptions import CobaException
from coba.primitives import Pipe, Source, Filter, Sink, Line
from coba.pipes.utilities import resolve_params

# There are three potential contexts -- spawn, fork, and forkserver. On Windows and Mac spawn is the only option.
# On Linux the default option is fork. Fork creates processes faster and can share memory but also doesn't play
# well with threads. Therefore, to make behavior consistent across Linux, Windows, and Mac -- and to avoid
# potential bugs -- we force our multiprocessors to always use spawn regardless of the host OS.
spawn_context = mp.get_context("spawn")

class ProcessLine(spawn_context.Process):

    ### We create a lock so that we can safely receive any possible exceptions. Empirical
    ### tests showed that creating a Pipe and Lock doesn't seem to slow us down too much.
    def __init__(self, line: Line, callback: Callable = None):
        self._line     = line
        self._callback = callback
        super().__init__(daemon=True)

    def start(self) -> None:
        callback   = self._callback
        recv, send = spawn_context.Pipe(False)

        self._send = send
        del self._callback

        # At this point we should only have
        # self._line and self._send defined
        super().start()

        self._lock = mt.Lock()
        self._recv = recv
        self._send = send

        if callback:
            def join_and_call(self=self):
                self.join()
                callback(self)
            mt.Thread(target=join_and_call,daemon=True).start()

    def run(self): #pragma: no cover (coverage can't be tracked for code that runs on background prcesses)
        try:
            self._line.run()
        except Exception as e:
            emsg = str(e)
            if emsg.startswith("Can't get attribute") and " from " not in emsg:
                ex,tb = CobaException(
                    "Pip install cloudpickle to use multiprocessing with custom classes in Jupyter."
                ),None
            elif str(e).startswith("Can't get attribute") and " from " in emsg:
                ex,tb = CobaException(
                    "Move classes outside of `if __name__ == '__main__'` to use multiprocessing."
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
        return try_else(lambda: self._traceback, None)

    @property
    def exception(self) -> Optional[Exception]:
        return try_else(lambda: self._exception, None)

    @property
    def poisoned(self) -> bool:
        return try_else(lambda: self._poisoned, False)

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

class SourceSink(Line):
    def __init__(self, *pipes: Union[Source,Filter,Sink]) -> None:
        self._pipes = sum((try_else(lambda: list(p),[p]) for p in pipes),[])

    def run(self) -> None:
        """Run the pipeline."""

        source  = self._pipes[0   ]
        filters = self._pipes[1:-1]
        sink    = self._pipes[-1  ]

        item = source.read()

        for filter in filters:
            item = filter.filter(item)

        sink.write(item)

    @property
    def params(self) -> Mapping[str, Any]:
        return resolve_params(list(self))

    def __str__(self) -> str:
        return ",".join(filter(None,map(str,self._pipes)))

    def __len__(self) -> int:
        return len(self._pipes)

    def __iter__(self) -> Iterator[Pipe]:
        yield from self._pipes

    def __getitem__(self, index:int) -> Union[Source,Filter,Sink]:
        return self._pipes[index]
