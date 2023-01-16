from multiprocessing import Process, Pipe, Lock
from threading import Thread
from traceback import format_exc
from typing import Optional

class SafeProcess(Process):

    ### We create a lock so that we can safely receive any possible exceptions.
    ### This could be a little heavy handed if many processes are being created.
    ### One possible solution to this is to only provide exceptions in the callback.
    ### This means we could *know* that the recv.recv() would only be called in our
    ### thread and wouldn't need to create a lock. We could even avoid creating a
    ### Pipe() all together in this case when a callback isn't provided.

    def __init__(self, *args, **kwargs):

        self._callback = kwargs.pop('callback',None)
        self._exception = None
        self._traceback = None
        super().__init__(*args, **kwargs)

    def start(self) -> None:
        callback = self._callback

        self._callback         = None
        self._recv, self._send = Pipe(False)
        self._lock             = Lock()

        super().start()

        if callback:
            def join_and_call():
                self.join()
                callback(self._exception,self._traceback)
            SafeThread(target=join_and_call).start()

    def run(self):#pragma: no cover (coverage can't be tracked for code that runs on background prcesses)
        try:
            super().run()
        except Exception as e:
            self._send.send((e, format_exc()))

    def join(self, timeout: float = None) -> None:
        super().join(timeout)
        self._get_ex()

    @property
    def exception(self) -> Optional[Exception]:
        return self._exception

    def _get_ex(self):
        with self._lock:
            if not self._recv.closed:
                if self._recv.poll():
                    ex,tb = self._recv.recv()
                    self._exception = ex
                    self._traceback = tb
                self._send.close()
                self._recv.close()

class SafeThread(Thread):

    def __init__(self, *args, **kwargs) -> None:
        self._callback = kwargs.pop("callback",None)
        super().__init__(*args, **kwargs)
        self._exception = None
        self._traceback = None

    def run(self) -> None:
        try:
            super().run()
        except Exception as e:
            self._exception = e
            self._traceback = format_exc()

        if self._callback: 
            self._callback(self._exception,self._traceback)

    @property
    def exception(self) -> Optional[Exception]:
        return self._exception
