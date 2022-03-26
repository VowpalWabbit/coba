import unittest
import pickle

from queue                       import Queue
from threading                   import Thread
from multiprocessing             import current_process, Event, Barrier
from typing                      import Iterable, Any

from coba.pipes import Filter, ListSink, Identity, QueueSource, QueueSink, NullSink

from coba.pipes.multiprocessing import Multiprocessor, PipesPool

class NotPicklableFilter(Filter):
    def __init__(self):
        self._a = lambda : None

    def filter(self, item):
        return 'a'

class ParamsFilter(Filter):
    @property
    def params(self):
        return {'param':1}
    
    def filter(self, item):
        pass

class ProcessNameFilter(Filter):
    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        yield f"pid-{current_process().pid}"

class BarrierNameFilter(Filter):
    def __init__(self):
        self._barrier = Barrier(2)
    
    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        self._barrier.wait()
        yield f"pid-{current_process().pid}"

class KillableFilter(Filter):
    def __init__(self, kill_event: Event) -> None:
        self._kill_event = kill_event

    def filter(self, item):
        self._kill_event.wait()

class ExceptionFilter(Filter):
    def __init__(self, exc = Exception("Exception Filter")):
        self._exc = exc

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        raise self._exc

class PipeMultiprocessor_Tests(unittest.TestCase):

    def test_foreach_false(self):
        items = list(Multiprocessor(Identity(), 1, 1, chunked=False).filter([[0,1,2,3]]))
        self.assertEqual(items, [[0,1,2,3]])

    def test_params(self):
        expected = ParamsFilter().params
        actual   = Multiprocessor(ParamsFilter(), 1, 1, chunked=False).params
        
        self.assertEqual(expected,actual)

    def test_singleprocess_singleperchild(self):
        items = list(Multiprocessor(ProcessNameFilter(), 1, 1).filter([0,1,2,3]))
        self.assertEqual(len(set(items)), 4)

    def test_singleprocess_multiperchild(self):
        items = list(Multiprocessor(ProcessNameFilter(), 1, None).filter(range(4)))
        self.assertEqual(len(set(items)), 1)

    def test_multiprocess_singleperchild(self):
        items = list(Multiprocessor(ProcessNameFilter(), 2, 1).filter(range(4)))
        self.assertEqual(len(set(items)), 4)

    def test_multiprocess_multiperchild(self):
        items = list(Multiprocessor(BarrierNameFilter(), 2).filter(range(2)))
        self.assertEqual(len(set(items)), 2)

    def test_filter_exception(self):
        stderr_sink = ListSink()

        list(Multiprocessor(ExceptionFilter(), 2, 1, stderr_sink).filter(range(4)))

        self.assertEqual(len(stderr_sink.items), 4)

        for item in stderr_sink.items:
            self.assertIn("Exception Filter", str(item))

    def test_items_not_picklable(self):
        stderr_sink = ListSink()

        list(Multiprocessor(ProcessNameFilter(), 2, 1, stderr_sink).filter([NotPicklableFilter()]))

        self.assertEqual(1, len(stderr_sink.items))
        self.assertIn("pickle", stderr_sink.items[0])

    def test_empty_list(self):
        items = list(Multiprocessor(ProcessNameFilter(), 1, 1).filter([]))
        self.assertEqual(len(items), 0)

    @unittest.skip("I have been unable to get this to work in the CI tests.")
    def test_class_definitions_not_loaded_in_main(self):

        stderr_sink = ListSink()

        global Test
        class Test:
            pass

        def test_function():
            list(Multiprocessor(ProcessNameFilter(), 2, 1, stderr_sink).filter([Test()]*2))

        t = Thread(target=test_function)
        t.start()
        t.join(5)

        self.assertFalse(t.is_alive())
        self.assertIn("unable to find", stderr_sink.items[0])

class PipeMultiprocessor_PipePool_Tests(unittest.TestCase):

    def test_worker_stdin_stdout_no_max(self):

        stdin  = QueueSource(Queue())
        stdout = QueueSink(Queue())
        stderr = QueueSink(Queue())

        filter = Identity()

        stdin._queue.put(pickle.dumps(1))
        stdin._queue.put(pickle.dumps(2))
        stdin._queue.put(None)

        self.assertEqual(0,stdout._queue.qsize())
        self.assertEqual(0,stderr._queue.qsize())
        PipesPool.worker(filter, stdin, stdout, stderr, None)
        self.assertEqual(2,stdout._queue.qsize())
        self.assertEqual(1,stdout._queue.get())
        self.assertEqual(2,stdout._queue.get())
        self.assertEqual(0,stderr._queue.qsize())

    def test_worker_stdin_stdout_max(self):

        stdin  = QueueSource(Queue())
        stdout = QueueSink(Queue())
        stderr = QueueSink(Queue())

        filter = Identity()

        stdin._queue.put(pickle.dumps(1))
        stdin._queue.put(pickle.dumps(2))
        stdin._queue.put(None)

        self.assertEqual(0,stdout._queue.qsize())
        self.assertEqual(0,stderr._queue.qsize())
        PipesPool.worker(filter, stdin, stdout, stderr, 1)
        self.assertEqual(1,stdout._queue.qsize())
        self.assertEqual(1,stdout._queue.get())
        self.assertEqual(0,stderr._queue.qsize())

    def test_worker_stderr(self):

        stdin  = QueueSource(Queue())
        stdout = QueueSink(Queue())
        stderr = QueueSink(Queue())

        filter = ExceptionFilter()

        stdin._queue.put(pickle.dumps(1))
        stdin._queue.put(None)

        self.assertEqual(0,stdout._queue.qsize())
        self.assertEqual(0,stderr._queue.qsize())
        PipesPool.worker(filter, stdin, stdout, stderr, None)
        self.assertEqual(0,stdout._queue.qsize())
        self.assertEqual(1,stderr._queue.qsize())
        self.assertEqual("Exception Filter", str(stderr._queue.get()[2]))

    def test_worker_iterable_filter_return(self):

        stdin  = QueueSource(Queue())
        stdout = QueueSink(Queue())
        stderr = QueueSink(Queue())

        filter = ProcessNameFilter()

        stdin._queue.put(pickle.dumps(1))
        stdin._queue.put(None)

        self.assertEqual(0,stdout._queue.qsize())
        self.assertEqual(0,stderr._queue.qsize())
        PipesPool.worker(filter, stdin, stdout, stderr, None)
        self.assertEqual(1,stdout._queue.qsize())
        self.assertEqual(1,len(stdout._queue.get()))
        self.assertEqual(0,stderr._queue.qsize())

    def test_terminate_stops_pipe(self):

        init_event = Event()
        term_event = Event()
        pool       = PipesPool(1,None,NullSink())

        def sleepy_iter():
            yield 1
            init_event.set()
            term_event.wait()

        def map_pool():
            list(pool.map(Identity(), sleepy_iter()))

        thread = Thread(target=map_pool)

        thread.start()
        init_event.wait() #make sure the pool has fully initialized
        pool.terminate()
        thread.join(4)

        self.assertFalse(thread.is_alive())

    def test_terminate_called_on_exception(self):

        with self.assertRaises(Exception):
            with PipesPool(1,None,NullSink()) as pool:
                raise Exception()

        self.assertTrue(pool.is_terminated)

    def test_attribute_error(self):
        stdin  = QueueSource(Queue())
        stdout = QueueSink(Queue())
        stderr = QueueSink(Queue())

        filter = ExceptionFilter(Exception("Can't get attribute"))

        stdin._queue.put(pickle.dumps(1))
        stdin._queue.put(pickle.dumps(2))
        stdin._queue.put(None)

        self.assertEqual(0,stdout._queue.qsize())
        self.assertEqual(0,stderr._queue.qsize())
        PipesPool.worker(filter, stdin, stdout, stderr, None)
        self.assertEqual(0,stdout._queue.qsize())
        self.assertEqual(1,stderr._queue.qsize())
        self.assertIn("We attempted to evaluate", stderr._queue.get())

    def test_keyboard_interrupt(self):
        stdin  = QueueSource(Queue())
        stdout = QueueSink(Queue())
        stderr = QueueSink(Queue())

        filter = ExceptionFilter(KeyboardInterrupt())

        stdin._queue.put(pickle.dumps(1))
        stdin._queue.put(pickle.dumps(2))
        stdin._queue.put(None)

        self.assertEqual(0,stdout._queue.qsize())
        self.assertEqual(0,stderr._queue.qsize())
        PipesPool.worker(filter, stdin, stdout, stderr, None)
        self.assertEqual(0,stdout._queue.qsize())
        self.assertEqual(0,stderr._queue.qsize())

if __name__ == '__main__':
    unittest.main()