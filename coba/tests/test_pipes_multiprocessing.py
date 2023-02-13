import unittest
import pickle

import multiprocessing as mp
import threading as mt

from typing import Iterable, Any

from coba.exceptions import CobaException
from coba.pipes import Filter, ListSink, Identity, QueueSink, IterableSource, QueueSource

from coba.pipes.multiprocessing import Multiprocessor, AsyncableLine, Unchunker, Pickler, Unpickler, EventSetter

spawn_context = mp.get_context("spawn")

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
        try:
            for item in items:
                yield f"pid-{spawn_context.current_process().pid}"
        except Exception as e:
            raise

class BarrierNameFilter(Filter):
    def __init__(self, n):
        self._barrier = spawn_context.Barrier(n)

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        for item in items:
            self._barrier.wait()
            yield f"pid-{spawn_context.current_process().pid}"

class ExceptionFilter(Filter):
    def __init__(self, exc = Exception("Exception Filter")):
        self._exc = exc

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        raise self._exc

class EventFilter:
    def __init__(self, event):
        self._event = event
    def filter(self, items):
        yield 1
        self._event.wait()
        yield 2

class Multiprocessor_Tests(unittest.TestCase):

    def test_single_process(self):
        items = list(Multiprocessor(Identity(), 1, None, False).filter([[0,1,2,3]]))
        self.assertEqual(items, [[0,1,2,3]])

    def test_chunk_false(self):
        items = list(Multiprocessor(Identity(), 1, 1, False).filter([[0,1,2,3]]))
        self.assertEqual(items, [[0,1,2,3]])

    def test_chunk_true(self):
        items = list(Multiprocessor(Identity(), 1, 1, True).filter([[0,1,2,3]]))
        self.assertEqual(items, [0,1,2,3])

    def test_params(self):
        expected = ParamsFilter().params
        actual   = Multiprocessor(ParamsFilter(), 1, 1, False).params
        self.assertEqual(expected,actual)

    def test_singleprocess_singleperchild(self):
        items = list(Multiprocessor(ProcessNameFilter(), 1, 1, False).filter([0,1,2,3]))
        self.assertEqual(len(set(items)), 4)

    def test_singleprocess_multiperchild(self):
        items = list(Multiprocessor(ProcessNameFilter(), 1, 0, False).filter(range(4)))
        self.assertEqual(len(set(items)), 1)

    def test_multiprocess_singleperchild(self):
        items = list(Multiprocessor(ProcessNameFilter(), 2, 1).filter(range(4)))
        self.assertEqual(len(set(items)), 4)

    def test_multiprocess_multiperchild(self):
        items = list(Multiprocessor(BarrierNameFilter(2), 2).filter(range(2)))
        self.assertEqual(len(set(items)), 2)

    def test_filter_exception(self):
        with self.assertRaises(Exception) as e:
            list(Multiprocessor(ExceptionFilter(), 2, 1).filter(range(4)))
        self.assertIn("Exception Filter", str(e.exception))

    def test_items_not_picklable(self):

        with self.assertRaises(CobaException) as e:
            list(Multiprocessor(ProcessNameFilter(), 2, 1).filter([NotPicklableFilter()]))

        self.assertIn("pickle", str(e.exception))

    def test_empty_list(self):
        items = list(Multiprocessor(ProcessNameFilter(), 1, 1).filter([]))
        self.assertEqual(len(items), 0)

    def test_class_definitions_not_loaded_in_main(self):

        #this makes Test picklable but not loadable by the process 
        global Test
        class Test:
            pass

        with self.assertRaises(Exception) as e:
            list(Multiprocessor(ProcessNameFilter(), 2, 1).filter([Test()]*2))

        self.assertIn("unable to find", str(e.exception))

class ProcessLine_Tests(unittest.TestCase):
    
    def setUp(self) -> None:
        self.mode = 'process'

    def test_run_async_process_no_callback(self):
        queue1    = spawn_context.Queue()
        queue2    = spawn_context.Queue()

        queue1.put(1)
        queue1.put(2)
        queue1.put(None)

        pipeline = AsyncableLine(QueueSource(queue1), QueueSink(queue2,True))
        proc     = pipeline.run_async(mode=self.mode)
        
        proc.join()

        self.assertEqual([1,2], [queue2.get(True),queue2.get(True)])
        self.assertIsNone(proc.exception)
        self.assertIsNone(proc.traceback)
        self.assertTrue(proc.poisoned)
        self.assertIs(proc.pipeline, pipeline)

    def test_run_async_process_with_callback(self):
        queue  = spawn_context.Queue()
        event  = mt.Event() 
        holder = []

        def callback(item):
            holder.append(item.exception)
            event.set()

        pipeline = AsyncableLine(IterableSource([1,2]), QueueSink(queue,True))
        proc     = pipeline.run_async(callback=callback,mode=self.mode)
        
        proc.join()
        event.wait()
        
        self.assertEqual([1,2], [queue.get(True),queue.get(True)])
        self.assertEqual(None, holder[0])
        self.assertIsNone(proc.exception)
        self.assertIsNone(proc.traceback)
        self.assertFalse(proc.poisoned)

    def test_run_async_process_with_exception(self):
        pipeline = AsyncableLine(IterableSource([1,2]), ExceptionFilter(), ListSink())
        proc = pipeline.run_async(mode=self.mode)
        proc.join()
        self.assertEqual(str(proc.exception),"Exception Filter")
        self.assertTrue(proc.traceback)
        self.assertFalse(proc.poisoned)

    def test_run_async_process_with_exception_and_callback(self):
        holder = []
        event  = mt.Event()

        def callback(item):
            holder.append(item.exception)
            event.set()

        pipeline = AsyncableLine(IterableSource([1,2]), ExceptionFilter(), ListSink())
        proc = pipeline.run_async(callback,mode=self.mode)
        proc.join()
        self.assertEqual(str(proc.exception),"Exception Filter")
        event.wait()
        self.assertEqual(str(holder[0]),"Exception Filter")

    def test_run_async_process_with_exception_and_callback(self):
        holder = []
        event  = mt.Event()

        def callback(item):
            holder.append(item.exception)
            event.set()

        pipeline = AsyncableLine(IterableSource([1,2]), ExceptionFilter(), ListSink())
        proc = pipeline.run_async(callback,mode=self.mode)
        proc.join()
        self.assertEqual(str(proc.exception),"Exception Filter")
        event.wait()
        self.assertEqual(str(holder[0]),"Exception Filter")

    def test_bad_async_mode(self):
        with self.assertRaises(CobaException):
            AsyncableLine(IterableSource([1,2]), ListSink()).run_async(None,mode="foobar")

class ThreadLine_Tests(ProcessLine_Tests):
    
    def setUp(self) -> None:
        self.mode = 'thread'

class Unchunker_Tests(unittest.TestCase):

    def test_chunked(self):
        unchunker = Unchunker(True)
        self.assertEqual(list(unchunker.filter([[1,2,3],[4,5,6]])), [1,2,3,4,5,6])

    def test_not_chunked(self):
        unchunker = Unchunker(False)
        self.assertEqual(list(unchunker.filter([[1,2,3],[4,5,6]])), [[1,2,3],[4,5,6]])

class Pickler_Tests(unittest.TestCase):

    def test_simple(self):
        self.assertEqual(list(Pickler().filter([1,2])), [pickle.dumps(1),pickle.dumps(2)])

class Unpickler_Tests(unittest.TestCase):

    def test_simple(self):
        self.assertEqual(list(Unpickler().filter([pickle.dumps(1),pickle.dumps(2)])), [1,2])

class EventSetter_Tests(unittest.TestCase):
    def test_simple(self):
        event = mt.Event()    
        setter = EventSetter(event)

        self.assertFalse(event.is_set())
        self.assertEqual(setter.filter([1,2]),[1,2])
        self.assertTrue(event.is_set())


if __name__ == '__main__':
    unittest.main()