import unittest

from threading       import Thread
from multiprocessing import current_process
from typing          import Iterable, Any

from coba.pipes import Filter, MemoryIO, PipeMultiprocessor, Identity

class NotPicklableFilter(Filter):
    def __init__(self):
        self._a = lambda : None

    def filter(self, item):
        return 'a'

class ProcessNameFilter(Filter):
    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        yield f"pid-{current_process().pid}"

class ExceptionFilter(Filter):
    def __init__(self, exc = Exception("Exception Filter")):
        self._exc = exc

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        raise self._exc

if current_process().name == 'MainProcess':
    class Test:
        pass

class PipeMultiprocessor_Tests(unittest.TestCase):

    def test_singleprocess_singletask(self):
        items = list(PipeMultiprocessor(ProcessNameFilter(), 1, 1).filter(range(4)))
        self.assertEqual(len(set(items)), 4)

        items = list(PipeMultiprocessor(Identity(), 1, 1, foreach=False).filter(range(4)))
        self.assertEqual(len(set(items) & set(range(4))), 4)

    def test_singleprocess_multitask(self):
        items = list(PipeMultiprocessor(ProcessNameFilter(), 1, None).filter(range(4)))
        self.assertEqual(len(set(items)), 1)

        items = list(PipeMultiprocessor(Identity(), 1, 1, foreach=False).filter(range(4)))
        self.assertEqual(len(set(items) & set(range(4))), 4)

    def test_multiprocess_singletask(self):
        items = list(PipeMultiprocessor(ProcessNameFilter(), 2, 1).filter(range(4)))
        self.assertEqual(len(set(items)), 4)

        items = list(PipeMultiprocessor(Identity(), 2, 1, foreach=False).filter(range(4)))
        self.assertEqual(len(set(items) & set(range(4))), 4)

    def test_multiprocess_multitask(self):
        items = list(PipeMultiprocessor(ProcessNameFilter(), 2).filter(range(40)))
        self.assertEqual(len(set(items)), 2)

        items = list(PipeMultiprocessor(Identity(), 2, foreach=False).filter(range(40)))
        self.assertEqual(len(set(items) & set(range(40))), 40)

    def test_filter_exception(self):
        stderr_sink = MemoryIO()

        list(PipeMultiprocessor(ExceptionFilter(), 2, 1, stderr_sink).filter(range(4)))

        self.assertEqual(len(stderr_sink.items), 4)

        for item in stderr_sink.items:
            self.assertIn("Exception Filter", str(item))

    def test_items_not_picklable(self):
        stderr_sink = MemoryIO()

        list(PipeMultiprocessor(ProcessNameFilter(), 2, 1, stderr_sink).filter([NotPicklableFilter()]))

        self.assertEqual(1, len(stderr_sink.items))
        self.assertIn("pickle", stderr_sink.items[0])

    def test_empty_list(self):
        items = list(PipeMultiprocessor(ProcessNameFilter(), 1, 1).filter([]))
        self.assertEqual(len(items), 0)

    def test_attribute_error_doesnt_freeze_process(self):

        stderr_sink = MemoryIO()

        def test_function():
            list(PipeMultiprocessor(ProcessNameFilter(), 2, 1, stderr_sink).filter([Test()]*2))

        t = Thread(target=test_function)
        t.start()
        t.join(5)

        self.assertFalse(t.is_alive())

class PipeMultiprocessor_Processor_Tests(unittest.TestCase):

    def test_filter_single_item(self):

        stdout = MemoryIO()
        stderr = MemoryIO()

        PipeMultiprocessor.Processor(Identity(),stdout,stderr,False).process(1)

        self.assertEqual(1, stdout.items[0])

    def test_filter_list_item(self):

        stdout = MemoryIO()
        stderr = MemoryIO()

        PipeMultiprocessor.Processor(Identity(),stdout,stderr,False).process([1,2])

        self.assertEqual([1,2], stdout.items[0])

    def test_filter_foreach_item(self):

        stdout = MemoryIO()
        stderr = MemoryIO()

        PipeMultiprocessor.Processor(Identity(),stdout,stderr,True).process([1,2])

        self.assertEqual(1, stdout.items[0])
        self.assertEqual(2, stdout.items[1])

    def test_ignore_keyboard_interrupt(self):

        stdout = MemoryIO()
        stderr = MemoryIO()

        PipeMultiprocessor.Processor(ExceptionFilter(KeyboardInterrupt()),stdout,stderr,True).process([1,2])
        
        self.assertEqual([], stdout.items)
        self.assertEqual([], stderr.items)

    def test_logged_exception(self):

        stdout = MemoryIO()
        stderr = MemoryIO()

        PipeMultiprocessor.Processor(ExceptionFilter(),stdout,stderr,True).process([1,2])
        
        self.assertEqual([], stdout.items)
        self.assertIn("Exception Filter", str(stderr.items[0][2]))

if __name__ == '__main__':
    unittest.main()