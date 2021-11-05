import unittest

from threading       import Thread
from multiprocessing import current_process
from typing          import Iterable, Any

from coba.pipes import Filter, MemoryIO, MultiprocessFilter, IdentityFilter

class NotPicklableFilter(Filter):
    def __init__(self):
        self._a = lambda : None

    def filter(self, item):
        return 'a'

class ProcessNameFilter(Filter):
    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        yield f"pid-{current_process().pid}"

class ExceptionFilter(Filter):
    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        raise Exception("Exception Filter")

if current_process().name == 'MainProcess':
    class Test:
        pass

class MultiprocessFilter_Tests(unittest.TestCase):

    def test_singleprocess_singletask(self):
        items = list(MultiprocessFilter([ProcessNameFilter()], 1, 1).filter(range(4)))
        self.assertEqual(len(set(items)), 4)

        items = list(MultiprocessFilter([IdentityFilter()], 1, 1).filter(range(4)))
        self.assertEqual(len(set(items) & set(range(4))), 4)

    def test_singleprocess_multitask(self):
        items = list(MultiprocessFilter([ProcessNameFilter()], 1, None).filter(range(4)))
        self.assertEqual(len(set(items)), 1)

        items = list(MultiprocessFilter([IdentityFilter()], 1, 1).filter(range(4)))
        self.assertEqual(len(set(items) & set(range(4))), 4)

    def test_multiprocess_singletask(self):
        items = list(MultiprocessFilter([ProcessNameFilter()], 2, 1).filter(range(4)))
        self.assertEqual(len(set(items)), 4)

        items = list(MultiprocessFilter([IdentityFilter()], 2, 1).filter(range(4)))
        self.assertEqual(len(set(items) & set(range(4))), 4)

    def test_multiprocess_multitask(self):
        items = list(MultiprocessFilter([ProcessNameFilter()], 2).filter(range(40)))
        self.assertEqual(len(set(items)), 2)

        items = list(MultiprocessFilter([IdentityFilter()], 2).filter(range(40)))
        self.assertEqual(len(set(items) & set(range(40))), 40)

    def test_filter_exception(self):
        stderr_sink = MemoryIO()
        
        list(MultiprocessFilter([ExceptionFilter()], 2, 1, stderr_sink).filter(range(4)))

        self.assertEqual(len(stderr_sink.items), 4)

        for item in stderr_sink.items:
            self.assertIn("Exception Filter", str(item))

    def test_items_not_picklable(self):
        stderr_sink = MemoryIO()

        list(MultiprocessFilter([ProcessNameFilter()], 2, 1, stderr_sink).filter([NotPicklableFilter()]))

        self.assertEqual(1, len(stderr_sink.items))
        self.assertIn("pickle", stderr_sink.items[0])

    def test_empty_list(self):
        items = list(MultiprocessFilter([ProcessNameFilter()], 1, 1).filter([]))
        self.assertEqual(len(items), 0)

    def test_attribute_error_doesnt_freeze_process(self):

        stderr_sink = MemoryIO()

        def test_function():
            list(MultiprocessFilter([ProcessNameFilter()], 2, 1, stderr_sink).filter([Test()]*2))

        t = Thread(target=test_function)
        t.start()
        t.join(5)

        self.assertFalse(t.is_alive())
        self.assertIn("We attempted to evaluate", stderr_sink.items[0])

if __name__ == '__main__':
    unittest.main()