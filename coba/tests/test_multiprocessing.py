import timeit
import time
import unittest

from multiprocessing import current_process, Process
from typing import Iterable, Any

from coba.config          import CobaConfig, IndentLogger
from coba.pipes           import Filter, MemorySink
from coba.multiprocessing import MultiprocessFilter

class NotPicklableFilter(Filter):
    def __init__(self):
        self._a = lambda : None

    def filter(self, item):
        return 'a'

class SleepingFilter(Filter):
    def filter(self, seconds: Iterable[float]) -> Any:
        second = next(iter(seconds)) #type: ignore
        #print(current_process().name + f" {seconds}")
        time.sleep(second)
        yield second

class ProcessNameFilter(Filter):
    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        process_name = current_process().name
        CobaConfig.Logger.log(process_name)
        yield process_name

class ExceptionFilter(Filter):
    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        raise Exception("Exception Filter")

class MultiprocessFilter_Tests(unittest.TestCase):

    def test_singleprocess_singletask(self):
        items = list(MultiprocessFilter([ProcessNameFilter()], 1, 1).filter(range(4)))
        self.assertEqual(len(set(items)), 4)

    def test_multiprocess_multitask(self):
        items = list(MultiprocessFilter([ProcessNameFilter()], 2).filter(range(40)))
        self.assertEqual(len(set(items)), 2)

    def test_multiprocess_singletask(self):
        items = list(MultiprocessFilter([ProcessNameFilter()], 2, 1).filter(range(4)))
        self.assertEqual(len(set(items)), 4)

    def test_multiprocess_sleeping_task(self):

        start_time = time.time()
        list(MultiprocessFilter([SleepingFilter()], 2, 1).filter([2,2,0.25,0.25]))
        end_time = time.time()

        self.assertLess(end_time-start_time, 4)

    def test(self):
        timeit.repeat(lambda: Process(target=time.sleep, args=(0,)).start(), repeat=1,number=100)

    def test_exception(self):
        with self.assertRaises(Exception):
            list(MultiprocessFilter([ExceptionFilter()], 2, 1).filter(range(4)))

    def test_logging(self):
        
        #this is an important example. Even if we set the main logger's
        #with_stamp to false it doesn't propogate to the processes.

        logger_sink = MemorySink()
        logger      = IndentLogger(logger_sink, with_stamp=False, with_name=True)

        CobaConfig.Logger = logger

        items = list(MultiprocessFilter([ProcessNameFilter()], 2, 1).filter(range(4)))

        self.assertEqual(len(logger_sink.items), 4)
        self.assertEqual(items, [ l.split(' ')[3] for l in logger_sink.items ] )
        self.assertEqual(items, [ l.split(' ')[5] for l in logger_sink.items ] )

    def test_not_picklable_sans_reduce(self):
        with self.assertRaises(Exception):
            list(MultiprocessFilter([NotPicklableFilter()], 2, 1).filter(range(4)))

    def test_empty_list(self):
        items = list(MultiprocessFilter([ProcessNameFilter()], 1, 1).filter([]))
        self.assertEqual(len(items), 0)

if __name__ == '__main__':
    unittest.main()