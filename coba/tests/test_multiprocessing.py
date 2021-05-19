import unittest

from multiprocessing import current_process
from typing import Iterable, Any

from coba.config          import CobaConfig, IndentLogger
from coba.pipes           import Filter, MemorySink
from coba.multiprocessing import MultiprocessFilter

class Multiprocess_Tests(unittest.TestCase):

    class ProcessNameFilter(Filter):
        def filter(self, items: Iterable[Any]) -> Iterable[Any]:

            process_name = current_process().name

            for _ in items:
                CobaConfig.Logger.log(process_name)
                yield process_name

    class ExceptionFilter(Filter):
        def filter(self, items: Iterable[Any]) -> Iterable[Any]:
            raise Exception("Exception Filter")

    def test_singleprocess_singletask(self):
        items = list(MultiprocessFilter([Multiprocess_Tests.ProcessNameFilter()], 1, 1).filter(range(4)))
        self.assertEqual(len(set(items)), 4)

    def test_multiprocess_multitask(self):
        items = list(MultiprocessFilter([Multiprocess_Tests.ProcessNameFilter()], 2).filter(range(40)))
        self.assertEqual(len(set(items)), 2)

    def test_multiprocess_singletask(self):
        items = list(MultiprocessFilter([Multiprocess_Tests.ProcessNameFilter()], 2, 1).filter(range(4)))
        self.assertEqual(len(set(items)), 4)

    def test_exception(self):
        with self.assertRaises(Exception):
            list(MultiprocessFilter([Multiprocess_Tests.ExceptionFilter()], 2, 1).filter(range(4)))

    def test_logging(self):
        
        #this is an important example. Even if we set the main logger's
        #with_stamp to false it doesn't propogate to the processes.

        logger_sink = MemorySink()
        logger      = IndentLogger(logger_sink, with_stamp=False, with_name=True)

        CobaConfig.Logger = logger

        items = list(MultiprocessFilter([Multiprocess_Tests.ProcessNameFilter()], 2, 1).filter(range(4)))

        self.assertEqual(len(logger_sink.items), 4)
        self.assertEqual(items, [ l.split(' ')[3] for l in logger_sink.items ] )
        self.assertEqual(items, [ l.split(' ')[5] for l in logger_sink.items ] )

if __name__ == '__main__':
    unittest.main()