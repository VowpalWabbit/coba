import time
import unittest

from multiprocessing import current_process
from typing import Iterable, Any

from coba.contexts        import CobaContext, NullLogger
from coba.contexts        import NullCacher, MemoryCacher
from coba.contexts        import IndentLogger, BasicLogger, ExceptLog, StampLog, NameLog, DecoratedLogger
from coba.pipes           import Filter, ListSink, Identity
from coba.multiprocessing import CobaMultiprocessor

class NotPicklableFilter(Filter):
    def __init__(self):
        self._a = lambda : None

    def filter(self, item):
        return 'a'

class SleepingFilter(Filter):
    def filter(self, seconds: Iterable[float]) -> Any:
        second = next(iter(seconds)) #type: ignore
        time.sleep(second)
        yield second

class ProcessNameFilter(Filter):
    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        for item in items:
            process_name = f"pid-{current_process().pid}"
            CobaContext.logger.log(process_name)
            yield process_name

class ExceptionFilter(Filter):
    def __init__(self, exc = Exception("Exception Filter")):
        self._exc = exc

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        for item in items:
            raise self._exc
            yield None

class CobaMultiprocessor_Tests(unittest.TestCase):

    def setUp(self) -> None:
        CobaContext.logger = NullLogger()
        CobaContext.store = {}

    def test_logging(self):
        logger_sink = ListSink()
        logger      = DecoratedLogger([],IndentLogger(logger_sink), [NameLog(), StampLog()])

        CobaContext.logger = logger
        CobaContext.cacher = NullCacher()

        items = list(CobaMultiprocessor(ProcessNameFilter(), 2, 1, False).filter(range(4)))

        self.assertEqual(len(logger_sink.items), 4)
        self.assertCountEqual(items, [ l.split(' ')[ 3] for l in logger_sink.items ] )
        self.assertCountEqual(items, [ l.split(' ')[-1] for l in logger_sink.items ] )

    def test_filter_exception_logging(self):
        CobaContext.logger = DecoratedLogger([ExceptLog()],BasicLogger(ListSink()),[])
        CobaContext.cacher = NullCacher()
        

        with self.assertRaises(Exception) as e:
            list(CobaMultiprocessor(ExceptionFilter(), 2, 1, False).filter(range(4)))

        self.assertIn("Exception Filter", str(e.exception))

    def test_read_exception_logging(self):
        CobaContext.logger = DecoratedLogger([ExceptLog()],BasicLogger(ListSink()),[])
        CobaContext.cacher = NullCacher()

        def broken_generator():
            yield [1]
            raise Exception("Generator Exception")

        with self.assertRaises(Exception) as e:
            list(CobaMultiprocessor(Identity(), 2, 1).filter(broken_generator()))

        self.assertIn("Generator Exception", str(e.exception))

    def test_keyboard_interrupt(self):
        CobaContext.logger = DecoratedLogger([ExceptLog()],BasicLogger(ListSink()),[])
        CobaContext.cacher = NullCacher()

        with self.assertRaises(KeyboardInterrupt):
            list(CobaMultiprocessor(ExceptionFilter(KeyboardInterrupt()), 2, 1).filter([1,2,3]))

    def test_not_picklable_logging(self):
        logger_sink = ListSink()
        CobaContext.logger = DecoratedLogger([ExceptLog()],BasicLogger(logger_sink),[])
        CobaContext.cacher = NullCacher()

        list(CobaMultiprocessor(ProcessNameFilter(), 2, 1).filter([lambda a:1]))

        self.assertEqual(1, len(logger_sink.items))
        self.assertIn("pickle", logger_sink.items[0])

    def test_class_definitions_not_found(self):
        #this makes Test picklable but not loadable by the process
        global Test
        class Test:
            pass

        logger_sink = ListSink()
        CobaContext.logger = DecoratedLogger([ExceptLog()],BasicLogger(logger_sink),[])
        CobaContext.cacher = NullCacher()

        list(CobaMultiprocessor(ProcessNameFilter(), 2, 1).filter([Test()]*2))

        self.assertEqual(1, len(logger_sink.items))
        self.assertIn("unable to find", logger_sink.items[0])

    def test_double_call(self):

        logger_sink = ListSink()
        logger      = DecoratedLogger([],IndentLogger(logger_sink), [NameLog(), StampLog()])

        CobaContext.logger = logger
        CobaContext.cacher = NullCacher()

        items = list(CobaMultiprocessor(ProcessNameFilter(), 2, 1).filter(range(4)))

        self.assertEqual(len(logger_sink.items), 4)
        self.assertCountEqual(items, [ l.split(' ')[ 3] for l in logger_sink.items ] )
        self.assertCountEqual(items, [ l.split(' ')[-1] for l in logger_sink.items ] )

class CobaMultiprocessor_ProcessFilter_Tests(unittest.TestCase):

    def test_coba_config_set_correctly(self):

        log_sink = ListSink()

        CobaContext.logger = NullLogger()
        CobaContext.cacher = NullCacher()
        CobaContext.store  = None

        filter = CobaMultiprocessor.ProcessFilter(Identity(),IndentLogger(),MemoryCacher(),{},log_sink)

        self.assertIsInstance(CobaContext.logger, NullLogger)
        self.assertIsInstance(CobaContext.cacher, NullCacher)
        self.assertIsNone(CobaContext.store)

        list(filter.filter([1]))

        self.assertIsInstance(CobaContext.logger, IndentLogger)
        self.assertIsInstance(CobaContext.cacher, MemoryCacher)
        self.assertIsInstance(CobaContext.store , dict)
        self.assertIsInstance(CobaContext.logger.sink, ListSink)

    def test_exception_logged_but_not_thrown(self):
        log_sink = ListSink()

        CobaContext.logger = NullLogger()
        CobaContext.cacher = NullCacher()
        CobaContext.store  = None

        logger = DecoratedLogger([ExceptLog()], IndentLogger(), [])

        with self.assertRaises(Exception) as e:
            list(CobaMultiprocessor.ProcessFilter(ExceptionFilter(),logger,None,None,log_sink).filter([1]))

        self.assertIn('Exception Filter', str(e.exception))

if __name__ == '__main__':
    unittest.main()