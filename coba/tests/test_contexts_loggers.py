import time
import unittest
import traceback
import multiprocessing
import unittest.mock
import datetime

from coba.exceptions import CobaException
from coba.pipes      import ListIO
from coba.contexts   import IndentLogger, BasicLogger, NullLogger, DecoratedLogger, ExceptLog, NameLog, StampLog
from coba.pipes.io   import NullIO

class LogDecorator:

    def __init__(self, name):
        self._name = name

    def filter(self, item):
        return f"{self._name} -- {item}"

class NullLogger_Tests(unittest.TestCase):
    def test_log_does_nothing(self):
        NullLogger().log("abc")

    def test_time_does_nothing(self):
        NullLogger().time("abc")

    def test_sink_is_set(self):
        logger = NullLogger()
        self.assertIsInstance(logger.sink, NullIO)
        logger.sink = None
        self.assertIsNone(logger.sink)

class BasicLogger_Tests(unittest.TestCase):

    def test_log(self):

        sink   = ListIO()
        logger = BasicLogger(sink)
        logs   = sink.items

        logger.log('a')
        logger.log('c')
        logger.log('d')

        self.assertEqual(logs[0], 'a' )
        self.assertEqual(logs[1], 'c' )
        self.assertEqual(logs[2], 'd' )

    def test_log_with(self):

        sink   = ListIO()
        logger = BasicLogger(sink)
        logs   = sink.items

        with logger.log('a'):
            logger.log('c')
            logger.log('d')
        logger.log('e')

        self.assertEqual(5, len(logs))

        self.assertEqual(logs[0], 'a')
        self.assertEqual(logs[1], 'c')
        self.assertEqual(logs[2], 'd')
        self.assertEqual(logs[3], 'a (completed)')
        self.assertEqual(logs[4], 'e')

    def test_log_with_exception(self):

        sink   = ListIO()
        logger = BasicLogger(sink)
        logs   = sink.items
        
        with self.assertRaises(Exception):
            with logger.log('a'):
                logger.log('c')
                logger.log('d')
                raise Exception()

        self.assertEqual(4, len(logs))

        self.assertEqual(logs[0], 'a')
        self.assertEqual(logs[1], 'c')
        self.assertEqual(logs[2], 'd')
        self.assertEqual(logs[3], 'a (exception)')

    def test_log_with_interrupt(self):

        sink   = ListIO()
        logger = BasicLogger(sink)
        logs   = sink.items
        
        with self.assertRaises(BaseException) as e:
            with logger.log('a'):
                logger.log('c')
                logger.log('d')
                raise KeyboardInterrupt()

        self.assertEqual(4, len(logs))

        self.assertEqual(logs[0], 'a')
        self.assertEqual(logs[1], 'c')
        self.assertEqual(logs[2], 'd')
        self.assertEqual(logs[3], 'a (interrupt)')

    def test_time_with_exception(self):

        sink   = ListIO()
        logger = BasicLogger(sink)
        logs   = sink.items
        
        with self.assertRaises(Exception) as e:
            with logger.time('a'):
                logger.log('c')
                logger.log('d')
                raise Exception()

        self.assertEqual(4, len(logs))

        self.assertEqual(logs[0], 'a')
        self.assertEqual(logs[1], 'c')
        self.assertEqual(logs[2], 'd')
        self.assertRegex(logs[3], '^a \\(\\d+\\.\\d+ seconds\\) \\(exception\\)$')

    def test_time_with_interrupt(self):

        sink   = ListIO()
        logger = BasicLogger(sink)
        logs   = sink.items
        
        with self.assertRaises(BaseException) as e:
            with logger.time('a'):
                logger.log('c')
                logger.log('d')
                raise KeyboardInterrupt()

        self.assertEqual(4, len(logs))

        self.assertEqual(logs[0], 'a')
        self.assertEqual(logs[1], 'c')
        self.assertEqual(logs[2], 'd')
        self.assertRegex(logs[3], '^a \\(\\d+\\.\\d+ seconds\\) \\(interrupt\\)$')

    def test_time_with_1(self):

        #This test is somewhat time dependent.
        #I don't think it should ever fail, but if it does
        #try running it again and see if it works that time.

        sink   = ListIO()
        logger = BasicLogger(sink)
        logs   = sink.items

        with logger.time('a'):
            logger.log('c')
            time.sleep(0.15)
            logger.log('d')
        logger.log('e')

        self.assertEqual(5, len(logs))

        self.assertEqual(logs[0], 'a')
        self.assertEqual(logs[1], 'c')
        self.assertEqual(logs[2], 'd')
        self.assertRegex(logs[3], '^a \\(\\d+\\.\\d+ seconds\\) \\(completed\\)$')
        self.assertEqual(logs[4], 'e')

        self.assertAlmostEqual(float(logs[3][3:7 ]), 0.15, 1)

    def test_time_with_3(self):

        #This test is somewhat time dependent.
        #I don't think it should ever fail, but if it does
        #try running it again and see if it works that time.

        sink   = ListIO()
        logger = BasicLogger(sink)
        logs   = sink.items

        with logger.time('a'):
            with logger.log('c'):
                time.sleep(0.05)
                with logger.time('d'):
                    logger.log('e')
                    time.sleep(0.05)
                with logger.time('d'):
                    logger.log('e')
                    time.sleep(0.05)
            logger.log('f')
        logger.log('g')

        self.assertEqual(12, len(logs))

        self.assertEqual(logs[0 ], 'a')
        self.assertEqual(logs[1 ], 'c')
        self.assertEqual(logs[2 ], 'd')
        self.assertEqual(logs[3 ], 'e')
        self.assertRegex(logs[4 ], '^d \\(\\d+\\.\\d+ seconds\\) \\(completed\\)$')
        self.assertEqual(logs[5 ], 'd')
        self.assertEqual(logs[6 ], 'e')
        self.assertRegex(logs[7 ], '^d \\(\\d+\\.\\d+ seconds\\) \\(completed\\)$')
        self.assertEqual(logs[8 ], 'c (completed)')
        self.assertEqual(logs[9 ], 'f')
        self.assertRegex(logs[10], '^a \\(\\d+\\.\\d+ seconds\\) \\(completed\\)$')
        self.assertEqual(logs[11], 'g')

        self.assertAlmostEqual(float(logs[4 ][3:7]), 0.05, 1)
        self.assertAlmostEqual(float(logs[7 ][3:7]), 0.05, 1)
        self.assertAlmostEqual(float(logs[10][3:7]), 0.15, 1)

    def test_time_two_separate(self):

        #This test is somewhat time dependent.
        #I don't think it should ever fail, but if it does
        #try running it again and see if it works that time.

        sink   = ListIO()
        logger = BasicLogger(sink)
        logs   = sink.items

        with logger.time('a'):
            time.sleep(0.05)
            with logger.time('d'):
                logger.log('e')
                time.sleep(0.05)

        logger.log('g')

        with logger.time('a'):
            time.sleep(0.05)
            with logger.time('d'):
                logger.log('e')
                time.sleep(0.05)

        self.assertEqual(11, len(logs))

        self.assertEqual(logs[0 ], 'a')
        self.assertEqual(logs[1 ], 'd')
        self.assertEqual(logs[2 ], 'e')
        self.assertRegex(logs[3 ], 'd \\(\\d+\\.\\d+ seconds\\)')
        self.assertRegex(logs[4 ], 'a \\(\\d+\\.\\d+ seconds\\)')
        self.assertEqual(logs[5 ], 'g')
        self.assertEqual(logs[6 ], 'a')
        self.assertEqual(logs[7 ], 'd')
        self.assertEqual(logs[8 ], 'e')
        self.assertRegex(logs[9 ], 'd \\(\\d+\\.\\d+ seconds\\)')
        self.assertRegex(logs[10], 'a \\(\\d+\\.\\d+ seconds\\)')

        self.assertAlmostEqual(float(logs[3 ][3:7]), 0.05, 1)
        self.assertAlmostEqual(float(logs[4 ][3:7]), 0.10, 1)
        self.assertAlmostEqual(float(logs[9 ][3:7]), 0.05, 1)
        self.assertAlmostEqual(float(logs[10][3:7]), 0.10, 1)

    def test_sink_is_set(self):
        logger = BasicLogger(ListIO())
        self.assertIsInstance(logger.sink, ListIO)
        logger.sink = NullIO()
        self.assertIsInstance(logger.sink, NullIO)

class IndentLogger_Tests(unittest.TestCase):

    def test_log(self):

        sink   = ListIO()
        logger = IndentLogger(sink)
        logs   = sink.items

        logger.log('a')
        logger.log('c')
        logger.log('d')

        self.assertEqual(logs[0], 'a' )
        self.assertEqual(logs[1], 'c' )
        self.assertEqual(logs[2], 'd' )

    def test_log_with_1(self):

        sink   = ListIO()
        logger = IndentLogger(sink)
        logs   = sink.items

        with logger.log('a'):
            logger.log('c')
            logger.log('d')
        logger.log('e')

        self.assertEqual(4, len(logs))
        self.assertEqual(logs[0], 'a'    )
        self.assertEqual(logs[1], '  * c')
        self.assertEqual(logs[2], '  * d')
        self.assertEqual(logs[3], 'e'    )

    def test_time_with_1(self):

        #This test is somewhat time dependent.
        #I don't think it should ever fail, but if it does
        #try running it again and see if it works that time.

        sink   = ListIO()
        logger = IndentLogger(sink)
        logs   = sink.items

        with logger.time('a'):
            logger.log('c')
            time.sleep(0.15)
            logger.log('d')
        logger.log('e')

        self.assertEqual(4, len(logs))
        self.assertRegex(logs[0], '^a \\(\\d+\\.\\d+ seconds\\) \\(completed\\)$')
        self.assertEqual(logs[1], '  * c')
        self.assertEqual(logs[2], '  * d')
        self.assertEqual(logs[3], 'e'    )

        self.assertAlmostEqual(float(logs[0][3:7 ]), 0.15, 1)

    def test_time_with_3(self):

        #This test is somewhat time dependent.
        #I don't think it should ever fail, but if it does
        #try running it again and see if it works that time.

        sink   = ListIO()
        logger = IndentLogger(sink)
        logs   = sink.items

        with logger.time('a'):
            with logger.log('c'):
                time.sleep(0.05)
                with logger.time('d'):
                    logger.log('e')
                    time.sleep(0.05)
                with logger.time('d'):
                    logger.log('e')
                    time.sleep(0.05)
            logger.log('f')
        logger.log('g')

        self.assertEqual(8, len(logs))
        self.assertRegex(logs[0], '^a \\(\\d+\\.\\d+ seconds\\) \\(completed\\)')
        self.assertEqual(logs[1], '  * c')
        self.assertRegex(logs[2], '^    > d \\(\\d+\\.\\d+ seconds\\) \\(completed\\)$')
        self.assertEqual(logs[3], '      - e')
        self.assertRegex(logs[4], '^    > d \\(\\d+\\.\\d+ seconds\\) \\(completed\\)$')
        self.assertEqual(logs[5], '      - e')
        self.assertEqual(logs[6], '  * f')
        self.assertEqual(logs[7], 'g'    )

        self.assertAlmostEqual(float(logs[0][3:7 ]), 0.15, 1)
        self.assertAlmostEqual(float(logs[2][9:13]), 0.05, 1)
        self.assertAlmostEqual(float(logs[4][9:13]), 0.05, 1)
    
    def test_time_two_separate(self):

        #This test is somewhat time dependent.
        #I don't think it should ever fail, but if it does
        #try running it again and see if it works that time.

        sink   = ListIO()
        logger = IndentLogger(sink)
        logs   = sink.items

        with logger.time('a'):
            time.sleep(0.05)
            with logger.time('d'):
                logger.log('e')
                time.sleep(0.05)
        
        logger.log('g')
        
        with logger.time('a'):
            time.sleep(0.05)
            with logger.time('d'):
                logger.log('e')
                time.sleep(0.05)
        
        self.assertEqual(7, len(logs))
        self.assertRegex(logs[0 ], 'a \\(\\d+\\.\\d+ seconds\\)')
        self.assertRegex(logs[1 ], '  \\* d \\(\\d+\\.\\d+ seconds\\)')
        self.assertEqual(logs[2 ], '    > e')
        self.assertEqual(logs[3 ], 'g')
        self.assertRegex(logs[4 ], 'a \\(\\d+\\.\\d+ seconds\\)')
        self.assertRegex(logs[5 ], '  \\* d \\(\\d+\\.\\d+ seconds\\)')
        self.assertEqual(logs[6 ], '    > e')

        self.assertAlmostEqual(float(logs[0][3:7 ]), 0.10, 1)
        self.assertAlmostEqual(float(logs[1][7:11]), 0.05, 1)
        self.assertAlmostEqual(float(logs[4][3:7 ]), 0.10, 1)
        self.assertAlmostEqual(float(logs[5][7:11]), 0.05, 1)

    def test_time_with_exception(self):

        sink   = ListIO()
        logger = IndentLogger(sink)
        logs   = sink.items

        try:
            with self.assertRaises(Exception) as e:
                with logger.time('a'):
                    logger.log('c')
                    logger.log('d')
                    raise Exception()
        except:
            pass

        self.assertEqual(3, len(logs))
        self.assertRegex(logs[0], '^a \\(\\d+\\.\\d+ seconds\\) \\(exception\\)$')
        self.assertEqual(logs[1], '  * c')
        self.assertEqual(logs[2], '  * d')

    def test_time_with_interrupt(self):

        #This test is somewhat time dependent.
        #I don't think it should ever fail, but if it does
        #try running it again and see if it works that time.

        sink   = ListIO()
        logger = IndentLogger(sink)
        logs   = sink.items

        try:
            with self.assertRaises(Exception) as e:
                with logger.time('a'):
                    logger.log('c')
                    logger.log('d')
                    raise KeyboardInterrupt()
        except:
            pass

        self.assertEqual(3, len(logs))
        self.assertRegex(logs[0], '^a \\(\\d+\\.\\d+ seconds\\) \\(interrupt\\)$')
        self.assertEqual(logs[1], '  * c')
        self.assertEqual(logs[2], '  * d')

    def test_sink_is_set(self):
        logger = IndentLogger(ListIO())
        self.assertIsInstance(logger.sink, ListIO)
        logger.sink = NullIO()
        self.assertIsInstance(logger.sink, NullIO)

class DecoratedLogger_Tests(unittest.TestCase):

    def test_no_decorators(self):
        pre_decorators  = []
        post_decorators = []

        sink   = ListIO()
        logger = DecoratedLogger(pre_decorators,BasicLogger(sink),post_decorators)

        logger.log('a')
        self.assertEqual(['a'], sink.items)

        sink   = ListIO()
        logger = DecoratedLogger(pre_decorators,BasicLogger(sink),post_decorators)

        with logger.log('a'):
            pass

        self.assertEqual(['a', 'a (completed)'], sink.items)

        sink   = ListIO()
        logger = DecoratedLogger(pre_decorators,BasicLogger(sink),post_decorators)

        with logger.time('a'):
            pass

        self.assertEqual(['a', 'a (0.0 seconds) (completed)'], sink.items)

    def test_pre_decorators(self):
        pre_decorators  = [LogDecorator(1),LogDecorator(2)]
        post_decorators = []

        sink   = ListIO()
        logger = DecoratedLogger(pre_decorators,BasicLogger(sink),post_decorators)

        logger.log('a')
        self.assertEqual(['2 -- 1 -- a'], sink.items)

        sink   = ListIO()
        logger = DecoratedLogger(pre_decorators,BasicLogger(sink),post_decorators)

        with logger.log('a'):
            pass

        self.assertEqual(['2 -- 1 -- a', '2 -- 1 -- a (completed)'], sink.items)

        sink   = ListIO()
        logger = DecoratedLogger(pre_decorators,BasicLogger(sink),post_decorators)

        with logger.time('a'):
            pass

        self.assertEqual(['2 -- 1 -- a', '2 -- 1 -- a (0.0 seconds) (completed)'], sink.items)

    def test_post_decorators(self):
        pre_decorators  = []
        post_decorators = [LogDecorator(1),LogDecorator(2)]

        sink   = ListIO()
        logger = DecoratedLogger(pre_decorators,BasicLogger(sink),post_decorators)

        logger.log('a')
        self.assertEqual(['2 -- 1 -- a'], sink.items)

        sink   = ListIO()
        logger = DecoratedLogger(pre_decorators,BasicLogger(sink),post_decorators)

        with logger.log('a'):
            pass

        self.assertEqual(['2 -- 1 -- a', '2 -- 1 -- a (completed)'], sink.items)

        sink   = ListIO()
        logger = DecoratedLogger(pre_decorators,BasicLogger(sink),post_decorators)

        with logger.time('a'):
            pass

        self.assertEqual(['2 -- 1 -- a', '2 -- 1 -- a (0.0 seconds) (completed)'], sink.items)

    def test_pre_post_decorators(self):
        pre_decorators  = [LogDecorator(1)]
        post_decorators = [LogDecorator(2)]

        sink   = ListIO()
        logger = DecoratedLogger(pre_decorators,BasicLogger(sink),post_decorators)

        logger.log('a')
        self.assertEqual(['2 -- 1 -- a'], sink.items)

        sink   = ListIO()
        logger = DecoratedLogger(pre_decorators,BasicLogger(sink),post_decorators)

        with logger.log('a'):
            pass

        self.assertEqual(['2 -- 1 -- a', '2 -- 1 -- a (completed)'], sink.items)

        sink   = ListIO()
        logger = DecoratedLogger(pre_decorators,BasicLogger(sink),post_decorators)

        with logger.time('a'):
            pass

        self.assertEqual(['2 -- 1 -- a', '2 -- 1 -- a (0.0 seconds) (completed)'], sink.items)

    def test_sink(self):
        pre_decorators  = [LogDecorator(1)]
        post_decorators = [LogDecorator(2)]

        sink   = ListIO()
        logger = DecoratedLogger(pre_decorators,BasicLogger(sink),post_decorators)

        self.assertIs(logger.sink, sink)

        sink = ListIO()
        logger.sink = sink

        self.assertIs(logger.sink, sink)

        logger.log('a')
        self.assertEqual(['2 -- 1 -- a'], sink.items)

class ExceptLog_Tests(unittest.TestCase):
    
    def test_filter_exception(self):

        decorator = ExceptLog()
        exception = Exception("Test Exception")

        log = decorator.filter(exception)

        tb  = ''.join(traceback.format_tb(exception.__traceback__))
        msg = ''.join(traceback.TracebackException.from_exception(exception).format_exception_only())

        expected_log = f"Unexpected exception:\n\n{tb}\n  {msg}"

        self.assertEqual(log, expected_log)


    def test_filter_exception_raise(self):
        
        decorator = ExceptLog()
        exception = Exception("Test Exception")

        try:
            raise exception
        except Exception as ex:
            log = decorator.filter(ex)

            tb = ''.join(traceback.format_tb(ex.__traceback__))
            msg = ''.join(traceback.TracebackException.from_exception(ex).format_exception_only())

            expected_log = f"Unexpected exception:\n\n{tb}\n  {msg}"

            self.assertEqual(log, expected_log)

    def test_filter_coba_exception(self):
        
        decorator = ExceptLog()
        exception = CobaException("Test Exception")

        log = decorator.filter(exception)

        self.assertEqual(log, "Test Exception")

class NameLog_Tests(unittest.TestCase):

    def test_filter(self):
        decorator = NameLog()
        name = f"pid-{multiprocessing.current_process().pid:<6}"
        self.assertEqual(decorator.filter('a'), f'{name} -- a' )

class StampLog_Tests(unittest.TestCase):

    def test_log(self):

        now = datetime.datetime.now()

        with unittest.mock.patch('coba.contexts.StampLog._now', return_value=now):
            decorator = StampLog()
            stamp = now.strftime('%Y-%m-%d %H:%M:%S')
            self.assertEqual(decorator.filter('a'), f'{stamp} -- a' )

if __name__ == '__main__':
    unittest.main()