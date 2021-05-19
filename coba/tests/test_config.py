import time
import shutil
import unittest
import traceback

from pathlib import Path

from coba.pipes import MemorySink
from coba.config import DiskCacher, IndentLogger, BasicLogger

class BasicLogger_Tests(unittest.TestCase):

    def test_log(self):

        sink   = MemorySink()
        logger = BasicLogger(sink, with_stamp=False, with_name=False)
        logs   = sink.items

        logger.log('a')
        logger.log('c')
        logger.log('d')

        self.assertEqual(logs[0], 'a' )
        self.assertEqual(logs[1], 'c' )
        self.assertEqual(logs[2], 'd' )

    def test_log_with_1(self):

        #This test is somewhat time dependent.
        #I don't think it should ever fail, but if it does
        #try running it again and see if it works that time.

        sink   = MemorySink()
        logger = BasicLogger(sink,with_stamp=False, with_name=False)
        logs   = sink.items

        with logger.log('a'):
            logger.log('c')
            logger.log('d')
        logger.log('e')

        self.assertEqual(5, len(logs))

        self.assertEqual(logs[0], 'a')
        self.assertEqual(logs[1], 'c')
        self.assertEqual(logs[2], 'd')
        self.assertEqual(logs[3], 'a (finish)')
        self.assertEqual(logs[4], 'e')

    def test_time_with_1(self):

        #This test is somewhat time dependent.
        #I don't think it should ever fail, but if it does
        #try running it again and see if it works that time.

        sink   = MemorySink()
        logger = BasicLogger(sink,with_stamp=False, with_name=False)
        logs   = sink.items

        with logger.time('a'):
            logger.log('c')
            time.sleep(0.15)
            logger.log('d')
        logger.log('e')

        self.assertEqual(5, len(logs))

        self.assertEqual        (logs[0], 'a')
        self.assertEqual        (logs[1], 'c')
        self.assertEqual        (logs[2], 'd')
        self.assertRegexpMatches(logs[3], 'a \\(\\d+\\.\\d+ seconds\\)')
        self.assertEqual        (logs[4], 'e')


        self.assertAlmostEqual(float(logs[3][3:7 ]), 0.15, 1)

    def test_time_with_3(self):

        sink   = MemorySink()
        logger = BasicLogger(sink, with_stamp=False)
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

        self.assertEqual        (logs[0 ], 'a')
        self.assertEqual        (logs[1 ], 'c')
        self.assertEqual        (logs[2 ], 'd')
        self.assertEqual        (logs[3 ], 'e')
        self.assertRegexpMatches(logs[4 ], 'd \\(\\d+\\.\\d+ seconds\\)')
        self.assertEqual        (logs[5 ], 'd')
        self.assertEqual        (logs[6 ], 'e')
        self.assertRegexpMatches(logs[7 ], 'd \\(\\d+\\.\\d+ seconds\\)')
        self.assertEqual        (logs[8 ], 'c (finish)')
        self.assertEqual        (logs[9 ], 'f')
        self.assertRegexpMatches(logs[10], 'a \\(\\d+\\.\\d+ seconds\\)')
        self.assertEqual        (logs[11], 'g')

        self.assertAlmostEqual(float(logs[4 ][3:7]), 0.05, 1)
        self.assertAlmostEqual(float(logs[7 ][3:7]), 0.05, 1)
        self.assertAlmostEqual(float(logs[10][3:7]), 0.15, 1)

    def test_time_two_separate(self):

        sink   = MemorySink()
        logger = BasicLogger(sink, with_stamp=False)
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

        self.assertEqual        (logs[0 ], 'a')
        self.assertEqual        (logs[1 ], 'd')
        self.assertEqual        (logs[2 ], 'e')
        self.assertRegexpMatches(logs[3 ], 'd \\(\\d+\\.\\d+ seconds\\)')
        self.assertRegexpMatches(logs[4 ], 'a \\(\\d+\\.\\d+ seconds\\)')
        self.assertEqual        (logs[5 ], 'g')
        self.assertEqual        (logs[6 ], 'a')
        self.assertEqual        (logs[7 ], 'd')
        self.assertEqual        (logs[8 ], 'e')
        self.assertRegexpMatches(logs[9 ], 'd \\(\\d+\\.\\d+ seconds\\)')
        self.assertRegexpMatches(logs[10], 'a \\(\\d+\\.\\d+ seconds\\)')

        self.assertAlmostEqual(float(logs[3 ][3:7]), 0.05, 1)
        self.assertAlmostEqual(float(logs[4 ][3:7]), 0.10, 1)
        self.assertAlmostEqual(float(logs[9 ][3:7]), 0.05, 1)
        self.assertAlmostEqual(float(logs[10][3:7]), 0.10, 1)

    def test_log_exception_1(self):
        
        sink   = MemorySink()
        logger = BasicLogger(sink, with_stamp=False)
        logs   = sink.items

        try:
            raise Exception("Test Exception")
        except Exception as ex:
            logger.log_exception('error:',ex)

            tb = ''.join(traceback.format_tb(ex.__traceback__))
            msg = ''.join(traceback.TracebackException.from_exception(ex).format_exception_only())

            expected_msg = f"error:\n\n{tb}\n  {msg}"

            self.assertTrue(ex.__logged__) #type:ignore
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0], expected_msg)

    def test_log_exception_2(self):
        
        sink   = MemorySink()
        logger = BasicLogger(sink, with_stamp=False)
        logs   = sink.items
        exception = Exception("Test Exception")

        logger.log('a')
        logger.log_exception('',exception)

        tb = ''.join(traceback.format_tb(exception.__traceback__))
        msg = ''.join(traceback.TracebackException.from_exception(exception).format_exception_only())

        expected_msg = f"\n\n{tb}\n  {msg}"

        self.assertTrue(exception.__logged__) #type:ignore
        self.assertEqual(logs[0], "a")
        self.assertEqual(logs[1], expected_msg)

class IndentLogger_Tests(unittest.TestCase):

    def test_log(self):

        sink   = MemorySink()
        logger = IndentLogger(sink, with_stamp=False, with_name=False)
        logs   = sink.items

        logger.log('a')
        logger.log('c')
        logger.log('d')

        self.assertEqual(logs[0], 'a' )
        self.assertEqual(logs[1], 'c' )
        self.assertEqual(logs[2], 'd' )

    def test_log_with_1(self):

        #This test is somewhat time dependent.
        #I don't think it should ever fail, but if it does
        #try running it again and see if it works that time.

        sink   = MemorySink()
        logger = IndentLogger(sink,with_stamp=False, with_name=False)
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

        sink   = MemorySink()
        logger = IndentLogger(sink,with_stamp=False, with_name=False)
        logs   = sink.items

        with logger.time('a'):
            logger.log('c')
            time.sleep(0.15)
            logger.log('d')
        logger.log('e')

        self.assertEqual(4, len(logs))
        self.assertRegexpMatches(logs[0], 'a \\(\\d+\\.\\d+ seconds\\)')
        self.assertEqual        (logs[1], '  * c')
        self.assertEqual        (logs[2], '  * d')
        self.assertEqual        (logs[3], 'e'    )

        self.assertAlmostEqual(float(logs[0][3:7 ]), 0.15, 1)

    def test_time_with_3(self):

        sink   = MemorySink()
        logger = IndentLogger(sink, with_stamp=False)
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
        self.assertRegexpMatches(logs[0], 'a \\(\\d+\\.\\d+ seconds\\)')
        self.assertEqual        (logs[1], '  * c')
        self.assertRegexpMatches(logs[2], '    > d \\(\\d+\\.\\d+ seconds\\)')
        self.assertEqual        (logs[3], '      - e')
        self.assertRegexpMatches(logs[4], '    > d \\(\\d+\\.\\d+ seconds\\)')
        self.assertEqual        (logs[5], '      - e')
        self.assertEqual        (logs[6], '  * f')
        self.assertEqual        (logs[7], 'g'    )

        self.assertAlmostEqual(float(logs[0][3:7 ]), 0.15, 1)
        self.assertAlmostEqual(float(logs[2][9:13]), 0.05, 1)
        self.assertAlmostEqual(float(logs[4][9:13]), 0.05, 1)
    
    def test_time_two_separate(self):

        sink   = MemorySink()
        logger = IndentLogger(sink, with_stamp=False)
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
        self.assertRegexpMatches(logs[0 ], 'a \\(\\d+\\.\\d+ seconds\\)')
        self.assertRegexpMatches(logs[1 ], '  \\* d \\(\\d+\\.\\d+ seconds\\)')
        self.assertEqual        (logs[2 ], '    > e')
        self.assertEqual        (logs[3 ], 'g')
        self.assertRegexpMatches(logs[4 ], 'a \\(\\d+\\.\\d+ seconds\\)')
        self.assertRegexpMatches(logs[5 ], '  \\* d \\(\\d+\\.\\d+ seconds\\)')
        self.assertEqual        (logs[6 ], '    > e')

        self.assertAlmostEqual(float(logs[0][3:7 ]), 0.10, 1)
        self.assertAlmostEqual(float(logs[1][7:11]), 0.05, 1)
        self.assertAlmostEqual(float(logs[4][3:7 ]), 0.10, 1)
        self.assertAlmostEqual(float(logs[5][7:11]), 0.05, 1)

    def test_log_exception_1(self):
        
        sink   = MemorySink()
        logger = IndentLogger(sink, with_stamp=False)
        logs   = sink.items

        try:
            raise Exception("Test Exception")
        except Exception as ex:
            logger.log_exception('error:',ex)

            tb = ''.join(traceback.format_tb(ex.__traceback__))
            msg = ''.join(traceback.TracebackException.from_exception(ex).format_exception_only())

            expected_msg = f"error:\n\n{tb}\n  {msg}"

            self.assertTrue(ex.__logged__) #type:ignore
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0], expected_msg)

    def test_log_exception_2(self):
        
        sink   = MemorySink()
        logger = IndentLogger(sink, with_stamp=False)
        logs   = sink.items
        exception = Exception("Test Exception")

        logger.log('a')
        logger.log_exception('',exception)

        tb = ''.join(traceback.format_tb(exception.__traceback__))
        msg = ''.join(traceback.TracebackException.from_exception(exception).format_exception_only())

        expected_msg = f"\n\n{tb}\n  {msg}"

        self.assertTrue(exception.__logged__) #type:ignore
        self.assertEqual(logs[0], "a")
        self.assertEqual(logs[1], expected_msg)

class DiskCache_Tests(unittest.TestCase):
    Cache_Test_Dir = Path("coba/tests/.temp/cache_tests/")
    
    def setUp(self):
        
        if self.Cache_Test_Dir.exists():
            shutil.rmtree(self.Cache_Test_Dir)
        
        self.Cache_Test_Dir.mkdir()

    def tearDown(self) -> None:
        
        if self.Cache_Test_Dir.exists():
            shutil.rmtree(self.Cache_Test_Dir)

    def test_creates_directory(self):
        cache = DiskCacher(self.Cache_Test_Dir / "folder1/folder2")
        cache.put("test.csv", b"test")
        self.assertTrue("test.csv" in cache)
            
    def test_write_csv_to_cache(self):

        cache = DiskCacher(self.Cache_Test_Dir)

        self.assertFalse("test.csv"    in cache)
        cache.put("test.csv", b"test")
        self.assertTrue("test.csv" in cache)

        self.assertEqual(cache.get("test.csv"), b"test")
    
    def test_rmv_csv_from_cache(self):

        cache = DiskCacher(self.Cache_Test_Dir)

        self.assertFalse("test.csv"    in cache)
        
        cache.put("test.csv", b"test")
        
        self.assertTrue("test.csv"    in cache)

        cache.rmv("test.csv")

        self.assertFalse("test.csv"    in cache)

if __name__ == '__main__':
    unittest.main()