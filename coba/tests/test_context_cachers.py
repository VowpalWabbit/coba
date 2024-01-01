import time
import shutil
import unittest

import threading as mt
import multiprocessing as mp
import ctypes as ct

from contextlib import contextmanager
from pathlib import Path

from coba.context import DiskCacher, MemoryCacher, NullCacher, ConcurrentCacher
from coba.exceptions import CobaException

def mp_run(cacher):
    while True:
        with cacher.get_set(1,None):
            pass

class IterCacher(MemoryCacher):
    def get(self,key):
        return iter(super().get(key))

class WaitCacher:
    def __init__(self, cacher, pause_once_on = None):
        self._cacher = cacher
        self._cur_count = 0
        self.max_count = 0

        self._paused      = False
        self._pause_on    = pause_once_on
        self._func_event  = mt.Event()
        self._first_event = mt.Event()

    @contextmanager
    def count_context(self):
        self._cur_count += 1
        self.max_count = max(self._cur_count, self.max_count)
        yield

        self._cur_count -= 1

    def wait(self):
        self._first_event.wait()

    def release(self):
        self._func_event.set()

    def _try_pause(self,method):
        if self._pause_on == method and not self._paused:
            self._paused = True
            self._first_event.set()
            self._func_event.wait()

    def __contains__(self,key):
        return key in self._cacher

    def rmv(self,key):
        with self.count_context():
            self._try_pause("rmv")
            self._cacher.rmv(key)

    def get_set(self, key, getter):
        with self.count_context():
            self._try_pause("get_set")
            return self._cacher.get_set(key, getter)

class NullCacher_Tests(unittest.TestCase):

    def test_contains(self):
        self.assertFalse("abc" in NullCacher())
        NullCacher().get_set("abc", lambda: 1)
        self.assertFalse("abc" in NullCacher())

    def test_rmv(self):
        NullCacher().rmv("abc")

    def test_get_set(self):
        my_iter = iter([1,2,3])

        with NullCacher().get_set("abc", lambda: my_iter) as out_iter:
            self.assertIs(my_iter, out_iter)
            self.assertEqual([1,2,3], list(my_iter))

class MemoryCacher_Tests(unittest.TestCase):

    def test_contains(self):
        cacher = MemoryCacher()
        self.assertNotIn("abc", cacher)
        cacher.get_set("abc", lambda: "abcd")
        self.assertIn("abc", cacher)

    def test_rmv(self):
        cacher = MemoryCacher()
        cacher.get_set("abc", lambda: "abcd")
        self.assertIn("abc", cacher)
        cacher.rmv("abc")
        self.assertNotIn("abc", cacher)
        cacher.rmv("abc")

    def test_get_set_callable(self):
        cacher = MemoryCacher()
        with cacher.get_set("abc", lambda: "abcd") as out:
            self.assertIn("abc", cacher)
            self.assertEqual(out,"abcd")

        with cacher.get_set("abc", None) as out:
            self.assertIn("abc", cacher)
            self.assertEqual(out,"abcd")

    def test_get_set_value(self):
        cacher = MemoryCacher()
        with cacher.get_set("abc", "abcd") as out:
            self.assertIn("abc", cacher)
            self.assertEqual(out,"abcd")

        with cacher.get_set("abc", None) as out:
            self.assertIn("abc", cacher)
            self.assertEqual(out,"abcd")

    def test_get_set_iter(self):
        cacher = MemoryCacher()
        with cacher.get_set("abc", lambda: iter([1,2,3])) as out:
            self.assertIn("abc", cacher)
            self.assertEqual(out,[1,2,3])

    def test_get_set_generator(self):
        def test():
            yield 1
            yield 2
        cacher = MemoryCacher()
        with cacher.get_set("abc", lambda: test()) as out:
            self.assertIn("abc", cacher)
            self.assertEqual(out,[1,2])

class DiskCacher_Tests(unittest.TestCase):
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
        cache.get_set("test.csv", lambda:["test"])
        self.assertTrue("test.csv" in cache)

    def test_creates_in_home_directory(self):
        try:
            cache = DiskCacher("~/can_delete")
            cache.get_set("test.csv", lambda:["test"])
            self.assertTrue("test.csv" in cache)
        finally:
            if Path("~/can_delete").expanduser().exists():
                shutil.rmtree(Path("~/can_delete").expanduser())

    def test_bad_directory(self):
        with self.assertRaises(CobaException):
            DiskCacher(None)

    def test_overwrite_empty_cache(self):
        cache = DiskCacher(self.Cache_Test_Dir)
        Path(cache._cache_path('test.csv')).touch()

        self.assertTrue("test.csv" in cache)
        with cache.get_set("test.csv", lambda:["test"]) as out:
            self.assertTrue("test.csv" in cache)
            self.assertEqual(list(out), ["test\r\n"])
        with cache.get_set("test.csv", None) as out:
            self.assertEqual(list(out), ["test\r\n"])

    def test_write_csv_to_cache(self):
        cache = DiskCacher(self.Cache_Test_Dir)
        self.assertFalse("test.csv" in cache)
        with cache.get_set("test.csv", lambda:["test"]) as out:
            self.assertTrue("test.csv" in cache)
            self.assertEqual(list(out), ["test\r\n"])
        with cache.get_set("test.csv", None) as out:
            self.assertEqual(list(out), ["test\r\n"])

    def test_write_multiline_csv_to_cache(self):
        cache = DiskCacher(self.Cache_Test_Dir)
        self.assertFalse("test.csv" in cache)
        with cache.get_set("test.csv", lambda:["test", "test2"]) as out:
            self.assertTrue("test.csv" in cache)
            self.assertEqual(list(out), ["test\r\n", "test2\r\n"])

    def test_rmv_csv_from_cache(self):
        cache = DiskCacher(self.Cache_Test_Dir)
        self.assertFalse("test.csv"    in cache)
        cache.get_set("test.csv", lambda:["test"])
        self.assertTrue("test.csv" in cache)
        cache.rmv("test.csv")
        self.assertFalse("test.csv" in cache)
        cache.rmv("test.csv")

    def test_get_corrupted_cache(self):

        (self.Cache_Test_Dir / "text.csv.gz").write_text("abcd")

        with self.assertRaises(Exception) as e:
            with DiskCacher(self.Cache_Test_Dir).get_set("text.csv",None) as out:
                list(out)

        self.assertTrue((self.Cache_Test_Dir / "text.csv.gz").exists())

        DiskCacher(self.Cache_Test_Dir).rmv("text.csv")
        self.assertFalse((self.Cache_Test_Dir / "text.csv.gz").exists())

    def test_put_corrupted_cache(self):

        def bad_data():
            yield b'test1'
            raise Exception()
            yield b'test2'

        with self.assertRaises(Exception) as e:
            DiskCacher(self.Cache_Test_Dir).get_set("text.csv", bad_data())

        self.assertNotIn("text.csv", DiskCacher(self.Cache_Test_Dir))
        self.assertFalse((self.Cache_Test_Dir / "text.csv.gz").exists())

    def test_bad_file_key(self):
        with self.assertRaises(CobaException):
            self.assertFalse("abcs/:/123/!@#" in DiskCacher(self.Cache_Test_Dir))

    def test_get_set_multiline_csv_to_cache(self):
        cache = DiskCacher(self.Cache_Test_Dir)
        self.assertFalse("test.csv" in cache)

        with cache.get_set("test.csv", lambda: ["test", "test2"]) as out:
            self.assertEqual(list(out), ["test\r\n", "test2\r\n"])
            self.assertTrue("test.csv" in cache)

        with cache.get_set("test.csv", None) as out:
            self.assertEqual(list(out), ["test\r\n", "test2\r\n"])
            self.assertTrue("test.csv" in cache)

class ConcurrentCacher_Test(unittest.TestCase):

    def test_rmv_works_correctly_single_thread(self):
        inner = MemoryCacher()
        inner.get_set("abc","abcd")
        outer = ConcurrentCacher(inner)
        self.assertIn("abc", outer)
        outer.rmv("abc")
        self.assertNotIn("abc", outer)
        outer.rmv("abc")

    def test_get_then_get_works_correctly_with_conflicting_keys_multi_thread(self):
        cacher      = MemoryCacher()
        wait_cacher = WaitCacher(cacher,pause_once_on="get_set")
        curr_cacher = ConcurrentCacher(wait_cacher)

        cacher.get_set(1,1)

        def thread_1():
            with curr_cacher.get_set(1,None): pass

        def thread_2():
            wait_cacher.wait()
            with curr_cacher.get_set(1,None): pass
            wait_cacher.release()

        t1 = mt.Thread(None, thread_1)
        t2 = mt.Thread(None, thread_2)

        t1.daemon = True
        t2.daemon = True

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        self.assertSetEqual(set(curr_cacher._locks.values()), {0})
        self.assertEqual(2, wait_cacher.max_count)
        self.assertEqual(curr_cacher.get_set(1,None).__enter__(), 1)

    def test_get_then_get_works_correctly_sans_conflicting_keys_multi_thread(self):
        cacher      = MemoryCacher()
        wait_cacher = WaitCacher(cacher,pause_once_on="get_set")
        curr_cacher = ConcurrentCacher(wait_cacher)

        cacher.get_set(1,1)
        cacher.get_set(2,2)

        def thread_1():
            with curr_cacher.get_set(1,None): pass

        def thread_2():
            wait_cacher.wait()
            with curr_cacher.get_set(2,None): pass
            wait_cacher.release()

        t1 = mt.Thread(None, thread_1)
        t2 = mt.Thread(None, thread_2)

        t1.daemon = True
        t2.daemon = True

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        self.assertSetEqual(set(curr_cacher._locks.values()), {0})
        self.assertEqual(2, wait_cacher.max_count)
        self.assertEqual(curr_cacher.get_set(1,None).__enter__(), 1)
        self.assertEqual(curr_cacher.get_set(2,None).__enter__(), 2)

    def test_put_then_put_works_correctly_with_conflicting_keys_multi_thread(self):
        base_cacher = WaitCacher(MemoryCacher(),pause_once_on="get_set")
        curr_cacher = ConcurrentCacher(base_cacher)

        def thread_1():
            with curr_cacher.get_set(1,2): pass

        def thread_2():
            base_cacher.wait()
            with curr_cacher.get_set(1,3): pass

        t1 = mt.Thread(None, thread_1)
        t2 = mt.Thread(None, thread_2)

        t1.daemon = True
        t2.daemon = True

        t1.start()
        t2.start()

        while curr_cacher._read_waits != 1:
            time.sleep(0.01)

        base_cacher.release()

        t1.join()
        t2.join()

        self.assertSetEqual(set(curr_cacher._locks.values()), {0})
        self.assertEqual(1, base_cacher.max_count)
        self.assertEqual(curr_cacher.get_set(1,None).__enter__(),2)

    def test_put_then_put_works_correctly_sans_conflicting_keys_multi_thread(self):
        base_cacher = WaitCacher(MemoryCacher(),pause_once_on="get_set")
        curr_cacher = ConcurrentCacher(base_cacher)

        def thread_1():
            with curr_cacher.get_set(1,2): pass

        def thread_2():
            base_cacher.wait()
            with curr_cacher.get_set(2,3): pass
            base_cacher.release()

        t1 = mt.Thread(None, thread_1)
        t2 = mt.Thread(None, thread_2)

        t1.daemon = True
        t2.daemon = True

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        self.assertSetEqual(set(curr_cacher._locks.values()), {0})
        self.assertEqual(2, base_cacher.max_count)
        self.assertEqual(curr_cacher.get_set(1,None).__enter__(), 2)
        self.assertEqual(curr_cacher.get_set(2,None).__enter__(), 3)

    def test_put_then_get_works_correctly_with_conflicting_keys_multi_thread(self):
        base_cacher = WaitCacher(MemoryCacher(),pause_once_on="get_set")
        curr_cacher = ConcurrentCacher(base_cacher)

        def thread_1():
            with curr_cacher.get_set(1,2): pass

        def thread_2():
            base_cacher.wait()
            with curr_cacher.get_set(1,None): pass

        t1 = mt.Thread(None, thread_1)
        t2 = mt.Thread(None, thread_2)

        t1.daemon = True
        t2.daemon = True

        t1.start()
        t2.start()

        while curr_cacher._read_waits != 1:
            time.sleep(0.01)

        base_cacher.release()

        t1.join()
        t2.join()

        self.assertSetEqual(set(curr_cacher._locks.values()), {0})
        self.assertEqual(1, base_cacher.max_count)
        self.assertEqual(curr_cacher.get_set(1,None).__enter__(), 2)

    def test_put_then_get_works_correctly_sans_conflicting_keys_multi_thread(self):
        base_cacher = MemoryCacher()
        wait_cacher = WaitCacher(base_cacher,pause_once_on="get_set")
        curr_cacher = ConcurrentCacher(wait_cacher)

        base_cacher.get_set(2,1)

        def thread_1():
            with curr_cacher.get_set(1,2): pass

        def thread_2():
            wait_cacher.wait()
            with curr_cacher.get_set(2,None): pass
            wait_cacher.release()

        t1 = mt.Thread(None, thread_1)
        t2 = mt.Thread(None, thread_2)

        t1.daemon = True
        t2.daemon = True

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        self.assertSetEqual(set(curr_cacher._locks.values()), {0})
        self.assertEqual(2, wait_cacher.max_count)
        self.assertEqual(curr_cacher.get_set(1,None).__enter__(), 2)

    def test_get_then_put_works_correctly_with_conflicting_keys_multi_thread(self):
        base_cacher = MemoryCacher()
        wait_cacher = WaitCacher(base_cacher,pause_once_on="get_set")
        curr_cacher = ConcurrentCacher(wait_cacher)

        base_cacher.get_set(1,1)

        def thread_1():
            with curr_cacher.get_set(1,None): pass

        def thread_2():
            wait_cacher.wait()
            curr_cacher.rmv(1)
            with curr_cacher.get_set(1,2): pass

        t1 = mt.Thread(None, thread_1)
        t2 = mt.Thread(None, thread_2)

        t1.daemon = True
        t2.daemon = True

        t1.start()
        t2.start()

        while curr_cacher._write_waits != 1:
            time.sleep(0.01)

        wait_cacher.release()

        t1.join()
        t2.join()

        self.assertSetEqual(set(curr_cacher._locks.values()), {0})
        self.assertEqual(1, wait_cacher.max_count)
        self.assertEqual(curr_cacher.get_set(1,None).__enter__(), 2)

    def test_get_then_put_works_correctly_sans_conflicting_keys_multi_thread(self):
        base_cacher = MemoryCacher()
        wait_cacher = WaitCacher(base_cacher,pause_once_on="get_set")
        curr_cacher = ConcurrentCacher(wait_cacher)

        base_cacher.get_set(1,1)

        def thread_1():
            with curr_cacher.get_set(1,None): pass

        def thread_2():
            wait_cacher.wait()
            with curr_cacher.get_set(2,2): pass
            wait_cacher.release()

        t1 = mt.Thread(None, thread_1)
        t2 = mt.Thread(None, thread_2)

        t1.daemon = True
        t2.daemon = True

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        self.assertSetEqual(set(curr_cacher._locks.values()), {0})
        self.assertEqual(2, wait_cacher.max_count)
        self.assertEqual(curr_cacher.get_set(1,None).__enter__(), 1)
        self.assertEqual(curr_cacher.get_set(2,None).__enter__(), 2)

    def test_get_set_put_then_get_set_get_works_correctly_with_conflicting_keys_multi_thread(self):
        base_cacher = WaitCacher(MemoryCacher(),pause_once_on="get_set")
        curr_cacher = ConcurrentCacher(base_cacher)

        def thread_1():
            with curr_cacher.get_set(1,lambda: 1): pass

        def thread_2():
            base_cacher.wait()
            with curr_cacher.get_set(1,lambda: 2): pass

        t1 = mt.Thread(None, thread_1)
        t2 = mt.Thread(None, thread_2)

        t1.daemon = True
        t2.daemon = True

        t1.start()
        t2.start()

        while curr_cacher._read_waits != 1:
            time.sleep(0.01)

        base_cacher.release()

        t1.join()
        t2.join()

        self.assertSetEqual(set(curr_cacher._locks.values()), {0})
        self.assertEqual(1, base_cacher.max_count)
        self.assertEqual(curr_cacher.get_set(1,None).__enter__(), 1)

    def test_get_set_put_then_get_set_get_works_correctly_sans_conflicting_keys_multi_thread(self):
        base_cacher = WaitCacher(MemoryCacher(),pause_once_on="get_set")
        curr_cacher = ConcurrentCacher(base_cacher)

        def thread_1():
            with curr_cacher.get_set(1,lambda: 1): pass

        def thread_2():
            base_cacher.wait()
            with curr_cacher.get_set(2,lambda: 2): pass
            base_cacher.release()

        t1 = mt.Thread(None, thread_1)
        t2 = mt.Thread(None, thread_2)

        t1.daemon = True
        t2.daemon = True

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        self.assertSetEqual(set(curr_cacher._locks.values()), {0})
        self.assertEqual(2, base_cacher.max_count)
        self.assertEqual(curr_cacher.get_set(1,None).__enter__(), 1)
        self.assertEqual(curr_cacher.get_set(2,None).__enter__(), 2)

    def test_get_set_put_then_get_set_get_iter_works_correctly_sans_conflicting_keys_multi_thread(self):
        base_cacher = WaitCacher(IterCacher(),pause_once_on="get_set")
        curr_cacher = ConcurrentCacher(base_cacher)

        def thread_1():
            with curr_cacher.get_set(1,lambda: [1]): pass

        def thread_2():
            base_cacher.wait()
            with curr_cacher.get_set(2,lambda: [2]): pass
            base_cacher.release()

        t1 = mt.Thread(None, thread_1)
        t2 = mt.Thread(None, thread_2)

        t1.daemon = True
        t2.daemon = True

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        self.assertSetEqual(set(curr_cacher._locks.values()), {0})
        self.assertEqual(2, base_cacher.max_count)
        self.assertEqual(list(curr_cacher.get_set(1,None).__enter__()), [1])
        self.assertEqual(list(curr_cacher.get_set(2,None).__enter__()), [2])

    def test_get_set_get_then_get_set_get_works_correctly_with_conflicting_keys_multi_thread(self):
        base_cacher = MemoryCacher()
        wait_cacher = WaitCacher(base_cacher,pause_once_on="get_set")
        curr_cacher = ConcurrentCacher(wait_cacher)

        base_cacher.get_set(1,1)

        def thread_1():
            with curr_cacher.get_set(1,lambda: 1): pass

        def thread_2():
            wait_cacher.wait()
            with curr_cacher.get_set(1,lambda: 2): pass
            wait_cacher.release()

        t1 = mt.Thread(None, thread_1)
        t2 = mt.Thread(None, thread_2)

        t1.daemon = True
        t2.daemon = True

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        self.assertSetEqual(set(curr_cacher._locks.values()), {0})
        self.assertEqual(2, wait_cacher.max_count)
        self.assertEqual(curr_cacher.get_set(1,None).__enter__(), 1)

    def test_rmv_during_get_set_same_process_without_release(self):

        base_cacher = IterCacher()
        curr_cacher = ConcurrentCacher(base_cacher)

        item = curr_cacher.get_set(1,range(4))

        with self.assertRaises(CobaException) as e:
            curr_cacher.rmv(1)

        self.assertTrue(curr_cacher._has_read_lock(1))
        self.assertEqual("The concurrent cacher was asked to enter an unrecoverable state.", str(e.exception))

    def test_put_during_get_set_same_process_causes_exception(self):

        base_cacher = IterCacher()
        curr_cacher = ConcurrentCacher(base_cacher)

        def getter():
            yield 1
            curr_cacher.get_set(1,1)
            yield 2

        with self.assertRaises(CobaException):
            curr_cacher.get_set(1,getter)

        self.assertFalse(curr_cacher._has_write_lock(1))

    def test_many_get_at_same_time(self):

        #In order to 'get' at the same time we have to use
        #multiple processes due to the GIL. Using processes
        #means we need to also use RawArray and Lock. Finally,
        #we need to sleep for a long time to increase the chance
        #that the problematic race condition occurrs. If this bug
        #is present this test will fail occasionally. If there is
        #no bug this test will never fail.

        spawn_context = mp.get_context("spawn")
        array         = spawn_context.RawArray(ct.c_short,[0]*2**16)
        lock          = spawn_context.Lock()

        curr_cacher = ConcurrentCacher(MemoryCacher(),array,lock)
        curr_cacher.get_set(1,1)

        processes = [ spawn_context.Process(None, mp_run, args=(curr_cacher,), daemon=True) for _ in range(5)]
        for p in processes: p.start()
        time.sleep(4)
        for p in processes: p.kill()
        self.assertNotIn(-1, set(curr_cacher._array))

if __name__ == '__main__':
    unittest.main()
