import time
import shutil
import threading
import unittest

from itertools import count
from contextlib import contextmanager
from pathlib import Path

from coba.config import DiskCacher, MemoryCacher, NullCacher, ConcurrentCacher
from coba.exceptions import CobaException

class IterCacher(MemoryCacher):
    def get(self,key):
        return iter(super().get(key))

class MaxCountCacher:
    def __init__(self, cacher):
        self._cacher = cacher
        self._cur_count = 0
        self.max_count = 0

    @contextmanager
    def count_context(self):
        self._cur_count += 1
        self.max_count = max(self._cur_count, self.max_count)
        time.sleep(0.1)
        yield

        self._cur_count -= 1

    def __contains__(self,key):
        return key in self._cacher

    def get(self,key):
        with self.count_context():
            return self._cacher.get(key)

    def put(self,key,val):
        with self.count_context():
            self._cacher.put(key,val)

    def rmv(self,key):
        with self.count_context():
            self._cacher.rmv(key)

class NullCacher_Tests(unittest.TestCase):

    def test_put_does_nothing(self):
        NullCacher().put("abc", "abc")
        self.assertNotIn("abc", NullCacher())
    
    def test_get_throws_exception(self):
        with self.assertRaises(Exception) as e:
            NullCacher().get("abc")

    def test_rmv_does_nothing(self):
        NullCacher().rmv("abc")

class MemoryCacher_Tests(unittest.TestCase):

    def test_put_works_correctly(self):
        cacher = MemoryCacher()
        cacher.put("abc", "abc")
        self.assertIn("abc", cacher)
    
    def test_get_works_correctly(self):
        cacher = MemoryCacher()
        cacher.put("abc", "abcd")
        self.assertIn("abc", cacher)
        self.assertEqual(cacher.get("abc"), "abcd")

    def test_rmv_works_correctly(self):
        cacher = MemoryCacher()
        cacher.put("abc", "abcd")
        self.assertIn("abc", cacher)
        cacher.rmv("abc")
        self.assertNotIn("abc", cacher)
        cacher.rmv("abc")

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
        cache.put("test.csv", [b"test"])
        self.assertTrue("test.csv" in cache)

    def test_creates_directory2(self):
        cache = DiskCacher(None)
        cache.cache_directory = self.Cache_Test_Dir / "folder1/folder2"
        cache.put("test.csv", [b"test"])
        self.assertTrue("test.csv" in cache)
            
    def test_write_csv_to_cache(self):

        cache = DiskCacher(self.Cache_Test_Dir)
        self.assertFalse("test.csv"    in cache)
        cache.put("test.csv", [b"test"])
        self.assertTrue("test.csv" in cache)
        self.assertEqual(list(cache.get("test.csv")), [b"test"])
    
    def test_write_multiline_csv_to_cache(self):

        cache = DiskCacher(self.Cache_Test_Dir)
        self.assertFalse("test.csv"    in cache)
        cache.put("test.csv", [b"test", b"test2"])
        self.assertTrue("test.csv" in cache)
        self.assertEqual(list(cache.get("test.csv")), [b"test", b"test2"])

    def test_rmv_csv_from_cache(self):

        cache = DiskCacher(self.Cache_Test_Dir)
        self.assertFalse("test.csv"    in cache)
        cache.put("test.csv", [b"test"])
        self.assertTrue("test.csv"    in cache)
        cache.rmv("test.csv")
        self.assertFalse("test.csv"    in cache)
        cache.rmv("test.csv")

    def test_get_corrupted_cache(self):

        #this is the md5 hexdigest of "text.csv" (531f844bd184e913b050d49856e8d438)
        (self.Cache_Test_Dir / "text.csv.gz").write_text("abcd")
        
        with self.assertRaises(Exception) as e:
            list(DiskCacher(self.Cache_Test_Dir).get("text.csv"))
        
        self.assertTrue((self.Cache_Test_Dir / "text.csv.gz").exists())
        
        DiskCacher(self.Cache_Test_Dir).rmv("text.csv")
        self.assertFalse((self.Cache_Test_Dir / "text.csv.gz").exists())

    def test_put_corrupted_cache(self):

        def bad_data():
            yield b'test1'
            raise Exception()
            yield b'test2'
        
        with self.assertRaises(Exception) as e:
            DiskCacher(self.Cache_Test_Dir).put("text.csv", bad_data())
        
        self.assertNotIn("text.csv", DiskCacher(self.Cache_Test_Dir))
        self.assertFalse((self.Cache_Test_Dir / "text.csv.gz").exists())

    def test_None_cach_dir(self):
        cacher = DiskCacher(None)

        self.assertNotIn('a', cacher)
        cacher.get("a")
        cacher.put("a", [b'123'])

    def test_bad_file_key(self):
        with self.assertRaises(CobaException):
            self.assertFalse("abcs/:/123/!@#" in DiskCacher(self.Cache_Test_Dir))

    def test_get_put_multiline_csv_to_cache(self):
        cache = DiskCacher(self.Cache_Test_Dir)
        self.assertFalse("test.csv" in cache)        
        self.assertEqual(list(cache.get_put("test.csv", lambda: [b"test", b"test2"])), [b"test", b"test2"])
        self.assertEqual(list(cache.get("test.csv")), [b"test", b"test2"])
        self.assertEqual(list(cache.get_put("test.csv", lambda: None)), [b"test", b"test2"])

    def test_get_put_None_cache_dir(self):
        cache = DiskCacher(None)
        self.assertFalse("test.csv" in cache)        
        self.assertEqual(list(cache.get_put("test.csv", lambda: [b"test", b"test2"])), [b"test", b"test2"])
        self.assertFalse("test.csv" in cache)        

class ConcurrentCacher_Test(unittest.TestCase):

    def test_put_works_correctly_single_thread(self):
        cacher = ConcurrentCacher(MemoryCacher(), {}, threading.Lock(), threading.Condition())
        cacher.put("abc", "abc")
        self.assertIn("abc", cacher)

    def test_get_works_correctly_single_thread(self):
        cacher = ConcurrentCacher(MemoryCacher(), {}, threading.Lock(), threading.Condition())
        cacher.put("abc", "abcd")
        self.assertIn("abc", cacher)
        self.assertEqual(cacher.get("abc"), "abcd")

    def test_get_iter_works_correctly_single_thread(self):
        cacher = ConcurrentCacher(IterCacher(), {}, threading.Lock(), threading.Condition())
        cacher.put("abc", [1,2,3])
        self.assertEqual(list(cacher.get("abc")), [1,2,3])
        self.assertEqual(list(cacher.get("abc")), [1,2,3])
        self.assertEqual(0, cacher._dict["abc"])

    def test_rmv_works_correctly_single_thread(self):
        cacher = ConcurrentCacher(MemoryCacher(), {}, threading.Lock(), threading.Condition())
        cacher.put("abc", "abcd")
        self.assertIn("abc", cacher)
        cacher.rmv("abc")
        self.assertNotIn("abc", cacher)
        cacher.rmv("abc")

    def test_put_works_correctly_separate_keys_multi_thread(self):
        cacher = ConcurrentCacher(MaxCountCacher(MemoryCacher()) , {}, threading.Lock(), threading.Condition())

        def thread_1():
            for i in range(0,10):
                cacher.put(i,(1,i))

        def thread_2():
            for i in range(10,20):
                cacher.put(i,(2,i))

        t1 = threading.Thread(None, thread_1)
        t2 = threading.Thread(None, thread_2)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        self.assertEqual(2, cacher._cache.max_count)

        for i in range(0,10):
            self.assertEqual(cacher.get(i), (1,i))

        for i in range(10,20):
            self.assertEqual(cacher.get(i), (2,i))

    def test_put_works_correctly_conflicting_keys_multi_thread(self):
        cacher = ConcurrentCacher(MaxCountCacher(MemoryCacher()) , {}, threading.Lock(), threading.Condition())

        def thread_1():
            for i in [1]*5:
                cacher.put(i,1)

        def thread_2():
            for i in [1]*5:
                cacher.put(i,2)

        t1 = threading.Thread(None, thread_1)
        t2 = threading.Thread(None, thread_2)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        self.assertEqual(1, cacher._cache.max_count)
        self.assertIn(cacher.get(1), [1,2])

    def test_get_and_put_works_correctly_conflicting_keys_multi_thread(self):
        cacher = ConcurrentCacher(MaxCountCacher(MemoryCacher()) , {}, threading.Lock(), threading.Condition())

        cacher.put(1,1)

        def thread_1():
            for _ in [1]*5:
                cacher.get(1)

        def thread_2():
            for _ in [1]*5:
                cacher.put(1,2)

        t1 = threading.Thread(None, thread_1)
        t2 = threading.Thread(None, thread_2)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        self.assertEqual(1, cacher._cache.max_count)
        self.assertEqual(cacher.get(1), 2)

    def test_get_works_correctly_separate_keys_multi_thread(self):
        cacher = ConcurrentCacher(MaxCountCacher(MemoryCacher()) , {}, threading.Lock(), threading.Condition())

        for i in range(0,20):
            cacher.put(i,i)

        def thread_1():
            for i in range(0,10):
                cacher.get(i)

        def thread_2():
            for i in range(10,20):
                cacher.get(i)

        t1 = threading.Thread(None, thread_1)
        t2 = threading.Thread(None, thread_2)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        self.assertEqual(2, cacher._cache.max_count)

    def test_get_works_correctly_conflicting_keys_multi_thread(self):
        cacher = ConcurrentCacher(MaxCountCacher(MemoryCacher()) , {}, threading.Lock(), threading.Condition())

        cacher.put(1,1)

        def thread_1():
            for i in [1]*10:
                cacher.get(i)

        def thread_2():
            for i in [1]*10:
                cacher.get(i)

        t1 = threading.Thread(None, thread_1)
        t2 = threading.Thread(None, thread_2)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        self.assertEqual(2, cacher._cache.max_count)

    def test_get_works_correctly_conflicting_keys_multi_thread(self):
        cacher  = ConcurrentCacher(MaxCountCacher(MemoryCacher()) , {}, threading.Lock(), threading.Condition())
        counter = count()

        def get_count():
            return next(counter)

        def thread_1():
            for i in [1]*10:
                cacher.get_put(i,get_count)

        def thread_2():
            for i in [1]*10:
                cacher.get_put(i,get_count)

        t1 = threading.Thread(None, thread_1)
        t2 = threading.Thread(None, thread_2)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        self.assertEqual(0, cacher.get(1))

if __name__ == '__main__':
    unittest.main()