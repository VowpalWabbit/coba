import shutil
from typing import Iterable
import unittest
import unittest.mock

from pathlib import Path

from coba.config import DiskCacher
from coba.config.cachers import MemoryCacher, NullCacher

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

    def test_get_corrupted_cache(self):

        #this is the md5 hexdigest of "text.csv" (531f844bd184e913b050d49856e8d438)
        (self.Cache_Test_Dir / "531f844bd184e913b050d49856e8d438.gz").write_text("abcd")
        
        with self.assertRaises(Exception) as e:
            list(DiskCacher(self.Cache_Test_Dir).get("test.csv"))
        
        self.assertTrue((self.Cache_Test_Dir / "531f844bd184e913b050d49856e8d438.gz").exists())
        
        DiskCacher(self.Cache_Test_Dir).rmv("test.csv")
        self.assertFalse((self.Cache_Test_Dir / "531f844bd184e913b050d49856e8d438.gz").exists())

    def test_put_corrupted_cache(self):

        def bad_data() -> Iterable[bytes]:
            yield b'test1'
            raise Exception()
            yield b'test2'
        
        with self.assertRaises(Exception) as e:
            DiskCacher(self.Cache_Test_Dir).put("test.csv", bad_data())
        
        self.assertNotIn("test.csv", DiskCacher(self.Cache_Test_Dir))
        self.assertFalse((self.Cache_Test_Dir / "531f844bd184e913b050d49856e8d438.gz").exists())

if __name__ == '__main__':
    unittest.main()