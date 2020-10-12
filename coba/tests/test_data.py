import unittest

from pathlib import Path

from coba.data import AsyncFileWriter

class Data_Tests(unittest.TestCase):
    
    def test_async_write(self):
        writer = AsyncFileWriter(".test/async.txt", 'a')

        try:
            writer.open()

            writer.async_write('line 0')
            writer.async_write('\n')
            writer.async_write('line 1')
            writer.async_write('\n')

            writer.close()

            with open(".test/async.txt", "r") as f:
                lines = f.readlines()

            self.assertEqual(lines[0], "line 0\n")
            self.assertEqual(lines[1], "line 1\n")

        finally:
            if Path(".test/async.txt").exists(): Path(".test/async.txt").unlink()

    def test_open_exception(self):

        try:
            with self.assertRaises(Exception):
                with AsyncFileWriter(".test/async.txt", 'a') as writer:
                    writer.open()
                    writer.open()
        finally:
            if Path(".test/async.txt").exists(): Path(".test/async.txt").unlink()

    def test_async_write_exception(self):
        writer = AsyncFileWriter(".test/async.txt", 'a')
        with self.assertRaises(Exception):
            writer.async_write("a")

if __name__ == '__main__':
    unittest.main()