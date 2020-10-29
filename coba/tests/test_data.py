import unittest

from multiprocessing import current_process
from typing import Iterable, Any

from coba.execution import UniversalLogger, ExecutionContext
from coba.data import Filter, MemorySink, MemorySource, Table, Pipe

class Pipe_Tests(unittest.TestCase):

    class ProcessNameFilter(Filter):
        def filter(self, items: Iterable[Any]) -> Iterable[Any]:

            process_name = current_process().name

            for _ in items:
                ExecutionContext.Logger.log(process_name)
                yield process_name

    class ExceptionFilter(Filter):
        def filter(self, items: Iterable[Any]) -> Iterable[Any]:
            raise Exception("Exception Filter")

    def test_single_process_multitask(self):
        source = MemorySource(list(range(10)))
        sink   = MemorySink()

        Pipe.join(source, [Pipe_Tests.ProcessNameFilter()], sink).run()

        self.assertEqual(sink.items, ['MainProcess']*10)

    def test_singleprocess_singletask(self):
        source = MemorySource(list(range(4)))
        sink   = MemorySink()

        Pipe.join(source, [Pipe_Tests.ProcessNameFilter()], sink).run(1,1)

        self.assertEqual(len(set(sink.items)), 4)

    def test_multiprocess_multitask(self):
        source = MemorySource(list(range(10)))
        sink   = MemorySink()

        Pipe.join(source, [Pipe_Tests.ProcessNameFilter()], sink).run(2)

        self.assertEqual(len(set(sink.items)), 2)

    def test_multiprocess_singletask(self):
        source = MemorySource(list(range(4)))
        sink   = MemorySink()

        Pipe.join(source, [Pipe_Tests.ProcessNameFilter()], sink).run(2,1)

        self.assertEqual(len(set(sink.items)), 4)

    def test_exception_multiprocess(self):
        source = MemorySource(list(range(4)))
        sink   = MemorySink()

        with self.assertRaises(Exception):
            Pipe.join(source, [Pipe_Tests.ExceptionFilter()], sink).run(2,1)

    def test_exception_singleprocess(self):
        source = MemorySource(list(range(4)))
        sink   = MemorySink()

        with self.assertRaises(Exception):
            Pipe.join(source, [Pipe_Tests.ExceptionFilter()], sink).run()

    def test_logging(self):
        
        actual_logs = []

        ExecutionContext.Logger = UniversalLogger(lambda msg,end: actual_logs.append((msg,end)))

        source = MemorySource(list(range(4)))
        sink   = MemorySink()

        Pipe.join(source, [Pipe_Tests.ProcessNameFilter()], sink).run(2,1)

        self.assertEqual(len(actual_logs), 4)
        self.assertEqual(sink.items, [ l[0][20:] for l in actual_logs ] )

class Table_Tests(unittest.TestCase):

    def test_add_row(self):
        table = Table("test", ['a'])

        table.add_row(a='A', b='B')
        table.add_row(a='a', b='B')

        self.assertTrue('A' in table)
        self.assertTrue('a' in table)
        self.assertTrue({'a':'A'} in table)
        self.assertTrue({'a':'a'} in table)

        self.assertEqual(table.get_row('a'), {'a':'a', 'b':'B'})
        self.assertEqual(table.get_row('A'), {'a':'A', 'b':'B'})

    def test_update_row(self):
        table = Table("test", ['a'])

        table.add_row(a='a', b='B')
        table.add_row('a','C')

        self.assertTrue('a' in table)
        self.assertTrue({'a':'a'} in table)
        self.assertFalse({'a':'C'} in table)

        self.assertEqual(table.get_row('a'), {'a':'a', 'b':'C'})

    def test_to_indexed_tuples(self):
        table = Table("test", ['a'])

        table.add_row(a='A', b='B')
        table.add_row(a='a', b='b')

        t = table.to_indexed_tuples()

        self.assertTrue('a' in t)
        self.assertTrue('A' in t)

        self.assertEqual(t['a'].a, 'a')
        self.assertEqual(t['a'].b, 'b')

        self.assertEqual(t['A'].a, 'A')
        self.assertEqual(t['A'].b, 'B')

if __name__ == '__main__':
    unittest.main()