import unittest
import unittest.mock
import time
import pickle

import multiprocessing as mp
import threading as mt

from typing import Iterable, Any

from coba.utilities  import PackageChecker
from coba.exceptions import CobaException
from coba.pipes      import Identity

from coba.pipes.multiprocessing import Multiprocessor, Pickler, Unpickler, EventSetter, Foreach, Safe

spawn_context = mp.get_context("spawn")

class NotPicklableFilter:
    def __init__(self):
        self._a = lambda : None

    def filter(self, item):
        return 'a'

class LiteralFilter:
    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        yield "A"

class ParamsFilter:
    @property
    def params(self):
        return {'param':1}

    def filter(self, item):
        pass

class ProcessNameFilter:
    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        yield f"pid-{spawn_context.current_process().pid}"

class BarrierNameFilter:
    def __init__(self, n):
        self._barrier = spawn_context.Barrier(n)

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        self._barrier.wait()
        yield f"pid-{spawn_context.current_process().pid}"

class ExceptionFilter:
    def __init__(self, exc = Exception("Exception Filter")):
        self._exc = exc

    def filter(self, items: Iterable[Any]) -> Iterable[Any]:
        raise self._exc

class Foreach_Tests(unittest.TestCase):
    def test_simple(self):
        #Why does it work this way? What problem was I trying to solve?
        self.assertEqual(list(Foreach(Identity()).filter([1])),[1])
        self.assertEqual(list(Foreach(Identity()).filter(['a'])),['a'])
        self.assertEqual(list(Foreach(Identity()).filter([[1,2,3]])),[[1,2,3]])
        self.assertEqual(list(Foreach(Identity()).filter([iter([1,2,3])])),[1,2,3])

class Multiprocessor_Tests(unittest.TestCase):

    def test_single_process(self):
        items = list(Multiprocessor(Identity(), 1, None).filter([[0,1,2,3]]))
        self.assertEqual(items, [[0,1,2,3]])

    def test_read_wait(self):
        items = list(Multiprocessor(Identity(), 1, None, read_wait=True).filter([[0,1,2,3]]))
        self.assertEqual(items, [[0,1,2,3]])

    @unittest.skipUnless(PackageChecker.torch(strict=False), "Requires pytorch")
    def test_torch(self):
        #https://stackoverflow.com/a/74364648/1066291
        import torch
        iterator = Multiprocessor(Identity(), 2, None, read_wait=True).filter(torch.tensor([0,1,2,3]))
        item = next(iterator)
        time.sleep(.2)
        remaining = list(iterator)

    def test_params(self):
        expected = ParamsFilter().params
        actual   = Multiprocessor(ParamsFilter(), 1, 1).params
        self.assertEqual(expected,actual)

    def test_singleprocess_singleperchild(self):
        items = list(Multiprocessor(ProcessNameFilter(), 1, 1).filter([0,1,2,3]))
        self.assertEqual(len(set(items)), 4)

    def test_singleprocess_multiperchild(self):
        items = list(Multiprocessor(ProcessNameFilter(), 1, 0).filter(range(4)))
        self.assertEqual(len(set(items)), 1)

    def test_multiprocess_singleperchild(self):
        items = list(Multiprocessor(ProcessNameFilter(), 2, 1).filter(range(4)))
        self.assertEqual(len(set(items)), 4)

    def test_multiprocess_multiperchild(self):
        items = list(Multiprocessor(BarrierNameFilter(2), 2).filter(range(2)))
        self.assertEqual(len(set(items)), 2)

    def test_filter_exception(self):
        with self.assertRaises(Exception) as e:
            list(Multiprocessor(ExceptionFilter(), 2, 1).filter(range(4)))
        self.assertIn("Exception Filter", str(e.exception))

    def test_empty_list(self):
        items = list(Multiprocessor(ProcessNameFilter(), 1, 1).filter([]))
        self.assertEqual(len(items), 0)

    def test_items_not_picklable_fails_without_cloudpickle(self):
        with unittest.mock.patch('importlib.util.find_spec', return_value=None):
            with self.assertRaises(CobaException) as e:
                list(Multiprocessor(ProcessNameFilter(), 2, 1).filter([NotPicklableFilter()]))
            self.assertIn("pickle", str(e.exception))

    @unittest.skipUnless(PackageChecker.cloudpickle(strict=False), "Cloudpickle is not installed.")
    def test_items_not_picklable_passes_with_cloudpickle(self):
        out = list(Multiprocessor(LiteralFilter(), 2, 1).filter([lambda: 1]))
        self.assertEqual(out,['A'])

    @unittest.skip("This throws ugly warning messages so we skip")
    def test_filter_not_picklable_fails_without_cloudpickle(self):
        with unittest.mock.patch('importlib.util.find_spec', lambda _: False):
            with self.assertRaises(AttributeError):
                class TestFilter:
                    def filter(self, items):
                        yield 'B'
                out = list(Multiprocessor(TestFilter(), 2, 1).filter([lambda: 1]))
                self.assertEqual(out,['B'])

    @unittest.skipUnless(PackageChecker.cloudpickle(strict=False), "Cloudpickle is not installed.")
    def test_filter_not_picklable_passes_with_cloudpickle(self):
        class TestFilter:
            def filter(self, items):
                yield 'B'
        out = list(Multiprocessor(TestFilter(), 2, 1).filter([lambda: 1]))
        self.assertEqual(out,['B'])

    def test_class_definitions_not_loaded_in_main_fails_without_cloudpickle(self):
        with unittest.mock.patch('importlib.util.find_spec', return_value=None):
            #this makes Test picklable but not loadable by the process
            global Test
            class Test:
                pass

            with self.assertRaises(Exception) as e:
                list(Multiprocessor(ProcessNameFilter(), 2, 1).filter([Test()]*2))

            self.assertIn("unable to find", str(e.exception))

    @unittest.skipUnless(PackageChecker.cloudpickle(strict=False), "Cloudpickle is not installed.")
    def test_class_definitions_not_loaded_in_main_pass_with_cloudpickle(self):
        class Test:
            _a = 'A'
        list(Multiprocessor(ProcessNameFilter(), 2, 1).filter([Test()]*2))

class Pickler_Tests(unittest.TestCase):

    def test_simple_without_cloudpickle(self):
        with unittest.mock.patch('importlib.util.find_spec', return_value=None):
            self.assertEqual(list(Pickler().filter([1,2])), [pickle.dumps(1),pickle.dumps(2)])

    @unittest.skipUnless(PackageChecker.cloudpickle(strict=False), "Cloudpickle is not installed.")
    def test_simple_with_cloudpickle(self):
        import cloudpickle
        self.assertEqual(list(Pickler().filter([1,2])), [cloudpickle.dumps(1),cloudpickle.dumps(2)])

class Unpickler_Tests(unittest.TestCase):

    def test_simple_without_cloudpickle(self):
        with unittest.mock.patch('importlib.util.find_spec', return_value=None):
            self.assertEqual(list(Unpickler().filter([pickle.dumps(1),pickle.dumps(2)])), [1,2])

    @unittest.skipUnless(PackageChecker.cloudpickle(strict=False), "Cloudpickle is not installed.")
    def test_simple_with_cloudpickle(self):
        import cloudpickle
        self.assertEqual(list(Unpickler().filter([cloudpickle.dumps(1),cloudpickle.dumps(2)])), [1,2])

class EventSetter_Tests(unittest.TestCase):
    def test_simple(self):
        event = mt.Event()
        setter = EventSetter(event)

        self.assertFalse(event.is_set())
        self.assertEqual(setter.filter([1,2]),[1,2])
        self.assertTrue(event.is_set())

class Safe_Tests(unittest.TestCase):

    @unittest.skipUnless(PackageChecker.cloudpickle(strict=False), "Cloudpickle is not installed.")
    def test_simple_with_cloudpickle(self):
        out = list(Safe(LiteralFilter()).filter([1]))
        self.assertEqual(['A'], out)

    @unittest.skipUnless(PackageChecker.cloudpickle(strict=False), "Cloudpickle is not installed.")
    def test_simple_with_cloudpickle_and_barrier(self):
        out = list(Safe(BarrierNameFilter(1)).filter([1]))

    def test_simple_without_cloudpickle(self):
        with unittest.mock.patch('importlib.util.find_spec', return_value=None):
            out = list(Safe(LiteralFilter()).filter([1]))
            self.assertEqual(['A'], out)

if __name__ == '__main__':
    unittest.main()