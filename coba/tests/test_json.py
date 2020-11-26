import unittest

from typing import Dict,Any

from coba.json import JsonSerializable, CobaJsonEncoder, CobaJsonDecoder

class SimpleJsonSerializable(JsonSerializable):
    
    def __init__(self, a) -> None:
        self._a = a

    def __to_json__(self) -> Dict[str,Any]:
        return { 'a': self._a }

    @staticmethod
    def __from_json__(obj: Dict[str,Any]) -> 'SimpleJsonSerializable':
        return SimpleJsonSerializable(obj['a'])

class JsonEncoder_Tests(unittest.TestCase):
    def test_simple_encode(self):
        obj               = SimpleJsonSerializable(1)
        actual_json_txt   = CobaJsonEncoder().encode(obj)
        expected_json_txt = '{"a": 1, "__type__": "SimpleJsonSerializable"}'

        self.assertEqual(actual_json_txt, expected_json_txt)

    def test_nested_encode(self):

        obj               = SimpleJsonSerializable(SimpleJsonSerializable(1))
        actual_json_txt   = CobaJsonEncoder().encode(obj)
        expected_json_txt = '{"a": {"a": 1, "__type__": "SimpleJsonSerializable"}, "__type__": "SimpleJsonSerializable"}'

        self.assertEqual(actual_json_txt, expected_json_txt)

class JsonDecoder_Tests(unittest.TestCase):

    def test_simple_decode(self):
        json_txt = '{"a": 1, "__type__": "SimpleJsonSerializable"}'
        obj = CobaJsonDecoder(types=[SimpleJsonSerializable]).decode(json_txt)

        self.assertEqual(obj._a, 1)

    def test_nested_decode(self):
        json_txt = '{"a": {"a": 1, "__type__": "SimpleJsonSerializable"}, "__type__": "SimpleJsonSerializable"}'
        obj = CobaJsonDecoder(types=[SimpleJsonSerializable]).decode(json_txt)

        self.assertEqual(obj._a._a, 1)

if __name__ == '__main__':
    unittest.main()