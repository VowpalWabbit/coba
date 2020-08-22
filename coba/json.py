"""The json module contains classes for serializing and deserializing coba classes"""

import json
import inspect

from abc import ABC, abstractmethod
from typing import Dict, Any, Sequence, Type

class JsonSerializable(ABC):
    @abstractmethod
    def __to_json_obj__(self) -> Dict[str,Any]:
        ...

    @staticmethod
    @abstractmethod
    def __from_json_obj__(obj: Dict[str,Any]) -> Any:
        ...

class CobaJsonEncoder(json.JSONEncoder):

    def default(self, obj):

        if getattr(obj, "__to_json_obj__", None):

            all_bases = [c.__name__ for c in inspect.getmro(obj.__class__)]
            json_obj  = obj.__to_json_obj__()
            json_obj['_types'] = all_bases[0:all_bases.index('JsonSerializable')]

            return json_obj

        return super().default(obj)

class CobaJsonDecoder(json.JSONDecoder):

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def decode(self, json_txt: str, types: Sequence[Type[JsonSerializable]] = []) -> Any:

        self._known_types = { t.__name__:t for t in types }
        return super().decode(json_txt)

    def object_hook(self, json_obj: Dict[str,Any]) -> Any:

        types = json_obj.get('_types', [])

        for tipe in types:
            if tipe in self._known_types:
                return self._known_types[tipe].__from_json_obj__(json_obj)

        return json_obj