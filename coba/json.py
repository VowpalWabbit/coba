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

        if hasattr(obj, "__to_json_obj__") and callable(obj.__to_json_obj__):

            all_bases = [c.__name__ for c in inspect.getmro(obj.__class__)]
            json_obj  = obj.__to_json_obj__()

            JS_index = all_bases.index('JsonSerializable')

            json_obj['__type__'] = all_bases[0] if JS_index == 1 else all_bases[0:JS_index]

            return json_obj

        return super().default(obj)

class CobaJsonDecoder:

    def __init__(self, *args, **kwargs):
        self._decoder = json.JSONDecoder(object_hook=self.object_hook, *args, **kwargs)

    def decode(self, json_txt: str, types: Sequence[Type[JsonSerializable]] = []) -> Any:

        #in order to avoid circular imports we can't actually import
        #the types ourselves (since those types will likely also be importing this
        # module). So, instead we rely on calling modules to provide us with all
        # types that we need to decode a given json string with our types
        self._known_types = { tipe.__name__:tipe for tipe in types }
        
        return self._decoder.decode(json_txt)

    def object_hook(self, json_obj: Dict[str,Any]) -> Any:

        __type__  = json_obj.get('__type__', [])
        __types__ = [__type__] if isinstance(__type__,str) else __type__

        for __type__ in __types__:
            if __type__ in self._known_types:
                return self._known_types[__type__].__from_json_obj__(json_obj)

        return json_obj