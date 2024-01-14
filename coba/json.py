import json
from math import isfinite
from typing import Any
from coba.registry import CobaRegistry

def dumps_registered(o:Any):
    #I think I just support one use case:
    #  (1) getstate/setstate
    # If I support more use cases things start to become kind of brittle or the json becomes bloated.
    # e.g., it is not clear how I'd differentiate between this and the default constructor behavior.

    name = CobaRegistry.registry.get(type(o))
    if name in CobaRegistry.setstate:
        try:
            return {name: o.__getstate__()}
        except:
            pass
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def loads_registered(o:dict):
    if len(o) == 1:
        name = next(iter(o.keys()))
        if name in CobaRegistry.setstate:
            klass = CobaRegistry.registry.get(name)
            obj = klass.__new__(klass)
            obj.__setstate__(o[name])
            return obj
    return o

def loads(s, *, cls=None, object_hook=loads_registered, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None, **kw) -> Any:
    return json.loads(s, cls=cls, object_hook=object_hook, parse_float=parse_float, parse_int=parse_int, parse_constant=parse_constant, object_pairs_hook=object_pairs_hook, **kw)

def dumps(obj, *, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, cls=None, indent=None, separators=None, default=dumps_registered, sort_keys=False, **kw) -> str:
    def _default(o: Any) -> Any:
        if hasattr(o,'ndim'): return o.tolist()
        if default: return default(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    return json.dumps(obj, skipkeys=skipkeys, ensure_ascii=ensure_ascii, check_circular=check_circular, allow_nan=allow_nan, cls=cls, indent=indent, separators=separators, default=_default, sort_keys=sort_keys, **kw)
