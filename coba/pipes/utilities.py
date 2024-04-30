from collections import Counter
from typing import Sequence

from coba.exceptions import CobaException
from coba.primitives import Pipe

def resolve_params(pipes: Sequence[Pipe]):

    params = []
    for p in pipes:

        if not hasattr(p,'params'):
            continue
        if callable(p.params):
            raise CobaException(f"The params on {type(p).__name__} is not decorated with @property")
        if not isinstance(p.params,dict):
            raise CobaException(f"The params on {type(p).__name__} is not a dict type")

        params.append(p.params)

    counts = Counter([k for p in params for k in p.keys()])
    index  = {}

    def resolve_key_conflicts(key):
        if counts[key] == 1:
            return key
        else:
            index[key] = index.get(key,0)+1
            return f"{key}{index[key]}"

    return { resolve_key_conflicts(k):v for p in params for k,v in p.items() }
